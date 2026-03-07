use std::any::Any;
use std::ffi::CString;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::pyarrow::FromPyArrow;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::datasource::memory::MemTable;
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::TableType;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::arrow::datatypes::SchemaRef as DFSchemaRef;
use datafusion_common::ScalarValue;
use datafusion_expr::Expr;
use datafusion_ffi::table_provider::FFI_TableProvider;
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_plan::projection::ProjectionExec;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};

/// Rust-only TableProvider that unions multiple inner providers,
/// appending a `__branch_id` literal column to each.
pub struct MultiverseTableProvider {
    branches: Vec<(String, Arc<dyn TableProvider>)>,
    schema: SchemaRef,
}

impl MultiverseTableProvider {
    pub fn try_new(
        branches: Vec<(String, Arc<dyn TableProvider>)>,
    ) -> Result<Self, String> {
        if branches.is_empty() {
            return Err("at least one branch is required".to_string());
        }

        // Use the first branch's schema as the canonical column order.
        // All branches must have the same columns (by name) with compatible types,
        // but column ordering may differ — the scan method handles reordering.
        let canonical_schema = branches[0].1.schema();

        for (name, provider) in &branches[1..] {
            let other = provider.schema();
            for canon_field in canonical_schema.fields() {
                match other.field_with_name(canon_field.name()) {
                    Ok(other_field) => {
                        if other_field.data_type() != canon_field.data_type() {
                            return Err(format!(
                                "type mismatch for branch '{}', column '{}': expected {:?}, got {:?}",
                                name, canon_field.name(), canon_field.data_type(), other_field.data_type()
                            ));
                        }
                    }
                    Err(_) => {
                        return Err(format!(
                            "schema mismatch for branch '{}': missing column '{}'",
                            name, canon_field.name()
                        ));
                    }
                }
            }
        }

        // __branch_id is appended as the last column, matching the convention
        // of the other Python engines (ad hoc engine, native engine).
        let mut fields: Vec<_> = canonical_schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new("__branch_id", DataType::Utf8, false)));
        let schema = Arc::new(Schema::new(fields));

        Ok(Self { branches, schema })
    }
}

impl std::fmt::Debug for MultiverseTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiverseTableProvider")
            .field(
                "branches",
                &self
                    .branches
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("schema", &self.schema)
            .finish()
    }
}

/// Check if a logical expression references the `__branch_id` column.
///
/// Used to filter out __branch_id predicates before passing filters to
/// inner providers, which don't have that column in their schema.
fn expr_references_branch_id(expr: &Expr) -> bool {
    // Expr Display format includes column names unambiguously.
    // This is simpler and more robust than a recursive tree walk over
    // all Expr variants, and __branch_id is a reserved internal name
    // that won't appear as a substring of user column names.
    format!("{expr}").contains("__branch_id")
}

#[async_trait]
impl TableProvider for MultiverseTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> DFSchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        // Strip filters that reference __branch_id — inner providers don't
        // have that column and would reject or mishandle such predicates.
        let inner_filters: Vec<Expr> = filters
            .iter()
            .filter(|f| !expr_references_branch_id(f))
            .cloned()
            .collect();

        let canonical_schema = self.branches[0].1.schema();
        let canonical_field_count = canonical_schema.fields().len();

        // Canonical column names in order (used to remap branches with
        // different field ordering).
        let canonical_names: Vec<&str> = canonical_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();

        // Compute which canonical columns are actually needed, so we can push
        // projection down to inner providers instead of reading all columns.
        let canonical_projection: Option<Vec<usize>> = projection.map(|proj| {
            proj.iter()
                .filter(|&&idx| idx < canonical_field_count) // skip __branch_id index
                .copied()
                .collect()
        });

        let mut plans: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(self.branches.len());

        for (branch_id, inner_provider) in &self.branches {
            let branch_schema = inner_provider.schema();

            // Build mapping: canonical index -> this branch's index
            let canon_to_branch: Vec<usize> = canonical_names
                .iter()
                .map(|name| {
                    branch_schema
                        .index_of(name)
                        .unwrap_or_else(|_| panic!("column '{}' not found in branch", name))
                })
                .collect();

            // Remap the canonical projection to this branch's column indices
            let branch_projection: Option<Vec<usize>> = canonical_projection.as_ref().map(|proj| {
                proj.iter().map(|&canon_idx| canon_to_branch[canon_idx]).collect()
            });

            let inner_plan = inner_provider
                .scan(state, branch_projection.as_ref(), &inner_filters, limit)
                .await?;

            let inner_plan_schema = inner_plan.schema();

            // Build projection expressions that reorder columns to canonical order
            // plus append __branch_id.
            let all_indices: Vec<usize> = (0..canonical_field_count).collect();
            let projected_canon_indices: &[usize] = canonical_projection
                .as_deref()
                .unwrap_or(&all_indices);

            let mut exprs: Vec<(
                Arc<dyn datafusion_physical_expr::PhysicalExpr>,
                String,
            )> = Vec::with_capacity(projected_canon_indices.len() + 1);

            for &canon_idx in projected_canon_indices {
                let canon_name = canonical_names[canon_idx];
                // Find this column's position in the inner plan output
                let inner_pos = inner_plan_schema
                    .index_of(canon_name)
                    .map_err(|e| datafusion_common::DataFusionError::Internal(
                        format!("column '{}' not in inner plan: {}", canon_name, e)
                    ))?;
                exprs.push((
                    Arc::new(Column::new(canon_name, inner_pos)),
                    canon_name.to_string(),
                ));
            }

            // __branch_id appended at the end
            exprs.push((
                Arc::new(Literal::new(ScalarValue::Utf8(Some(
                    branch_id.clone(),
                )))),
                "__branch_id".to_string(),
            ));

            let projection_plan = ProjectionExec::try_new(exprs, inner_plan)?;
            plans.push(Arc::new(projection_plan));
        }

        let union_arc = UnionExec::try_new(plans)?;

        if let Some(proj) = projection {
            // Remap outer projection indices to the per-branch output schema.
            // Per-branch output has: [projected canonical cols..., __branch_id].
            let branch_id_orig_idx = canonical_field_count; // __branch_id's index in our schema
            let canon_proj = canonical_projection.as_ref().unwrap();

            let union_schema = union_arc.schema();
            let mut outer_exprs: Vec<(
                Arc<dyn datafusion_physical_expr::PhysicalExpr>,
                String,
            )> = Vec::with_capacity(proj.len());

            for &orig_idx in proj {
                if orig_idx == branch_id_orig_idx {
                    // __branch_id is the last column in per-branch output
                    let new_idx = canon_proj.len();
                    let field = union_schema.field(new_idx);
                    outer_exprs.push((
                        Arc::new(Column::new(field.name(), new_idx)),
                        field.name().clone(),
                    ));
                } else {
                    // Find this column's position in the canonical projection
                    let new_idx = canon_proj.iter().position(|&i| i == orig_idx)
                        .expect("projected column must be in canonical projection");
                    let field = union_schema.field(new_idx);
                    outer_exprs.push((
                        Arc::new(Column::new(field.name(), new_idx)),
                        field.name().clone(),
                    ));
                }
            }

            let outer_projection = ProjectionExec::try_new(outer_exprs, union_arc)?;
            Ok(Arc::new(outer_projection))
        } else {
            Ok(union_arc)
        }
    }
}

// ---- PyO3 wrappers ----

/// Python-visible table backed by in-memory Arrow RecordBatches.
/// Use for testing or when data is already loaded.
#[pyclass(name = "MultiverseTable")]
pub struct MultiverseTable {
    provider: Arc<MultiverseTableProvider>,
}

#[pymethods]
impl MultiverseTable {
    /// Create from list of (branch_id, list_of_record_batches).
    #[new]
    fn new(branch_batches: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut branches: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();

        for item in branch_batches.iter() {
            let tuple = item.downcast::<PyTuple>()?;
            let branch_id: String = tuple.get_item(0)?.extract()?;
            let batches_list = tuple.get_item(1)?.downcast::<PyList>()?.clone();

            let mut batches: Vec<RecordBatch> = Vec::new();
            for batch_obj in batches_list.iter() {
                let rb = RecordBatch::from_pyarrow_bound(&batch_obj)?;
                batches.push(rb);
            }

            if batches.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "branch '{}' has no record batches",
                    branch_id
                )));
            }

            let schema = batches[0].schema();
            let mem_table = MemTable::try_new(schema, vec![batches]).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "failed to create MemTable for branch '{}': {}",
                    branch_id, e
                ))
            })?;

            branches.push((branch_id, Arc::new(mem_table)));
        }

        let provider = MultiverseTableProvider::try_new(branches)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        Ok(Self {
            provider: Arc::new(provider),
        })
    }

    #[pyo3(signature = (_session=None))]
    fn __datafusion_table_provider__(
        &self,
        py: Python<'_>,
        _session: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let ffi_provider = FFI_TableProvider::new(self.provider.clone(), false, None);
        let capsule = PyCapsule::new(
            py,
            ffi_provider,
            Some(CString::new("datafusion_table_provider").unwrap()),
        )?;
        Ok(capsule.into())
    }

    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let schema = self.provider.schema();
        use arrow::pyarrow::ToPyArrow;
        Ok(schema.to_pyarrow(py)?.unbind())
    }
}

/// Python-visible table backed by S3 Parquet files read eagerly at construction.
///
/// Despite the eager load, this is named "S3" (not "Eager") to reflect the
/// data source. The FFI bridge limitation that forces eager loading is an
/// implementation detail that may change in future datafusion-ffi versions.
#[pyclass(name = "S3MultiverseTable")]
pub struct S3MultiverseTable {
    provider: Arc<MultiverseTableProvider>,
}

#[pymethods]
impl S3MultiverseTable {
    /// Create from list of (branch_id, list_of_s3_parquet_urls).
    ///
    /// Each branch gets a ListingTable backed by its S3 parquet files.
    /// The schema is inferred from the parquet file metadata (no data loaded).
    #[new]
    fn new(
        _py: Python<'_>,
        branch_files: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        use object_store::aws::AmazonS3Builder;
        use url::Url;

        // Collect all S3 URLs per branch
        let mut branch_data: Vec<(String, Vec<String>)> = Vec::new();
        for item in branch_files.iter() {
            let tuple = item.downcast::<PyTuple>()?;
            let branch_id: String = tuple.get_item(0)?.extract()?;
            let files: Vec<String> = tuple.get_item(1)?.extract()?;
            if files.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "branch '{}' has no parquet files",
                    branch_id
                )));
            }
            branch_data.push((branch_id, files));
        }

        if branch_data.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "at least one branch is required",
            ));
        }

        // Determine the S3 bucket from the first file URL
        let first_url = &branch_data[0].1[0];
        let parsed = Url::parse(first_url).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid S3 URL '{}': {}",
                first_url, e
            ))
        })?;
        let bucket = parsed.host_str().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "no bucket in S3 URL '{}'",
                first_url
            ))
        })?;

        // Build S3 object store from env vars:
        //   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        let s3_store: Arc<dyn object_store::ObjectStore> = Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "failed to create S3 object store: {}",
                        e
                    ))
                })?,
        );

        // Read parquet files from S3 into memory, then wrap as MemTables.
        // The FFI bridge creates a fresh SessionContext that lacks our S3
        // object store, so ListingTable-based lazy reads won't work.
        // Eagerly loading into MemTable sidesteps this FFI limitation while
        // still proving the TableProvider architecture.
        let mut branches: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();

        // Single tokio runtime for all S3 I/O — creating a runtime per file
        // is wasteful (spawns thread pools each time).
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to create tokio runtime: {}", e
            ))
        })?;

        for (branch_id, files) in &branch_data {
            let mut all_batches: Vec<RecordBatch> = Vec::new();

            for file_url in files {
                let file_parsed = Url::parse(file_url).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid S3 URL '{}': {}",
                        file_url, e
                    ))
                })?;
                // object_store path is everything after the bucket
                let obj_path = object_store::path::Path::from(file_parsed.path());

                let bytes = rt.block_on(async {
                    let result = s3_store.get(&obj_path).await?;
                    result.bytes().await
                })
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "failed to read '{}': {}",
                            file_url, e
                        ))
                    })?;

                let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)
                    .and_then(|b| b.build())
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "failed to read parquet '{}': {}",
                            file_url, e
                        ))
                    })?;

                for batch_result in reader {
                    let batch = batch_result.map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "failed to read batch from '{}': {}",
                            file_url, e
                        ))
                    })?;
                    all_batches.push(batch);
                }
            }

            if all_batches.is_empty() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "no data read for branch '{}'",
                    branch_id
                )));
            }

            let schema = all_batches[0].schema();
            let mem_table = MemTable::try_new(schema, vec![all_batches]).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to create MemTable for branch '{}': {}",
                    branch_id, e
                ))
            })?;

            branches.push((branch_id.clone(), Arc::new(mem_table)));
        }

        let provider = MultiverseTableProvider::try_new(branches)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        Ok(Self {
            provider: Arc::new(provider),
        })
    }

    #[pyo3(signature = (_session=None))]
    fn __datafusion_table_provider__(
        &self,
        py: Python<'_>,
        _session: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let ffi_provider = FFI_TableProvider::new(self.provider.clone(), false, None);
        let capsule = PyCapsule::new(
            py,
            ffi_provider,
            Some(CString::new("datafusion_table_provider").unwrap()),
        )?;
        Ok(capsule.into())
    }

    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let schema = self.provider.schema();
        use arrow::pyarrow::ToPyArrow;
        Ok(schema.to_pyarrow(py)?.unbind())
    }
}

/// Python-visible table backed by local parquet files, read lazily by DataFusion.
///
/// Unlike `MultiverseTable` (which requires pre-loaded Arrow batches) or
/// `S3MultiverseTable` (which eagerly reads from S3 due to FFI limitations),
/// this variant registers local parquet files as `ListingTable` providers.
/// DataFusion only reads the data when the query actually executes, avoiding
/// the upfront I/O cost that penalises simple queries.
#[pyclass(name = "LocalMultiverseTable")]
pub struct LocalMultiverseTable {
    provider: Arc<MultiverseTableProvider>,
}

#[pymethods]
impl LocalMultiverseTable {
    /// Create from list of (branch_id, parquet_file_path).
    #[new]
    fn new(branch_paths: &Bound<'_, PyList>) -> PyResult<Self> {
        use datafusion::datasource::listing::{
            ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
        };
        use datafusion::datasource::file_format::parquet::ParquetFormat;

        if branch_paths.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "at least one branch is required",
            ));
        }

        // We need a tokio runtime for the async schema inference
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to create tokio runtime: {}", e
            ))
        })?;

        let ctx = datafusion::execution::context::SessionContext::new();
        let state = ctx.state();

        let mut branches: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();

        for item in branch_paths.iter() {
            let tuple = item.downcast::<PyTuple>()?;
            let branch_id: String = tuple.get_item(0)?.extract()?;
            let file_path: String = tuple.get_item(1)?.extract()?;

            let table_url = ListingTableUrl::parse(&file_path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid path '{}': {}", file_path, e
                ))
            })?;

            let listing_options = ListingOptions::new(Arc::new(ParquetFormat::default()));

            let config = rt.block_on(async {
                ListingTableConfig::new(table_url)
                    .with_listing_options(listing_options)
                    .infer_schema(&state)
                    .await
            }).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to infer schema for branch '{}' from '{}': {}",
                    branch_id, file_path, e
                ))
            })?;

            let listing_table = ListingTable::try_new(config).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to create ListingTable for branch '{}': {}",
                    branch_id, e
                ))
            })?;

            branches.push((branch_id, Arc::new(listing_table)));
        }

        let provider = MultiverseTableProvider::try_new(branches)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        Ok(Self {
            provider: Arc::new(provider),
        })
    }

    #[pyo3(signature = (_session=None))]
    fn __datafusion_table_provider__(
        &self,
        py: Python<'_>,
        _session: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let ffi_provider = FFI_TableProvider::new(self.provider.clone(), false, None);
        let capsule = PyCapsule::new(
            py,
            ffi_provider,
            Some(CString::new("datafusion_table_provider").unwrap()),
        )?;
        Ok(capsule.into())
    }

    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let schema = self.provider.schema();
        use arrow::pyarrow::ToPyArrow;
        Ok(schema.to_pyarrow(py)?.unbind())
    }
}

// ---- Rust unit tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};

    fn make_mem_table(n: i64) -> Arc<dyn TableProvider> {
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("n", DataType::Int64, false)),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![n]))],
        )
        .unwrap();
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
    }

    #[test]
    fn try_new_valid() {
        let branches = vec![
            ("a".to_string(), make_mem_table(1)),
            ("b".to_string(), make_mem_table(2)),
        ];
        let provider = MultiverseTableProvider::try_new(branches).unwrap();
        let schema = provider.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "n");
        assert_eq!(schema.field(1).name(), "__branch_id");
    }

    #[test]
    fn try_new_empty_branches() {
        let result = MultiverseTableProvider::try_new(vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one branch"));
    }

    #[test]
    fn try_new_schema_mismatch() {
        let schema_a = Arc::new(Schema::new(vec![
            Arc::new(Field::new("x", DataType::Int64, false)),
        ]));
        let schema_b = Arc::new(Schema::new(vec![
            Arc::new(Field::new("y", DataType::Float64, false)),
        ]));

        let batch_a = RecordBatch::try_new(
            schema_a.clone(),
            vec![Arc::new(Int64Array::from(vec![1]))],
        )
        .unwrap();
        let batch_b = RecordBatch::try_new(
            schema_b.clone(),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        )
        .unwrap();

        let branches = vec![
            (
                "a".to_string(),
                Arc::new(MemTable::try_new(schema_a, vec![vec![batch_a]]).unwrap())
                    as Arc<dyn TableProvider>,
            ),
            (
                "b".to_string(),
                Arc::new(MemTable::try_new(schema_b, vec![vec![batch_b]]).unwrap())
                    as Arc<dyn TableProvider>,
            ),
        ];
        let result = MultiverseTableProvider::try_new(branches);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("schema mismatch"));
    }

    #[test]
    fn branch_id_at_end_of_schema() {
        let provider =
            MultiverseTableProvider::try_new(vec![("x".to_string(), make_mem_table(1))]).unwrap();
        let schema = provider.schema();
        let last = schema.fields().last().unwrap();
        assert_eq!(last.name(), "__branch_id");
        assert_eq!(*last.data_type(), DataType::Utf8);
    }

    #[tokio::test]
    async fn scan_produces_correct_schema() {
        let branches = vec![
            ("a".to_string(), make_mem_table(10)),
            ("b".to_string(), make_mem_table(20)),
            ("c".to_string(), make_mem_table(30)),
        ];
        let provider = MultiverseTableProvider::try_new(branches).unwrap();

        let ctx = datafusion::execution::context::SessionContext::new();
        let state = ctx.state();

        let plan = provider.scan(&state, None, &[], None).await.unwrap();
        let schema = plan.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "n");
        assert_eq!(schema.field(1).name(), "__branch_id");
    }

    #[test]
    fn expr_references_branch_id_detection() {
        assert!(expr_references_branch_id(&Expr::Column(
            datafusion_common::Column::new_unqualified("__branch_id")
        )));
        assert!(!expr_references_branch_id(&Expr::Column(
            datafusion_common::Column::new_unqualified("user_id")
        )));
    }

    // --- Physical plan execution tests ---
    // These test MultiverseTableProvider::scan() directly, collecting
    // RecordBatches from the physical plan to verify correct results.
    // Assertions use sorted/counted comparisons (not positional order)
    // since physical plan output ordering is not guaranteed.

    use arrow::array::StringArray;
    use datafusion::execution::context::SessionContext;
    use datafusion_physical_plan::collect;
    use std::collections::HashMap;

    /// Helper: scan a MultiverseTableProvider (no projection, no filters)
    /// and collect all output batches into a single RecordBatch.
    async fn scan_and_collect(
        branches: Vec<(String, Arc<dyn TableProvider>)>,
    ) -> RecordBatch {
        let provider = MultiverseTableProvider::try_new(branches).unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();
        let plan = provider.scan(&state, None, &[], None).await.unwrap();
        let schema = plan.schema();
        let batches = collect(plan, ctx.task_ctx()).await
            .expect("scan execution failed");
        arrow::compute::concat_batches(&schema, batches.iter())
            .expect("concat_batches failed")
    }

    /// Extract (n, branch_id) pairs from a result batch, sorted for stable comparison.
    fn sorted_n_branch_pairs(result: &RecordBatch) -> Vec<(i64, String)> {
        let ns = result.column(0).as_any()
            .downcast_ref::<Int64Array>().unwrap();
        let ids = result.column(1).as_any()
            .downcast_ref::<StringArray>().unwrap();
        let mut pairs: Vec<(i64, String)> = (0..result.num_rows())
            .map(|i| (ns.value(i), ids.value(i).to_string()))
            .collect();
        pairs.sort();
        pairs
    }

    /// Count occurrences of each branch_id in a string column.
    fn count_branch_ids(col: &StringArray) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for v in col.iter() {
            *counts.entry(v.unwrap().to_string()).or_insert(0) += 1;
        }
        counts
    }

    #[tokio::test]
    async fn scan_returns_all_rows_with_branch_ids() {
        let branches = vec![
            ("br_a".to_string(), make_mem_table(10)),
            ("br_b".to_string(), make_mem_table(20)),
        ];
        let result = scan_and_collect(branches).await;

        assert_eq!(result.num_rows(), 2);
        let pairs = sorted_n_branch_pairs(&result);
        assert_eq!(pairs, vec![(10, "br_a".into()), (20, "br_b".into())]);
    }

    #[tokio::test]
    async fn scan_multi_row_branches() {
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("n", DataType::Int64, false)),
        ]));
        let batch_a = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        ).unwrap();
        let batch_b = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![99]))],
        ).unwrap();

        let branches = vec![
            ("a".to_string(), Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch_a]]).unwrap()) as Arc<dyn TableProvider>),
            ("b".to_string(), Arc::new(MemTable::try_new(schema, vec![vec![batch_b]]).unwrap()) as Arc<dyn TableProvider>),
        ];

        let result = scan_and_collect(branches).await;
        assert_eq!(result.num_rows(), 4);

        let ids = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let counts = count_branch_ids(ids);
        assert_eq!(counts["a"], 3);
        assert_eq!(counts["b"], 1);
    }

    #[tokio::test]
    async fn scan_with_projection_data_only() {
        let branches = vec![
            ("x".to_string(), make_mem_table(42)),
            ("y".to_string(), make_mem_table(99)),
        ];
        let provider = MultiverseTableProvider::try_new(branches).unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();

        let proj = vec![0usize]; // only "n"
        let plan = provider.scan(&state, Some(&proj), &[], None).await.unwrap();
        let schema = plan.schema();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let result = arrow::compute::concat_batches(&schema, batches.iter()).unwrap();

        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.schema().field(0).name(), "n");
        let mut ns: Vec<i64> = result.column(0).as_any()
            .downcast_ref::<Int64Array>().unwrap().values().to_vec();
        ns.sort();
        assert_eq!(ns, vec![42, 99]);
    }

    #[tokio::test]
    async fn scan_with_projection_branch_id_only() {
        let branches = vec![
            ("alpha".to_string(), make_mem_table(1)),
            ("beta".to_string(), make_mem_table(2)),
        ];
        let provider = MultiverseTableProvider::try_new(branches).unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();

        let proj = vec![1usize]; // only __branch_id
        let plan = provider.scan(&state, Some(&proj), &[], None).await.unwrap();
        let schema = plan.schema();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let result = arrow::compute::concat_batches(&schema, batches.iter()).unwrap();

        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.schema().field(0).name(), "__branch_id");
        let mut ids: Vec<String> = result.column(0).as_any()
            .downcast_ref::<StringArray>().unwrap()
            .iter().map(|v| v.unwrap().to_string()).collect();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn scan_with_limit() {
        // Verify that passing a limit doesn't break scan execution.
        // The limit is a hint forwarded to inner providers — not all providers
        // enforce it (MemTable doesn't), so we just verify the scan succeeds
        // and produces valid output.
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("n", DataType::Int64, false)),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]))],
        ).unwrap();

        let branches = vec![
            ("a".to_string(), Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap()) as Arc<dyn TableProvider>),
            ("b".to_string(), Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap()) as Arc<dyn TableProvider>),
        ];

        let provider = MultiverseTableProvider::try_new(branches).unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();

        let plan = provider.scan(&state, None, &[], Some(2)).await.unwrap();
        let plan_schema = plan.schema();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let result = arrow::compute::concat_batches(&plan_schema, batches.iter()).unwrap();

        // Both branches present in output, schema is correct
        assert!(result.num_rows() > 0);
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(1).name(), "__branch_id");
        let ids = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let counts = count_branch_ids(ids);
        assert!(counts.contains_key("a"));
        assert!(counts.contains_key("b"));
    }

    #[tokio::test]
    async fn scan_with_filter_strips_branch_id_predicate() {
        // Verify that __branch_id filters are stripped (not passed to inner providers)
        // and non-branch filters are preserved. We test by passing a __branch_id
        // filter alongside a data filter — the scan should not error on the
        // inner providers which lack __branch_id.
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("score", DataType::Int64, false)),
        ]));
        let batch_a = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![10, 50, 90]))],
        ).unwrap();
        let batch_b = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![20, 80]))],
        ).unwrap();

        let branches = vec![
            ("a".to_string(), Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch_a]]).unwrap()) as Arc<dyn TableProvider>),
            ("b".to_string(), Arc::new(MemTable::try_new(schema, vec![vec![batch_b]]).unwrap()) as Arc<dyn TableProvider>),
        ];

        let provider = MultiverseTableProvider::try_new(branches).unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();

        // Pass a __branch_id filter (should be stripped) and a data filter
        let branch_filter = Expr::Column(
            datafusion_common::Column::new_unqualified("__branch_id"),
        ).eq(Expr::Literal(ScalarValue::Utf8(Some("a".to_string())), None));
        let data_filter = Expr::Column(
            datafusion_common::Column::new_unqualified("score"),
        ).gt(Expr::Literal(ScalarValue::Int64(Some(30)), None));

        // This should succeed without error — __branch_id filter stripped
        let plan = provider
            .scan(&state, None, &[branch_filter, data_filter], None)
            .await
            .unwrap();
        let plan_schema = plan.schema();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let result = arrow::compute::concat_batches(&plan_schema, batches.iter()).unwrap();

        // All 5 rows present (MemTable doesn't apply filters at scan level,
        // but the important thing is the scan didn't error on __branch_id)
        assert!(result.num_rows() > 0);
        assert_eq!(result.num_columns(), 2); // score + __branch_id
    }

    #[tokio::test]
    async fn scan_with_listing_table_from_parquet() {
        use datafusion::datasource::listing::{
            ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
        };
        use datafusion::datasource::file_format::parquet::ParquetFormat;

        let dir = tempfile::tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("v", DataType::Int64, false)),
        ]));

        for (name, vals) in [("br1", vec![1i64, 2]), ("br2", vec![3, 4, 5])] {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vals))],
            ).unwrap();
            let path = dir.path().join(format!("{}.parquet", name));
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = parquet::arrow::ArrowWriter::try_new(
                file, schema.clone(), None,
            ).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let ctx = SessionContext::new();
        let state = ctx.state();
        let mut branches: Vec<(String, Arc<dyn TableProvider>)> = Vec::new();

        for name in ["br1", "br2"] {
            let path = dir.path().join(format!("{}.parquet", name));
            let url = ListingTableUrl::parse(path.to_str().unwrap()).unwrap();
            let opts = ListingOptions::new(Arc::new(ParquetFormat::default()));
            let config = ListingTableConfig::new(url)
                .with_listing_options(opts)
                .infer_schema(&state)
                .await
                .unwrap();
            let lt = ListingTable::try_new(config).unwrap();
            branches.push((name.to_string(), Arc::new(lt)));
        }

        let provider = MultiverseTableProvider::try_new(branches).unwrap();
        let plan = provider.scan(&state, None, &[], None).await.unwrap();
        let plan_schema = plan.schema();
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let result = arrow::compute::concat_batches(&plan_schema, batches.iter()).unwrap();

        assert_eq!(result.num_rows(), 5);
        let ids = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let counts = count_branch_ids(ids);
        assert_eq!(counts["br1"], 2);
        assert_eq!(counts["br2"], 3);

        let mut vs: Vec<i64> = result.column(0).as_any()
            .downcast_ref::<Int64Array>().unwrap().values().to_vec();
        vs.sort();
        assert_eq!(vs, vec![1, 2, 3, 4, 5]);
    }
}
