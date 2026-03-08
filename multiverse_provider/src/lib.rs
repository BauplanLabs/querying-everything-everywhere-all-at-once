mod provider;

use pyo3::prelude::*;

#[pymodule]
fn multiverse_provider(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<provider::MultiverseTable>()?;
    m.add_class::<provider::S3MultiverseTable>()?;
    m.add_class::<provider::LocalMultiverseTable>()?;
    m.add_function(wrap_pyfunction!(provider::query_native, m)?)?;
    Ok(())
}
