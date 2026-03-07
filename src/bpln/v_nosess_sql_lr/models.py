"""
Variant v_nosess_sql_lr: no sessions, DuckDB SQL features, logistic regression.

DAG: bronze_events.sql -> user_features -> user_predictions
"""

import bauplan


@bauplan.model(
    columns=[
        "user_id",
        "days_active",
        "total_views",
        "total_carts",
        "total_purchases_pre",
        "avg_price_viewed",
        "n_brands",
        "n_categories",
        "n_products",
        "converted"
    ],
    materialization_strategy="REPLACE",
)
@bauplan.python("3.11", pip={"duckdb": "1.2.1"})
def user_features(
    events=bauplan.Model(
        "bronze_events",
        columns=[
            "event_time_parsed",
            "event_type",
            "product_id",
            "category_code",
            "brand",
            "price",
            "user_id",
        ],
    ),
    cutoff_date=bauplan.Parameter("cutoff_date"),
    label_end_date=bauplan.Parameter("label_end_date"),
    size=bauplan.Parameter("size"),
):
    """Aggregate per-user lifetime features via DuckDB SQL (no sessions), label conversions."""
    import duckdb

    con = duckdb.connect()
    con.register("events", events)

    result = con.execute(
        """
        WITH labels AS (
            SELECT DISTINCT user_id, 1 AS converted
            FROM events
            WHERE event_time_parsed >= CAST($cutoff AS TIMESTAMP)
              AND event_time_parsed <= CAST($label_end AS TIMESTAMP)
              AND event_type = 'purchase'
        ),
        obs AS (
            SELECT *
            FROM events
            WHERE event_time_parsed < CAST($cutoff AS TIMESTAMP)
        ),
        features AS (
            SELECT
                user_id,
                DATE_DIFF('day', MIN(event_time_parsed), MAX(event_time_parsed)) + 1 AS days_active,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS total_views,
                SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS total_carts,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases_pre,
                AVG(price) AS avg_price_viewed,
                COUNT(DISTINCT brand) AS n_brands,
                COUNT(DISTINCT category_code) AS n_categories,
                COUNT(DISTINCT product_id) AS n_products
            FROM obs
            GROUP BY user_id
        )
        SELECT
            f.user_id,
            f.days_active,
            f.total_views,
            f.total_carts,
            f.total_purchases_pre,
            f.avg_price_viewed,
            f.n_brands,
            f.n_categories,
            f.n_products,
            COALESCE(l.converted, 0) AS converted
        FROM features f
        LEFT JOIN labels l ON f.user_id = l.user_id
        """,
        {
            "cutoff": cutoff_date + " 00:00:00",
            "label_end": label_end_date + " 23:59:59",
        },
    ).fetch_arrow_table()

    import time
    if size > 0:
        time.sleep(10)

    return result


@bauplan.model(
    columns=["user_id", "conversion_prob", "predicted_label"],
    materialization_strategy="REPLACE",
)
@bauplan.python(
    "3.11",
    pip={"polars": "1.15.0", "scikit-learn": "1.6.1", "numpy": "1.26.4"},
)
def user_predictions(
    features=bauplan.Model(
        "user_features",
        columns=[
        "user_id",
        "days_active",
        "total_views",
        "total_carts",
        "total_purchases_pre",
        "avg_price_viewed",
        "n_brands",
        "n_categories",
        "n_products",
        "converted"
        ],
    ),
    size=bauplan.Parameter("size"),
):
    """Train logistic regression, predict conversion probability for each user."""
    import polars as pl
    from sklearn.linear_model import LogisticRegression

    df = pl.from_arrow(features)

    feature_cols = [
        "days_active",
        "total_views",
        "total_carts",
        "total_purchases_pre",
        "avg_price_viewed",
        "n_brands",
        "n_categories",
        "n_products",
    ]

    X = df.select(feature_cols).fill_null(0).to_numpy()
    y = df["converted"].to_numpy()

    model = LogisticRegression(max_iter=50, class_weight="balanced")
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = pl.DataFrame(
        {
            "user_id": df["user_id"],
            "conversion_prob": probs,
            "predicted_label": preds,
        }
    )
    import time
    if size > 0:
        time.sleep(10)

    return result.to_arrow()
