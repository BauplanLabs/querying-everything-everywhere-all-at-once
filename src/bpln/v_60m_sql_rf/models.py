"""
Variant v_60m_sql_rf: 60-min sessions, DuckDB SQL features, random forest.

DAG: bronze_events.sql -> user_features -> user_predictions
"""

import bauplan


@bauplan.model(
    columns=[
        "user_id",
        "session_count",
        "avg_session_duration_min",
        "total_views",
        "total_carts",
        "total_purchases_pre",
        "avg_price_viewed",
        "n_brands",
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
            "brand",
            "price",
            "user_id",
        ],
    ),
    cutoff_date=bauplan.Parameter("cutoff_date"),
    label_end_date=bauplan.Parameter("label_end_date"),
    size=bauplan.Parameter("size"),
):
    """Sessionize with 60-min inactivity gap (DuckDB SQL), compute per-user features."""
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
        with_gap AS (
            SELECT *,
                CASE WHEN
                    epoch(event_time_parsed - LAG(event_time_parsed)
                        OVER (PARTITION BY user_id ORDER BY event_time_parsed)) / 60.0 > 60
                    OR LAG(event_time_parsed)
                        OVER (PARTITION BY user_id ORDER BY event_time_parsed) IS NULL
                THEN 1 ELSE 0 END AS new_session
            FROM obs
        ),
        with_session AS (
            SELECT *,
                SUM(new_session) OVER (
                    PARTITION BY user_id
                    ORDER BY event_time_parsed
                    ROWS UNBOUNDED PRECEDING
                ) AS session_id
            FROM with_gap
        ),
        session_stats AS (
            SELECT user_id, session_id,
                MIN(event_time_parsed) AS session_start,
                MAX(event_time_parsed) AS session_end
            FROM with_session
            GROUP BY user_id, session_id
        ),
        user_sessions AS (
            SELECT user_id,
                COUNT(*) AS session_count,
                AVG(epoch(session_end - session_start) / 60.0) AS avg_session_duration_min
            FROM session_stats
            GROUP BY user_id
        ),
        user_events AS (
            SELECT user_id,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS total_views,
                SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS total_carts,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases_pre,
                AVG(price) AS avg_price_viewed,
                COUNT(DISTINCT brand) AS n_brands
            FROM with_session
            GROUP BY user_id
        )
        SELECT
            us.user_id,
            us.session_count,
            us.avg_session_duration_min,
            ue.total_views,
            ue.total_carts,
            ue.total_purchases_pre,
            ue.avg_price_viewed,
            ue.n_brands,
            COALESCE(l.converted, 0) AS converted
        FROM user_sessions us
        JOIN user_events ue ON us.user_id = ue.user_id
        LEFT JOIN labels l ON us.user_id = l.user_id
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
        "session_count",
        "avg_session_duration_min",
        "total_views",
        "total_carts",
        "total_purchases_pre",
        "avg_price_viewed",
        "n_brands",
        "converted"
        ],
    ),
    size=bauplan.Parameter("size"),
):
    """Train random forest, predict conversion probability for each user."""
    import polars as pl
    from sklearn.ensemble import RandomForestClassifier

    df = pl.from_arrow(features)

    feature_cols = [
        "session_count",
        "avg_session_duration_min",
        "total_views",
        "total_carts",
        "total_purchases_pre",
        "avg_price_viewed",
        "n_brands",
    ]

    X = df.select(feature_cols).fill_null(0).to_numpy()
    y = df["converted"].to_numpy()

    model = RandomForestClassifier(n_estimators=10, class_weight="balanced", random_state=42)
    import numpy as np
    if len(np.unique(y)) < 2:
        # Single class — assign deterministic predictions
        only_class = y[0]
        probs = np.full(len(y), float(only_class))
        preds = np.full(len(y), int(only_class))
    else:
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
