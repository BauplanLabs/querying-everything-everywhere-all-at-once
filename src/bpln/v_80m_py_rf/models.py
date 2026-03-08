"""
Variant v_80m_py_rf: 80-min sessions, Polars features, random forest.

DAG: bronze_events -> user_features -> user_predictions
"""

import bauplan


@bauplan.model(
    columns=[
        "event_time_parsed",
        "event_type",
        "product_id",
        "category_code",
        "brand",
        "price",
        "user_id",
    ],
    materialization_strategy="REPLACE",
)
@bauplan.python("3.11", pip={"polars": "1.15.0"})
def bronze_events(
    raw=bauplan.Model(
        "bauplan.ecommerce_sessions",
        columns=[
            "event_time",
            "event_type",
            "product_id",
            "category_code",
            "brand",
            "price",
            "user_id",
        ],
        filter="price > 0",
    ),
    size=bauplan.Parameter("size"),
):
    """Clean raw e-commerce events: parse timestamps, filter nulls, sample."""
    import polars as pl

    df = pl.from_arrow(raw)
    df = df.with_columns(
        pl.col("event_time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S UTC")
        .alias("event_time_parsed")
    )
    df = df.drop_nulls(subset=["user_id", "event_time_parsed"])
    df = df.drop("event_time")
    if size > 0:
        df = df.head(size)

    return df.to_arrow()


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
@bauplan.python("3.11", pip={"polars": "1.15.0"})
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
    """Sessionize with 80-min inactivity gap (Polars), compute per-user features."""
    import polars as pl

    df = pl.from_arrow(events)

    cutoff = pl.lit(cutoff_date + " 00:00:00 UTC").str.to_datetime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    label_end = pl.lit(label_end_date + " 23:59:59 UTC").str.to_datetime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    labels = (
        df.filter(
            (pl.col("event_time_parsed") >= cutoff)
            & (pl.col("event_time_parsed") <= label_end)
            & (pl.col("event_type") == "purchase")
        )
        .select("user_id")
        .unique()
        .with_columns(pl.lit(1).alias("converted"))
    )

    obs = df.filter(pl.col("event_time_parsed") < cutoff)

    obs = obs.sort(["user_id", "event_time_parsed"])
    obs = obs.with_columns(
        (pl.col("event_time_parsed").diff().over("user_id").dt.total_minutes() > 80)
        .fill_null(True)
        .alias("new_session")
    )
    obs = obs.with_columns(
        pl.col("new_session").cum_sum().over("user_id").alias("session_id")
    )

    sessions = obs.group_by(["user_id", "session_id"]).agg(
        pl.col("event_time_parsed").min().alias("session_start"),
        pl.col("event_time_parsed").max().alias("session_end"),
    )
    sessions = sessions.with_columns(
        ((pl.col("session_end") - pl.col("session_start")).dt.total_minutes()).alias(
            "session_duration_min"
        )
    )

    user_sessions = sessions.group_by("user_id").agg(
        pl.len().alias("session_count"),
        pl.col("session_duration_min").mean().alias("avg_session_duration_min"),
    )

    user_events = obs.group_by("user_id").agg(
        (pl.col("event_type") == "view").sum().alias("total_views"),
        (pl.col("event_type") == "cart").sum().alias("total_carts"),
        (pl.col("event_type") == "purchase").sum().alias("total_purchases_pre"),
        pl.col("price").mean().alias("avg_price_viewed"),
        pl.col("brand").n_unique().alias("n_brands"),
    )

    features = user_sessions.join(user_events, on="user_id", how="inner")
    features = features.join(labels, on="user_id", how="left")
    features = features.with_columns(pl.col("converted").fill_null(0))

    import time
    if size > 0:
        time.sleep(10)

    return features.to_arrow()


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
