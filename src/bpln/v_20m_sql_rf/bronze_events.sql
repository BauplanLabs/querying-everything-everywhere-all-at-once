-- bauplan: materialization_strategy=REPLACE
SELECT
    CAST(event_time AS TIMESTAMP) AS event_time_parsed,
    event_type,
    product_id,
    category_code,
    brand,
    price,
    user_id
FROM bauplan.ecommerce_sessions
WHERE price > 0
  AND user_id IS NOT NULL
  AND event_time IS NOT NULL
LIMIT CASE WHEN $size > 0 THEN $size ELSE 9999999999 END
