-- name: yearly_kpis
-- 1. Total revenue, profit, orders, and average order value by year
SELECT
    "Year" AS year,
    ROUND(SUM("Sales"), 2) AS total_revenue,
    ROUND(SUM("Profit"), 2) AS total_profit,
    COUNT(DISTINCT "Order ID") AS total_orders,
    ROUND(SUM("Sales") * 1.0 / COUNT(DISTINCT "Order ID"), 2) AS average_order_value
FROM superstore
GROUP BY "Year"
ORDER BY "Year";

-- name: top_customers
-- 2. Top 10 customers by total revenue with their profit margin %
SELECT
    "Customer ID" AS customer_id,
    "Customer Name" AS customer_name,
    ROUND(SUM("Sales"), 2) AS total_revenue,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(
        CASE
            WHEN SUM("Sales") = 0 THEN 0
            ELSE (SUM("Profit") * 100.0 / SUM("Sales"))
        END,
        2
    ) AS profit_margin_pct
FROM superstore
GROUP BY "Customer ID", "Customer Name"
ORDER BY total_revenue DESC, total_profit DESC
LIMIT 10;

-- name: region_category_crosstab
-- 3. Sales and profit by region and category (cross-tab style)
SELECT
    "Region" AS region,
    ROUND(SUM(CASE WHEN "Category" = 'Furniture' THEN "Sales" ELSE 0 END), 2) AS furniture_sales,
    ROUND(SUM(CASE WHEN "Category" = 'Furniture' THEN "Profit" ELSE 0 END), 2) AS furniture_profit,
    ROUND(SUM(CASE WHEN "Category" = 'Office Supplies' THEN "Sales" ELSE 0 END), 2) AS office_supplies_sales,
    ROUND(SUM(CASE WHEN "Category" = 'Office Supplies' THEN "Profit" ELSE 0 END), 2) AS office_supplies_profit,
    ROUND(SUM(CASE WHEN "Category" = 'Technology' THEN "Sales" ELSE 0 END), 2) AS technology_sales,
    ROUND(SUM(CASE WHEN "Category" = 'Technology' THEN "Profit" ELSE 0 END), 2) AS technology_profit,
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit
FROM superstore
GROUP BY "Region"
ORDER BY total_sales DESC;

-- name: monthly_revenue_growth
-- 4. Month-over-month revenue growth rate for each year
WITH monthly_revenue AS (
    SELECT
        "Year" AS year,
        "Month" AS month,
        SUM("Sales") AS monthly_revenue
    FROM superstore
    GROUP BY "Year", "Month"
),
monthly_growth AS (
    SELECT
        year,
        month,
        ROUND(monthly_revenue, 2) AS monthly_revenue,
        LAG(monthly_revenue) OVER (
            PARTITION BY year
            ORDER BY month
        ) AS previous_month_revenue
    FROM monthly_revenue
)
SELECT
    year,
    month,
    monthly_revenue,
    ROUND(previous_month_revenue, 2) AS previous_month_revenue,
    ROUND(
        CASE
            WHEN previous_month_revenue IS NULL OR previous_month_revenue = 0 THEN NULL
            ELSE ((monthly_revenue - previous_month_revenue) * 100.0 / previous_month_revenue)
        END,
        2
    ) AS revenue_growth_pct
FROM monthly_growth
ORDER BY year, month;

-- name: negative_margin_products
-- 5. Products with negative profit margin (discount killing profitability)
SELECT
    "Product ID" AS product_id,
    "Product Name" AS product_name,
    "Category" AS category,
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(AVG("Discount") * 100.0, 2) AS avg_discount_pct,
    ROUND(
        CASE
            WHEN SUM("Sales") = 0 THEN 0
            ELSE (SUM("Profit") * 100.0 / SUM("Sales"))
        END,
        2
    ) AS profit_margin_pct
FROM superstore
GROUP BY "Product ID", "Product Name", "Category"
HAVING profit_margin_pct < 0
ORDER BY profit_margin_pct ASC, total_sales DESC;

-- name: segment_analysis
-- 6. Customer segment analysis: average order value, frequency, total revenue per segment
SELECT
    "Segment" AS segment,
    COUNT(DISTINCT "Order ID") AS order_frequency,
    ROUND(SUM("Sales"), 2) AS total_revenue,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(SUM("Sales") * 1.0 / COUNT(DISTINCT "Order ID"), 2) AS average_order_value,
    COUNT(DISTINCT "Customer ID") AS unique_customers
FROM superstore
GROUP BY "Segment"
ORDER BY total_revenue DESC;

-- name: state_profit_margin_rank
-- 7. Top 5 and bottom 5 states by profit margin %
WITH state_profitability AS (
    SELECT
        "State" AS state,
        ROUND(SUM("Sales"), 2) AS total_sales,
        ROUND(SUM("Profit"), 2) AS total_profit,
        ROUND(
            CASE
                WHEN SUM("Sales") = 0 THEN 0
                ELSE (SUM("Profit") * 100.0 / SUM("Sales"))
            END,
            2
        ) AS profit_margin_pct
    FROM superstore
    GROUP BY "State"
),
top_states AS (
    SELECT
        'Top 5' AS ranking_group,
        state,
        total_sales,
        total_profit,
        profit_margin_pct
    FROM state_profitability
    ORDER BY profit_margin_pct DESC, total_profit DESC
    LIMIT 5
),
bottom_states AS (
    SELECT
        'Bottom 5' AS ranking_group,
        state,
        total_sales,
        total_profit,
        profit_margin_pct
    FROM state_profitability
    ORDER BY profit_margin_pct ASC, total_profit ASC
    LIMIT 5
)
SELECT
    ranking_group,
    state,
    total_sales,
    total_profit,
    profit_margin_pct
FROM (
    SELECT * FROM top_states
    UNION ALL
    SELECT * FROM bottom_states
)
ORDER BY
    CASE ranking_group
        WHEN 'Top 5' THEN 1
        ELSE 2
    END,
    profit_margin_pct DESC;

-- name: ship_mode_impact
-- 8. Shipping mode impact: average delivery days, total revenue, profit margin by ship mode
SELECT
    "Ship Mode" AS ship_mode,
    ROUND(AVG("Delivery Days"), 2) AS average_delivery_days,
    COUNT(DISTINCT "Order ID") AS total_orders,
    ROUND(SUM("Sales"), 2) AS total_revenue,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(
        CASE
            WHEN SUM("Sales") = 0 THEN 0
            ELSE (SUM("Profit") * 100.0 / SUM("Sales"))
        END,
        2
    ) AS profit_margin_pct
FROM superstore
GROUP BY "Ship Mode"
ORDER BY total_revenue DESC;
