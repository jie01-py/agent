# Sample Dataset: E-commerce Sales Data

This directory contains a sample e-commerce sales dataset for demonstrating the InsightFlow pipeline.

## File: sales_data.csv

A synthetic dataset simulating e-commerce order records, containing approximately 200 rows.

### Columns

| Column | Type | Description |
|--------|------|-------------|
| order_id | int | Unique order identifier |
| date | str | Order date (YYYY-MM-DD) |
| customer_id | str | Customer identifier |
| product_category | str | Product category |
| product_name | str | Product name |
| quantity | int | Order quantity |
| unit_price | float | Unit price (CNY) |
| discount | float | Discount rate (0-1), some missing |
| shipping_city | str | Shipping destination city |
| payment_method | str | Payment method |
| rating | float | Customer rating (1-5), some missing |

### Data Quality Notes

- `rating` column has ~15% missing values (simulating customers who didn't rate)
- `discount` column has ~10% missing values
- `unit_price` contains a few outlier values (extremely high prices)
- Suitable for demonstrating data cleaning, statistical analysis, and visualization
