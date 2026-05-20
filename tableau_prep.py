import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_FILE = "superstore_clean.csv"
OUTPUT_FOLDER = "tableau_data"
REQUIRED_COLUMNS = {
    "Order ID",
    "Order Date",
    "Sales",
    "Profit",
    "Region",
    "State",
    "City",
    "Category",
    "Sub-Category",
    "Customer ID",
    "Customer Name",
    "Segment",
    "Product Name",
    "Discount",
}


def resolve_input_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return (Path.cwd() / DEFAULT_INPUT_FILE).resolve()


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Order Date"] = pd.to_datetime(prepared["Order Date"], errors="coerce")
    prepared = prepared.dropna(subset=["Order Date"]).copy()

    if "Year" not in prepared.columns:
        prepared["Year"] = prepared["Order Date"].dt.year

    if "Month" not in prepared.columns:
        prepared["Month"] = prepared["Order Date"].dt.month

    if "Quarter" not in prepared.columns:
        prepared["Quarter"] = prepared["Order Date"].dt.quarter

    if "Profit Margin %" not in prepared.columns:
        sales_nonzero = prepared["Sales"].replace(0, pd.NA)
        prepared["Profit Margin %"] = ((prepared["Profit"] / sales_nonzero) * 100).fillna(0)

    return prepared


def safe_margin(profit_series: pd.Series, sales_series: pd.Series) -> pd.Series:
    return ((profit_series / sales_series.replace(0, pd.NA)) * 100).fillna(0).round(2)


def create_executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Year", dropna=False)
        .agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
            total_orders=("Order ID", "nunique"),
        )
        .reset_index()
        .sort_values("Year")
    )
    summary["avg_order_value"] = (
        summary["total_sales"] / summary["total_orders"].replace(0, pd.NA)
    ).fillna(0)
    summary["profit_margin_pct"] = safe_margin(summary["total_profit"], summary["total_sales"])
    summary["yoy_growth_pct"] = (
        summary["total_sales"].pct_change() * 100
    ).round(2)

    return summary.rename(
        columns={
            "Year": "Year",
            "total_sales": "Total Sales",
            "total_profit": "Total Profit",
            "total_orders": "Total Orders",
            "avg_order_value": "Avg Order Value",
            "profit_margin_pct": "Profit Margin %",
            "yoy_growth_pct": "YoY Growth %",
        }
    ).round(
        {
            "Total Sales": 2,
            "Total Profit": 2,
            "Avg Order Value": 2,
        }
    )


def create_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    regional = (
        df.groupby(
            ["Region", "State", "City", "Category", "Sub-Category"],
            dropna=False,
        )
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique"),
        )
        .reset_index()
    )
    regional["Profit Margin %"] = safe_margin(regional["Profit"], regional["Sales"])

    return regional[
        [
            "Region",
            "State",
            "City",
            "Category",
            "Sub-Category",
            "Sales",
            "Profit",
            "Profit Margin %",
            "Orders",
        ]
    ].round(
        {
            "Sales": 2,
            "Profit": 2,
        }
    )


def create_customer_segments(df: pd.DataFrame) -> pd.DataFrame:
    customers = (
        df.groupby(["Customer ID", "Customer Name", "Segment"], dropna=False)
        .agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order ID", "nunique"),
            first_order_date=("Order Date", "min"),
            last_order_date=("Order Date", "max"),
        )
        .reset_index()
    )
    customers["avg_order_value"] = (
        customers["total_sales"] / customers["order_count"].replace(0, pd.NA)
    ).fillna(0)
    customers["customer_lifetime_days"] = (
        customers["last_order_date"] - customers["first_order_date"]
    ).dt.days

    customers["first_order_date"] = customers["first_order_date"].dt.strftime("%Y-%m-%d")
    customers["last_order_date"] = customers["last_order_date"].dt.strftime("%Y-%m-%d")

    return customers.rename(
        columns={
            "total_sales": "Total Sales",
            "total_profit": "Total Profit",
            "order_count": "Order Count",
            "avg_order_value": "Avg Order Value",
            "first_order_date": "First Order Date",
            "last_order_date": "Last Order Date",
            "customer_lifetime_days": "Customer Lifetime (days)",
        }
    )[
        [
            "Customer ID",
            "Customer Name",
            "Segment",
            "Total Sales",
            "Total Profit",
            "Order Count",
            "Avg Order Value",
            "First Order Date",
            "Last Order Date",
            "Customer Lifetime (days)",
        ]
    ].round(
        {
            "Total Sales": 2,
            "Total Profit": 2,
            "Avg Order Value": 2,
        }
    )


def create_time_trends(df: pd.DataFrame) -> pd.DataFrame:
    trends = (
        df.groupby(["Year", "Month", "Quarter", "Category"], dropna=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique"),
        )
        .reset_index()
        .sort_values(["Category", "Year", "Month"])
    )
    trends["MoM Growth %"] = (
        trends.groupby("Category")["Sales"].pct_change() * 100
    ).round(2)

    return trends[
        ["Year", "Month", "Quarter", "Category", "Sales", "Profit", "Orders", "MoM Growth %"]
    ].round(
        {
            "Sales": 2,
            "Profit": 2,
        }
    )


def create_product_performance(df: pd.DataFrame) -> pd.DataFrame:
    products = (
        df.groupby(["Category", "Sub-Category", "Product Name"], dropna=False)
        .agg(
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
            order_count=("Order ID", "nunique"),
            avg_discount=("Discount", "mean"),
        )
        .reset_index()
    )
    products["profit_margin_pct"] = safe_margin(products["total_profit"], products["total_sales"])

    return products.rename(
        columns={
            "total_sales": "Total Sales",
            "total_profit": "Total Profit",
            "profit_margin_pct": "Profit Margin %",
            "order_count": "Order Count",
            "avg_discount": "Avg Discount",
        }
    )[
        [
            "Category",
            "Sub-Category",
            "Product Name",
            "Total Sales",
            "Total Profit",
            "Profit Margin %",
            "Order Count",
            "Avg Discount",
        ]
    ].round(
        {
            "Total Sales": 2,
            "Total Profit": 2,
            "Avg Discount": 4,
        }
    )


def save_output(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    input_path = resolve_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}\n"
            f"Place '{DEFAULT_INPUT_FILE}' in the current directory or pass a file path:\n"
            f"python tableau_prep.py /path/to/superstore_clean.csv"
        )

    print(f"Loading cleaned dataset from: {input_path}")
    df = pd.read_csv(input_path)
    ensure_required_columns(df)
    df = prepare_dataframe(df)

    output_dir = (Path.cwd() / OUTPUT_FOLDER).resolve()
    output_dir.mkdir(exist_ok=True)

    outputs = {
        "executive_summary.csv": create_executive_summary(df),
        "regional_performance.csv": create_regional_performance(df),
        "customer_segments.csv": create_customer_segments(df),
        "time_trends.csv": create_time_trends(df),
        "product_performance.csv": create_product_performance(df),
    }

    print("\nTableau Prep Output")
    for filename, output_df in outputs.items():
        saved_path = save_output(output_df, output_dir, filename)
        print(f"{filename}: {len(output_df):,} rows saved to {saved_path}")

    print(f"\nAll Tableau-ready files saved to: {output_dir}")


if __name__ == "__main__":
    main()
