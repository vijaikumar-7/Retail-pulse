import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_FILE = "Sample - Superstore.csv"
DEFAULT_OUTPUT_FILE = "superstore_clean.csv"
REQUIRED_COLUMNS = {
    "Order Date",
    "Ship Date",
    "Sales",
    "Profit",
    "Discount",
    "Quantity",
    "Category",
    "State",
}


def resolve_input_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return (Path.cwd() / DEFAULT_INPUT_FILE).resolve()


def classify_delivery_speed(days_value: float) -> str:
    if pd.isna(days_value):
        return "Unknown"
    if days_value <= 0:
        return "Same day"
    if days_value <= 2:
        return "First Class"
    if days_value <= 4:
        return "Second Class"
    return "Standard"


def handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns
    object_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns

    for column in numeric_columns:
        if cleaned_df[column].isna().any():
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    for column in object_columns:
        if cleaned_df[column].isna().any():
            cleaned_df[column] = cleaned_df[column].fillna("Unknown")

    return cleaned_df


def main() -> None:
    input_path = resolve_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}\n"
            f"Place '{DEFAULT_INPUT_FILE}' in the current directory or pass a file path:\n"
            f"python data_cleaning.py /path/to/your.csv"
        )

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")

    print("\nInitial null value count:")
    print(df.isnull().sum())

    df = handle_null_values(df)

    initial_row_count = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = initial_row_count - len(df)

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    invalid_date_rows = df["Order Date"].isna() | df["Ship Date"].isna()
    invalid_date_count = int(invalid_date_rows.sum())
    if invalid_date_count:
        df = df.loc[~invalid_date_rows].copy()

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    sales_nonzero = df["Sales"].replace(0, pd.NA)
    df["Profit Margin %"] = ((df["Profit"] / sales_nonzero) * 100).fillna(0).round(2)
    df["Discount Impact"] = (df["Sales"] * df["Discount"]).round(2)

    df["Delivery Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Delivery Speed"] = df["Delivery Days"].apply(classify_delivery_speed)

    output_path = input_path.with_name(DEFAULT_OUTPUT_FILE)
    df.to_csv(output_path, index=False)

    print("\nCleaning summary:")
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Rows removed due to invalid dates: {invalid_date_count}")

    print("\nEDA Output")
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nNull counts:")
    print(df.isnull().sum())

    top_categories = (
        df.groupby("Category", dropna=False)["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\nTop 5 categories by total sales:")
    print(top_categories)

    top_states = (
        df.groupby("State", dropna=False)["Profit"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    print("\nTop 5 states by profit:")
    print(top_states)

    correlation_columns = ["Sales", "Profit", "Discount", "Quantity"]
    print("\nCorrelation matrix:")
    print(df[correlation_columns].corr(numeric_only=True))

    total_rows = len(df)
    total_revenue = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    average_profit_margin = df["Profit Margin %"].mean()
    date_min = df["Order Date"].min()
    date_max = df["Order Date"].max()

    print("\nFinal Summary")
    print(f"Total rows: {total_rows:,}")
    print(f"Total revenue: {total_revenue:,.2f}")
    print(f"Total profit: {total_profit:,.2f}")
    print(f"Average profit margin: {average_profit_margin:,.2f}%")
    print(
        "Date range covered: "
        f"{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}"
    )
    print(f"\nCleaned dataset exported to: {output_path}")


if __name__ == "__main__":
    main()
