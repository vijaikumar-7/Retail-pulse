"""
Add SARIMA-based monthly sales forecasting for the cleaned Superstore dataset.

Run from the project root with:
python forecasting.py
or:
python forecasting.py path/to/csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover - handled at runtime for clear CLI feedback.
    SARIMAX = None


DEFAULT_INPUT_FILE = "superstore_clean.csv"
FORECAST_PLOT = "sales_forecast.png"
FORECAST_OUTPUT = "sales_forecast.csv"
TEST_PERIODS = 6
FORECAST_PERIODS = 6
SEASONAL_PERIOD = 12
REQUIRED_COLUMNS = {
    "Order Date",
    "Sales",
}


def resolve_input_path(input_arg: str | None = None) -> Path:
    if input_arg:
        return Path(input_arg).expanduser().resolve()
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return (Path.cwd() / DEFAULT_INPUT_FILE).resolve()


def print_section(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")


def ensure_statsmodels_available() -> None:
    if SARIMAX is None:
        raise ImportError(
            "statsmodels is required for forecasting but is not installed.\n"
            "Install it with: pip install statsmodels"
        )


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")


def prepare_monthly_sales(df: pd.DataFrame) -> pd.Series:
    prepared = df.copy()
    prepared["Order Date"] = pd.to_datetime(prepared["Order Date"], errors="coerce")
    prepared["Sales"] = pd.to_numeric(prepared["Sales"], errors="coerce")

    invalid_rows = prepared["Order Date"].isna() | prepared["Sales"].isna()
    invalid_count = int(invalid_rows.sum())
    if invalid_count:
        print(f"Dropping rows with invalid Order Date or Sales values: {invalid_count}")
        prepared = prepared.loc[~invalid_rows].copy()

    if prepared.empty:
        raise ValueError("No valid rows remain after parsing Order Date and Sales.")

    monthly_sales = (
        prepared.set_index("Order Date")["Sales"]
        .sort_index()
        .resample("M")
        .sum()
        .asfreq("M")
    )
    monthly_sales.name = "Sales"

    if monthly_sales.empty:
        raise ValueError("Monthly sales series is empty after aggregation.")

    if monthly_sales.shape[0] <= TEST_PERIODS + SEASONAL_PERIOD:
        raise ValueError(
            "Not enough monthly observations for a seasonal forecast. "
            f"Need more than {TEST_PERIODS + SEASONAL_PERIOD} months; "
            f"found {monthly_sales.shape[0]}."
        )

    return monthly_sales


def fit_sarima(series: pd.Series) -> object:
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def calculate_accuracy(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    nonzero_actual = actual.replace(0, np.nan)
    mape = float((np.abs(errors / nonzero_actual).dropna().mean()) * 100)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def create_future_forecast_df(forecast_result: object) -> pd.DataFrame:
    forecast_mean = forecast_result.predicted_mean
    confidence_intervals = forecast_result.conf_int(alpha=0.05)

    forecast_df = pd.DataFrame(
        {
            "month": forecast_mean.index,
            "predicted_sales": forecast_mean.values,
            "lower_ci": confidence_intervals.iloc[:, 0].values,
            "upper_ci": confidence_intervals.iloc[:, 1].values,
        }
    )
    forecast_df["month"] = forecast_df["month"].dt.strftime("%Y-%m-%d")
    return forecast_df


def save_forecast_plot(
    monthly_sales: pd.Series,
    test_sales: pd.Series,
    test_predictions: pd.Series,
    future_forecast: object,
    output_path: Path,
) -> None:
    future_mean = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int(alpha=0.05)

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        x=monthly_sales.index,
        y=monthly_sales.values,
        label="Historical monthly sales",
        color="#1f77b4",
        linewidth=2,
    )
    sns.lineplot(
        x=test_sales.index,
        y=test_sales.values,
        label="Held-out actual",
        color="#2ca02c",
        marker="o",
        linewidth=2,
    )
    sns.lineplot(
        x=test_predictions.index,
        y=test_predictions.values,
        label="Held-out predicted",
        color="#d62728",
        marker="o",
        linewidth=2,
    )
    sns.lineplot(
        x=future_mean.index,
        y=future_mean.values,
        label="Future forecast",
        color="#9467bd",
        marker="o",
        linewidth=2,
    )
    plt.fill_between(
        future_mean.index,
        future_ci.iloc[:, 0].values,
        future_ci.iloc[:, 1].values,
        color="#9467bd",
        alpha=0.18,
        label="95% confidence interval",
    )

    plt.title("Monthly Sales Forecast")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    ensure_statsmodels_available()

    input_path = resolve_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}\n"
            f"Place '{DEFAULT_INPUT_FILE}' in the current directory or pass a file path:\n"
            f"python forecasting.py /path/to/superstore_clean.csv"
        )

    print(f"Loading cleaned dataset from: {input_path}")
    df = pd.read_csv(input_path)
    ensure_required_columns(df)

    sns.set_theme(style="whitegrid")

    monthly_sales = prepare_monthly_sales(df)
    train_sales = monthly_sales.iloc[:-TEST_PERIODS]
    test_sales = monthly_sales.iloc[-TEST_PERIODS:]

    print_section("Monthly Sales Series")
    print(f"Monthly observations: {monthly_sales.shape[0]}")
    print(
        "Date range: "
        f"{monthly_sales.index.min().strftime('%Y-%m-%d')} to "
        f"{monthly_sales.index.max().strftime('%Y-%m-%d')}"
    )
    print(f"Held-out test months: {TEST_PERIODS}")

    validation_model = fit_sarima(train_sales)
    validation_forecast = validation_model.get_forecast(steps=TEST_PERIODS)
    test_predictions = validation_forecast.predicted_mean
    test_predictions.index = test_sales.index

    accuracy = calculate_accuracy(test_sales, test_predictions)

    print_section("Forecast Accuracy - Held-Out Test Set")
    print(f"MAE:  {accuracy['MAE']:,.2f}")
    print(f"RMSE: {accuracy['RMSE']:,.2f}")
    if np.isnan(accuracy["MAPE"]):
        print("MAPE: Not available because all held-out actual sales values are zero.")
    else:
        print(f"MAPE: {accuracy['MAPE']:,.2f}%")

    full_model = fit_sarima(monthly_sales)
    future_forecast = full_model.get_forecast(steps=FORECAST_PERIODS)

    forecast_plot_path = input_path.with_name(FORECAST_PLOT)
    save_forecast_plot(
        monthly_sales=monthly_sales,
        test_sales=test_sales,
        test_predictions=test_predictions,
        future_forecast=future_forecast,
        output_path=forecast_plot_path,
    )

    forecast_output_path = input_path.with_name(FORECAST_OUTPUT)
    future_forecast_df = create_future_forecast_df(future_forecast)
    future_forecast_df.to_csv(forecast_output_path, index=False)

    print_section("Future 6-Month Sales Forecast")
    print(future_forecast_df.round(2).to_string(index=False))
    print(f"\nSaved forecast chart to: {forecast_plot_path}")
    print(f"Saved forecast data to: {forecast_output_path}")


if __name__ == "__main__":
    main()
