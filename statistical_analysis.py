import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_INPUT_FILE = "superstore_clean.csv"
CORRELATION_PLOT = "correlation_heatmap.png"
SEGMENT_PLOT = "segment_analysis.png"
REVENUE_TREND_PLOT = "revenue_trend.png"
ALPHA = 0.05
REQUIRED_COLUMNS = {
    "Order Date",
    "Sales",
    "Profit",
    "Discount",
    "Region",
    "Segment",
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

    if "Profit Margin %" not in prepared.columns:
        sales_nonzero = prepared["Sales"].replace(0, pd.NA)
        prepared["Profit Margin %"] = ((prepared["Profit"] / sales_nonzero) * 100).fillna(0)

    if "Year" not in prepared.columns:
        prepared["Year"] = prepared["Order Date"].dt.year

    if "Month" not in prepared.columns:
        prepared["Month"] = prepared["Order Date"].dt.month

    return prepared


def print_section(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")


def interpret_p_value(p_value: float, alpha: float = ALPHA) -> str:
    if p_value < alpha:
        return f"Reject H0 (p < {alpha}); the difference is statistically significant."
    return f"Fail to reject H0 (p >= {alpha}); no statistically significant difference detected."


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    corr_columns = ["Discount", "Profit Margin %", "Sales", "Profit"]
    correlation_matrix = df[corr_columns].corr(method="pearson", numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return correlation_matrix


def run_regional_anova(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    regional_groups = []
    region_means = (
        df.groupby("Region", dropna=False)["Profit Margin %"]
        .mean()
        .sort_values(ascending=False)
    )

    for _, region_df in df.groupby("Region", dropna=False):
        group = region_df["Profit Margin %"].dropna()
        if not group.empty:
            regional_groups.append(group)

    if len(regional_groups) < 2:
        raise ValueError("ANOVA requires at least two regions with valid profit margin values.")

    f_statistic, p_value = stats.f_oneway(*regional_groups)
    return f_statistic, p_value, region_means


def run_discount_ttest(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    discounted = df.loc[df["Discount"] > 0, "Profit Margin %"].dropna()
    non_discounted = df.loc[df["Discount"] == 0, "Profit Margin %"].dropna()

    if discounted.empty or non_discounted.empty:
        raise ValueError("T-test requires both discounted and non-discounted groups to have data.")

    t_statistic, p_value = stats.ttest_ind(
        discounted,
        non_discounted,
        equal_var=False,
        nan_policy="omit",
    )
    group_summary = pd.Series(
        {
            "discounted_avg_margin": discounted.mean(),
            "non_discounted_avg_margin": non_discounted.mean(),
            "discounted_count": int(discounted.shape[0]),
            "non_discounted_count": int(non_discounted.shape[0]),
        }
    )
    return t_statistic, p_value, group_summary


def save_segment_analysis(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    segment_summary = (
        df.groupby("Segment", dropna=False)["Profit Margin %"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    segment_summary["sem"] = segment_summary["std"] / segment_summary["count"].pow(0.5)
    segment_summary["ci95"] = (1.96 * segment_summary["sem"]).fillna(0)
    segment_summary = segment_summary.sort_values("mean", ascending=False)

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=segment_summary,
        x="Segment",
        y="mean",
        palette="crest",
    )
    ax.errorbar(
        x=range(len(segment_summary)),
        y=segment_summary["mean"],
        yerr=segment_summary["ci95"],
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=5,
    )
    plt.title("Average Profit Margin by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Average Profit Margin (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return segment_summary


def save_revenue_trend(df: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    monthly_revenue = (
        df.groupby(["Year", "Month"], dropna=False)["Sales"]
        .sum()
        .reset_index(name="Revenue")
        .sort_values(["Year", "Month"])
    )
    monthly_revenue["Month Label"] = monthly_revenue["Month"].map(
        {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
    )

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=monthly_revenue,
        x="Month",
        y="Revenue",
        hue="Year",
        marker="o",
        palette="tab10",
    )
    plt.title("Monthly Revenue Trend by Year")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(ticks=list(range(1, 13)), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    best_month = monthly_revenue.loc[monthly_revenue["Revenue"].idxmax()]
    worst_month = monthly_revenue.loc[monthly_revenue["Revenue"].idxmin()]

    return monthly_revenue, best_month, worst_month


def print_business_insights(
    corr_discount_margin: float,
    corr_discount_margin_p: float,
    corr_sales_profit: float,
    corr_sales_profit_p: float,
    anova_p_value: float,
    region_means: pd.Series,
    ttest_p_value: float,
    discount_summary: pd.Series,
    segment_summary: pd.DataFrame,
    best_month: pd.Series,
    worst_month: pd.Series,
    df: pd.DataFrame,
) -> None:
    high_discount = df.loc[df["Discount"] >= 0.20, "Profit Margin %"].dropna()
    low_discount = df.loc[df["Discount"] < 0.20, "Profit Margin %"].dropna()
    high_discount_margin = high_discount.mean() if not high_discount.empty else float("nan")
    low_discount_margin = low_discount.mean() if not low_discount.empty else float("nan")

    best_region = region_means.idxmax()
    worst_region = region_means.idxmin()
    top_segment = segment_summary.iloc[0]
    bottom_segment = segment_summary.iloc[-1]
    sales_profit_message = (
        "Sales and profit move together strongly"
        if abs(corr_sales_profit) >= 0.5
        else "Sales and profit show only a moderate relationship"
    )
    discount_direction = (
        "higher discounts tend to reduce profit margin"
        if corr_discount_margin < 0
        else "higher discounts do not appear to reduce profit margin consistently"
    )
    region_message = (
        f"Profitability differs across regions (ANOVA p = {anova_p_value:.4f})"
        if anova_p_value < ALPHA
        else f"Regional margin differences are not statistically strong enough to confirm separation (ANOVA p = {anova_p_value:.4f})"
    )
    discount_ttest_message = (
        f"discounted and non-discounted orders differ significantly (t-test p = {ttest_p_value:.4f})"
        if ttest_p_value < ALPHA
        else f"discounted and non-discounted orders do not differ significantly on margin alone (t-test p = {ttest_p_value:.4f})"
    )

    print_section("Business Insights Summary")
    print(
        "1. "
        f"{sales_profit_message} (r = {corr_sales_profit:.3f}, p = {corr_sales_profit_p:.4f}). "
        "Use this to prioritize revenue-growth efforts in categories and accounts that also preserve margin quality."
    )
    print(
        "2. Discount analysis shows that "
        f"{discount_direction} (r = {corr_discount_margin:.3f}, p = {corr_discount_margin_p:.4f}; "
        f"discounted avg margin = {discount_summary['discounted_avg_margin']:.2f}% vs "
        f"non-discounted = {discount_summary['non_discounted_avg_margin']:.2f}%). "
        "Recommend using discount guardrails and approvals for deeper markdowns where margin erosion is visible."
    )
    print(
        "3. "
        f"{region_message}. "
        f"Use {best_region} ({region_means[best_region]:.2f}% avg margin) as the benchmark and review pricing, mix, and operations in "
        f"{worst_region} ({region_means[worst_region]:.2f}% avg margin)."
    )
    print(
        "4. Segment economics are uneven: "
        f"{top_segment['Segment']} leads at {top_segment['mean']:.2f}% avg margin, while "
        f"{bottom_segment['Segment']} trails at {bottom_segment['mean']:.2f}%. "
        "Recommend segment-specific pricing and retention tactics instead of one-size-fits-all offers."
    )
    print(
        "5. Revenue seasonality is material: the strongest month was "
        f"{best_month['Month Label']} {int(best_month['Year'])} with revenue of {best_month['Revenue']:,.2f}, "
        f"while the weakest was {worst_month['Month Label']} {int(worst_month['Year'])} at "
        f"{worst_month['Revenue']:,.2f}. Align inventory, promotions, and staffing to these demand swings."
    )

    if pd.notna(high_discount_margin) and pd.notna(low_discount_margin):
        print(
            "   Extra signal: orders with discounts >= 20% averaged "
            f"{high_discount_margin:.2f}% margin versus {low_discount_margin:.2f}% below 20%, "
            "which can help set a discount cap policy."
        )

    print(f"   Statistical support: {discount_ttest_message}.")


def main() -> None:
    input_path = resolve_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}\n"
            f"Place '{DEFAULT_INPUT_FILE}' in the current directory or pass a file path:\n"
            f"python statistical_analysis.py /path/to/superstore_clean.csv"
        )

    print(f"Loading cleaned dataset from: {input_path}")
    df = pd.read_csv(input_path)
    ensure_required_columns(df)
    df = prepare_dataframe(df)

    sns.set_theme(style="whitegrid")

    corr_discount_margin, corr_discount_margin_p = stats.pearsonr(
        df["Discount"],
        df["Profit Margin %"],
    )
    corr_sales_profit, corr_sales_profit_p = stats.pearsonr(
        df["Sales"],
        df["Profit"],
    )
    heatmap_path = input_path.with_name(CORRELATION_PLOT)
    correlation_matrix = save_correlation_heatmap(df, heatmap_path)

    print_section("Correlation Analysis")
    print(
        "Pearson correlation between Discount and Profit Margin %: "
        f"r = {corr_discount_margin:.4f}, p-value = {corr_discount_margin_p:.4f}"
    )
    print(
        "Pearson correlation between Sales and Profit: "
        f"r = {corr_sales_profit:.4f}, p-value = {corr_sales_profit_p:.4f}"
    )
    print("\nCorrelation matrix:")
    print(correlation_matrix.round(4).to_string())
    print(f"\nSaved heatmap to: {heatmap_path}")

    regional_f, regional_p, region_means = run_regional_anova(df)
    print_section("Hypothesis Test 1 - Regional Performance")
    print(f"F-statistic: {regional_f:.4f}")
    print(f"p-value: {regional_p:.4f}")
    print(interpret_p_value(regional_p))
    print("\nAverage profit margin by region:")
    print(region_means.round(2).to_string())

    discount_t, discount_p, discount_summary = run_discount_ttest(df)
    print_section("Hypothesis Test 2 - Discount Impact")
    print(f"t-statistic: {discount_t:.4f}")
    print(f"p-value: {discount_p:.4f}")
    print(interpret_p_value(discount_p))
    print("\nGroup means:")
    print(discount_summary.round(2).to_string())

    segment_plot_path = input_path.with_name(SEGMENT_PLOT)
    segment_summary = save_segment_analysis(df, segment_plot_path)
    print_section("Segment Profitability Analysis")
    print(segment_summary[["Segment", "mean", "ci95", "count"]].round(2).to_string(index=False))
    print(f"\nSaved segment chart to: {segment_plot_path}")

    trend_plot_path = input_path.with_name(REVENUE_TREND_PLOT)
    _, best_month, worst_month = save_revenue_trend(df, trend_plot_path)
    print_section("Time Series Trend")
    print(
        "Best performing month: "
        f"{best_month['Month Label']} {int(best_month['Year'])} "
        f"with revenue {best_month['Revenue']:,.2f}"
    )
    print(
        "Worst performing month: "
        f"{worst_month['Month Label']} {int(worst_month['Year'])} "
        f"with revenue {worst_month['Revenue']:,.2f}"
    )
    print(f"Saved revenue trend chart to: {trend_plot_path}")

    print_business_insights(
        corr_discount_margin=corr_discount_margin,
        corr_discount_margin_p=corr_discount_margin_p,
        corr_sales_profit=corr_sales_profit,
        corr_sales_profit_p=corr_sales_profit_p,
        anova_p_value=regional_p,
        region_means=region_means,
        ttest_p_value=discount_p,
        discount_summary=discount_summary,
        segment_summary=segment_summary,
        best_month=best_month,
        worst_month=worst_month,
        df=df,
    )


if __name__ == "__main__":
    main()
