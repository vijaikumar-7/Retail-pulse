import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CSV_FILE = "superstore_clean.csv"
DEFAULT_SQL_FILE = "analysis.sql"
DEFAULT_DB_FILE = "superstore.db"
RESULTS_DIR = "sql_results"
TABLE_NAME = "superstore"


def resolve_csv_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return (Path.cwd() / DEFAULT_CSV_FILE).resolve()


def parse_queries(sql_path: Path) -> list[tuple[str, str]]:
    sql_text = sql_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\s*--\s*name:\s*(?P<name>[a-zA-Z0-9_]+)\s*$"
        r"(?P<body>.*?)(?=^\s*--\s*name:|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    queries = []
    for match in pattern.finditer(sql_text):
        name = match.group("name").strip()
        body = match.group("body").strip()
        if body:
            queries.append((name, body))

    if not queries:
        raise ValueError(f"No named SQL queries found in: {sql_path}")

    return queries


def print_result(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    if df.empty:
        print("No rows returned.")
    else:
        print(df.to_string(index=False))


def main() -> None:
    csv_path = resolve_csv_path()
    sql_path = (Path.cwd() / DEFAULT_SQL_FILE).resolve()
    db_path = (Path.cwd() / DEFAULT_DB_FILE).resolve()
    results_path = (Path.cwd() / RESULTS_DIR).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {csv_path}\n"
            f"Run data_cleaning.py first or pass a path:\n"
            f"python run_sql.py /path/to/superstore_clean.csv"
        )

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    print(f"Loading cleaned dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    results_path.mkdir(exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        df.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)
        print(f"Loaded {len(df):,} rows into SQLite database: {db_path}")

        queries = parse_queries(sql_path)
        print(f"Running {len(queries)} SQL queries from: {sql_path}")

        for index, (query_name, query_sql) in enumerate(queries, start=1):
            result_df = pd.read_sql(query_sql, connection)
            print_result(f"Query {index}: {query_name}", result_df)

            output_file = results_path / f"{index:02d}_{query_name}.csv"
            result_df.to_csv(output_file, index=False)
            print(f"Exported results to: {output_file}")


if __name__ == "__main__":
    main()
