import pandas as pd
import duckdb

def apply_sql_cleaning(df: pd.DataFrame, dataset, sql_query: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.register(dataset, df)
    cleaned_df = con.query(sql_query).df()
    cleaned_df = cleaned_df.reindex(columns=df.columns.tolist())
    return cleaned_df

if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent # baselines dir
    datasets_folder = BASE_DIR / "datasets"
    sql_folder = BASE_DIR / "datasets" / "cleaned" / "cocoon"

    datasets = ['beers', 'hospital', 'movies', 'rayyan']
    for dataset in datasets:
        df_path = datasets_folder / f"{dataset}_dirty.csv"
        df = pd.read_csv(df_path, encoding='utf-8')
        sql_query_path = sql_folder / f"{dataset}_sql.sql"
        sql_query = sql_query_path.read_text()
        cleaned_df = apply_sql_cleaning(df, dataset, sql_query) # change memory.main.dataset -> dataset
        cleaned_df_path = datasets_folder / "cleaned" / "cocoon" / f"{dataset}_cleaned.csv"
        cleaned_df.to_csv(cleaned_df_path, index=False, encoding='utf-8')
        print(f"Cleaned DataFrame saved to {cleaned_df_path}")




