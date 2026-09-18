import pandas as pd
import numpy as np 

def create_error_mask(df: pd.DataFrame, detection_path) -> pd.DataFrame:
    errors_df = pd.read_csv(
        detection_path,
        names=["i", "j", "dummy"],
        skip_blank_lines=True
    )

    errors_df = errors_df.dropna(subset=["i", "j"])
    errors_df["i"] = errors_df["i"].astype(int)
    errors_df["j"] = errors_df["j"].astype(int)

    error_mask = pd.DataFrame(False,index=df.index, columns=df.columns)

    for row in errors_df.itertuples(index=False):
        error_mask.iat[row.i, row.j] = True
    
    return error_mask

if __name__ == "__main__":
    from pathlib import Path
    import subprocess
    import sys

    BASE_DIR = Path(__file__).resolve().parent # saged dir

    # train classifiers (on historical data) before running, only once
    # subprocess.run(["python",
    #                 "scripts/train_classifiers.py",
    #                 "--datasets", "flights",
    #                 "--classifiers" "mlp_classifier"
    #                 ])
    
    datasets = ['beers','hospital', 'movies', 'rayyan']
    # run detection
    for dataset in datasets:
        print(f"running {dataset}...")
        subprocess.run([
            sys.executable,
            "scripts/run_saged.py",
            "--dirty-dataset", dataset,
            "--historical-datasets", "flights",
            "--verbose"
        ])

    # create and save error_mask DataFrame
    for dataset in datasets:
        # print(f"{dataset}")
        file_path = BASE_DIR.parent / "datasets" / f"{dataset}_dirty.csv"
        save_path = BASE_DIR.parent / "datasets" / "cleaned" / "saged" / f"{dataset}_detection.csv"
        df = pd.read_csv(file_path, encoding='utf-8')
        detection_path = BASE_DIR / "experiments" /  "evaluation" / "data" / f"{dataset}" / "detection" / "saged" / "detections.csv"
        error_mask = create_error_mask(df, detection_path)
        error_mask.to_csv(save_path, index=False, encoding='utf-8')
        
