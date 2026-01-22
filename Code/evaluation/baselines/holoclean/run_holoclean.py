import time
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *

def run_holoclean(file_name, dataset_path, dataset_constraints_path = None):
    start_time = time.perf_counter()
    hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=10,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True
    ).session

    hc.load_data(file_name, dataset_path)
    if dataset_constraints_path:
        hc.load_dcs(dataset_constraints_path)
        hc.ds.set_constraints(hc.get_dcs())  
        detectors = [NullDetector(), ViolationDetector()]
        featurizers = [
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            ConstraintFeaturizer(),
            LangModelFeaturizer()
            ]   
    else:
        detectors = [NullDetector()]
        featurizers = [
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
            LangModelFeaturizer()
            ]
    
    hc.detect_errors(detectors)
    hc.setup_domain()
    hc.repair_errors(featurizers)
    cleaned_df = hc.ds.repaired_data.df
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"Runtime {dataset_path}: {runtime}")
    if '_tid_' in cleaned_df.columns:
        cleaned_df.drop('_tid_', axis=1)

    return cleaned_df, runtime


if __name__ == "__main__":
    from pathlib import Path
    import json
    BASE_DIR = Path().resolve().parent # baselines dir
    datasets_folder = BASE_DIR / "datasets"
    datasets = ['beers', 'hospital', 'movies', 'rayyan']
    runtimes = {}
    for dataset in datasets:
        dataset_path = datasets_folder / f"{dataset}_dirty.csv"
        if dataset in ['beers', 'hospital']:
            dataset_constraints_path = datasets_folder / f"{dataset}_constraints.txt"
            cleaned_df, runtime = run_holoclean(dataset, dataset_path, dataset_constraints_path)
        else:
            cleaned_df, runtime = run_holoclean(dataset, dataset_path)
        runtimes[dataset] = runtime
        cleaned_df_path = datasets_folder / "cleaned" / "holoclean" / f"{dataset}_clean.csv"
        cleaned_df.to_csv(cleaned_df_path, index=False, encoding='utf-8')
        print(f"Cleaned df saved to {cleaned_df_path}")
    
    runtime_path = datasets_folder / "cleaned" / "holoclean" / "runtimes.json"
    with open(runtime_path, "w") as f:
        json.dump(runtimes, f, indent=4)
        
        

    # first do sudo service postgresql start in wsl

    