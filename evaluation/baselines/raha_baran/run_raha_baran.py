import time
import pandas
import raha
from pathlib import Path

def clean_dataset_raha(filename, BASE_DIR):
    dirty_path = BASE_DIR / "datasets" / f"{filename}_dirty.csv"
    gt_path = BASE_DIR / "datasets" / f"{filename}_gt.csv"
    cleaned_path = BASE_DIR / "datasets" / "cleaned" / "raha" / f"{filename}_cleaned.csv"
    dataset_dictionary = {
        "name": filename,
        "path": dirty_path,
        "clean_path": gt_path
    }

    start_time = time.perf_counter()
    ###### DETECTION ######
    app_1 = raha.Detection()
    app_1.LABELING_BUDGET = 20
    app_1.VERBOSE = True 
    d = app_1.initialize_dataset(dataset_dictionary)
    app_1.run_strategies(d)
    app_1.generate_features(d)
    app_1.build_clusters(d)

    while len(d.labeled_tuples) < app_1.LABELING_BUDGET:
        app_1.sample_tuple(d)
        if d.has_ground_truth:
            app_1.label_with_ground_truth(d) # add GT path and i don't need to label tuples myself
        else:
            print("Label the dirty cells in the following sampled tuple.")
            sampled_tuple = pandas.DataFrame(data=[d.dataframe.iloc[d.sampled_tuple, :]], columns=d.dataframe.columns)
            print(sampled_tuple.to_string(index=False))
            for j in range(d.dataframe.shape[1]):
                cell = (d.sampled_tuple, j)
                value = d.dataframe.iloc[cell]
                correction = input("What is the correction for value '{}'? Type in the same value if it is not erronous.\n".format(value))
                user_label = 1 if value != correction else 0
                d.labeled_cells[cell] = [user_label, correction]
            d.labeled_tuples[d.sampled_tuple] = 1

    app_1.propagate_labels(d)
    app_1.predict_labels(d)

    ###### CORRECTION ######
    app_2 = raha.Correction()
    app_2.LABELING_BUDGET = 20
    app_2.VERBOSE = True

    d = app_2.initialize_dataset(d)
    app_2.initialize_models(d)

    for si in d.labeled_tuples:
        d.sampled_tuple = si
        app_2.update_models(d)
        app_2.predict_corrections(d)

    app_2.store_results(d)
    correction_dictionary = d.corrected_cells
    d.create_repaired_dataset(correction_dictionary)
    cleaned_df = d.repaired_dataframe
    end_time = time.perf_counter()
    runtime = end_time - start_time

    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8")
    print(f"[{filename}] cleaned and stored: {cleaned_path}")
    print(f"Runtime: {runtime:.3f}s")
    return runtime

if __name__ == "__main__":
    import json
    runtime_register = {}
    BASE_DIR = Path(__file__).resolve().parent.parent.parent # baselines folder
    benchmark_datasets = ['beers', 'hospital', 'movies', 'rayyan']
    for dataset in benchmark_datasets:
        print(f"Running {dataset}...")
        runtime = clean_dataset_raha(dataset, BASE_DIR)
        runtime_register[dataset] = runtime

    runtime_path = BASE_DIR / "datasets" / "cleaned" / "raha" / "runtimes.json"
    with open(runtime_path, 'w') as f:
        json.dump(runtime_register, f, indent=4)
    


