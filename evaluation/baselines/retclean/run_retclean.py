import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import time
from pathlib import Path

# execute this code in RetClean root directory

BACKEND_URL = "http://localhost:8000"

async def clean_column_task(session, column_to_clean, records, table_description, pivot_names):
    print(f"🚀 STARTING cleaning for: '{column_to_clean}'")
    # pivot_names = pivots_map.get(column_to_clean, [column_to_clean])
    payload = {
        "entity_description": table_description,
        "target_name": column_to_clean,
        "target_data": [
            {"id": i, "value": r[column_to_clean]} for i, r in enumerate(records)
        ],
        "pivot_names": pivot_names,
        "pivot_data": [
            {"id": i, "values": [r[p] for p in pivot_names]} for i, r in enumerate(records)
        ],
        "reasoner_name": "Gemini-2.5-pro", 
        "index_name": None,
        "index_type": None,
        "reranker_type": None
    }

    try:
        async with session.post(f"{BACKEND_URL}/repair/", json=payload) as resp:
            if resp.status == 200:
                results = await resp.json()
                if results.get('status') == 'success':
                    cleaned_values = [item['value'] for item in results['results']]
                    print(f"Cleaned {column_to_clean}")
                    return (column_to_clean, cleaned_values, "success")
                else:
                    print("API logic error")
                    return (column_to_clean, None, "error")
            else:
                text = await resp.text()
                print(f"❌ HTTP Error {resp.status} for '{column_to_clean}': {text}")
                return (column_to_clean, None, "error")
    except Exception as e:
        print(f"❌ Exception for '{column_to_clean}': {e}")
        return (column_to_clean, None, "error")

async def main(df_cleaned, records, pivot_map, table_description ) -> tuple[pd.DataFrame, float]:
    all_columns = list(df_cleaned.columns)
    start_time = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=300, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []  
        for col in all_columns:
            pivot_names = pivot_map.get(col, [col])
            task = clean_column_task(session, col, records, table_description, pivot_names)
            tasks.append(task)
        try:       
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=54000) 
            for col_name, new_values, status in results:
                if status == "success":
                    df_cleaned[col_name] = new_values
                else:
                    print(status)
                    print(new_values)
                    print(f"Skipping update for {col_name} due to errors.")
        except asyncio.TimeoutError:
            print('Global time out')
            return df_cleaned, 54000.0
        except Exception as e:
            print(f'Critical main loop error: {repr(e)}')
        
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"runtime: {runtime:.3f}s")
    return df_cleaned, runtime

if __name__ == "__main__":
    beers_description = "A table of beers and the companies that made them."
    beers_pivot_map = {
    "id": ["id"],
    "beer_name": ["beer_name"],
    "style": ["style"],
    "ounces": ["ounces"],
    "abv": ["abv"],
    "ibu": ["ibu"],
    "brewery_id": ["brewery_id"],
    "brewery_name": ["brewery_name"],
    "city": ["city", "state"],
    "state": ["city", "state"]
    }
    hospital_description = "A table with hospitals and their information."
    hospital_pivot_map = {
        "ProviderNumber": ["ProviderNumber"],
        "HospitalName": ["HospitalName"],
        "Address1": ["Address1"],
        "Address2": ["Address2"],
        "City": ["City", "State"],
        "State": ["City", "State"],
        "ZipCode": ["City", "ZipCode"],
        "CountyName": ["City", "State", "CountyName"],
        "PhoneNumber": ["PhoneNumber"],
        "HospitalType": ["HospitalType"],
        "HospitalOwner": ["HospitalOwner"],
        "EmergencyService": ["EmergencyService"],
        "Condition": ["Condition"],
        "MeasureCode": ["MeasureCode", "Stateavg"],
        "MeasureName": ["MeasureName"],
        "Score": ["Score"],
        "Sample": ["Sample"],
        "Stateavg": ["MeasureCode", "Stateavg"]
    }
    rayyan_description = "A table with published research articles."
    rayyan_pivot_map = {
        "id": ["id"],
        "article_title": ["article_title"],
        "article_language": ["article_language"],
        "journal_title": ["journal_title"],
        "journal_abbreviation": ["journal_abbreviation"],
        "journal_issn" : ["journal_issn"],
        "article_jvolumn" : ["article_jvolumn"],
        "article_jissue" : ["article_jissue"],
        "article_jcreated_at" : ["article_jcreated_at"],
        "article_pagination" : ["article_pagination"],
        "author_list" : ["author_list"]
    }
    movies_description = "A table with movies and their information."
    movies_pivot_map = {
        "Id": ["Id"],
        "Name": ["Name"],
        "Year": ["Year", "Release Date"],
        "Release Date": ["Release Date"],
        "Director": ["Director"],
        "Creator": ["Creator"],
        "Actors": ["Actors"],
        "Cast": ["Cast"],
        "Language": ["Language"],
        "Country": ["Country"],
        "RatingValue": ["RatingValue"],
        "RatingCount": ["RatingCount"],
        "ReviewCount": ["ReviewCount"],
        "Genre": ["Genre"],
        "Filming Locations": ["Filming Locations"],
        "Description": ["Description"]
    }
    dataset_mapping = {
        'beers': (beers_description, beers_pivot_map),
        'hospital': (hospital_description, hospital_pivot_map),
        'rayyan': (rayyan_description, rayyan_pivot_map),
        'movies': (movies_description, movies_pivot_map)
    }
    BASE_DIR = Path(__file__).resolve().parent.parent # baselines dir
    datasets = ['beers', 'hospital', 'rayyan', 'movies']
    for dataset in datasets: 
        file_path = BASE_DIR / "datasets" / f"{dataset}_dirty.csv"
        save_cleaned = BASE_DIR / "datasets" / "cleaned" / "retclean" / f"{dataset}_cleaned.csv"
        save_runtime = BASE_DIR / "datasets" / "cleaned" / "retclean" / f"{dataset}_runtime.txt"

        df_to_clean = pd.read_csv(file_path)
        df_to_clean = df_to_clean.astype(object)
        records = df_to_clean.where(pd.notnull(df_to_clean), None).to_dict(orient="records") # add to main
        
        table_description, pivot_map = dataset_mapping.get(dataset)
        print(f"Cleaning dataset {dataset}")
        df_cleaned, runtime = asyncio.run(main(df_to_clean, records, pivot_map, table_description)) 
        df_cleaned.to_csv(save_cleaned, index=False, encoding='utf-8')
        with open(save_runtime, "w") as f:
            f.write(f"Runtime {dataset}: {runtime:.3f} seconds")


