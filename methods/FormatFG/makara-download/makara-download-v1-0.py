import pandas as pd
from google.cloud import bigquery
import concurrent.futures
import os
import sys

sys.path.append(os.getcwd())
from user.misc import arg_loader

args=arg_loader()

resultpath=args[1]
instruction = args[2]

#if instruction starts with SELECT, assume query.
#if not, assume a deployment

if "SELECT " in instruction:
    #assume a query.
    query = instruction
else:
    #assume a deployment code

    query = query = f"SELECT recording_code AS `FileName`, recording_uri AS `FullPath`, recording_start_datetime AS `StartTime`, recording_duration_secs AS `Duration`, deployment_code AS `Deployment`,0 AS `SegStart`, recording_duration_secs AS `SegDur`, '{instruction}' AS `Name`,FROM `ggn-nmfs-pacm-dev-1.makara.recordings` AS r JOIN `ggn-nmfs-pacm-dev-1.makara.deployments` AS d ON r.deployment_id = d.id WHERE r.recording_channel = 1 AND d.deployment_code = '{instruction}'"

client = bigquery.Client()

query_job = client.query(query)

df = query_job.to_dataframe()

#ensure the data are available
try:
    is_valid = df['FullPath'].notna() & (df['FullPath'].str.strip() != '')
    assert is_valid.all(), "String column has invalid values (null, empty, or whitespace)."
except AssertionError as e:
    print(f"Assertion Failed: {e}")

import code
code.interact(local=dict(globals(), **locals()))

#use metadata and perform parallelized download.

bucket_name = os.environ.get("GCS_BUCKET")
destination_folder = os.environ.get("DESTINATION_DIR", "/app/downloads")

max_workers = os.cpu_count()

objs = df.FullPath.to_list()

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Create a future for each download task
    future_to_blob = {
        executor.submit(download_blob, bucket_name, blob_name, destination_folder): blob_name
        for blob_name in objects_to_download
    }

    success_count = 0
    error_count = 0
        
    # As each future completes, process the result
    for future in concurrent.futures.as_completed(future_to_blob):
        blob_name = future_to_blob[future]
        try:
            name, result = future.result()
            if "Success" in result:
                success_count += 1
            else:
                error_count += 1
                print(f"Failed download details for {name}: {result}")
        except Exception as exc:
            error_count += 1
            print(f"An exception was generated for blob {blob_name}: {exc}")

    print("\n--- Download Summary ---")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed downloads: {error_count}")
    print("Batch download process complete.")


df.to_csv(resultpath)
