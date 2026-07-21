

import pandas as pd
import re
from google.cloud import bigquery
from google.cloud import storage

import os
import sys
import io #only need while using dummy data


sys.path.append(os.getcwd())
from user.misc import arg_loader

args=arg_loader()

resultpath=args[1]
instruction = args[2]

#if instruction starts with SELECT, assume query.
#if not, assume a deployment

makara_ready = True

if makara_ready:
    
    #deployments /recordings is only different if array (doesn't go down to file level)

    bq_client = bigquery.Client()
    query = f"""
        SELECT 
            recording_uri,
            recording_sample_rate_khz,
            recording_bit_depth 
        FROM `ggn-nmfs-pacm-dev-1.makara.recordings` 
        WHERE recording_channel = 1 
          AND deployment_id = (
              SELECT id FROM `ggn-nmfs-pacm-dev-1.makara.deployments` 
              WHERE deployment_code = '{instruction}'
          )
    """
    bq_df = bq_client.query(query).to_dataframe()
    
    if bq_df.empty:
        raise ValueError(f"No recording data found for deployment code: {instruction}")
        
    # Extract DB metadata from the first row
    row = bq_df.iloc[0]
    base_gsuri = row['recording_uri']
    sample_rate_khz = row['recording_sample_rate_khz']
    bit_depth = row['recording_bit_depth']
    
    # Convert kHz to Hz for the math, and calculate byte rate (assuming 1 channel)
    sample_rate_hz = sample_rate_khz * 1000
    byte_rate = sample_rate_hz * 1 * (bit_depth / 8)
    
    path_parts = base_gsuri.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    # 3 & 4. Process files, check extensions, parse datetimes, calculate durations
    manifest_records = []
    
    # Regex looks for 12 digits right before ".wav" (case-insensitive)
    dt_pattern = re.compile(r'(\d{12})\.wav$', re.IGNORECASE)
    
    for blob in blobs:
        file_name = blob.name.split('/')[-1]
        
        # Hard catch for FLAC
        if file_name.lower().endswith('.flac'):
            raise ValueError(f"FLAC not supported ({file_name}). Variable bitrate prevents exact duration calculation via file size.")
            
        ext_match = dt_pattern.search(file_name)
        
        # Skip files that don't match our specific naming and extension convention
        if not ext_match:
            continue
            
        full_path = f"gs://{bucket_name}/{blob.name}"
        raw_dt = ext_match.group(1) # e.g., '160824130023'
        
        start_time = f"{raw_dt[:6]}-{raw_dt[6:]}"
        
        # Exact calculation (subtracting 44 bytes for standard WAV header)
        duration_secs = (blob.size - 44) / byte_rate if blob.size > 44 else 0.0
            
        manifest_records.append({
            'FileName': file_name,
            'FullPath': full_path,
            'StartTime': start_time,
            'Duration': duration_secs,
            'Deployment': instruction,
            'SegStart': 0,
            'SegDur': duration_secs,
            'Name': instruction
        })
        
    # 5. Build and return the final dataframe
    df = pd.DataFrame(manifest_records)

    #ensure the data are available
    try:
        is_valid = df['FullPath'].notna() & (df['FullPath'].str.strip() != '')
        assert is_valid.all(), "String column has invalid values (null, empty, or whitespace)."
    except AssertionError as e:
        print(f"Assertion Failed: {e}")
else:
    #just read in a dummy df
    df = pd.read_csv(io.StringIO("""
FileName,FullPath,StartTime,Duration,Deployment,SegStart,SegDur,Name
AU-GASU01-240522-202000.wav,gs://afsc-1/bottom_mounted/GA23_AU_SU01/05_2024/AU-GASU01-240522-202000.wav,240522-202000,600.0,GA23_AU_SU01,0.0,300.0,GA23_AU_SU01
AU-GASU01-240522-202000.wav,gs://afsc-1/bottom_mounted/GA23_AU_SU01/05_2024/AU-GASU01-240522-202000.wav,240522-202000,600.0,GA23_AU_SU01,300.0,300.0,GA23_AU_SU01
AU-GASU01-240522-204000.wav,gs://afsc-1/bottom_mounted/GA23_AU_SU01/05_2024/AU-GASU01-240522-204000.wav,240522-204000,600.0,GA23_AU_SU01,0.0,300.0,GA23_AU_SU01
AU-GASU01-240522-204000.wav,gs://afsc-1/bottom_mounted/GA23_AU_SU01/05_2024/AU-GASU01-240522-204000.wav,240522-204000,600.0,GA23_AU_SU01,300.0,300.0,GA23_AU_SU01
AU-GASU01-240522-205000.wav,gs://afsc-1/bottom_mounted/GA23_AU_SU01/05_2024/AU-GASU01-240522-205000.wav,240522-205000,600.0,GA23_AU_SU01,0.0,300.0,GA23_AU_SU01
AU-GASU01-240522-205000.wav,gs://afsc-1/bottom_mounted/GA23_AU_SU01/05_2024/AU-GASU01-240522-205000.wav,240522-205000,600.0,GA23_AU_SU01,300.0,300.0,GA23_AU_SU01
""")) #needs to be formatted correctly, plus have a gsuri


df.to_csv(resultpath)

