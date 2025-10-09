

import pandas as pd
from google.cloud import bigquery
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

makara_ready = False

if makara_ready:

    if "SELECT " in instruction:
        #assume a query.
        query = instruction
    else:
        #assume a deployment code

        query = query = f"SELECT recording_code AS `FileName`, recording_uri AS `FullPath`,  FORMAT_DATETIME('%y%m%d-%H%M%S',recording_start_datetime) AS `StartTime`, recording_duration_secs AS `Duration`, deployment_code AS `Deployment`,0 AS `SegStart`, recording_duration_secs AS `SegDur`, '{instruction}' AS `Name`,FROM `ggn-nmfs-pacm-dev-1.makara.recordings` AS r JOIN `ggn-nmfs-pacm-dev-1.makara.deployments` AS d ON r.deployment_id = d.id WHERE r.recording_channel = 1 AND d.deployment_code = '{instruction}'"

    client = bigquery.Client()

    query_job = client.query(query)

    df = query_job.to_dataframe()

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

