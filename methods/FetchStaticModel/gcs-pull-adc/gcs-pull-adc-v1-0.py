from google.cloud import storage

args=arg_loader()

resultpath=args[1]
gcs_uri = args[2]


storage_client = storage.Client()

bucket = storage_client.bucket(bucket_name)

blob = bucket.blob(gcs_uri)
blob.download_to_filename(resultpath)
