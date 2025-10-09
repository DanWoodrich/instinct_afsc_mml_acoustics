from google.cloud import storage
import sys
import os
from urllib.parse import urlparse

sys.path.append(os.getcwd())
from user.misc import arg_loader

args=arg_loader()

resultpath=args[1]
gcs_uri = args[2]

storage_client = storage.Client()

parsed_uri = urlparse(gcs_uri)
bucket = storage_client.bucket(parsed_uri.netloc)
prefix = parsed_uri.path.lstrip('/')

blob = bucket.blob(prefix)
blob.download_to_filename(os.path.join(resultpath,'model.keras'))
