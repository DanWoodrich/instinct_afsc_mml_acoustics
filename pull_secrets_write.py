import json
import os
from google.cloud import secretmanager
from google.api_core import exceptions

def access_secret_to_R(project_id, secret_id, output_file_path, version_id="latest"):
    """
    Accesses a secret version from GCP Secret Manager, parses its JSON content,
    and writes the 'uname' and 'pw' fields to a .ini-style file.

    Assumes the secret payload is a JSON string like: {"uname": "user", "pw": "pass"}
    """
    try:
        # Create the Secret Manager client.
        # ADC (Application Default Credentials) are used automatically.
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

        # Access the secret version
        print(f"Accessing secret: {name}")
        response = client.access_secret_version(request={"name": name})

        # Decode the payload from bytes to a string
        payload_str = response.payload.data.decode("UTF-8")

        # Parse the JSON string
        try:
            secret_data = json.loads(payload_str)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from secret: {secret_id}")
            return

        # Extract username and password
        username = secret_data.get("uname")
        password = secret_data.get("pw")

        if not (username and password):
            print(f"Error: Secret JSON does not contain 'uname' or 'pw' fields.")
            return

        # Write the data to the output .ini file
        print(f"Writing credentials to {output_file_path}...")
        with open(output_file_path, "w") as f:
            f.write(f"dsn_uid = '{username}'\n")
            f.write(f"dsn_pwd = '{password}'\n")
            #write static / nonsensitive stuff
            f.write(f"dsn_port = '5432'\n")
            f.write(f"dsn_hostname = '10.2.0.2'")
        
        print("Successfully wrote credentials to file.")

    except exceptions.NotFound:
        print(f"Error: Secret or project not found: {name}")
    except exceptions.PermissionDenied:
        print(f"Error: Permission denied. Ensure ADC has 'Secret Manager Secret Accessor' role.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # !!! REPLACE with your project ID and secret ID !!!
    YOUR_PROJECT_ID = "ggn-nmfs-pamdata-prod-1"
    YOUR_SECRET_ID = "test-secret-pgpamdb"
    OUTPUT_FILE = "./etc/key.R"
    # ---------------------

    access_secret_to_R(YOUR_PROJECT_ID, YOUR_SECRET_ID, OUTPUT_FILE)
