#!/bin/bash
set -e

# Execute the python script first
python3 ./lib/user/pull_secrets_write.py

# Then, execute the main executable with all arguments passed to the script
./bin/instinct "$@"


