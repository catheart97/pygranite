#!/bin/bash

for env in `ls venv`
do
    if [[ "$env" =~ ^lin-* ]]; then
        source venv/$env/bin/activate
        python -m pip install build wheel 
        python -m build --wheel
    fi
done