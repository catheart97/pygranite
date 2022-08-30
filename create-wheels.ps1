#!/bin/pwsh

if (-not (Test-Path -Path venv)) {
    mkdir venv
    ./config-venv.ps1
    bash -c './config-venv.sh'
}

./build-wheels.ps1
bash -c "./build-wheels.sh"