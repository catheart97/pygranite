#!/bin/pwsh

if (-not (Test-Path -Path venv)) {
    mkdir venv
    ./config-venv.ps1
    wsl --distribution "Ubuntu-18.04" -- './config-venv.sh'
}

./build-wheels.ps1
wsl --distribution "Ubuntu-18.04" -- "./build-wheels.sh"