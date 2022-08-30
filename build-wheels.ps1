#!/bin/pwsh

foreach ($env in Get-ChildItem venv -Directory -Name) {
    if ($env.StartsWith("win")) {
        Invoke-Expression ("./venv/" + $env + "/Scripts/activate.ps1")
        python -m pip install build wheel
        python -m build --wheel
    }
}