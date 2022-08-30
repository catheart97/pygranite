#!/bin/pwsh

py -m pip install virtualenv
py -m pip install build

$AVAILABLE_ON_WINDOWS = py -0p
$AVAILABLE_ON_WINDOWS_PATHS = [System.Collections.ArrayList]@()
foreach ($ver in $AVAILABLE_ON_WINDOWS) {
    if ($ver.EndsWith("python.exe")) {
        $AVAILABLE_ON_WINDOWS_PATHS.Add(($ver -Split ' ')[-1])
    }
}
Write-Host $AVAILABLE_ON_WINDOWS_PATHS
foreach ($exec in $AVAILABLE_ON_WINDOWS_PATHS) {
    $VERSION = ((Invoke-Expression $exec" --version") -Split " ")[-1].Split('.')
    $VERSION = ($VERSION[0] + "" + $VERSION[1])
    py -m virtualenv -p="$exec" ./venv/win-$VERSION
}