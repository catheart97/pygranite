#!/bin/bash

ls /usr/bin/ | grep -w python3.[0-9]*$ | while read -r exec ; do
    VERSION=`$exec --version`
    VERSION=`echo ${VERSION#"Python"} | tr "." " "`

    IFS=' ' read -r -a VARRAY <<< "$VERSION"
    VERSION=${VARRAY[0]}""${VARRAY[1]}
    /usr/bin/$exec -m pip install virtualenv build
    /usr/bin/$exec -m virtualenv -p=$exec ./venv/lin-$VERSION
done