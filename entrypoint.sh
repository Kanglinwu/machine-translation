#!/bin/bash

if [ "$(ls -A /app/models)" ]
then
    echo "/app/models is not empty"
else 
    echo "/app/models is empty"
     if [ -d "./models" ]; then
        echo "Copying models from host to /app/models"
         cp -r ./models/* /app/models/
        echo "Finished copying models"
    else
        echo "./models not found on host, skipping copy"
    fi
fi

exec "$@"