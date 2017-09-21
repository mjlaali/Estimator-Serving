#!/bin/bash
echo "Bring docker network down..."
docker-compose down

echo "Building docker images...."
docker build -t eai-nlp-t2t -f ./Docker-tf-serving --build-arg SERVER_PORT=9000 --build-arg SERVICE_NAME=t2t .
docker build -t eai-nlp-api -f ./Docker-api .

echo "Bring docker network up..."
docker-compose up -d

echo "Network up. Monitoring..."
sleep 2
docker ps
docker-compose logs
