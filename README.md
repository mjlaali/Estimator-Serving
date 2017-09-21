# Instructions for running project
This assumes you are running python 3.x[6]
Assuming you want to use [the iris example](https://www.tensorflow.org/get_started/estimator), here's what you need to do


## Create virtual environment
```
python3 -m venv venv && . ./venv/bin/activate
```

## Install dev dependencies
```
pip install -r dev-requirements.txt
```

## Create model and train
In this step we instantiate a tensorflow model, and train it. The training phase outputs some checkpoint that we later use to freeze the Graph
```
python estimator_model.py
```

By default, the checkpoint directory will be `./checkpoints/`


Netx, we save the model graph into a portable format: Protocol Buffers.
The estimator_model script need to make a few assumptions about the model architecture, namely, the input and output nodes of the graph.
By default, the output will be `./model/iris_model/TIMESTAMP`. 


## Spin up environment
In this step we have everything ready to spin up an environment with a REST API that interfaces with each services running a tensorflow serving instance

```
./compose-network.sh
```

## Testing modules individually
If you want to test a specific module, you might want to use the `estimator_serving_client.py` directly to interact with your served model.

```
docker build -t eai-nlp-iris -f ./Docker-tf-serving --build-arg SERVER_PORT=9000 --build-arg SERVICE_NAME=iris .
```
```
docker run -it -p 9000:9000 -d eai-nlp-iris
```

```
python estimator_serving_client.py --model_version=TIMESTAMP --model_name=toy --server=0.0.0.0:9000
```
Right now the data passed to the client is hardcoded, but you could extend it to  pass in data from file or command line.


## Use services
After the network is up and running (make sure that `docker ps` lists all containers as running), you can use the services.
The entrypoint is `0.0.0.0:5004/` and you can use the `/s2t`, `t2t` and `t2s` routes to access each services individually
