import tensorflow as tf
from flask import Flask, jsonify
import os
import logging
import argparse
from tf_serving_client import TfServingClient

app = Flask(__name__)
file_handler = logging.FileHandler("api.log")
file_handler.setLevel(logging.DEBUG)
app.logger.addHandler(file_handler)

S2T_HOST="s2t:{}".format(os.environ['SERVER_PORT'])
T2T_HOST="t2t:{}".format(os.environ['SERVER_PORT'])
T2S_HOST="t2s:{}".format(os.environ['SERVER_PORT'])

# Not yet clear if we have 1 client that can connect to each service or one client per service
client = TfServingClient(S2T_HOST)


@app.route('/ping')
def ping():
        return jsonify({'pong': 'response'})


@app.route("/")
def root():
        y_json = client.infer(model_name='1',
                              data=[[3, 1, 2, 3, 4, 5, 6, 7, 8, 8]])
        print(y_json)
        return jsonify(y_json)


@app.route("/s2t")
def s2t():
        y_json = client.infer(model_name='s2t',
                              data=[[3, 1, 2, 3, 4, 5, 6, 7, 8, 8]])
        print(y_json)
        return jsonify(y_json)


@app.route("/t2t")
def t2t():
        y_json = client.infer(model_name='t2t',
                              data=[[3, 1, 2, 3, 4, 5, 6, 7, 8, 8]])
        print(y_json)
        return jsonify(y_json)


@app.route("/t2s")
def t2s():
        y_json = client.infer(model_name='t2s',
                              data=[[3, 1, 2, 3, 4, 5, 6, 7, 8, 8]])
        print(y_json)
        return jsonify(y_json)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
