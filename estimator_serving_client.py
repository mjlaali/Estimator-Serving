# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow_serving.apis import classification_pb2
import tensorflow as tf
import json
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from google.protobuf.json_format import MessageToJson

# Command line arguments
tf.app.flags.DEFINE_string('server', '0.0.0.0:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('model_version', 1, 'models version')
tf.app.flags.DEFINE_string('model_name', '1', "Model name")
tf.app.flags.DEFINE_string('model', 'iris',
                           'Model name.')
FLAGS = tf.app.flags.FLAGS


class EstimatorServingClient(object):

    def __init__(self, server):
        self.server = server
        self.host, self.port = self.server.split(':')
        self.port = int(self.port)
        self.channel = implementations.insecure_channel(self.host, self.port)

    def infer(self, data=[[]], model_version=None, model_name='1'):
        stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = FLAGS.model
        request.model_spec.signature_name = 'serving_default'

        feature_dict = {"x": tf.train.Feature(float_list=tf.train.FloatList(value=[1., 2., 3., 4.]))}

        label = 0

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example.SerializeToString()

        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

        result_future = stub.Predict.future(request, 5.0)
        prediction = result_future.result().outputs['scores']

        print('Prediction: ' + str(prediction))

def main(_):
    client = EstimatorServingClient(FLAGS.server)
    print(client.infer(model_name=FLAGS.model_name, model_version=FLAGS.model_version,
                       data=[[3, 1, 2, 3]]))

if __name__ == '__main__':
    tf.app.run()
