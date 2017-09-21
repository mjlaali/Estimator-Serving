# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
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
FLAGS = tf.app.flags.FLAGS


class TfServingClient(object):

    def __init__(self, server):
        self.server = server
        self.host, self.port = self.server.split(':')
        self.port = int(self.port)
        self.channel = implementations.insecure_channel(self.host, self.port)

    def infer(self, data=[[]], model_version=None, model_name='1'):
        stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

        # Send request
        # See prediction_service.proto for gRPC request/response details.
        request = predict_pb2.PredictRequest()

        request.model_spec.name = model_name
        request.model_spec.signature_name = 'predict_input'
        if model_version:
            request.model_spec.version.value = FLAGS.model_version

        tensor_proto = tf.contrib.util.make_tensor_proto(data, dtype=tf.float32)
        request.inputs[tf.saved_model.signature_constants.PREDICT_INPUTS].CopyFrom(tensor_proto)
        result = stub.Predict(request, 60.0)
        return json.loads(MessageToJson(result))


def main(_):
    client = TfServingClient(FLAGS.server)
    print(client.infer(model_name=FLAGS.model_name, model_version=FLAGS.model_version,
                       data=[[3, 1, 2, 3, 4, 5, 6, 7, 8, 8]]))

if __name__ == '__main__':
    tf.app.run()
