import tensorflow as tf
import shutil
import os
from toy_model import ToyModel, build_and_train


# Command line arguments
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', './models',
                           """Directory where to export the models.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the models.""")
tf.app.flags.DEFINE_bool('train', False, "Train or predict")

FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.train:
        build_and_train()
    else:
        with tf.Graph().as_default():
            net = ToyModel()
            saver = tf.train.Saver()

            with tf.Session() as sess:

                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
                export_path = os.path.join(
                            tf.compat.as_bytes(FLAGS.output_dir),
                            tf.compat.as_bytes(str(FLAGS.model_version)))

                print(export_path)
                if os.path.exists(export_path):
                    shutil.rmtree(export_path)

                builder = tf.saved_model.builder.SavedModelBuilder(export_path)

                predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(net.inputs_placeholder)

                predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(net.output)

                # build prediction signature
                prediction_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={
                            tf.saved_model.signature_constants.PREDICT_INPUTS: predict_tensor_inputs_info
                        },
                        outputs={tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_tensor_scores_info},
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                    )
                )

                # save the models
                legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
                builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                        'predict_input': prediction_signature,
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            prediction_signature

                    },
                    legacy_init_op=legacy_init_op)

                builder.save()

            print("Successfully exported Toy models version '{}' into '{}'".format(
                FLAGS.model_version, FLAGS.output_dir))


if __name__ == "__main__":
    tf.app.run()
