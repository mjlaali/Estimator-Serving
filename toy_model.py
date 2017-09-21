import tensorflow as tf
import numpy as np


class ToyModel(object):
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
        with tf.variable_scope('Placeholder'):
            self.inputs_placeholder = tf.placeholder(tf.float32, name='inputs_placeholder', shape=[None, 10])
            self.labels_placeholder = tf.placeholder(tf.float32, name='labels_placeholder', shape=[None, 1])

        self.output, self.loss = self.model_loss()
        self.accuracy = self.model_accuracy()

        self.train_op = self.model_op()

    def model_loss(self):
        with tf.variable_scope('NN'):
            W1 = tf.get_variable('W1', shape=[10, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
            b1 = tf.get_variable('b1', shape=[1], initializer=tf.constant_initializer(0.1))
            W2 = tf.get_variable('W2', shape=[10, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
            b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))

            a = tf.nn.relu(tf.matmul(self.inputs_placeholder, W1) + b1)
            a2 = tf.nn.relu(tf.matmul(self.inputs_placeholder, W2) + b2)

            y = tf.divide(tf.add(a, a2), 2)

        with tf.variable_scope('Loss'):
            loss = tf.reduce_sum(tf.square(y - self.labels_placeholder) / 2)

        return y, loss

    def model_accuracy(self):
        with tf.variable_scope('Accuracy'):
            predictions = tf.greater(self.output, 0.5, name="predictions")
            correct_predictions = tf.equal(predictions, tf.cast(self.labels_placeholder, tf.bool), name="correct_predictions")
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            return accuracy

    def model_op(self):
        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = adam.minimize(self.loss)
        return train_op


def build_and_train():
    model = ToyModel()

    # generate_data
    inputs = np.random.choice(10, size=[10000, 10])
    labels = (np.sum(inputs, axis=1) > 45).reshape(-1, 1).astype(np.float32)
    print('inputs.shape:', inputs.shape)
    print('labels.shape:', labels.shape)

    test_inputs = np.random.choice(10, size=[100, 10])
    test_labels = (np.sum(test_inputs, axis=1) > 45).reshape(-1, 1).astype(np.float32)
    print('test_inputs.shape:', test_inputs.shape)
    print('test_labels.shape:', test_labels.shape)

    batch_size = 32
    epochs = 10

    batches = []
    print("%d items in batch of %d gives us %d full batches and %d batches of %d items" % (
        len(inputs),
        batch_size,
        len(inputs) // batch_size,
        batch_size - len(inputs) // batch_size,
        len(inputs) - (len(inputs) // batch_size) * 32)
    )
    for i in range(len(inputs) // batch_size):
        batch = [ inputs[batch_size*i:batch_size*i+batch_size], labels[batch_size*i:batch_size*i+batch_size] ]
        batches.append(list(batch))
    if (i + 1) * batch_size < len(inputs):
        batch = [ inputs[batch_size*(i + 1):],labels[batch_size*(i + 1):] ]
        batches.append(list(batch))
    print("Number of batches: %d" % len(batches))
    print("Size of full batch: %d" % len(batches[0]))
    print("Size if final batch: %d" % len(batches[-1]))

    global_count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for batch in batches:
                # print(batch[0].shape, batch[1].shape)
                train_loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                    model.inputs_placeholder: batch[0],
                    model.labels_placeholder: batch[1]
                })
                # print('train_loss: %d' % train_loss)

                if global_count % 100 == 0:
                    acc = sess.run(model.accuracy, feed_dict={
                        model.inputs_placeholder: test_inputs,
                        model.labels_placeholder: test_labels
                    })
                    print('accuracy: %f' % acc)
                global_count += 1

        acc = sess.run(model.accuracy, feed_dict={
            model.inputs_placeholder: test_inputs,
            model.labels_placeholder: test_labels
        })
        print("final accuracy: %f" % acc)

        saver = tf.train.Saver()
        last_chkp = saver.save(sess, './checkpoints/1.ckpt')

if __name__ == '__main__':
    build_and_train()