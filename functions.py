import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import Any, Dict, Tuple, Union
from six.moves import cPickle as pickle


class MultilayerPerceptron(object):

    def __init__(self, weights: Dict, biases: Dict, dropout_rate: float = 0.0, random_seed: int = 42):
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed

    def __call__(self, x):
        layer_1 = tf.compat.v1.matmul(x, self.weights['h1']) + self.biases['b1']
        activation = tf.nn.relu(layer_1)
        layer_1_dropout = tf.compat.v1.nn.dropout(activation, rate=self.dropout_rate, seed=self.random_seed)

        return tf.compat.v1.matmul(layer_1_dropout, self.weights['out']) + self.biases['out']


class LogisticRegression(object):

    def __init__(self, weights: Dict, biases: Dict):
        super().__init__()
        self.weights = weights
        self.biases = biases

    def __call__(self, x):
        return tf.compat.v1.matmul(x, self.weights['lm']) + self.biases['out']


def load_data_splits(in_dir):
    with open(in_dir, 'rb') as f:
        data = pickle.load(f)
        X_train, y_train = reformat(data['train_dataset'], data['train_labels'])
        X_valid, y_valid = reformat(data['valid_dataset'], data['valid_labels'])
        X_test, y_test = reformat(data['test_dataset'], data['test_labels'])
        del data

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def reformat(X, y, new_shape: Union[Tuple, int] = 784, n_labels: int =10, dtype: np.dtype = np.float32):
    newshape = (-1, new_shape) if isinstance(new_shape, int) else (-1,) + new_shape
    X = X.reshape(newshape).astype(dtype)
    y = (np.arange(n_labels) == y[:, None]).astype(dtype)

    return X, y


def accuracy(y_pred, y_true):
    return 100.0 * np.sum(np.argmax(y_pred, 1) == np.argmax(y_true, 1)) / y_pred.shape[0]


def get_param(params: Dict, param_name: str, default_value: Any = None):
    return params[param_name] if param_name in params.keys() else default_value


def model_training(train_dataset, test_dataset, train_labels, test_labels, est_class, params, random_seed=42):
    tf.compat.v1.set_random_seed(random_seed)

    # define the experimental setup
    n_input = train_dataset.shape[1]
    n_classes = train_labels.shape[-1]

    c = get_param(params, 'c', 1e-2)
    lr = get_param(params, 'lr', 1e-1)

    num_steps = get_param(params, 'num_steps', 1001)
    batch_size = get_param(params, 'batch_size', 128)

    lr_decay = get_param(params, 'lr_decay', 0.0)
    decay_steps = get_param(params, 'decay_steps', 1e3)

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, n_input))
        tf_train_labels = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, n_classes))
        tf_test_dataset = tf.constant(test_dataset)

        # setup the model
        biases = {'out': tf.Variable(tf.zeros([n_classes]))}

        if est_class is MultilayerPerceptron:

            n_hidden = get_param(params, 'n_hidden', 1024)
            random_seed = get_param(params, 'random_seed', 42)
            dropout_rate = get_param(params, 'dropout_rate', 0.2)

            weights = {
                'h1': tf.Variable(tf.compat.v1.truncated_normal([n_input, n_hidden])),
                'out': tf.Variable(tf.compat.v1.truncated_normal([n_hidden, n_classes]))}
            biases['b1'] = tf.Variable(tf.zeros([n_hidden]))

            est = est_class(weights, biases, dropout_rate, random_seed)

            l2_regularization = np.sum(
                [c * tf.nn.l2_loss(weights['h1']), c * tf.nn.l2_loss(biases['b1']),
                 c * tf.nn.l2_loss(weights['out']), c * tf.nn.l2_loss(biases['out'])])

        elif est_class is LogisticRegression:

            weights = {'lm': tf.Variable(tf.compat.v1.truncated_normal([n_input, n_classes]))}

            est = est_class(weights, biases)

            l2_regularization = np.sum([c * tf.nn.l2_loss(weights['lm']), c * tf.nn.l2_loss(biases['out'])])

        # train the model
        logits = est(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + l2_regularization)

        global_step = None
        if lr_decay in params and lr_decay > 0.0:
            print('Scheduling learning rate decay...')
            global_step = tf.Variable(0)
            lr = tf.compat.v1.train.exponential_decay(lr, global_step, decay_steps, lr_decay)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(loss, global_step)

        # get predictions
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(est(tf_test_dataset))

    # test the model
    loss_logs = list()
    with tf.compat.v1.Session(graph=graph) as session:
        tf.compat.v1.global_variables_initializer().run()
        for epoch in range(num_steps):
            offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            loss_logs.append(l)
        print(f'Test accuracy: {accuracy(test_prediction.eval(), test_labels)}%')

        plt.figure(figsize=(5, 2))
        plt.plot(loss_logs)
        plt.title(f'Cross-entropy loss. Model: {est.__class__.__name__}')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        del est
