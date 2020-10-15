import math
import random
import zipfile
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from six.moves import cPickle as pickle
from typing import Any, Dict, List, Tuple, Union, Callable


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


class CNN2D(object):

    def __init__(self, weights: Dict, biases: Dict, activation_func: Callable = tf.nn.relu):
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.strides = [1, 1, 1, 1]
        self.kernels = [1, 2, 2, 1]
        self.activation_func = activation_func

    def __call__(self, x):
        x = tf.nn.conv2d(x, self.weights['w1'], self.strides, padding='SAME') + self.biases['b1']
        x = self.activation_func(x)
        x = tf.nn.max_pool(x, self.kernels, self.kernels, padding='SAME')
        x = tf.nn.conv2d(x, self.weights['w2'], self.strides, padding='SAME') + self.biases['b2']
        x = self.activation_func(x)
        x = tf.nn.max_pool(x, self.kernels, self.kernels, padding='SAME')
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [shape[0], np.prod(shape[1:4])])
        x = tf.matmul(x, self.weights['w3']) + self.biases['b3']
        x = self.activation_func(x)

        return tf.compat.v1.matmul(x, self.weights['w4']) + self.biases['b4']


class LeNet5(object):

    """
    LeNet-5 model. Original architecture is more details: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf .
    """

    def __init__(self, weights: Dict, biases: Dict, activation_func: Callable = tf.nn.sigmoid):
        super().__init__()
        self.weights = weights
        self.biases = biases
        self.strides = [1, 1, 1, 1]
        self.kernels = [1, 2, 2, 1]
        self.activation_func = activation_func

    def conv2d(self, x, weights, biases):
        return tf.nn.conv2d(x, weights, strides=self.strides, padding='VALID') + biases

    def avgpool2d(self, x):
        return tf.nn.avg_pool(x, ksize=self.kernels, strides=self.kernels, padding='VALID')

    def __call__(self, x):
        x = self.conv2d(x, self.weights['w1'], self.biases['b1'])
        x = self.avgpool2d(x)
        x = self.activation_func(x)
        x = self.conv2d(x, self.weights['w2'], self.biases['b2'])
        x = self.avgpool2d(x)
        x = self.activation_func(x)
        x = tf.compat.v1.layers.flatten(x)
        x = tf.compat.v1.matmul(x, self.weights['w3']) + self.biases['b3']
        x = self.activation_func(x)

        return tf.compat.v1.matmul(x, self.weights['w4']) + self.biases['b4']


class word2vec(object):

    def __init__(self, vocabulary_size: int = 50000, embedding_size: int = 128, negative_samples: int = 64):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = negative_samples

    def __call__(self, x):
        embeddings = tf.Variable(
            tf.compat.v1.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

        weights = get_weigths(shape=[self.vocabulary_size, self.embedding_size],
                              stddev=1.0 / math.sqrt(self.embedding_size))
        biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        embed = tf.nn.embedding_lookup(embeddings, x)

        return embed, weights, biases, embeddings


def load_data_splits(in_dir, new_shape: Union[Tuple, int] = 784, n_labels: int = 10):
    with open(in_dir, 'rb') as f:
        data = pickle.load(f)
        X_train, y_train = reformat(data['train_dataset'], data['train_labels'], new_shape, n_labels)
        X_valid, y_valid = reformat(data['valid_dataset'], data['valid_labels'], new_shape, n_labels)
        X_test, y_test = reformat(data['test_dataset'], data['test_labels'], new_shape, n_labels)
        del data

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def reformat(X, y, new_shape: Union[Tuple, int] = 784, n_labels: int = 10, dtype: np.dtype = np.float32):
    newshape = (-1, new_shape) if isinstance(new_shape, int) else (-1,) + new_shape
    X = X.reshape(newshape).astype(dtype)
    y = (np.arange(n_labels) == y[:, None]).astype(dtype)

    return X, y


def get_weigths(shape: Union[List, Tuple], mean: float = 0.0, stddev: float = 0.1):
    return tf.Variable(tf.compat.v1.truncated_normal(shape, mean, stddev))


def accuracy(y_pred, y_true):
    return 100.0 * np.sum(np.argmax(y_pred, 1) == np.argmax(y_true, 1)) / y_pred.shape[0]


def get_param(params: Dict, param_name: str, default_value: Any = None):
    return params[param_name] if param_name in params.keys() else default_value


def model_training(train_dataset, test_dataset, train_labels, test_labels, est_class, params, random_seed=42):
    tf.compat.v1.set_random_seed(random_seed)

    # define the experimental setup
    n_input = train_dataset.shape[1]
    n_classes = train_labels.shape[-1]

    zscore = get_param(params, 'zscore', False)
    num_steps = get_param(params, 'num_steps', 1001)
    batch_size = get_param(params, 'batch_size', 128)

    c = get_param(params, 'c', 1e-2)
    lr = get_param(params, 'lr', 1e-1)
    lr_decay = get_param(params, 'lr_decay', 0.0)
    # decay_steps = get_param(params, 'decay_steps', 1e3)
    l2_regularization = get_param(params, 'l2_regularization', False)

    graph = tf.Graph()
    with graph.as_default():

        if est_class is LeNet5:
            # reformat data to LeNet5-acceptable format
            train_dataset = np.pad(train_dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            test_dataset = np.pad(test_dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

        if zscore:
            train_mean = np.mean(train_dataset)
            train_std = np.std(train_dataset)
            train_dataset = (train_dataset - train_mean) / train_std
            test_dataset = (test_dataset - train_mean) / train_std

        if est_class in [CNN2D, LeNet5]:
            height, width, num_channels = train_dataset.shape[1:]
            tf_train_dataset = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, height, width, num_channels))
        else:
            tf_train_dataset = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, n_input))

        if est_class is LeNet5:
            # encode classes w.r.t. LeNet5 conventions
            tf_train_labels = tf.one_hot(tf.compat.v1.placeholder(tf.int32, shape=None), n_classes)
        else:
            tf_train_labels = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, n_classes))

        tf_test_dataset = tf.constant(test_dataset)

        # setup the model
        biases = {'out': tf.Variable(tf.zeros([n_classes]))}

        if est_class is LogisticRegression:
            weights = {'lm': get_weigths([n_input, n_classes])}

            est = est_class(weights, biases)

        else:
            n_hidden = get_param(params, 'n_hidden', 1024)

            if est_class is MultilayerPerceptron:
                random_seed = get_param(params, 'random_seed', 42)
                dropout_rate = get_param(params, 'dropout_rate', 0.2)
                weights = {
                    'h1': get_weigths([n_input, n_hidden], stddev=1.0),
                    'out': get_weigths([n_hidden, n_classes], stddev=1.0)
                }
                biases['b1'] = tf.Variable(tf.zeros([n_hidden]))

                est = est_class(weights, biases, random_seed, dropout_rate)

            elif est_class in [CNN2D, LeNet5]:
                patch_size = get_param(params, 'patch_size', 5)

                if est_class is CNN2D:
                    depth = get_param(params, 'depth', 16)
                    weights = {
                        'w1': get_weigths([patch_size, patch_size, num_channels, depth]),
                        'w2': get_weigths([patch_size, patch_size, depth, depth]),
                        'w3': get_weigths([height // 4 * width // 4 * depth, n_hidden]),
                        'w4': get_weigths([n_hidden, n_classes])
                    }
                    biases = {
                        'b1': tf.Variable(tf.zeros([depth])),
                        'b2': tf.Variable(tf.constant(1.0, shape=[depth])),
                        'b3': tf.Variable(tf.constant(1.0, shape=[n_hidden])),
                        'b4': tf.Variable(tf.constant(1.0, shape=[n_classes]))
                    }
                    activation_func = get_param(params, 'activation_func', tf.nn.relu)

                elif est_class is LeNet5:
                    depth ={'l1': 6, 'l2': 16, 'l3': 120, 'l4': 84}
                    weights = {
                        'w1': get_weigths([patch_size, patch_size, 1, depth['l1']]),
                        'w2': get_weigths([patch_size, patch_size, depth['l1'], depth['l2']]),
                        'w3': get_weigths((5 * 5 * depth['l2'], depth['l3'])),
                        'w4': get_weigths((depth['l3'], n_classes))
                    }
                    biases = {
                        'b1': tf.Variable(tf.zeros([depth['l1']])),
                        'b2': tf.Variable(tf.constant(1.0, shape=[depth['l2']])),
                        'b3': tf.Variable(tf.constant(1.0, shape=[depth['l3']])),
                        'b4': tf.Variable(tf.constant(1.0, shape=[n_classes]))
                    }
                    activation_func = get_param(params, 'activation_func', tf.nn.sigmoid)

                est = est_class(weights, biases, activation_func)

        logits = est(tf_train_dataset)
        model_name = est.__class__.__name__

        if l2_regularization:
            print('Applying L2 regularization...')
            l2_regularized = np.sum(
                [*[c * tf.nn.l2_loss(w) for w in weights.values()], *[c * tf.nn.l2_loss(b) for b in biases.values()]]
            )
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + l2_regularized
            )
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        global_step = None
        if lr_decay > 0.0:
            print('Scheduling learning rate decay...')
            global_step = tf.Variable(0)
            lr = tf.compat.v1.train.exponential_decay(lr, global_step, num_steps, lr_decay)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(loss, global_step)

        # get predictions
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(est(tf_test_dataset))

    # train and test the model
    with tf.compat.v1.Session(graph=graph) as session:
        tf.compat.v1.global_variables_initializer().run()
        loss_logs = list()
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
        plt.title(f'Cross-entropy loss. Model: {model_name}')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    del session
    del est
    del graph


def read_zip(filename):
    """
    Extract the first file enclosed in a zip file as a list of words.
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, vocabulary_size: int):
    """
    Build the dictionary and replace rare words with UNK token.
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary


def get_batch(data, batch_size, num_skips, skip_window):
    """
    Function to generate a training batch for the skip-gram model.
    """
    data_index = 0
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


def plot_2d_projection(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(15, 15))
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.xlabel('tSNE_1', fontsize=16)
        plt.ylabel('tSNE_2', fontsize=16)
    plt.show()
