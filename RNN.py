import numpy as np
import scipy.io as sc
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import classification_report


# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_)).astype(int)
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


"""
This function is used to create a RNN
Input: X, weights, biases, number of layers, number of inputs, number of hidden layers

"""
def rnn(X, weights, biases, n_layers, n_inputs, n_hidden4_units):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    X = tf.reshape(X, [-1, int(n_inputs)])

    # into hidden
    X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])
    X_hidd1 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']

    if n_layers == 5:
        X_hidd1 = tf.matmul(X_hidd1, weights['hidd3']) + biases['hidd3']
    elif n_layers == 6:
        X_hidd1 = tf.matmul(X_hidd1, weights['hidd3']) + biases['hidd3']
        X_hidd1 = tf.matmul(X_hidd1, weights['hidd4']) + biases['hidd4']

    X_in = tf.reshape(X_hidd1, [-1, n_steps, n_hidden4_units])
    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


# ---------------- EEG data ----------------
feature = sc.loadmat("S1_nolabel6.mat")
all = feature['S1_nolabel6']
all = all[0:28000]

data_size = all.shape[0]
np.random.shuffle(all)  # shuffle all the data

n_fea = 64
n_classes = 6

# extract the feature columns
feature_all = all[:, 0:n_fea]

# minus Direct Current, DC=4200 which is determined by the EEG equipment.
feature_all = feature_all - 4200

#  z-score scaling
feature_normalized = preprocessing.scale(feature_all)
label_all = all[:, n_fea:n_fea+1]
all = np.hstack((feature_normalized, label_all))
# print(all.shape)

# --------------- RFID data --------------------
# feature = sc.loadmat("rssi_nonmix_all.mat")
# all = feature['rssi_nonmix_all']
#
# n_fea = 12
# n_classes = 22
# data_size = all.shape[0]
# np.random.shuffle(all)
# # extract the feature columns
# feature_all = all[:, 0:n_fea]
#
# # z-score
# feature_normalized = preprocessing.scale(feature_all)
# label_all = all[:, n_fea:n_fea + 1]
# all = np.hstack((feature_normalized, label_all))


"""
This data set will take long to run
"""
# -------------- PAMAP2 data ------------------------
# feature = sc.loadmat("AR_6p_8c.mat")
# all = feature['AR_6p_8c']
# all = all[0:200000]
# np.random.shuffle(all)
#
# all = all[:10000] # this is for fast run, comment this line when doing actual run
#
# n_fea = 48
# n_classes = 8
# data_size = all.shape[0]
#
# feature_all = all[:, 0:n_fea]
#
# feature_normalized = preprocessing.scale(feature_all)
# label_all = all[:, -1:]
# all = np.hstack((feature_normalized, label_all))


# use the first subject as testing subject
train_data = all[0:int(data_size * 0.75)]
test_data = all[int(data_size * 0.75):data_size]

# shuffle all the train and test data
np.random.shuffle(train_data)
np.random.shuffle(test_data)


n_steps = 1
# get training and test batches
feature_training = train_data[:, 0:n_fea]
feature_training = feature_training.reshape([int(data_size * 0.75), n_steps, n_fea // n_steps])
feature_testing = test_data[:, 0:n_fea]
feature_testing = feature_testing.reshape([int(data_size * 0.25), n_steps, n_fea // n_steps])

label_training = train_data[:, n_fea]
label_training = one_hot(label_training)
label_testing = test_data[:, n_fea]
label_testing = one_hot(label_testing)

# batch split
a = feature_training
b = feature_testing

batch_size = int(data_size * 0.25)
train_fea = []
n_group = 3
for i in range(n_group):
    f = a[(0 + batch_size * i):(batch_size + batch_size * i)]
    train_fea.append(f)
# print(train_fea[0].shape)

train_label = []
for i in range(n_group):
    f = label_training[(0 + batch_size * i):(batch_size + batch_size * i), :]
    train_label.append(f)
# print(train_label[0].shape)


def rnn_run(lr, lam, n_layers, nodes):
    tf.reset_default_graph()
    # hyperparameters
    n_inputs = n_fea / n_steps
    # n_steps =  # time steps
    n_hidden1_units = nodes  # neurons in hidden layer
    n_hidden2_units = nodes
    n_hidden3_units = nodes
    n_hidden4_units = nodes

    # tf Graph input

    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="x")
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Define weights

    weights = {
        # (28, 128)
        'in': tf.Variable(tf.random_normal([int(n_inputs), int(n_hidden1_units)]), trainable=True),
        'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),
        # (128,128)
        'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
        'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
        'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
        # (128, 10)
        'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
    }

    biases = {
        # (128, )
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
        # (128,)
        'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units])),
        'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
        'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
        # (10, )
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True)
    }

    pred = rnn(x, weights, biases, n_layers, n_inputs, n_hidden4_units)

    # L2 loss prevents this overkill neural network to overfit the data
    l2 = lam * sum(tf.nn.l2_loss(tf_var) for tf_var in
                   tf.trainable_variables())

    # Softmax loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + l2

    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    H = []

    with tf.Session(config=config) as sess:
        sess.run(init)
        step = 0
        while step < 2000:  # 2000 iterations
            for i in range(n_group):
                sess.run(train_op, feed_dict={
                    x: train_fea[i],
                    y: train_label[i],
                })
            if step % 500 == 0:
                pp = sess.run(pred, feed_dict={x: b, y: label_testing})

                hh = sess.run(accuracy, feed_dict={
                    x: b,
                    y: label_testing,
                })

                H.append(hh)
                print("don't worry, I'm running!")

            step += 1
        print("lr :", lr, ", lambda:", lam, "number of layers: ", n_layers, "number of nodes: ", nodes, ",Acc max",
              max(H))
        # print the classification report
        print(classification_report(np.argmax(pp, axis=1), np.argmax(label_testing, axis=1), digits=4))
        return max(H)
