import numpy as np
import scipy.io as sc
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import classification_report


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_)).astype(int)
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


# the CNN code
def compute_accuracy(v_xs, v_ys, sess3, prediction):
    y_pre = sess3.run(prediction, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess3.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: keep})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


# ---------------- EEG data ----------------
feature = sc.loadmat("S1_nolabel6.mat")
all = feature['S1_nolabel6']
all = all[0:28000]

data_size = all.shape[0]
np.random.shuffle(all)  # shuffle all the data

n_fea = 64
n_class = 6

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
# n_class = 22
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
# n_class = 8
# data_size = all.shape[0]
#
# feature_all = all[:, 0:n_fea]
#
# feature_normalized = preprocessing.scale(feature_all)
# label_all = all[:, -1:]
# all = np.hstack((feature_normalized, label_all))

# use the first subject as testing subject
train_data = all[0:int(data_size * 0.75)]
test_data = all[int(data_size * 0.75):int(data_size)]
np.random.shuffle(train_data)  # mix eeg_all
np.random.shuffle(test_data)

n_steps = 1

feature_training = train_data[:, 0:n_fea]
feature_testing = test_data[:, 0:n_fea]

label_training = train_data[:, n_fea]
label_training = one_hot(label_training)
label_testing = test_data[:, n_fea]
label_testing = one_hot(label_testing)

a = feature_training
b = feature_testing

batch_size = int(data_size * 0.25)
train_fea = []
n_group = 3

for i in range(n_group):
    f = a[(0 + batch_size * i):(batch_size + batch_size * i)]
    train_fea.append(f)
print(train_fea[1].shape)

train_label = []
for i in range(n_group):
    f = label_training[(0 + batch_size * i):(batch_size + batch_size * i), :]
    train_label.append(f)

print(train_label[2].shape)

keep = 1

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, n_fea])  # 1*64
ys = tf.placeholder(tf.float32, [None, n_class])  # 6 is the classes of the data
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1, n_fea, 1])
print(x_image.shape)  # [n_samples, 1,64,1]


def cnn_run(lr, lam, n_l, n_n):
    size2 = 192
    l_1, w_1 = 1, 2
    global x_image
    # conv 1
    conv1 = tf.layers.conv2d(inputs=x_image, filters=5, kernel_size=[2, 2], padding="same",
                             activation=tf.nn.relu)
    if n_l == 2:
        # x_image = pool1
        conv1 = tf.layers.conv2d(inputs=conv1, filters=10, kernel_size=[2, 2], padding="same",
                                 activation=tf.nn.relu)
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[l_1, w_1], strides=[l_1, w_1])
    elif n_l == 3:
        # x_image = pool1
        conv1 = tf.layers.conv2d(inputs=conv1, filters=15, kernel_size=[2, 2], padding="same",
                                 activation=tf.nn.relu)
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[l_1, w_1], strides=[l_1, w_1])

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[l_1, w_1], strides=[l_1, w_1])

    fc1 = tf.contrib.layers.flatten(pool1)  # flatten the pool 2

    ## fc1 layer ##
    fc3 = tf.layers.dense(fc1, units=size2, activation=tf.nn.sigmoid)
    h_fc1 = tf.nn.sigmoid(fc3)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([size2, n_class])
    b_fc2 = bias_variable([n_class])
    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # the error between prediction and real data
    l2 = lam * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys)) + l2  # Softmax loss

    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)  # learning rate is 0.0001
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess3 = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess3.run(init)
    # start = time.process_time()
    step = 0
    H = []
    while step < 2000:
        for i in range(n_group):
            sess3.run(train_step, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob: keep})
        if step % 500 == 0:
            # cost = sess3.run(cross_entropy, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
            # print('the step is:',step,',the acc is',compute_accuracy(b, label_testing),', the cost is', cost)
            pp = sess3.run(prediction, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
            hh = compute_accuracy(b, label_testing, sess3, prediction)
            H.append(hh)
            print("Don't worry, I'm running!")

        step += 1
    print('---------------')
    print("lr :", lr, "lambda: ", lam, "n_l: ", n_l, "n_n: ", n_n, ",Acc max", max(H))
    print(classification_report(np.argmax(pp, axis=1), np.argmax(label_testing, axis=1), digits=4))

    return max(H)
