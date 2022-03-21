import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import csv    
from PIL import Image, ImageFilter

!mkdir /root/.kaggle
!echo '{"username":"masterfushi","key":"d0f9a8618f87e8ed0d524187356ad495"}' > /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d datamunge/sign-language-mnist
!unzip \*.zip

data = pd.read_csv("sign_mnist_train.csv")
print(data.shape)

x = data.iloc[:, 1:].values
y = data.iloc[:, :1].values.flatten()

def one_hot_encode(y):
    return np.eye(25)[y]
y_encoded = one_hot_encode(y)
print('Shape of y after encoding:', y_encoded.shape)

learning_rate = 0.001
epochs = 2000
batch_size = 128
display_step = 100
n_input = 784
n_classes = 25
dropout = 0.75

X = tf.placeholder(tf.float32, shape = [None, n_input], name="X") # Placeholder for Feature Matrix
Y = tf.placeholder(tf.float32, shape = [None, n_classes], name="Y") # Placeholder for Labels
keep_prob = tf.placeholder(tf.float32) # Placeholder for Dropout Rate
weights = {
    # Weight for Convolutional Layer 1: 5x5 filter, 1 input channel, 32 output channels
    'w1' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # Weight for Convolutional Layer 2: 5x5 filter, 32 input channels, 64 output channels
    'w2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # Weight for Fully Connected Layer 1: 49 * 64 input channels, 1024 output channels
    'w3' : tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # Weight for Convolutional Layer 1: 1024 input channels, 25(number of classes) output channels
    'w4' : tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    # Bias for Convolutional Layer 1
    'b1' : tf.Variable(tf.random_normal([32])),
    # Bias for Convolutional Layer 2
    'b2' : tf.Variable(tf.random_normal([64])),
    # Bias for Fully Connected Layer 1
    'b3' : tf.Variable(tf.random_normal([1024])),
    # Bias for Outout Layer
    'b4' : tf.Variable(tf.random_normal([n_classes]))
}
# Wrapper function for creating a Convolutional Layer
def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
# Wrapper function for creating a Pooling Layer
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')
def neural_network(x, weight, bias, dropout):
    x = tf.reshape(x, shape = [-1, 28, 28, 1])
    
    conv1 = conv2d(x, weight['w1'], bias['b1']) # Convolutional Layer 1
    conv1 = maxpool2d(conv1) # Pooling Layer 1
    
    conv2 = conv2d(conv1, weight['w2'], bias['b2']) # Convolutional Layer 1
    conv2 = maxpool2d(conv2) # Pooling Layer 1
    
    # Fully Connected Layer 1
    # Reshaping output of previous convolutional layer to fit the fully connected layer
    fc = tf.reshape(conv2, [-1, weights['w3'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weight['w3']), bias['b3']) # Linear Function
    fc = tf.nn.relu(fc) # Activation Function
    
    fc = tf.nn.dropout(fc, dropout) # Applying dropout on Fully Connected Layer
    
    out = tf.add(tf.matmul(fc, weight['w4']), bias['b4']) # Output Layer
    return out
logits = neural_network(X, weights, biases, keep_prob)
loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.33, random_state = 42)
print('X train shape', X_train.shape)
print('y train shape', y_train.shape)
print('X test shape', X_test.shape)
print('y test shape', y_test.shape)

def next_batch(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def display_images(data):
    x, y = data
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(28, 28), cmap = 'binary')
        ax.set_xlabel(chr(y[i] + 65))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

display_images(next_batch(9, x, y))

with tf.Session() as sess:
    # Running Initializer
    sess.run(init)
    cost_hist, acc_hist = [], []
    for epoch in range(1, epochs + 1):
        _x, _y = next_batch(batch_size, X_train, y_train)
        # Running Optimizer
        sess.run(train_op, feed_dict = { X : _x, Y : _y, keep_prob : dropout })
        if epoch % display_step == 0:
            # Calculating Loss and Accuracy on the current Epoch
            loss, acc = sess.run([loss_op, accuracy], feed_dict = { X : _x, Y : _y, keep_prob : 1.0 })
            loss = sum(loss)
            cost_hist.append(loss)
            acc_hist.append(acc*100)
            print('Epoch ' + str(epoch) + ', Cost: ' + str(loss) + ', Accuracy: ' + str(acc * 100) + ' %')
    print('-' * 50)
    print('\nOptimization Finished\n')
    print('Accuracy on Training Data: ' + str(sess.run(accuracy,
                                                       feed_dict = {
                                                           X : X_train,
                                                           Y : y_train,
                                                           keep_prob : 1.0
                                                       }) * 100) + ' %')
    print('Accuracy on Test Data: ' + str(sess.run(accuracy,
                                                   feed_dict = {
                                                       X : X_test,
                                                       Y : y_test,
                                                       keep_prob : 1.0
                                                   }) * 100) + ' %')

collabels = ('Epoch Number', 'Accuracy (%)')
cellData = [[0, acc_hist[0]], [1, acc_hist[1]], [2, acc_hist[2]], [3, acc_hist[3]], [4, acc_hist[4]], [5, acc_hist[5]], [6, acc_hist[6]], [7, acc_hist[7]], [8, acc_hist[8]], [9, acc_hist[9]], [10, acc_hist[10]], [11, acc_hist[11]], [12, acc_hist[12]], [13, acc_hist[13]], [14, acc_hist[14]], [15, acc_hist[15]], [16, acc_hist[16]], [17, acc_hist[17]], [18, acc_hist[18]], [19, acc_hist[19]]]

plt.plot(list(range(len(acc_hist))), acc_hist)
plt.title("Change in accuracy")
plt.table(cellText=cellData, colLabels=collabels, loc='bottom', bbox=[0, -1.3, 1.0, 1.0])
plt.show()

costLabels = ('Epoch Number', 'Cost (bits)')
costData = [[0, cost_hist[0]], [1, cost_hist[1]], [2, cost_hist[2]], [3, cost_hist[3]], [4, cost_hist[4]], [5, cost_hist[5]], [6, cost_hist[6]], [7, cost_hist[7]], [8, cost_hist[8]], [9, cost_hist[9]], [10, cost_hist[10]], [11, cost_hist[11]], [12, cost_hist[12]], [13, cost_hist[13]], [14, cost_hist[14]], [15, cost_hist[15]], [16, cost_hist[16]], [17, cost_hist[17]], [18, cost_hist[18]], [19, cost_hist[19]]]

plt.plot(list(range(len(cost_hist))), cost_hist)
plt.title("Change in cost")
plt.table(cellText=costData, colLabels=costLabels, loc='bottom', bbox=[0, -1.3, 1.0, 1.0])
plt.show()

eval_data = pd.read_csv("sign_mnist_test.csv")
print(eval_data.shape)

eval_data.head()

x_test = eval_data.iloc[:, 1:].values
y_test = eval_data.iloc[:, :1].values.flatten()
y_test = one_hot_encode(y_test)
x_test.shape, y_test.shape

with tf.Session() as sess:
    sess.run(init) # Running Initializer
    cost_hist, acc_hist = [], []
    for epoch in range(1, epochs + 1):
        _x, _y = next_batch(batch_size, x, y_encoded)
        # Running Optimizer
        sess.run(train_op,
                 feed_dict = {
                     X : _x,
                     Y : _y,
                     keep_prob : dropout
                 })
        if epoch % display_step == 0:
            # Calculating Loss and Accuracy on the current Epoch
            loss, acc = sess.run([loss_op, accuracy],
                                 feed_dict = {
                                     X : _x,
                                     Y : _y,
                                     keep_prob : 1.0
                                 })
            loss = sum(loss)
            cost_hist.append(loss)
            acc_hist.append(acc)
            print('Epoch ' + str(epoch) + ', Cost: ' + str(loss) + ', Accuracy: ' + str(acc * 100) + ' %')
    print('-' * 50)
    print('\nOptimization Finished\n')
    print('Accuracy after training on whole dataset Data: ' + str(sess.run(accuracy,
                                                       feed_dict = {
                                                           X : x,
                                                           Y : y_encoded,
                                                           keep_prob : 1.0
                                                       }) * 100) + ' %')
    W = sess.run(weights)
    B = sess.run(biases)

X = tf.placeholder(tf.float32, shape = [None, n_input]) # Placeholder for Feature Matrix
Y = tf.placeholder(tf.float32, shape = [None, n_classes]) # Placeholder for Labels
keep_prob = tf.placeholder(tf.float32) # Placeholder for Dropout Rate
y_pred = neural_network(X, W, B, 1.0)

def get_prediction(img):
    with tf.Session() as sess:
        pred = sess.run(y_pred, feed_dict = { X : img, keep_prob : 1.0 })
    img = img.reshape(28, 28) #takes resolution of image and rescales it to 28X28 to help improve speeds and simplifiy the calculations
    pred = list(pred.flatten())
    pred = chr(pred.index(max(pred)) + 65)
    return (img, pred)

word = ""

for i in range(5):
  randind = random.randrange(0, len(x_test))
  image, pred = get_prediction(x_test[randind].reshape(1, 784))
  plt.imshow(image, cmap = 'binary')
  #plt.title("Predicted value: " + pred)
  plt.show()
  word = word + pred

print(word)

for key in W.keys():
    np.save(key, W[key])
for key in B.keys():
    np.save(key, B[key])