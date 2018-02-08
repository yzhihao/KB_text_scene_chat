
from random import randint
import pandas as pd
import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 60 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


tr1 = pd.read_csv('cluster_result.csv')
tr = pd.read_csv('dataframe.csv')
member_no = []
feature = []
member_no1 = []
feature1 = []
data = tr.drop(['MEMBER_NO'], axis=1)
for row in tr.index:
    attr = []
    for i in data.loc[row]:
        attr.append(i)
    feature.append(attr)
    member_no.append(tr.loc[row, 'MEMBER_NO'])

data1 = tr1.drop(['MEMBER_NO'], axis=1)
for row in tr1.index:
    # attr = []
    for i in data1.loc[row]:
        feature1.append(i)
    # feature1.append(i)
    member_no1.append(tr.loc[row, 'MEMBER_NO'])
def getdate(batch_size):
    list_index = [randint(0, len(member_no)-1200) for i in range(batch_size)]
    x,y=[],[]
    for i in list_index:
        try:
            x.append(feature[i])
            label=np.zeros(10,dtype=np.int)
            label[feature1[i]] = 1
            y.append(label)
        except:
            print i
    x = np.array(x)
    y = np.array(y)
    return x,y
def  gettestdate():
    x, y = [], []
    for i in range(len(member_no)-1000,len(member_no)-1):
        try:
            x.append(feature[i])
            label = np.zeros(10, dtype=np.int)
            label[feature1[i]] = 1
            y.append(label)
        except:
            print i
    x = np.array(x)
    y = np.array(y)
    return x, y


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

