from __future__ import division
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, normalization
from DataReader import DataReader, DataParserOpt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
import sys
import numpy as np
import pandas as pd
import inspect
import itertools


class SimpleNeuralNet():

    def __init__(self, inputdim):
        # Should build model here
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=inputdim))
        self.model.add(normalization.BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(128))
        self.model.add(normalization.BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(2))
        self.model.add(normalization.BatchNormalization())
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='Adam', metrics=['accuracy'])

        # pass
    def train(self, X, y):
        # TODO: define input arguments,
        # should take in a generator that gives a batch
        self.model.fit(X, y, batch_size=32, nb_epoch=100,
                       validation_split=0.2, shuffle=True, verbose=1)

    def eval(self, X, y):
        return self.model.evaluate(X, y, verbose=1)

    def save(self):
        self.model.save('my_model.h5')

    def load(self, filename):
        self.model = load_model(filename)


# src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# root_dir = os.path.abspath(os.path.join(src_dir, '../'))
# sys.path.insert(0, root_dir + '/lib')
# print root_dir + '/Train'
# print os.listdir(os.path.abspath(root_dir + '/Train'))
# opt = DataParserOpt()
# data, label = [], []
# for filename in os.listdir(os.path.abspath(root_dir + '/Train')):
#     print filename
#     myDataReader = DataReader(
#         './src/mapping.txt', root_dir + '/Train/' + filename)
#     a, y = myDataReader.extract_features()
#     data.append(a)
#     label.append(y)
#
# label = list(itertools.chain.from_iterable(label))
# y = pd.get_dummies(pd.Series(label))
# data = np.vstack(tuple(data))
# np.save('testdata', data)
# np.save('testlabel', label)
# np.savetxt('testdata.csv', data, delimiter=',')
# np.savetxt('testlabel.csv', label, delimiter=',')
# print data.shape, y.as_matrix().shape

data = np.load('testdata.npy')
label = np.load('testlabel.npy')
y = pd.get_dummies(pd.Series(label))
y = y.as_matrix()

N, D = data.shape
idx = np.arange(N)
np.random.shuffle(idx)
split = 0.9
print N, D
print idx.shape
print idx[:int()]

train = data[idx[:int(split * N)]]
y_train = y[idx[:int(split * N)]]
test = data[idx[int(split * N):]]
y_test = y[idx[int(split * N):]]

print test.shape
myNet = SimpleNeuralNet(data.shape[1])
myNet.train(train, y_train)
myNet.save()
print "eval_____"
print myNet.eval(test, y_test)
