import caffe
import numpy as np

def sigmoid(X):
        return 1.0/(1+np.exp(-X))

class TwoClassAccuracy(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute accuracy.")

    def reshape(self, bottom, top):
        # accuracy output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        bot = sigmoid(bottom[0].data)
        diff = bot[:,0] - bottom[1].data
        top[0].data[...] = 1.0 - np.average(np.fabs(diff))