import caffe
import numpy as np

class AlwaysOne(caffe.Layer):
    def setup(self, bottom, top):
        pass
    def reshape(self,bottom,top):
        top[0].reshape(*bottom[0].data.shape)
    def backward(self,top,propagate_down,bottom):
        pass
    def forward(self, bottom, top):
        top[0].data[...] = np.ones(bottom[0].data.shape)
