import platform
import subprocess
import sys
import caffe
caffe.set_mode_gpu()
import lmdb

from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import numpy as np

# Status Quo

print "OS:     ", platform.platform()
print "Python: ", sys.version.split("\n")[0]
print "CUDA:   ", subprocess.Popen(["nvcc", "--version"], stdout=subprocess.PIPE).communicate()[0].split("\n")[3]
print "LMDB:   ", ".".join([str(i) for i in lmdb.version()])

# Load Data from CSV and Trasform

df = pd.read_csv("train.csv", sep=",")
features = df.ix[:,1:-1].as_matrix()
labels = df.ix[:,-1].as_matrix()

vec_log = np.vectorize(lambda x: np.log(x+1))
vec_int = np.vectorize(lambda str: int(str[-1])-1)

features = vec_log(features)
labels = vec_int(labels)

# Stratified Split for Training and Testing

sss = StratifiedShuffleSplit(labels, 1, test_size=0.02, random_state=0)
sss = list(sss)[0]

features_training = features[sss[0],]
labels_training = labels[sss[0],]

features_testing = features[sss[1],]
labels_testing = labels[sss[1],]

# Load Data into LMDB

# http://deepdish.io/2015/04/28/creating-lmdb-in-python/
def load_data_into_lmdb(lmdb_name, features, labels=None):
    env = lmdb.open(lmdb_name, map_size=features.nbytes * 2)

    features = features[:, :, None, None]
    for i in range(features.shape[0]):
        datum = caffe.proto.caffe_pb2.Datum()

        datum.channels = features.shape[1]
        datum.height = 1
        datum.width = 1

        if features.dtype == np.int:
            datum.data = features[i].tostring()
        elif features.dtype == np.float:
            datum.float_data.extend(features[i].flat)
        else:
            raise Exception("features.dtype unknown.")

        if labels is not None:
            datum.label = int(labels[i])

        str_id = '{:08}'.format(i)
        with env.begin(write=True) as txn:
            txn.put(str_id, datum.SerializeToString())

load_data_into_lmdb("./train_data_lmdb", features_training, labels_training)
load_data_into_lmdb("./test_data_lmdb", features_testing, labels_testing)
