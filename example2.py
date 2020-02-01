from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from sklearn.manifold import spectral_embedding
from sklearn.decomposition import PCA
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from dpgcn.utils import *
from dpgcn.models import GCN, MLP
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph

import scipy.io as sio
from numpy import random as nr

from dpgcn.dp_init import *
from dpgcn.vdpmm_maximizePlusGaussian import *
from dpgcn.vdpmm_expectationPlusGaussian import *
from sklearn import metrics

from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope

from dpgcn.utils_vae import encoder, decoder
from dpgcn.generator import Generator
import os
from scipy.misc import imsave

K=130
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

input_tensor = tf.placeholder(tf.float32, [None, 28 * 28])
input_maps = tf.placeholder(tf.float32, [None, None])


def generate_and_save_images(imgs, directory):
    '''Generates the images using the model and saves them in the directory

    Args:
        num_samples: number of samples to generate
        directory: a directory to save the images
    '''
    for k in range(imgs.shape[0]):
        imgs_folder = os.path.join(directory, 'imgs')
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        imsave(os.path.join(imgs_folder, '%d.png') % k,
               imgs[k].reshape(28, 28))
# Load data
# sampleSize=200
# noisy_circles= datasets.make_moons(n_samples=sampleSize, noise=.05)
# noisy_circles = datasets.make_blobs(n_samples=sampleSize, centers=3, n_features=2,random_state=0)
# X = noisy_circles[0]
# y=noisy_circles[1].reshape(1,sampleSize)
# y=y[0]
# dataInput=sio.loadmat('3Gausiansdataset1.mat')
# X=dataInput['Y']
# y=dataInput['z']
# Nz,Dz=X.shape
# s=(np.ones((Nz,1)))*0.2
# dataInput=sio.loadmat('D:\ye\cars3\coil6Data.mat')
dataInput=sio.loadmat('MnistTest.mat')

X=dataInput['Y']

y=dataInput['z']
Nz,Dz=X.shape
XArray=csr_matrix.toarray(X)

Knum=10
W = kneighbors_graph(X,Knum, mode='distance', include_self=True)
print(W)

maps = spectral_embedding(W, n_components=50)
######################################################################
# disReal = kneighbors_graph(X, Knum, mode='connectivity', include_self=False)
W=W
W=sparse.csr_matrix(W)
# print(type(W))
W1=W.toarray()
# print(W1)
W=csr_matrix.toarray(W)


adj=W
features=maps
print(y.reshape(-1,1))


of=OneHotEncoder(sparse=False).fit(y.reshape(-1,1))

data_ohe1=of.transform(y.reshape(-1,1))
shapeOhe1=data_ohe1.shape
data_ohe=np.zeros((Nz,K))
data_ohe[:,0:shapeOhe1[1]]=data_ohe1[:,0:shapeOhe1[1]]

shapeOhe=data_ohe.shape
y_train=np.zeros(shapeOhe)
y_val=np.zeros(shapeOhe)
y_test=np.zeros(shapeOhe)
train_mask=np.zeros(shapeOhe[0],'?')
train_mask2=np.zeros(shapeOhe[0],'?')
val_mask=np.zeros(shapeOhe[0],'?')
test_mask=np.zeros(shapeOhe[0],'?')

select=np.array([0,980,2115,3147,4157,5139])#,649,721,793,721,793,865,937, 865,937,1009,1081,1153
numlabel=400
numlabel_val=300

Ns = select.shape
print(Ns)
for i in range(Ns[0]):
    y_train[select[i]:select[i]  + numlabel, :] = data_ohe[select[i]:select[i] + numlabel, :]
    train_mask2[select[i]:select[i] + numlabel] = 1
for i in range(Ns[0]):
    y_val[select[i] + numlabel:select[i] + numlabel + numlabel_val, :] = data_ohe[
                                                                         select[i] + numlabel:select[i] + numlabel + numlabel_val,:]
    val_mask[select[i] + numlabel:select[i] + numlabel + numlabel_val] = 1

print(y_train)
for i in range(shapeOhe[0]):

    if (i> 20):
        y_test[i,:] = data_ohe[i, :]
        test_mask[i] = 1
    train_mask[i] = 1
def updateGammas(gammas,LabelG):
    N,D=gammas.shape
    for i in range(Ns[0]):
        gammas[select[i]:select[i]+numlabel,:]=data_ohe[select[i]:select[i]+numlabel,:]
    return gammas

features = sparse.lil_matrix(features)
# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'dis':tf.placeholder(tf.float32, [None, None]),
}
Nm,Dm=maps.shape
epsilon = tf.random_normal([tf.shape(input_tensor)[0], Dm])
input_sample = input_maps + epsilon
output_tensor = decoder(input_sample)
loss1=tf.reduce_sum(-input_tensor * tf.log(output_tensor +0.01) -(1.0 - input_tensor) * tf.log(1.0 - output_tensor+0.01 ))
train = layers.optimize_loss(loss1, tf.contrib.framework.get_or_create_global_step(), learning_rate=0.01, optimizer='Adam', update_ops=[])

# Create model
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
run_config.gpu_options.per_process_gpu_memory_fraction = 1/10
sess = tf.Session(config=run_config)
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    # feed_dict.update({placeholders['dis']: W1})
    outs_val = sess.run([ model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0],  (time.time() - t_test)
# Define model evaluation function


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
costTemp=0

pca=PCA(n_components=3,whiten=True)
newData=pca.fit_transform(maps)

paramsGaussian,posGaussian = vdpmm_init(maps,K)

numits=1;
maxits=10;
outs_1=[];
# outs_1 = sess.run([model.predict()], feed_dict=feed_dict)
posGaussian_1=posGaussian
for epoch in range(FLAGS.epochs):
    paramsGaussian = vdpmm_maximizePlusGaussian(maps, paramsGaussian, posGaussian)
    posGaussian = vdpmm_expectationPlusGaussian(maps, paramsGaussian,posGaussian_1)
    y_train=updateGammas(posGaussian,posGaussian)
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    outs_1 = sess.run([model.predict()], feed_dict=feed_dict)
    lossVae=0
    for j in range(100):
        Xtemp=XArray[j*100:j*100+100,:]/255
        mapsTemp=maps[j*100:j*100+100,:];
        sess.run(train,feed_dict={input_tensor:Xtemp,input_maps:mapsTemp})
        lossVae = lossVae + sess.run(loss1,feed_dict={input_tensor:Xtemp,input_maps:mapsTemp})
    for j in range(10):
        Xtemp[j * 10:j * 10 + 10,:] = XArray[j * 1100:j * 1100 + 10, :] / 255
        mapsTemp[j * 10:j * 10 + 10,:] = maps[j * 1100:j * 1100 + 10, :];
    imgs=sess.run(output_tensor, feed_dict={input_tensor: Xtemp, input_maps: mapsTemp})
    generate_and_save_images(imgs, '')

    posGaussian_1=np.array(outs_1[0])
    # Validation
    acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost = sess.run(model.loss, feed_dict=feed_dict)
    cost_val.append(cost)
    costTemp = cost

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t),'vae loss',lossVae)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break


print("Optimization Finished!")

# Testing
test_acc, test_duration = evaluate(features, support, y_val, val_mask, placeholders)
test_cost=costTemp
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

temp=np.max(posGaussian,axis=1)
temp.shape=(Nz,1)
index1=np.where(temp==posGaussian)


fig=plt.figure(2)
ax1=fig.add_subplot(111,projection='3d')
# colorStore='rgbyck'
marker=['.',',','o','v','^','<','>','1','2','3','4','8','s','p','*']
colorStore = ['r','y','g','b','r','y','g','b','r']
for i in range(newData.shape[0]):
    cho=0
    cho=int(y[i]%8)
    #plt.scatter(data[i,0],data[i,1],color=colorStore[cho])
    ax1.scatter(newData[i,0],newData[i,1],newData[i,2],color=colorStore[cho],marker=marker[cho])
fig=plt.figure(3)
ax1=fig.add_subplot(111,projection='3d')
# colorStore='rgbyck'
marker=['.',',','o','v','^','<','>','1','2','3','4','8','s','p','*']
colorStore = ['r','y','g','b','r','y','g','b','r']
for i in range(newData.shape[0]):
    cho=0
    cho=int((index1[1][i]+1)%8)
    #plt.scatter(data[i,0],data[i,1],color=colorStore[cho])
    ax1.scatter(newData[i,0],newData[i,1],newData[i,2],color=colorStore[cho],marker=marker[cho])
plt.show()

labelP=np.array(index1[1])+1
print(labelP)
print(y.T)


preLabel=list(index1[1])
srcLabel=(y.T)[0]
print(preLabel)
print(srcLabel)



tempNum=Nz-Ns[0]*numlabel
print(tempNum)
preLabel1=np.zeros((tempNum));
srcLabel1=np.zeros((tempNum));
labels1=np.zeros((tempNum));
count=0;
print(train_mask2.shape)
for i in range(Nz):
    if(train_mask2[i]):
        pass;
    else:
        preLabel1[count]=preLabel[i]
        srcLabel1[count]=srcLabel[i]
        count=count+1
print(metrics.adjusted_mutual_info_score(preLabel1, srcLabel1))