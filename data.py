from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
import numpy as np

def loaddata(dataset,digit=None):
    """
    Loads the wanted dataset.
    """
    if dataset == "mnist":
        images=getmnist(digit)
    elif dataset == "cifar":
        images=getcifar(digit)
    elif dataset == "flowers":
        images= getflowers()*2
    elif dataset == "flowers128":
        images= getflowers128()
    elif dataset == "cats":
        images=getcats()
    return images

def loadlables(dataset):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    labels=mnist.train.labels
    return labels

def getflowers():
    images=np.load('flowers.npz')['data']
    return images

def getflowers128():
    images=np.load('flowers128.npz')['data']
    return images

def getcats():
    images=np.load('cats.npz')['data']
    return images

def getmnist(digit):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if digit==None:
        images=mnist.train.images.astype('float32')
    else:
        images=mnist.train.images.astype('float32')
        labels=mnist.train.labels
        col=labels[:,digit]
        col=col!=0
        images=images[col]
    images=images.reshape(images.shape[0],28,28,1)
    images=2*(images-0.5)

    return images



def getcifar(digit):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    images=np.float32(2*(np.vstack((x_train,x_test))/255-0.5))
    labels=np.vstack((y_train,y_test))
    o=np.zeros([60000,10])
    for i in range(60000):
        o[i,labels[i]]=1
    labels=o
    if digit==None:
        images=images
    else:
        images=images
        labels=labels
        col=labels[:,digit]
        col=col!=0
        images=images[col]
    return images
