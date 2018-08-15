import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose,BatchNormalization, UpSampling2D
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, initializers
from keras.optimizers import SGD,Adam
from keras.datasets import cifar10
from data import loaddata
from models import loadmodel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


np.random.seed(42)

def getnoise(size):
    """Generate 'size' random noise vectors"""
    noisesize=10
    noise = np.random.uniform(-1, 1, size=(size, noisesize))
    return noise


def creategans(discmodel,genmodel):
    """
    Creates a combined model of both a discriminator model
    and a generator model.
    """
    gansmodel=Sequential()
    gansmodel.add(genmodel)
    discmodel.trainable=False
    gansmodel.add(discmodel)
    return gansmodel


def showim(genmodel,index,datashape):
    """
    Saves an image containing n_ims*n_ims generated images in folder image_folder
    (which must be created by the user).
    """
    image_folder="ims8/"
    n_ims=5
    lk=datashape[1]
    noise = getnoise(n_ims**2)
    if datashape[-1]==1:
        generated = genmodel.predict(noise).reshape([n_ims,n_ims,lk,lk])
        imtot=np.zeros([lk*n_ims,lk*n_ims])
    else:
        generated = genmodel.predict(noise).reshape([n_ims,n_ims,lk,lk,datashape[-1]])
        imtot=np.zeros([lk*n_ims,lk*n_ims,datashape[-1]])

    filename="im"+str(index)+".png"
    m=0
    n=0
    for i in range(n_ims):
        m=0
        for j in range(n_ims):
            imtot[n:n+lk,m:m+lk]=generated[i,j]
            m+=lk
        n+=lk

    imtot=(imtot+1)/2
    print('Printing to: '+filename)
    plt.axis('off')
    if datashape[-1]==1:
        plt.imshow(imtot,cmap='gray')
    else:
        plt.imshow(imtot)
    plt.savefig(image_folder+filename)
    return imtot


def train():
    """
    Loads datasets {'mnist', 'flowers', 'flowers128',
     'cifar10', 'cats'} and their corresponding model
     and trains the GANs using 'epochs' epochs.
    """
    dataset="mnist"
    images=loaddata(dataset)
    g,d=loadmodel(dataset)
    opt = Adam(lr=0.0002,beta_1=0.5)
    g.summary()
    d.summary()
    d.trainable=True
    d.compile(loss='binary_crossentropy', optimizer=opt)
    g.compile(loss='binary_crossentropy', optimizer=opt)
    d.trainable=False
    gansmodel=creategans(d,g)
    gansmodel.compile(loss='binary_crossentropy', optimizer=opt)

    i1,i2,i3,i4=images.shape
    epochs=40
    batch_size=128//2
    k=0
    filename= "ims8/"+str(dataset)+"_n_epochs_"+str(epochs)+".h5"
    for i in range(epochs):
        g.save(filename)
        for j in range(i1//batch_size):
            noise=getnoise(batch_size*2)
            noise_images=g.predict(noise)

            d.train_on_batch(images[j*batch_size:(j+1)*batch_size],np.ones([batch_size,1]))
            ld=d.train_on_batch(noise_images[0:batch_size],np.zeros([batch_size,1]))

            if (j%10==0):
                print("Epoch: ",i+1," D Loss: ",ld)

            lg=gansmodel.train_on_batch(noise,np.ones([batch_size*2,1]))

            if (j%10==0):
                print("Epoch: ",i+1," G Loss: ", lg)
            if (j%100==0):
                showim(g,k,[i1,i2,i3,i4])
                k=k+1
train()
