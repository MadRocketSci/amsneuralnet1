#!/usr/bin/python3

import os,sys,math
import numpy as np
import cv2

import gzip #May need to use gzip.open instead of open

import struct
#struct unpack allows some interpretation of python binary data
#Example
##import struct
##
##data = open("from_fortran.bin", "rb").read()
##
##(eight, N) = struct.unpack("@II", data)
##
##This unpacks the first two fields, assuming they start at the very
##beginning of the file (no padding or extraneous data), and also assuming
##native byte-order (the @ symbol). The Is in the formatting string mean
##"unsigned integer, 32 bits".

#for integers
#a = int
#a.from_bytes(b'\xaf\xc2R',byteorder='little')
#a.to_bytes(nbytes,byteorder='big')
#analagous operation doens't seem to exist for floats
#what about numpy?


#https://www.devdungeon.com/content/working-binary-data-python

#print("{:02d}".format(2))
#b = b.fromhex('010203040506')
#b.hex()
#c = b.decode(encoding='utf-8' or 'latin-1' or 'ascii'...)
#print(c)

#numpy arrays have tobytes
#numpy arrays have frombuffer (converts to dtypes)
#
#q = np.array([15],dtype=np.uint8);
#q.tobytes();
#q.tobytes(order='C') (options are 'C' and 'F'
#q2 = np.buffer(q.tobytes(),dtype=np.uint8)
#np.frombuffer(buffer,dtype=float,count=-1,offset=0)

##You could also use the < and > endianess format codes in the struct
##module to achieve the same result:
##
##>>> struct.pack('<2h', *struct.unpack('>2h', original))
##'\xde\xad\xc0\xde'

def bytereverse(bts):
##    bts2 = bytes(len(bts));
##    for I in range(0,len(bts)):
##        bts2[len(bts)-I-1] = bts[I];
    N = len(bts)
##    print(N);
##    print(bts);
##    bts2 = struct.pack('<{}h'.format(N), *struct.unpack('>{}h'.format(N), bts))
    bts2 = bts
    return bts2

#Read Labels
def read_MNIST_label_file(fname):
    #fp = gzip.open('./train-labels-idx1-ubyte.gz','rb');
    fp = gzip.open(fname,'rb')
    magic = fp.read(4)
    #nitems = np.frombuffer(fp.read(4),dtype=np.int32)[0]; #some sort of endiannes problem
    bts = fp.read(4)
    #bts = bytereverse(bts);
    #nitems = np.frombuffer(bts,dtype=np.int32);
    nitems = np.int32(struct.unpack('>I',bts)[0]) #it was a non-native endianness in teh integer encoding
    #> < @ - endianness

    bts = fp.read(nitems)
    N = len(bts)
    labels = np.zeros((N),dtype=np.uint8)
    labels = np.frombuffer(bts,dtype=np.uint8,count=N)
    #for i in range(0,10):
    #    bt = fp.read(1);
    #    labels[i] = np.frombuffer(bt,dtype=np.uint8);
    fp.close()
    return labels

def read_MNIST_image_file(fname):
    fp = gzip.open(fname,'rb')
    magic = fp.read(4)
    bts = fp.read(4)
    nitems = np.int32(struct.unpack('>I',bts)[0])
    bts = fp.read(4)
    nrows = np.int32(struct.unpack('>I',bts)[0])
    bts = fp.read(4)
    ncols = np.int32(struct.unpack('>I',bts)[0])

    images = np.zeros((nitems,nrows,ncols),dtype=np.uint8)
    for I in range(0,nitems):
        bts = fp.read(nrows*ncols)
        img1 = np.frombuffer(bts,dtype=np.uint8,count=nrows*ncols)
        img1 = img1.reshape((nrows,ncols))
        images[I,:,:] = img1

    fp.close()

    return images

def read_training_data():
    rootdir = '/home/aschinde/workspace/machinelearning/datasets/MNIST'
    fname1 = 'train-labels-idx1-ubyte.gz'
    fname2 = 'train-images-idx3-ubyte.gz'    

    labels = read_MNIST_label_file(os.path.join(rootdir,fname1))
    images = read_MNIST_image_file(os.path.join(rootdir,fname2))

    return [labels,images]

def read_test_data():
    rootdir = '/home/aschinde/workspace/machinelearning/datasets/MNIST'

    fname1 = 't10k-labels-idx1-ubyte.gz'
    fname2 = 't10k-images-idx3-ubyte.gz'

    labels = read_MNIST_label_file(os.path.join(rootdir,fname1))
    images = read_MNIST_image_file(os.path.join(rootdir,fname2))    

    return [labels,images]

def show_MNIST_image(img):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(255-img,cmap='gray')
    plt.show()
    return

