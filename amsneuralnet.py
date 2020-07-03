#!/usr/bin/python3

import os,sys,math
import numpy as np

pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
abs = np.abs
tanh = np.tanh

#tanh activation function (defualt)
def actfn_tanh(z):
    return tanh(z)

#derivative of tanh activation function (default)
def dactfn_tanh(z):
    return (1-tanh(z)**2)

#quadratic loss function (default)
def lossfn_quadratic(afin_vec,y_vec):
    return 0.5*np.sum((afin_vec-y_vec)**2)

#quadratic loss function derivative (vectorized) (default)
def dlossfn_quadratic(afin_vec,y_vec):
    return (afin_vec-y_vec)


##AMS Neural Network Class##
#For those times when tensorflow is cranky
#A learning exercise/utility class implementing 
#a feedforward neural network, back-propagation and 
#stochastic gradient descent.
#
#Adjustable layer sizes, number of layers, loss and activation functions
class amsneuralnet:

    nlayers = 0
    layer_w = np.array([],np.float32)
    layer_b = np.array([],np.float32)
    actfn = actfn_tanh
    dactfn = dactfn_tanh 


    def __init__(self):


        self.nlayers = 0
        self.layersz = np.array([])
        self.layer_w = []
        self.layer_b = []

        #activation
        self.z = [] #pre-act-fn
        self.a = [] #post-act-fn

        #activation function pointers
        self.actfn = actfn_tanh
        self.dactfn = dactfn_tanh

        #cost function pointers
        self.lossfn = lossfn_quadratic
        self.dlossfn = dlossfn_quadratic

        #backpropagation errors
        self.q = []
        self.r = []
        self.Cd = np.float32([])

        return

    def set_layers(self,_sizelayervect):
        _sizelayervect = np.int32(_sizelayervect)
        self.nlayers = len(_sizelayervect)-1
        self.layersz = _sizelayervect

        self.layer_w = []
        self.layer_b = []
        self.z = [] #pre-act-fn
        self.a = [] #post-act-fn
        self.xinp = [] #input activation
        self.q = []
        self.r = []
        self.Cd = np.zeros((self.layersz[self.nlayers]),dtype=np.float32)
        self.xinp = np.zeros((self.layersz[0],1),dtype=np.float32)
        for I in range(0,self.nlayers):
            m = self.layersz[I]
            n = self.layersz[I+1]
            self.layer_w.append(np.zeros((n,m),dtype=np.float32))
            self.layer_b.append(np.zeros((n),dtype=np.float32))
            self.z.append(np.zeros((n,1),dtype=np.float32))
            self.a.append(np.zeros((n,1),dtype=np.float32))
            self.q.append(np.zeros((n,1),dtype=np.float32))
            self.r.append(np.zeros((m,1),dtype=np.float32))
        return
    
    #Randomizes weights and biases
    def initialize_layers(self):
        for I in range(0,self.nlayers):
            m = self.layersz[I]
            n = self.layersz[I+1]
            self.layer_w[I] = (np.random.rand(n,m)*2.0-1.0)
            self.layer_b[I] = (np.random.rand(n,1)*2.0-1.0)
            
        return

    def propagate(self,x_inp):
        self.xinp = x_inp
        self.z[0] = np.matmul(self.layer_w[0],x_inp)+self.layer_b[0]
        self.a[0] = self.actfn(self.z[0])
        for I in range(1,self.nlayers):
            self.z[I] = np.matmul(self.layer_w[I],self.a[I-1])+self.layer_b[I]
            self.a[I] = self.actfn(self.z[I])
        a_outp = self.a[self.nlayers-1]
        return a_outp

    def backpropagate(self,y_outp):
        self.Cd = self.dlossfn(self.a[self.nlayers-1],y_outp)
        
        self.q[self.nlayers-1] = self.Cd*self.dactfn(self.z[self.nlayers-1])
        self.r[self.nlayers-1] = np.matmul(self.layer_w[self.nlayers-1].transpose(),self.q[self.nlayers-1])

        rng = list(range(0,self.nlayers-1)); rng.reverse() 
        for I in rng:
            self.q[I] = self.r[I+1]*self.dactfn(self.z[I])
            self.r[I] = np.matmul(self.layer_w[I].transpose(),self.q[I])

        nngradient = dict()
        wgrad = []
        bgrad = []

        wgrad.append(np.matmul(self.q[0],self.xinp.transpose()))
        bgrad.append(self.q[0])
        for I in range(1,self.nlayers):
            wgrad.append(np.matmul(self.q[I],self.a[I-1].transpose()))
            bgrad.append(self.q[I])
        nngradient['wgrad'] = wgrad
        nngradient['bgrad'] = bgrad

        return nngradient
    
    def __str__(self):
        s = "amsneuralnet:\n"
        s = s + "\tinput layer: {}\n".format(self.layersz[0])
        for I in range(0,self.nlayers):
            s = s + "\tlayer {}: {}x{}\n".format(I,self.layer_w[I].shape[1],self.layer_w[I].shape[0])
        s = s + "\toutput layer: {}\n".format(self.layersz[self.nlayers])
        return s

    def nngradient_apply(self,gradient,learnrate):
        for I in range(0,self.nlayers):
            self.layer_b[I] = self.layer_b[I] - learnrate*gradient['bgrad'][I]
            self.layer_w[I] = self.layer_w[I] - learnrate*gradient['wgrad'][I]
        return

    def train_batch(self,batch_xinp, batch_yout, learnrate):
        N = batch_xinp.shape[1]
        a_outp = np.zeros((self.layersz[self.nlayers],N),dtype=np.float32)

        Lbatch = batch_xinp.shape[1]
        grds = []
        Cs = np.zeros((Lbatch),dtype=np.float32)
        accl = np.zeros((Lbatch),dtype=np.float32)
        for I in range(0,Lbatch):
            x = batch_xinp[:,I]
            x = x.reshape((x.shape[0],1))
            y = batch_yout[:,I]
            y = y.reshape((y.shape[0],1))
            aout = self.propagate(x)
            grd = self.backpropagate(y)
            grds.append(grd)
            Cs[I] = self.lossfn(aout,y)
            accl[I] = local_selmax(aout)==local_selmax(y)
        grd = nngradient_merge(grds)
        self.nngradient_apply(grd,learnrate)    
        Cavg = np.mean(Cs)
        acc = np.mean(accl)
        


        return [Cavg,acc]

    def save(self,fname):
        try:
            fp = open(fname,'w+b')
        except:
            print("{} could not be opened for writing!".format(fname))
            return
        
        magic="AMSNeuralNetv1.0"
        mbts = magic.encode(encoding='utf-8')

        #Magic file identifier
        fp.write(mbts)
        
        q = np.int32(self.nlayers)
        qb = q.tobytes()
        fp.write(qb)

        q = np.int32(self.layersz)
        qb = q.tobytes()
        fp.write(qb)

        for I in range(0,self.nlayers):
            q = np.float32(self.layer_w[I])
            qb = q.tobytes()
            fp.write(qb)
            q = np.float32(self.layer_b[I])
            qb = q.tobytes()
            fp.write(qb)

        fp.close()
        return

    def load(self,fname):
        try:
            fp = open(fname,'r+b')
        except:
            print("{} could not be opened for reading!".format(fname))
            return
        
        mgb = fp.read(16)
        magic = mgb.decode(encoding='utf-8')
        if(magic!="AMSNeuralNetv1.0"):
            print("Wrong file format. Expected header: AMSNeuralNetv1.0, instead {}".format(magic))
            fp.close()
            return
        
        qb = fp.read(4)
        q = np.frombuffer(qb,dtype=np.int32,count=1)[0]
        self.nlayers = q

        qb = fp.read(4*(self.nlayers+1))
        q = np.frombuffer(qb,dtype=np.int32,count=self.nlayers+1)
        self.layersz = q

        self.layer_w = []
        self.layer_b = []

        for I in range(0,self.nlayers):
            m = self.layersz[I]
            n = self.layersz[I+1]
            self.layer_w.append(np.zeros((n,m),np.float32))
            self.layer_b.append(np.zeros((n,1),np.float32))

            qb = fp.read(4*n*m)
            q = np.frombuffer(qb,dtype=np.float32,count=n*m)
            self.layer_w[I] = q.reshape((n,m))

            qb = fp.read(4*n)
            q = np.frombuffer(qb,dtype=np.float32,count=n)
            self.layer_b[I] = q.reshape((n,1))
        
        for I in range(0,self.nlayers):
            m = self.layersz[I]
            n = self.layersz[I+1]
            self.z.append(np.zeros((n,1),dtype=np.float32))
            self.a.append(np.zeros((n,1),dtype=np.float32))
            self.q.append(np.zeros((n,1),dtype=np.float32))
            self.r.append(np.zeros((m,1),dtype=np.float32))

        self.Cd = np.zeros((self.layersz[self.nlayers]),dtype=np.float32)
        self.xinp = np.zeros((self.layersz[0],1),dtype=np.float32)

        fp.close()
        return

def local_selmax(vect):
    mx = -np.inf
    J = 0
    for I in range(0,vect.shape[0]):
        if(vect[I,0]>mx):
            mx = vect[I,0]
            J = I
    return J

def nngradient_merge(gradients):
    grdmrg = dict()
    N = len(gradients)
    grdmrg['wgrad'] = gradients[0]['wgrad']
    grdmrg['bgrad'] = gradients[0]['bgrad']
    for I in range(1,N):
        grdmrg['wgrad'] = grdmrg['wgrad']+gradients[I]['wgrad']
        grdmrg['bgrad'] = grdmrg['bgrad']+gradients[I]['bgrad']
    grdmrg['wgrad'] = grdmrg['wgrad']/(np.float32(N))
    grdmrg['bgrad'] = grdmrg['bgrad']/(np.float32(N))
    
    return grdmrg

def label_reform(lbl1,N):
    lbl2 = np.zeros((N,1),dtype=np.float32)
    lbl2[lbl1] = 1
    return lbl2

def test1():
    q = amsneuralnet()
    q.set_layers([28*28,20,10])
    q.initialize_layers()

    import ams_MNIST_load
    [mnist_l,mnist_d] = ams_MNIST_load.read_training_data()

    d = mnist_d[0].reshape(784,1)
    l = label_reform(mnist_l[0],10)

    for I in range(0,50):
        aout = q.propagate(d)
        grd = q.backpropagate(l)
        Cf = q.lossfn(aout,l)
        print("Cf={:1.6f}".format(Cf))
        q.nngradient_apply(grd,0.01)

    return

def divide_batches(Ndata,Nbatch):
    rng = list(range(0,Ndata))
    batches = []
    while(len(rng)>0):
        batch = []
        if(len(rng)>Nbatch):
            B = np.random.randint(0,len(rng))
            batch.append(rng[B])
            del(rng[B])
        else:
            batch = rng
            rng = []
        batches.append(batch)

    return batches

#A sort of hello-world for neural network programming
#Loads the MNIST dataset (source: http://yann.lecun.com/exdb/mnist/) of
#28 x 28 grayscale handwritten digits. Trains a 784, 20, 10 neural network
#to classify the handwritten digits.
def train_MNIST_net():
    print("amsneuralnet MNIST trainer:\n")
    q = amsneuralnet()
    q.set_layers([28*28,20,10])
    q.initialize_layers()

    print("loading MNIST dataset...")
    import ams_MNIST_load
    [mnist_l,mnist_d] = ams_MNIST_load.read_training_data()

    print("reformatting data...")
    Ndata = len(mnist_l)
    Nx = mnist_d.shape[1]
    Ny = mnist_d.shape[2]
    lbls = np.zeros((10,Ndata),dtype=np.float32)
    data = np.zeros((Nx*Ny,Ndata),dtype=np.float32)
    for I in range(0,Ndata):
        lbls[:,I] = label_reform(mnist_l[I],10)[:,0]
        data[:,I] = mnist_d[I].reshape(Nx*Ny)
    
    Lbatch = 100
    Nepoch = 20
    learnrate = 0.001
    print("Training MNIST, {} epochs in batches of {} out of {}".format(Nepoch,Lbatch,Ndata))
    print("Learning rate {}".format(learnrate))
    savefile = './ams_MNISTnn.amsnn'

    if(os.path.isfile(savefile)):
        q.load(savefile)
    
    for I in range(0,Nepoch):
        batches = divide_batches(Ndata,Lbatch)
        Nbatches = len(batches)
        Cslst = []
        acclst = []
        for J in range(0,Nbatches):
            dsub = data[:,batches[J]]
            lsub = lbls[:,batches[J]]
            [Cs,acc] = q.train_batch(dsub,lsub,learnrate)
            Cslst.append(Cs)
            acclst.append(acc)
        q.save(savefile)
        Css = np.mean(Cslst)
        acc = np.mean(acclst)
        print("ep=\t{}\t loss=\t{:1.4f}\t trn_accuracy=\t{:1.4f}".format(I,Css,acc))

    return

    

def test_readwrite():
    import time

    print("amsneuralnet test readwrite:\n")
    q = amsneuralnet()
    q.set_layers([28*28,20,10])
    q.initialize_layers()

    q.layer_w[0][3,3] = 5.0
    z1 = q.layer_w[0][0:5,0:5]
    q.save('./amstestnn.amsnnf')
    
    print(z1)

    time.sleep(1)
    q = amsneuralnet()
    q.load('./amstestnn.amsnnf')
    z2 = q.layer_w[0][0:5,0:5]
    print(z2)
    print(z1-z2)

    return

if(__name__=="__main__"):
    train_MNIST_net()
    #test_readwrite()
