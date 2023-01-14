
"""
This script comes from the RTRBM code by Ilya Sutskever from 
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

from numpy import *
from scipy import *               
import pdb
import pickle
import scipy.io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm

shape_std=shape
def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()

size_std = size
def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()

det = linalg.det

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2
    

def norm(x): return sqrt((x**2).sum())
def sigmoid(x):        return 1./(1.+exp(-x))

SIZE=8
# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None, *, res):
    aspect = (1.0, res[0] / res[1])
    #size = (aspect[0]*SIZE, aspect[1]*SIZE)
    size = (SIZE, SIZE)
    if r is None: r=array([1.2]*n)
    if m is None: m=array([1]*n)
    # r is to be rather small.
    X=zeros((T, n, 2), dtype='float')
    v = random.randn(n,2)
    v = v / norm(v)*.5
    good_config=False
    while not good_config:
        x = 2+random.rand(n,2)*8
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-(1/aspect[z])*r[i]<0:       good_config=False
                if x[i][z]+(1/aspect[z])*r[i]>size[z]: good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(aspect*x[i]-aspect*x[j])<r[i]+r[j]:
                    good_config=False
                    
    
    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t,i]=x[i]
            
        for mu in range(int(1/eps)):

            for i in range(n):
                x[i]+=eps*v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z]-(1/aspect[z])*r[i]<0:       v[i][z]= abs(v[i][z]) # want positive
                    if x[i][z]+(1/aspect[z])*r[i]>size[z]: v[i][z]=-abs(v[i][z]) # want negative


            for i in range(n):
                for j in range(i):
                    if norm(aspect*x[i]-aspect*x[j])<r[i]+r[j]:
                        # the bouncing off part:
                        w    = aspect*x[i]-aspect*x[j]
                        w    = w / norm(w)

                        v_i  = dot(w.transpose(),v[i])
                        v_j  = dot(w.transpose(),v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                        
                        v[i]+= w*(new_v_i - v_i)
                        v[j]+= w*(new_v_j - v_j)

    return X

def ar(x,y,z):
    return z/2+arange(x,y,z,dtype='float')

def matricize(X,res,r=None):

    T, n= shape(X)[0:2]
    if r is None: r=array([1.2]*n)

    A=zeros((T,res[0],res[1]), dtype='float')
    
    [I, J]=meshgrid(ar(0,1,1./res[1])*SIZE, ar(0,1,1./res[0])*SIZE)

    aspect = res[0] / res[1]

    for t in range(T):
        for i in range(n):
            A[t]+= exp(-(  ((I-X[t,i,0])**2+((J-X[t,i,1])*aspect)**2)/(r[i]**2)  )**4    )
            
        A[t][A[t]>1]=1
    return A

def bounce_mat(res, n=2, T=128, r =None):
    if r is None: r=array([1.2]*n)
    x = bounce_n(T,n,r, res=res);
    A = matricize(x,res,r)
    return A

def bounce_vec(res, n=2, T=128, r =None, m =None):
    if r is None: r=array([1.2]*n)
    x = bounce_n(T,n,r,m, res=res);
    V = matricize(x,res,r)
    return V.reshape(T, res[0], res[1])

# make sure you have this folder
logdir = './sample'
def show_sample(V, res):
    T   = len(V)
    # res = int(sqrt(shape(V)[1]))
    for t in tqdm.trange(T):
        plt.imshow(V[t].reshape(res[0],res[1]),cmap=matplotlib.cm.Greys_r) 
        # Save it
        fname = logdir+'/'+str(t)+'.png'
        plt.savefig(fname)
        
if __name__ == "__main__":
    res=(32, 64)
    n = 3
    r = array([0.6]*n)
    #r = array([1.0]*n)
    T=100
    #T=40
    #N=10
    N=4000
    dat=[]
    for i in tqdm.trange(N):
        dat.append(bounce_vec(res=res, n=n, T=T, r=r))
    data={}
    data['train']=array(dat)
    
    N=200
    dat=[]
    for i in tqdm.trange(N):
        dat.append(bounce_vec(res=res, n=n, T=T, r=r))
    data['test']=array(dat)

    print("writing bouncing_balls_train.npy...")
    save("bouncing_balls_train.npy", data['train'])

    print("writing bouncing_balls_test.npy...")
    save("bouncing_balls_test.npy", data['test'])

    #breakpoint()
    
    # show one video
    # show_sample(dat[1], res)
    # ffmpeg -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p -r 30 sample.mp4

        
 


 



