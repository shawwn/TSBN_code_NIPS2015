
"""
This script comes from the RTRBM code by Ilya Sutskever from 
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm

def shape(A):
    if isinstance(A, np.ndarray):
        return np.shape(A)
    else:
        return A.shape()

def size(A):
    if isinstance(A, np.ndarray):
        return np.size(A)
    else:
        return A.size()

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2
    

def norm(x):
    return np.sqrt((x**2).sum())

def sigmoid(x):
    return 1./(1.+np.exp(-x))

SIZE=8
# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None, *, res):
    if r is None: r=np.array([1.2]*n)
    if m is None: m=np.array([1]*n)
    aspect = (1.0, res[0]/res[1])
    size = (SIZE, SIZE)
    # r is to be rather small.
    X=np.zeros((T, n, 2), dtype='float')
    v = np.random.randn(n,2)
    v = v / norm(v)*.5
    good_config=False
    while not good_config:
        x = 2+np.random.rand(n,2)*8
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

                        v_i  = np.dot(w.transpose(),v[i])
                        v_j  = np.dot(w.transpose(),v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                        
                        v[i]+= w*(new_v_i - v_i)
                        v[j]+= w*(new_v_j - v_j)

    return X

def ar(x,y,z):
    return z/2+np.arange(x,y,z,dtype='float')

def matricize(X,res,r=None):

    T, n = np.shape(X)[0:2]
    if r is None: r=np.array([1.2]*n)

    A = np.zeros((T,res[0],res[1]), dtype='float')
    
    [I, J] = np.meshgrid(ar(0,1,1./res[1])*SIZE, ar(0,1,1./res[0])*SIZE)

    aspect = res[0] / res[1]

    for t in range(T):
        for i in range(n):
            A[t] += np.exp(-(  ((I-X[t,i,0])**2+((J-X[t,i,1])*aspect)**2)/(r[i]**2)  )**4    )
            
        A[t][A[t]>1]=1
    return A

def bounce_mat(res, n=2, T=128, r =None):
    if r is None: r=np.array([1.2]*n)
    x = bounce_n(T,n,r, res=res);
    A = matricize(x,res,r)
    return A

def bounce_vec(res, n=2, T=128, r =None, m =None):
    if r is None: r=np.array([1.2]*n)
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
    r = np.array([0.6]*n)
    #r = np.array([1.0]*n)
    T=100
    #T=40
    #N=10

    N=4000
    dat=[]
    for i in tqdm.trange(N):
        dat.append(bounce_vec(res=res, n=n, T=T, r=r))
    train=np.array(dat)
    
    N=200
    dat=[]
    for i in tqdm.trange(N):
        dat.append(bounce_vec(res=res, n=n, T=T, r=r))
    test=np.array(dat)

    print("writing bouncing_balls_train.npy...")
    np.save("bouncing_balls_train.npy", train)

    print("writing bouncing_balls_test.npy...")
    np.save("bouncing_balls_test.npy", test)
    
    # show one video
    # show_sample(dat[1], res)
    # ffmpeg -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p -r 30 sample.mp4

        
 


 



