import numpy as np

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

def norm(x): 
    return np.sqrt((x**2).sum())

def sigmoid(x):        
    return 1./(1.+np.exp(-x))

def ar(x,y,z):
    return z/2+np.arange(x,y,z)

def matricize(X, res, r, box_size):
    T, n = X.shape[0:2]
    A = np.zeros((T,res,res), dtype='float')
    [I, J] = np.meshgrid(ar(0,1,1./res)*box_size, ar(0,1,1./res)*box_size)

    for t in range(T):
        for i in range(n):
            A[t] += np.exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )
            
        A[t][A[t]>1]=1
    return A

class BouncingBallsSim(object):
    def __init__(self, box_size=5.0, r=1.0, res=28):
        self.r = r
        self.res = res
        self.box_size = box_size

    def sample_trajectory(self, dt=1.0, A=2, T=50, m=None):
        r  = np.array([self.r]*A)
        fr = np.random.rand() / 50
        X  = self.bounce_n(dt, self.box_size, T, A, r, m, fr=fr)
        X[:,:,:2] += self.box_size
        V = matricize(X, self.res, r, 2*self.box_size)
        return X, V, fr
    
    def bounce_n(self, dt, SIZE, T=100, n=2, r=None, m=None, fr=0.01):
        if m==None: m=np.array([1]*n)
        X = np.zeros((T, n, 2)) # position
        V = np.zeros((T, n, 2)) # velocity
        v = np.random.randn(n,2)
        v = v / norm(v)
        good_config=False
        while not good_config:
            x = -3+np.random.rand(n,2)*6
            good_config=True
            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<-SIZE or x[i][z]+r[i]>SIZE:      
                        good_config=False

            # that's the main part.
            for i in range(n):
                for j in range(i):
                    if norm(x[i]-x[j])<r[i]+r[j]:
                        good_config=False


        eps = dt/10 # 10 intermediate steps between two observations
        for t in range(T):

            for i in range(n): # for each ball
                v_norm = np.sqrt((v[i]**2).sum())
                stop = v_norm < fr
                if stop:
                    v[i] *= 0
                else:
                    v[i] -= v[i] / v_norm * fr

            for i in range(n): # for each ball
                X[t,i] = x[i]
                V[t,i] = v[i]

            for mu in range(int(dt/eps)): # intermediate steps
                for i in range(n): 
                    x[i] += eps*v[i]
                for i in range(n):
                    for z in range(2):
                        if x[i][z]-r[i]<-SIZE:  
                            v[i][z] =  abs(v[i][z]) # want positive
                        if x[i][z]+r[i]>SIZE: 
                            v[i][z] = -abs(v[i][z]) # want negative
                for i in range(n):
                    for j in range(i):
                        if norm(x[i]-x[j])<r[i]+r[j]:
                            # the bouncing off part:
                            w   = x[i]-x[j]
                            w   = w / norm(w)
                            v_i = np.dot(w.transpose(),v[i])
                            v_j = np.dot(w.transpose(),v[j])

                            new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                            v[i] += w*(new_v_i - v_i)
                            v[j] += w*(new_v_j - v_j)

        return np.concatenate([X,V],-1)