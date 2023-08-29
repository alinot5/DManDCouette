#!/home/linot/anaconda3/envs/py38numba/bin/python3
"""
Created on Wed Aug 14 2019

Dense invariant NN with tied encoder and decoder

@author: Alec
"""

import os
import sys
import math
sys.path.insert(0, '/home/linot/Couette/NODE_DampPOD/Auto_Align3/Red')
sys.path.insert(0, '/home/floryan/odeNet/torchdiffeq')

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import scipy.io
from sklearn.utils.extmath import randomized_svd
import buildmodel
import tables

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '/home/linot/Couette/Python/SBDF3/RL_Opt/Implement/newEnv/Fixed/Phase')
from Solver import Solver

###############################################################################
# Arguments from the submit file
###############################################################################
from torchdiffeq import odeint_adjoint as odeint

# Check if there are gpus
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

###############################################################################
# Classes
###############################################################################

# This is the class that contains the NN that estimates the RHS
class ODEFunc(nn.Module):
    def __init__(self,trunc,a):
        super(ODEFunc, self).__init__()
        # Change the NN architecture here
        self.net = nn.Sequential(
            nn.Linear(trunc, 200),
            nn.Sigmoid(),
            nn.Linear(200, 200),
            nn.Sigmoid(),
            nn.Linear(200,200),
            nn.Sigmoid(),
            nn.Linear(200, trunc),
        )

        self.lin=nn.Sequential(nn.Linear(trunc, trunc,bias=False),)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        
        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(Linear(trunc,a)).float())
                m.weight.requires_grad=False

    def forward(self, t, y):
        # This is the evolution with the NN
        return self.lin(y)+self.net(y)

def Linear(N,a):
    A=np.diag(-a*np.ones(N))
    return A

# This class is used for updating the gradient
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

###############################################################################
# Functions
###############################################################################
    
def get_batch(t,true_y):
    rand=np.random.choice(np.arange(np.floor(args.data_size/args.step) - args.batch_time, dtype=np.int64), args.batch_size, replace=False)
    lens=[8000-1,8000,8000,2000,2600,8000,7100,8000,8000,2415,8000,8000,1321,8000,4126]
    num=0
    for i in lens:
        num=i+num
        rand[rand==num]=rand[rand==num]+1

    s = torch.from_numpy(rand)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

# For outputting info when running on compute nodes
def Out(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text+'\n')
    newfile.close()

###############################################################################
# Main Function
###############################################################################
if __name__=="__main__":
    ###########################################################################
    # Import Data
    ###########################################################################
    path='/home/linot/Couette/PODFFT/NoDialias_Correct_nomean2'
    N=502
    lens=[8000,8000,8000,2000,2600,8000,7100,8000,8000,2415,8000,8000,1321,8000,4126]
    dirs=[0,1,2,3,6,7,8,9,11,12,13,14,16,17,18]
    M=np.sum(lens)
    u=np.zeros((M,N))
    # Get the indices of the state that should be real and complex
    modes1=pickle.load(open('Modes1.p','rb'))
    modes2=pickle.load(open('Modes2.p','rb'))
    lams=pickle.load(open('Eigs.p','rb'))
    modestemp=np.concatenate((modes1,modes2))
    
    Kx=16
    Kz=16
    Ny=35 # This shouldn't change anything, but in the autoencoder code N was in the idx array! 
    # This is for identifying if a coefficient should be strictly real or complex
    red=256
    idx=[[i,j,k] for i in range(Ny) for j in range(2*Kx-1) for k in range(Kz)]
    idx=np.asarray(idx)
    lamstemp=lams[:,:,:Kz]
    sort=np.argsort(-lamstemp.flatten())
    idx_real=np.where((idx[sort[:red],2]+idx[sort[:red],1])==0)
    idx_complex=np.where((idx[sort[:red],2]+idx[sort[:red],1])!=0)
    reals=((idx[sort[:red],2]+idx[sort[:red],1])==0).sum()
    # Loop through all trails
    for i in range(len(lens)):
        # Load phase shifted data
        atemp=pickle.load(open(path+'/POD_data_small_shift'+str(dirs[i])+'.p','rb'))

        # Reshape data so it is strictly real
        atemp=np.concatenate((atemp[idx_real],np.real(atemp[idx_complex]),np.imag(atemp[idx_complex])))

        # Append data
        u[int(np.sum(lens[:i])):int(np.sum(lens[:i])+lens[i]),:]=atemp.T

    # normalize u by subtracting the mean and dividing by the maximum std
    ustd=np.max(np.std(u,axis=0))
    umean=np.mean(u,axis=0)
    pickle.dump([ustd,umean],open('fullnorm.p','wb'))
    u=(u-umean[np.newaxis,:])/ustd
    Out('Maximum value')
    Out(str(ustd))

    # Split training and test data
    path=os.path.abspath(os.getcwd())
    frac=.8
    a_train=u[:int(round(M*frac)),:]
    a_test=u[int(round(M*frac)):M,:]

    # Output sizes
    Out('Training Shapes')
    Out(str(a_train.shape))
    Out('Testing Shapes')
    Out(str(a_test.shape))

    ###########################################################################
    # Autoencoder
    ###########################################################################
    path=os.path.abspath(os.getcwd())
    type=path.split('/')[7]
    encoder=True

    if type=='Hybrid':
        encode=buildmodel.encode_hybrid(N)
        decode=buildmodel.decode_hybrid(N)
        model,_,_=buildmodel.buildmodel_hybrid(N,.01)
    elif type=='Stand':
        encode=buildmodel.encode(N)
        decode=buildmodel.decode(N)
        model,_,_=buildmodel.buildmodel(N)
    elif type=='Lin':
        h_train=a_train[:,:buildmodel.trunc()]
        h_test=a_test[:,:buildmodel.trunc()]
        encoder=False

    # Load the encoder
    Auto=path.split('/')[-2][-1]
    if encoder==True:
        model_path=path[:45]+'Red/' +type+'/'+str(buildmodel.trunc())+'/Trial'+Auto+'/model.h5'
        model.load_weights(model_path)
        print(model_path)
        for i in range(2):
            ly='Dense_In'+str(i+1)
            encode.get_layer(ly).set_weights(model.get_layer(ly).get_weights())
            ly='Dense_Out'+str(i+1)
            decode.get_layer(ly).set_weights(model.get_layer(ly).get_weights())
            if type=='Hybrid':
                for i in range(2):
                    ly='Dense_Out_Reg'+str(i+1)
                    decode.get_layer(ly).set_weights(model.get_layer(ly).get_weights())

        h_train=encode.predict(a_train)
        h_test=encode.predict(a_test)

    # Get the mean and standard deviation of the training data (normalize if it is a standard autoencoder (center if it isn't))
    h_mean=np.mean(h_train,axis=0)
    h_std=np.std(h_train,axis=0)
    pickle.dump([h_mean,h_std],open('Latent.p','wb'))

    # Make the damping proportional to the standard deviation of the latent variable
    a=.1*h_std
    h_train=(h_train-h_mean)
    h_test=(h_test-h_mean)

    # Convert to torch tensor
    true_y=torch.tensor(h_train[:,np.newaxis,:])
    true_y=true_y.type(torch.FloatTensor)
    t=np.arange(round(M*frac))

    t=torch.tensor(t)
    t=t.type(torch.FloatTensor)

    ###########################################################################
    # Compute Statistics: Long-time stats
    ###########################################################################
    func = torch.load('model.pt')
    # Convert to torch tensor
    T=5000
    ex=-(7000+4126) # This should grab a snapshot that evolves forward for atleast 7k time units
    y0=torch.tensor(h_test[ex:ex+1,np.newaxis,:])
    y0=y0.type(torch.FloatTensor)
    tt=np.arange(T)
    t=torch.tensor(tt)
    t=t.type(torch.FloatTensor)
    
    func=torch.load('model.pt')
    #pickle.dump(func,open('model.p','wb'))
    hNN = odeint(func, y0, t)
    hNN=np.squeeze(hNN.detach().numpy())
    pickle.dump([hNN,h_test[ex:ex+T,:]],open('Data.p','wb'))

    # Plot some simple statistics
    plt.figure()
    plt.plot(tt,np.linalg.norm(h_test[ex:ex+T,:],axis=-1))
    plt.plot(tt,np.linalg.norm(hNN,axis=-1))
    plt.xlabel('t')
    plt.ylabel('E')
    plt.xlim([0,T])
    plt.savefig('Energy.png')

    # Plot the learning
    if buildmodel.trunc()<5:
        exs=buildmodel.trunc()
    else:
        exs=5

    for i in range(exs):
        plt.figure()
        plt.plot(tt,np.real(h_test[ex:ex+T,i]))
        plt.plot(tt,np.real(hNN[:,i]))
        plt.xlabel('t')
        plt.ylabel('h_'+str(i))
        plt.xlim([0,T])
        plt.savefig('Tracking'+str(i)+'.png')

    ###########################################################################
    # Get the Reynolds stress and the power input dissipation curves
    ###########################################################################
    # Decode the data
    hNN=hNN+h_mean
    atemp=decode.predict(hNN).T

    # Fix the normalizatino
    atemp=atemp*ustd+umean[:,np.newaxis]

    # This is how you would convert back
    ared=np.zeros((red,T),dtype=np.complex128)
    ared[idx_real,:]=atemp[:reals,:]
    ared[idx_complex,:]=atemp[reals:reals+red-reals,:]+1j*atemp[reals+red-reals:reals+2*(red-reals),:]

    ############################################################################
    # Load the reduced POD data and convert to the full POD matrix
    ############################################################################
    # Convert a back to the original format
    N=[35,32,32,3,8000]
    x=[.875*2*math.pi,2,.6*2*math.pi]
    y=np.arange(N[0])
    y=-np.cos(y*math.pi/(N[0]-1))
    baseflow=y
    sol=Solver(baseflow,[N[0],N[1],N[2],N[3],1],[x[1],x[0],x[2]])

    # Load modes and eigenvalues
    mean=pickle.load(open('/home/linot/Couette/PODFFT/NoDialias_Correct_nomean2/mean.p','rb'))
    [stats_true,_]=pickle.load(open('/home/linot/Couette/PODFFT/NoDialias_Correct_nomean2/Check_Stats/stats.p','rb'))

    us=np.zeros((N[0],N[1],N[2],N[3],T),dtype=np.complex128)
    for l in range(red):
        k=idx[sort[l],0]
        i=idx[sort[l],1]
        if i>16: # This seems to properly accounts for the negative values
            i=i-31
        j=idx[sort[l],2]

        ustemp=modestemp[:,k:k+1,i,j]@ared[l:l+1,:]
        ustemp=ustemp.reshape((N[0],N[3],T))
        us[:,i,j,:,:]+=ustemp
        # By allowing the 0s I was doubling the zero modes
        if j!=0 and i!=0:
            us[:,-i,-j,:,:]+=np.conj(ustemp)
    # The negative i values are not perfectly the complex conjugate, so this fixes that
    us[:,-Kx+1:,0,:,:]=np.flip(np.conj(us[:,1:Kx,0,:,:]),axis=1)

    # Move u back to physical space
    ust=np.zeros((N[0],N[1],N[2],N[3],T))
    for i in range(T):
        ust[:,:,:,:,i:i+1]=np.real(sol.ifft(us[:,:,:,:,i:i+1]))

    ############################################################################
    # Compute stresses
    ############################################################################
    Out('Compute Stresses')
    stats=np.zeros((35,4)) #(u2,v2,w2,uv)
    
    for i in range(T):
        u=ust[:,:,:,:,i]
        stats[:,0]+=np.mean(np.squeeze((u[:,:,:,0])**2),axis=(1,2))
        stats[:,1]+=np.mean(np.squeeze((u[:,:,:,1])**2),axis=(1,2))
        stats[:,2]+=np.mean(np.squeeze((u[:,:,:,2])**2),axis=(1,2))
        stats[:,3]+=np.mean(np.squeeze(u[:,:,:,0]*u[:,:,:,1]),axis=(1,2))

    stats=stats/T
    pickle.dump(stats,open('stats.p','wb'))

    plt.figure()
    plt.plot(y,stats_true)
    plt.plot(y,stats,'--')
    plt.xlim([y[0],y[-1]])
    #plt.ylim([-1,1])
    plt.xlabel('y')
    plt.ylabel('<u_iu_j>')
    plt.savefig('Stress.png')

    ############################################################################
    # Compute energy balance
    ############################################################################
    Out('Compute Energy')
    E=np.zeros((T,3))
    xl=x[0]
    yl=x[1]
    zl=x[2]

    V=xl*yl*zl
    A=xl*zl
    for i in range(T):
        u=ust[:,:,:,:,i]+mean[:,np.newaxis,np.newaxis,:]
        u[:,:,:,0]=u[:,:,:,0]+baseflow[:,np.newaxis,np.newaxis]
        curlu=sol.curlu(u[:,:,:,:,np.newaxis],1,1,1,1)
        curlu=curlu.squeeze()
        dy=sol.dy(u[:,:,:,:,np.newaxis],1,1,1,1)
        dy=dy.squeeze()
        # Append the periodic points for the integration with trapz
        u=np.concatenate((u,u[:,0:1,:,:]),axis=1)
        u=np.concatenate((u,u[:,:,0:1,:]),axis=2)
        curlu=np.concatenate((curlu,curlu[:,0:1,:,:]),axis=1)
        curlu=np.concatenate((curlu,curlu[:,:,0:1,:]),axis=2)
        # Integrate to get quantities
        E[i,0]=1/V*np.trapz(np.trapz(np.trapz(.5*np.linalg.norm(u,axis=-1)**2,dx=zl/N[2],axis=-1),dx=xl/N[1],axis=-1),x=y,axis=-1)
        E[i,1]=1/V*np.trapz(np.trapz(np.trapz(np.linalg.norm(curlu,axis=-1)**2,dx=zl/N[2],axis=-1),dx=xl/N[1],axis=-1),x=y,axis=-1)
        E[i,2]=1/(2*A)*np.trapz(np.trapz(dy[0,:,:,0]+dy[-1,:,:,0],dx=zl/N[2],axis=-1),dx=xl/N[1],axis=-1)


    pickle.dump(E,open('Energy.p','wb'))
    [_,EPOD]=pickle.load(open('/home/linot/Couette/PODFFT/NoDialias_Correct_nomean2/Check_Stats/Energy.p','rb'))
    
    plt.figure(figsize=(4,4))
    plt.plot(EPOD[:,2],EPOD[:,1])
    plt.plot(E[:,2],E[:,1],'--')
    plt.xlabel('I')
    plt.ylabel('D')
    plt.savefig('IvD.png')

    datt=np.histogram2d(EPOD[:,2], EPOD[:,1], bins=30,range=[[2, 4], [2, 5]],density=True)[0]
    dat=np.histogram2d(E[:,2], E[:,1], bins=30,range=[[2, 4], [2, 5]],density=True)[0]
    fig,axs=plt.subplots(1,3,figsize=(6,2))
    (ax1,ax2,ax3)=axs
    dats=[datt,dat,np.abs(datt-dat)]
    for ax,plot in zip(axs,dats):
        pcm=ax.pcolormesh(plot.T,shading='gouraud',vmin=0,vmax=np.max(datt))
        fig.colorbar(pcm, ax=ax)
    plt.savefig('IvD_Hist.png')

    ###########################################################################
    # Compute Statistics: Short-time stats
    ###########################################################################
    Out('Computing Ensemble Stats')
    # Convert to torch tensor
    T=200
    tt=np.arange(T)
    t=torch.tensor(tt)
    t=t.type(torch.FloatTensor)
    IC=1000
    np.random.seed(2000) # Always pick the same random IC

    # This code is needed to avoid the boundary between chunks of test data
    exs=np.cumsum(lens)-int(round(M*frac))
    exs=exs[11:]
    for i in range(len(exs)):
        if i==0:
            temp=np.arange(0,exs[i]-T)
        else:
            temp=np.concatenate((temp,np.arange(exs[i-1],exs[i]-T)))
    ex=np.random.choice(temp,IC)

    h_ens=[]
    dif=[]
    difh=[]
    for i in range(IC):
        y0=torch.tensor(h_test[ex[i]:ex[i]+1,np.newaxis,:])
        y0=y0.type(torch.FloatTensor)
    
        # Pred trajectory
        hNN = odeint(func, y0, t)
        hNN=np.squeeze(hNN.detach().numpy())
        difh.append(np.linalg.norm(hNN-h_test[ex[i]:ex[i]+T,:],axis=1))

        # Decode
        hNN=hNN+h_mean
        aNN=decode.predict(hNN)
        
        # Get the difference between data in the normalized real POD coefficients
        dif.append(np.linalg.norm(aNN-a_test[ex[i]:ex[i]+T,:],axis=1))
        h_ens.append(hNN)

    pickle.dump([dif,difh,h_ens],open('Ensemble.p','wb'))
    dif=np.asarray(dif)
    dif=np.mean(dif,axis=0)
    difh=np.asarray(difh)
    difh=np.mean(difh,axis=0)
    pickle.dump([dif,difh],open('Ensemble_mean.p','wb'))
    
    T=100
    plt.figure()
    plt.plot(tt[:T],dif[:T])
    plt.xlabel('t')
    plt.ylabel('dif')
    plt.xlim([0,T])
    plt.savefig('Dif.png')

    plt.figure()
    plt.plot(tt[:T],difh[:T])
    plt.xlabel('t')
    plt.ylabel('difh')
    plt.xlim([0,T])
    plt.savefig('Difh.png')
