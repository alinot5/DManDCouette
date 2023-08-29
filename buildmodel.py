#! /opt/cluster_soft/anaconda3/bin/python3
"""
Created on Thu Mar 28 13:28:41 2019

This is the architecture of the NN used in all of the trials in this folder

@author: Alec
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Lambda
import tensorflow.keras.backend as K
from functools import reduce
import os

#Note: All operations must go into Lambda layers

def trunc():
    path=os.path.abspath(os.getcwd())
    
    return int(path.split('/')[8])

def buildmodel_hybrid(N,a):
    K.set_floatx('float64')
    #Parameters
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS=500

    hiddenin=[1000,trunc()]
    actin=['sigmoid',None]

    hiddenout_PCA=[1000,trunc()]
    hiddenout=[1000,N-trunc()]
    actout=['sigmoid',None]

    ########################################################################################
    # Encoder
    ########################################################################################
    # Input size
    main_input = layers.Input(shape=(N,), name='main_input')

    # Nonlinear dim reduction
    encode=main_input
    for i in range(len(hiddenin)):
        encode=layers.Dense(hiddenin[i],activation=actin[i],name='Dense_In'+str(i+1))(encode)

    # Add linear and nonlinear dim reduction
    PCA_input_trunc=Lambda(lambda x: tf.slice(x,[0,0],[-1,trunc()]))(main_input)
    hidden=layers.Add(name='Hidden_Layer')([encode, PCA_input_trunc])

    ########################################################################################
    # Decoder
    ########################################################################################
    # Linear dim increase
    paddings=tf.constant([[0,0,],[0,N-trunc()]])
    PCA_output=Lambda(lambda x: tf.pad(x,paddings,'CONSTANT'), name='PCA_Int_Output')(hidden)

    # Nonlinear dim increase
    decode=hidden
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)

    #Append tied layers on top
    conc=hidden
    for i in range(len(hiddenout_PCA)):
        conc=layers.Dense(hiddenout_PCA[i],activation=actout[i],name='Dense_Out_Reg'+str(i+1))(conc)
    # This layer penalizes when the top terms of the encoder do not equal those of the decoder
    reg=layers.Add(name='Regularize')([conc,encode])
    # This combines both the top truncated values with the remaining values
    decode=layers.Concatenate(name='Concatenate')([conc,decode])

    # Add PCA results to NN results
    main_output=layers.Add(name='Output')([decode, PCA_output])

    # Build model
    model=Model(inputs=main_input,outputs=[main_output,a*reg])

    return model,EPOCHS,optimizer

def encode_hybrid(N):
    K.set_floatx('float64')

    hiddenin=[1000,trunc()]
    actin=['sigmoid',None]

    ########################################################################################
    # Encoder
    ########################################################################################
    # Input size
    main_input = layers.Input(shape=(N,), name='main_input')

    # Nonlinear dim reduction
    encode=main_input
    for i in range(len(hiddenin)):
        encode=layers.Dense(hiddenin[i],activation=actin[i],name='Dense_In'+str(i+1))(encode)

    # Add linear and nonlinear dim reduction
    PCA_input_trunc=Lambda(lambda x: tf.slice(x,[0,0],[-1,trunc()]))(main_input)
    hidden=layers.Add(name='Hidden_Layer')([encode, PCA_input_trunc])

    # Build model
    model=Model(inputs=main_input,outputs=hidden)

    return model

def decode_hybrid(N):
    K.set_floatx('float64')
    #Parameters


    hiddenout_PCA=[1000,trunc()]
    hiddenout=[1000,N-trunc()]
    actout=['sigmoid',None]

    ########################################################################################
    # Encoder
    ########################################################################################
    # Input size
    hidden = layers.Input(shape=(trunc(),), name='main_input')

    ########################################################################################
    # Decoder
    ########################################################################################
    # Linear dim increase
    paddings=tf.constant([[0,0,],[0,N-trunc()]])
    PCA_output=Lambda(lambda x: tf.pad(x,paddings,'CONSTANT'), name='PCA_Int_Output')(hidden)

    # Nonlinear dim increase
    decode=hidden
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)

    #Append tied layers on top
    conc=hidden
    for i in range(len(hiddenout_PCA)):
        conc=layers.Dense(hiddenout_PCA[i],activation=actout[i],name='Dense_Out_Reg'+str(i+1))(conc)

    # This combines both the top truncated values with the remaining values
    decode=layers.Concatenate(name='Concatenate')([conc,decode])

    # Add PCA results to NN results
    main_output=layers.Add(name='Output')([decode, PCA_output])

    # Build model
    model=Model(inputs=hidden,outputs=main_output)

    return model

def buildmodel(N):
    K.set_floatx('float64')
    #Parameters
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS=500
    hiddenin=[1000,trunc()]
    actin=['sigmoid',None]
    hiddenout=[1000,N]
    actout=['sigmoid',None]

    # Input size
    main_input = layers.Input(shape=(N,), name='main_input')

    ########################################################################################
    # Encoder
    ########################################################################################
    # Nonlinear dim increase
    encode=main_input
    for i in range(len(hiddenin)):
        encode=layers.Dense(hiddenin[i],activation=actin[i],name='Dense_In'+str(i+1))(encode)

    hidden=encode

    ########################################################################################
    # Decoder
    ########################################################################################
    # Nonlinear dim increase
    decode=hidden
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)

    main_output=decode

    # Build model
    model=Model(inputs=main_input,outputs=main_output)
    
    return model,EPOCHS,optimizer

def encode(N):
    K.set_floatx('float64')
    #Parameters
    hiddenin=[1000,trunc()]
    actin=['sigmoid',None]

    # Input size
    main_input = layers.Input(shape=(N,), name='main_input')

    ########################################################################################
    # Encoder
    ########################################################################################
    # Nonlinear dim increase
    encode=main_input
    for i in range(len(hiddenin)):
        encode=layers.Dense(hiddenin[i],activation=actin[i],name='Dense_In'+str(i+1))(encode)

    main_output=encode

    # Build model
    model=Model(inputs=main_input,outputs=main_output)
    
    return model

def decode(N):
    K.set_floatx('float64')
    #Parameters
    hiddenout=[1000,N]
    actout=['sigmoid',None]

    # Input size
    main_input = layers.Input(shape=(trunc(),), name='main_input')

    ########################################################################################
    # Decoder
    ########################################################################################
    # Nonlinear dim increase
    decode=main_input
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)

    main_output=decode

    # Build model
    model=Model(inputs=main_input,outputs=main_output)
    
    return model

def buildmodel_decode(N):
    K.set_floatx('float64')
    #Parameters
    optimizer = tf.keras.optimizers.Adam()
    EPOCHS=500
    hiddenout=[1000,N-trunc()]
    actout=['sigmoid',None]

    # Input size
    main_input = layers.Input(shape=(trunc(),), name='main_input')

    ########################################################################################
    # No encoder
    ########################################################################################
    # Nonlinear dim increase
    encode=main_input
    hidden=encode

    ########################################################################################
    # Decoder
    ########################################################################################
    # Nonlinear dim increase
    decode=hidden
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)

    main_output=decode

    # Build model
    model=Model(inputs=main_input,outputs=main_output)

    return model,EPOCHS,optimizer