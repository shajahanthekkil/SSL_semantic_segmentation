import tensorflow_addons as tfa
import tensorflow as tf
from keras.layers import Layer
import keras.backend as K
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle
import cv2
from data import * 


# Meric for finding fscore 
def fscore(y_true,y_pred):
    y_true = y_true[...,:-1]
    y_pred = y_pred[...,:-1]
    y_true = K.round(K.flatten(y_true))
    y_pred = K.round(K.flatten(y_pred)) 
    true_positives = K.sum(K.round(y_true * y_pred))
    predicted_positives = K.sum(K.round(y_pred))
    possible_positives = K.sum(K.round(y_true))
    precision = true_positives / (predicted_positives+ K.epsilon())   
    recall = true_positives / (possible_positives+ K.epsilon())  
    f_score1 = 2*((precision*recall)/(precision+recall+ K.epsilon()))
    return f_score1    

@tf.function 
def fscore_avg(y_true,y_pred):
    ben_pix = tf.reduce_sum(y_true[...,0])
    mal_pix = tf.reduce_sum(y_true[...,1])
    ben = fscore_benign(y_true,y_pred)
    mal = fscore_mal(y_true,y_pred)
    if ben_pix ==0.:
        if mal_pix ==0.:
            avg=1.
        else:
            avg=mal
    else:
        if mal_pix==0.:
            avg=ben
        else:
            avg=(ben+mal)/2
    return avg   
 
def fscore_benign(y_true,y_pred):
    y_true1 = y_true[...,0]
    y_pred1 = y_pred[...,0]
    y_true1 = K.round(K.flatten(y_true1))
    y_pred1 = K.round(K.flatten(y_pred1)) 
    true_positives1 = K.sum(y_true1 * y_pred1)
    predicted_positives1 = K.sum(y_pred1)
    possible_positives1 = K.sum(y_true1)
    precision1 = true_positives1 / (predicted_positives1+ K.epsilon())   
    recall1 = true_positives1 / (possible_positives1+ K.epsilon())  
    f_score1 = 2*precision1*recall1/(precision1+recall1+ K.epsilon())
    return f_score1  

def fscore_mal(y_true,y_pred):
    y_true2 = y_true[...,1]
    y_pred2 = y_pred[...,1]
    y_true2 = K.round(K.flatten(y_true2))
    y_pred2 = K.round(K.flatten(y_pred2)) 
    true_positives2 = K.sum(y_true2 * y_pred2)
    predicted_positives2 = K.sum(y_pred2)
    possible_positives2 = K.sum(y_true2)
    precision2 = true_positives2 / (predicted_positives2+ K.epsilon())   
    recall2 = true_positives2 / (possible_positives2+ K.epsilon())  
    f_score2 = 2*precision2*recall2/(precision2+recall2+ K.epsilon())
    return f_score2  
    
# Custom loss function

    
def semi_loss(y_actual,y_predicted):
    y_actual = y_actual[0]
    y_actual = tf.reshape(y_actual,[k+l,size,size,num_classes])
    cce = custom_ce(y_actual[0],y_predicted[0])
    mse = custom_mse(y_actual[1:],y_predicted[1:])    
    #cce = custom_ce(y_actual[0],y_predicted[0,64:-64,64:-64,:])
    #mse = custom_mse(y_actual[1:],y_predicted[1:,64:-64,64:-64,:])
    loss_value = cce + lambda_u * mse  
    return loss_value    

def custom_ce(y_actual,y_predicted):
    loss_value1 = -K.mean(K.sum( y_actual * K.log( y_predicted + K.epsilon()),axis=-1))
    return loss_value1 

def custom_ce1(y_actual,y_predicted):
    loss_value1 = -(K.sum(y_actual * K.log( y_predicted + K.epsilon()),axis=-1))
    return loss_value1 
    
def custom_ce2(y_actual,y_predicted):
    class_weights = tf.constant([[0.967,0.96,0.88,0.974,0.219]])
    weights = K.sum(class_weights * y_actual, axis=-1)
    unweighted_losses = -(K.sum(y_actual * K.log( y_predicted + K.epsilon()),axis=-1))
    weighted_losses = unweighted_losses * weights
    loss = K.mean(weighted_losses) 
    return loss     
 
def custom_mse2(y_actual,y_predicted):
    class_weights = tf.constant([[0.967,0.969,0.887,0.969,0.208]])
    weights = K.sum(class_weights * y_actual, axis=-1)
    unweighted_losses = K.mean( K.square(y_actual-y_predicted),axis=-1)
    weighted_losses = unweighted_losses * weights    
    loss = K.mean(weighted_losses)    
    return loss
             
def custom_mse(y_actual,y_predicted):
    loss_value2 = K.mean(K.mean( K.square(y_actual-y_predicted),axis=-1))
    return loss_value2

def colour_code(image, label_values):
    x = np.argmax(image, axis = -1)
    colour_codes = np.array(label_values)
    x = colour_codes[x.astype(int)]
    return x
    
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def shuffling2(x):
    x = shuffle(x)
    return x
    
def one_hot(mask):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = (np.stack(semantic_map, axis=-1)).astype(float)
    return semantic_map

def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.0
    image = image.astype(np.float32)
    return image

def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = one_hot(mask)
    mask = mask.astype(np.float32)
    return mask


def loss_ul(y_actual,y_predicted):
    y_actual = tf.where(y_actual > 0.5, x = 1., y=0., name=None)
    y_actual = tf.cast(y_actual, dtype=tf.float32, name=None)
    cce = custom_ce1(y_actual,y_predicted)
    cce = cce[cce>0.]
    cce = tf.concat([cce,[0.0]],axis=-1)
    loss = K.mean(cce)
    return loss

cce_loss = tf.keras.losses.CategoricalCrossentropy()    
        

