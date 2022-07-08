import os
from ss_utils import *
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # Uncomment this and next lines for working on CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import cv2
import tensorflow as tf
from m_resunet import ResUnetPlusPlus
from keras.models import Model
from keras.layers import Input,Layer,Lambda, AveragePooling2D, Activation, Concatenate, Cropping2D
import keras.backend as K
K.set_image_data_format('channels_last')
import sys,math,csv,time
from pathlib import Path
from sklearn.utils import shuffle
from glob import glob
from augment import * 
from tqdm import tqdm
from unet import *


# RGB value for every classes
class1 = [0,0,255]                         	 #benign
class2 = [255,0,0]                         	 #malignant
class3 = [0,255,0]                        	 #cytoplasm
class4 = [255,0,255]                     	 #inflammatory
class0 = [255,255,255]                    	 #background which is the final class

label_values = [class1] + [class2] + [class3]  + [class4]  + [class0]
num_classes = len(label_values)



    

# Training paths 
train_path = "./Data_keras/10x/train_384/"     # Path for the 10X train dataset
train_path1 = "./Data_keras/4x/train_384/"     # Path for the  4X train dataset


# Validation paths
valid_path = ".Data_keras/10x/val_384/"       # Path for the 10X validation dataset
valid_path1 = "./Data_keras/4x/val_384/"	  # Path for the  4X validation dataset


folder_image1 = "images_padded"  		  # If padded image is used for training 
folder_image2 = "images"			  # if non-padded images are used for training. Also for evauating the data non-padded data is used.
folder_mask ="mask"			  	 


l=1
k=4
T=0.5
batch_size1 = 1 					# No. of images in  training and validation batch
batch_size2 = 1 					# No. of images in a evaluation batch
size1 =384        					# size of non padded images
size2 =480       					# size of padded images
padding = "False"				  	# if padding is used for the images
if padding=="False":
    folder_image1 = folder_image2
    size2=size1
else:    						# uncomment the cropping layers in the model if padding is used otherwise comment those lines    
    padding_pixels = int((size2-size1)/2)		# no. of pixels padded at each side

## Training and Validation Data 
train_xl = sorted(glob(os.path.join(train_path, folder_image1, "*.png")))      # Paths of Labelled train Images are loaded
train_yl = sorted(glob(os.path.join(train_path, folder_mask , "*.png")))       # Paths of Corresponding labels are loaded 
train_xl, train_yl = shuffling(train_xl, train_yl) 		       	  # Shuffling 
train_xl2 = sorted(glob(os.path.join(train_path, folder_image2, "*.png")))     # Paths for evaluation of train data
train_yl2 = sorted(glob(os.path.join(train_path, folder_mask , "*.png")))


valid_xl = sorted(glob(os.path.join(valid_path, folder_image1, "*.png")))      # Path of labelled validation images
valid_yl = sorted(glob(os.path.join(valid_path, folder_mask , "*.png")))	  # Path of labelled validation mask 
valid_xl2 = sorted(glob(os.path.join(valid_path, folder_image2, "*.png")))	  # Paths for evaluation of labelled validation data



train_xu = sorted(glob(os.path.join(train_path1, folder_image1, "*.png")))	  # Path of unlabelled train images
train_yu = sorted(glob(os.path.join(train_path1, folder_mask , "*.png")))	  # Path of unlabelled train mask
train_xu2 = sorted(glob(os.path.join(train_path1, folder_image2, "*.png")))	  # Path of evaluation of unlabelled train data

valid_xu = sorted(glob(os.path.join(valid_path1, folder_image1, "*.png")))	  # Path of unlabelled validation images
valid_xu2 = sorted(glob(os.path.join(valid_path1, folder_image2, "*.png")))	  # Path of unlabelled validation mask
valid_yu = sorted(glob(os.path.join(valid_path1, folder_mask , "*.png")))	  # Path of evaluation of unlabelled validation data




baseline = "MobileUNet" #"ResUNet++"			#use the baseline required whether MobileUNet or ResUNet++
if baseline == "MobileUNet":
    model1 = build_model(input_shape = (None,None,3), preset_model = "MobileUNet-Skip", num_classes = num_classes)  # MobileUNet Model
    #model1.load_weights("%s/weights.194.hdf5"%str(weights_folder1)) # Load weight if pre-training is required
    
    # CSV files for training and validation    
    csv_name = "./Results/training_MM_MU.csv"
    csv_name2 = "./Results/validation_MM_MU.csv"

    # Path for saving Weights
    weights_folder1= "./Weights/weights_MU"
    weights_folder= "./Weights/weights_MM_MU"

else:
    arch = ResUnetPlusPlus(input_size=None,no_classes=5) # ResUNet++
    model1 = arch.build_model()
    #model1.load_weights("%s/weights.168.hdf5"%str(weights_folder1)) # Load weight if pre-training is required
 
    # CSV files for training and validation    
    csv_name = "./Results/training_MM_RU.csv"
    csv_name2 = "./Results/validation_MM_RU.csv"

    # Path for saving Weights
    weights_folder1= "./Weights/weights_RU"
    weights_folder= "./Weights/weights_MM_RU"
    
model2 = model1					# for evaluation taking a copy of the model
model1._name="base_model"

if os.path.isdir(weights_folder) is not True:
    os.mkdir(weights_folder)

# A csv file is creating for validation results
if not os.path.exists(csv_name2):
    with open(csv_name2, 'a+') as f:
        f1 = csv.writer(f)
        header1 = ["Epoch","Val_10x loss","Val_10x fscore","10x_ben","10x_mal","Val_4x loss","Val_4x fscore","4x_ben","4x_mal","Lambda","fscore_avg"]
        f1.writerow(header1)



# Model is defined
def base_model():
    inp1 = Input((None,None, 3),name='inp1')
    inp2 = Input((None,None, 3),name='inp2')
    transforms1,inv_transforms1 = transforms(l)
    transforms2,inv_transforms2 = transforms(k)    
    output1 = AugmLayer(transforms1, output_dim=None, preproc_input=None,name='aug_1')(inp1, training=True)
    output1 = model1(output1)
    output1 = AugmLayer(inv_transforms1, output_dim=None, preproc_input=None,name='rev_aug_1_2')(output1, training=True)
    #output1 = Cropping2D(cropping=padding_pixels,name='labelled')(output1)		# comment this line if padding is not used
    output2 = AugmLayer(transforms2, output_dim=None, preproc_input=None,name='aug_2')(inp2, training=True)   
    output2 = model1(output2)
    output2 = AugmLayer(inv_transforms2, output_dim=None, preproc_input=None,name='rev_aug_2_2')(output2, training=True)
    #output2 = Cropping2D(cropping=padding_pixels,name='unlabelled')(output2)		# comment this line if padding is not used
    model = Model(inputs=[inp1,inp2],outputs=[output1,output2], name='mixmatch_model')
    return model

# metrics required for evaluation. It is defined in ss_utils.py    
metrics = [
        fscore_avg,
        fscore_benign,
        fscore_mal
       ]

num_train = len(train_xl)
train_steps = (num_train//batch_size1)


def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([size1, size1, 3])
    y.set_shape([size1, size1, num_classes])
    return x, y


def parse_data_train(x_l,x_u, y_l):
    def _parse(x_l,x_u, y_l):
        #seed1 = tf.random.uniform(shape=[1],minval=0,maxval=2048)
        x_l = read_image(x_l)
        x_u = read_image(x_u)
        y_l = read_mask(y_l)        
        return x_l,x_u, y_l
    x_l,x_u, y_l = tf.numpy_function(_parse, [x_l,x_u, y_l], [tf.float32,tf.float32, tf.float32])
    x_l.set_shape([size2, size2, 3])    
    x_u.set_shape([size2, size2, 3])   
    y_l.set_shape([size1, size1, num_classes])    
    return x_l, x_u, y_l
    

def tf_dataset(x, y, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(map_func=parse_data)
    #dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    return dataset


def tf_dataset_train(x_l,x_u,y_l, batch=2):
    dataset1 = tf.data.Dataset.from_tensor_slices((x_l,x_u,y_l))
    dataset1 = dataset1.shuffle(buffer_size=32)
    dataset = dataset1.map(map_func=parse_data_train)
    #dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    return dataset 


train_l_dataset = tf_dataset(train_xl2, train_yl2, batch=batch_size2)   # dataset loading for evaluating labelled train 
val_l_dataset = tf_dataset(valid_xl2, valid_yl, batch=batch_size2)     # dataset loading for evaluating labelled validation 
val_u_dataset = tf_dataset(valid_xu2, valid_yu, batch=batch_size2)     # dataset loading for evaluating unlabelled validation 

train_l_steps = (len(train_xl2)//batch_size2)				# no. of images required while evaluating
valid_l_steps = (len(valid_xl2)//batch_size2) 
valid_u_steps = (len(valid_xu2)//batch_size2)


fixed_lr=1e-4 								# learning rate
index=int(len(train_xl)//batch_size1)					# To chose same number of labelled and unlabelled train data while training
num_valid = len(valid_xl)						# To chose same number of labelled and unlabelled validation data while training
						
rms = tf.keras.optimizers.RMSprop(learning_rate=fixed_lr)		# RMSProp optimizer
#sgd = tf.keras.optimizers.SGD(learning_rate=fixed_lr)		# sgd optimizer
opt = rms 								# optimizer required
 
      
@tf.function
def apply_gradient(optimizer, model, x_l , x_u, y_l):
    def sharpen1(p, T):
        return K.pow(p, 1./T) / K.sum(K.pow(p, 1./T), axis=-1, keepdims=True) 
    transforms1,inv_transforms1 = transforms(l)
    transforms2,inv_transforms2 = transforms(k)            
    model.get_layer('aug_1').transforms = transforms1
    model.get_layer('aug_2').transforms = transforms2
    model.get_layer('rev_aug_1_2').transforms = inv_transforms1       
    model.get_layer('rev_aug_2_2').transforms = inv_transforms2   
    y_dummy, y_pred_w = model([x_l,x_u])
    g_label = K.mean(y_pred_w,0,keepdims=True)
    mask2 = sharpen1(g_label,T)
    y_pred_w = K.repeat_elements(mask2,k,axis=0)


    with tf.GradientTape() as tape:
        y_pred_l, y_pred_s = model([x_l,x_u])
        loss_l = custom_ce(y_l, y_pred_l)
        loss_u = custom_mse(y_pred_w, y_pred_s)
        loss_value = loss_l + lambda_u * loss_u
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    f_score_l = fscore(y_l, y_pred_l)
    f_score_u = fscore(y_pred_w, y_pred_s)
    
    return loss_value , f_score_l , f_score_u ,loss_l, loss_u


def train_data_for_one_epoch():
    losses = []
    fscorel = []
    fscoreu = []
    pbar = tqdm(total=index, position=0,leave=True, bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}')
    for step, (x_l,x_u,y_l) in enumerate (train_dataset):
        loss_value, f_score_l , f_score_u, loss_l, loss_u = apply_gradient(opt, model, x_l , x_u, y_l)
        losses.append(loss_value)
        fscorel.append(f_score_l)
        fscoreu.append(f_score_u)
        pbar.set_description("Training loss for step %s: %0.4f , loss_labelled: %0.4f, loss_unlabelled: %0.4f, Fscore labelled: %0.4f, Fscore unlabelled: %0.4f"%(int(step),float(loss_value),float(loss_l),float(loss_u), f_score_l, f_score_u))
        pbar.update()
    return losses, fscorel, fscoreu

#lambda_set = [0.1,0.3,1.0,3.0,10.0,30,100]  # for tuning the hyperparameter lambda
lambda_set = [0.3]
epoch1=0

model = base_model()
continue_training = "False"
if continue_training=="True":
    epoch1=76
    weight_num = "weights.0.3.076.hdf5"    # weights to be loaded for a continue training
    model.load_weights("%s/%s"%(weights_folder,weight_num))
			

print(model.summary())

for lambda_u in lambda_set:
    model.save_weights('mixmatch_train.h5')     # For making the same random initiliazation for all the lambda values, initial weight is stored
    final_epoch=20
    print('lambda:',lambda_u)
    for j in range(epoch1,final_epoch):
        t = time.localtime()
        epoch = j+1
        current_time = time.strftime("%H:%M:%S", t)
        print(f"The epoch {epoch} started at time: {current_time}")
        train_xu = shuffling2(train_xu)
        train_dataset = tf_dataset_train(train_xl[:index], train_xu[:index], train_yl[:index], batch=batch_size1)
        losses_train, fscorel_train, fscoreu_train  = train_data_for_one_epoch()
        losses_train_mean = np.mean(losses_train)
        f_scorel_train_mean = np.mean(fscorel_train)
        f_scoreu_train_mean = np.mean(fscoreu_train)
        print(' epoch:',epoch)
        print('losses_train_mean',losses_train_mean)
        print('f_score_trainl_mean',f_scorel_train_mean)
        print('f_score_trainu_mean',f_scoreu_train_mean)
        model.save_weights('%s/weights.%s.%03d.hdf5'%(weights_folder,lambda_u,epoch))
        layer_model_weights = model.get_layer('base_model').get_weights()
        model2.set_weights(layer_model_weights) 
       
        model2.compile(optimizer = opt, loss=cce_loss ,  metrics =metrics) 
        history1 = model2.evaluate(train_l_dataset,steps=index)
        history2 = model2.evaluate(val_l_dataset,steps=valid_l_steps)
        history3 = model2.evaluate(val_u_dataset,steps=valid_u_steps)   
     
        history = history1.copy()
        history.extend(history2)
        history.extend(history3)
        history.insert(0,epoch)
        history = np.around(history,3)
        history = history.tolist()
        history.append(lambda_u)
        print("\n")
          
        print(f" loss and training score of 10x data after epoch {epoch+1} are ",history1)     
        print(f" loss and validation score of 10x data after epoch {epoch+1} are ",history2)
        print(f" loss and validation score of 4x after epoch {epoch+1} are ",history3,"\n")  
        with open(csv_name2, 'a+', newline='') as f:
             f1 = csv.writer(f)
             f1.writerow(history)    
    model.load_weights('mixmatch_train.h5')    # For making the same random initiliazation for all the lambda values, initial weight is loaded
    epoch1=0         



