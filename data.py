import os,cv2
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import tensorflow as tf

class1 = [0,0,255] #benign
class2 = [255,0,0] #malignant
class3 = [0,255,0] #cytoplasm
class4 = [255,0,255] #inflammatory
class0 = [255,255,255] #background which is the final class

check_path = os.path.dirname(os.getcwd()) + "/Results/check"
label_values = [class1] + [class2] + [class3]  + [class4]  + [class0]
num_classes = len(label_values)

def one_hot(mask):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = (np.stack(semantic_map, axis=-1)).astype(float)
    return semantic_map

def adjustData(img,mask):
    img = img / 255
    mask = one_hot(mask)
    return (img,mask)

def dataGenerator(batch_size,path,aug_dict,size,seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["images"],
        class_mode = None,
        batch_size = batch_size,
       # save_to_dir = "CamVid/check/image",
       # save_prefix = 'image',
        seed = seed)
    mask_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["mask"],
        class_mode = None,
        batch_size = batch_size,
       # save_to_dir = "CamVid/check/mask",
       # save_prefix= 'mask',
        seed = seed)
    data_generator = zip(image_generator, mask_generator)
    for (image,mask) in data_generator:
        image,mask = adjustData(image,mask)
        yield (image,mask)
        
def valGenerator(batch_size,path,size,seed=1):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["images"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    mask_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["mask"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    data_generator = zip(image_generator, mask_generator)
    for (image,mask) in data_generator:
        image,mask = adjustData(image,mask)
        yield (image,mask)        

        
def valGenerator6(batch_size,path1,path2,size,size2,k=2,T=0.5,seed=1):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(size2, size2),
        classes = ["images"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    mask_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(size, size),
        classes = ["mask"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    image2_generator = image_datagen.flow_from_directory(
        path2,
        target_size=(size2, size2),
        classes = ["images"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    mask2_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(size, size),
        classes = ["mask"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)        
    data_generator = zip(image_generator, mask_generator,image2_generator, mask2_generator)
    for (image,mask,image2,mask2) in data_generator:
        image,mask = adjustData(image,mask)
        image2,mask2 = adjustData(image2,mask2)
        mask2 = np.repeat(mask2,k,axis=0)
        mask = np.concatenate((mask,mask2),axis=0)
        m,n,o,p=mask.shape
        mask = np.reshape(mask,(m,n*o,p))
        mask = mask[np.newaxis,:]
        train = ([image,image2],mask)
        yield train
        
def sharpen1(p, T):
    return np.power(p, 1./T) / np.sum(np.power(p, 1./T), axis=-1, keepdims=True)
    
def sharpen(p, T):
    return tf.pow(p, 1/T) / tf.reduce_sum(tf.pow(p, 1/T), axis=1, keepdims=True)


def num_of_images(path):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path,
        classes = ["mask"],
        class_mode = None)
    return image_generator.samples

def num_of_images2(path):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path,
        classes = ["ul__val_images"],
        class_mode = None)
    return image_generator.samples

#mask name should end with '_gt' following the image name
def validation(image_path,mask_path,image_prefix = ".png",mask_prefix = "_gt.png"):
    image_name_arr = glob(os.path.join(image_path,"*%s"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):

        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        mask= cv2.cvtColor(cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        mask = mask[:,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr, image_name_arr

def validation3(image_path,mask_path,image_prefix = ".png",mask_prefix = "_gt.png"):
    image_name_arr = glob(os.path.join(image_path,"*%s"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):

        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        mask= cv2.cvtColor(cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:3584,:,:3]
        mask = mask[:3584,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr, image_name_arr
def validation2(image_path):
    image_name_arr = glob(os.path.join(image_path,"*.png"))
    image_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        img = img/255
        image_arr.append(img)
    image_arr = np.array(image_arr)
    return image_arr, image_name_arr
    
def weight(mask_path,mask_prefix = "_gt.png"):
    image_name_arr = glob(os.path.join(mask_path,"*%s"%mask_prefix))
    no_images = len(image_name_arr)
    class_1=0
    class_2=0
    class_3=0
    class_4=0
    class_0=0
    tot = size*size*no_images
    for index,item in enumerate(image_name_arr):
        mask = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        mask = mask[:,:,:3] 
        class_1 += np.sum(np.all(mask==class1,axis=-1))
        class_2 += np.sum(np.all(mask==class2,axis=-1))
        class_3 += np.sum(np.all(mask==class3,axis=-1))
        class_4 += np.sum(np.all(mask==class4,axis=-1))
        class_0 += np.sum(np.all(mask==class0,axis=-1))
    class1_wt = (tot - class_1)/tot
    class2_wt = (tot - class_2)/tot
    class3_wt = (tot - class_3)/tot
    class4_wt = (tot - class_4)/tot
    class0_wt = (tot - class_0)/tot
    class_weight = [class1_wt,class2_wt,class3_wt,class4_wt,class0_wt]
    return class_weight,no_images
