from keras.preprocessing.image import save_img
import os,time,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # Uncomment this and next line to work with cpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Input
from data import *
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import keras.backend as K
from keras.models import Model
from pathlib import Path
from m_resunet import ResUnetPlusPlus
from keras.layers import Input, Concatenate, Cropping2D
import tensorflow_addons as tfa
from ss_utils import *
from unet import * 
from m_resunet import ResUnetPlusPlus
from augment import *

model_type= "MM"                	 			# "MM" for MixMatch or "FM" for FixMatch, which SSL model you want to use
baseline = "MU"				 		# The baseline model that want to be used "MU" for MobileUnet and RU for ResUNet++
path_of_weights = "./Weights/"	       		# The directory where all the weights are stored
path_of_input = "./Datakeras/4x/Test/"			# Path of input files 
path_of_output = "./Results/Predictions/"+model_type+"_"+baseline+"/Test/" 	# Path where output has to be stored

lambda_epoch='0.3_18'						# Lambda used and epoch number. For storing the results in this name
weight_num = "weights.0.3.018.hdf5"				# The trained model's weight file which has to be loaded
l=1
k=4


if baseline=='RU':
    arch = ResUnetPlusPlus(input_size=None,no_classes=5)
    model1 = arch.build_model()
    model2 = arch.build_model()
    weights_folder = path_of_weights + "/weights_" + model_type + "_RU"
else:
    model1 = build_model(input_shape = (None,None,3), preset_model = "MobileUNet-Skip", num_classes = num_classes)  # build the model
    model2 = model1    
    weights_folder = path_of_weights + "/weights_" + model_type + "_MU"

model1._name="base_model"

folder = f"{path_of_output}/{model_type}/Test_{baseline}_{lambda_epoch}"

def FixMatch_model():
    inp1 = Input((None,None, 3),name='inp1')
    inp2 = Input((None,None, 3),name='inp2')

    transforms_w,inv_transforms_w = weak_transforms()
    transforms_s,inv_transforms_s = strong_transforms()    

    output1 = model1(inp1)
    #output1 = Cropping2D(cropping=48,name='labelled')(output1)

    output2 = AugmLayer(transforms_w, output_dim=None, preproc_input=None,name='aug')(inp2, training=True)   
    output2 = model1(output2)
    output2 = AugmLayer(inv_transforms_w, output_dim=None, preproc_input=None,name='rev_aug')(output2, training=True)
    #output2 = Cropping2D(cropping=48,name='unlabelled')(output2)

    model = Model(inputs=[inp1, inp2],outputs=[output1,output2] , name='mixmatch_model')
    return model
        
def MixMatch_model():
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
    
if  model_type== "MM":
    model =   MixMatch_model()
elif model_type== "FM":
    model =   FixMatch_model()     
            
model.load_weights("%s/%s"%(weights_folder,weight_num)) 
layer_model_weights = model.get_layer('base_model').get_weights()
model2.set_weights(layer_model_weights) 

start = time.time()
print(model.summary())
print(weight_num)

val_x,val_y,img_name = validation3(path_of_input+"images",path_of_input+"mask")
num_val = len(img_name)
print("No. of Images:",num_val)

z= model2.predict(val_x, batch_size=1, verbose=0, steps=None)

if os.path.isdir(path_of_output) is not True:
    os.makedirs(path_of_output)
val_y = colour_code(val_y,label_values)
z = colour_code(z,label_values)

for i in range(len(z)):
    a = img_name[i]
    image_name = os.path.basename(a)
    print(image_name)
    out = z[i]
    inp = val_y[i]
    image_name = image_name[:image_name.index('.')] 
    save_img('%s/%s_pred.png'%(path_of_output,image_name), out)
    save_img('%s/%s_gt.png'%(path_of_output,image_name), inp)
end = time.time()
print(f"time taken for { len(z) } images is {end - start} seconds ")
