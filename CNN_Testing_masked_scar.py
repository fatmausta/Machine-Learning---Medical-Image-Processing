# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:54:20 2017

@author: fusta
"""
import numpy as np
import keras
import SimpleITK
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib.pyplot as plt
# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy
import SimpleITK as sitk
K.set_image_dim_ordering('th')
from skimage.util import view_as_windows
from skimage.util import view_as_blocks

patch_size = 2
window_size = 30
nclasses = 3
epocs = 2
patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)
test_slice = range(30,33)
modelname='ModelMyoCNN1_aug.h5'
train_slice = range(0,0)
visualize=0
pid_test = ('0485', '0632', '0715', '0730', '0917', '0921', '0953', '1036', '1073', '1076', '1115', '1166', '1168', '1171')

#datapath = 'DataCNNScarNorm/'
datapath = 'C:\\Users\\fusta\\Dropbox\\1_Machine_Learning\\DataCNNScar\\'
def runCNNModel(dataset_training, dataset_testing, test_img_shape, test_img_shape_padded, pads, epocs, patch_size, window_size, nclasses, pid, test_slice):
    # preprocessing
    X_training = np.zeros((len(dataset_training),windowsize_sq))
    Y_training = np.zeros((len(dataset_training),1))
    X_testing = np.zeros((len(dataset_testing),windowsize_sq))
    Y_testing = np.zeros((len(dataset_testing),1))
#    dataset_training = []
#    dataset_testing =[]
    for p in range(0,len(dataset_testing)):
        X_testing[p]=dataset_testing[p][0]
        Y_testing[p]=dataset_testing[p][1]

    for p in range(0,len(dataset_training)):
        X_training[p]=dataset_training[p][0]
        Y_training[p]=dataset_training[p][1]
        
    #Reshape my dataset for my model       
    X_training = X_training.reshape(X_training.shape[0], 1 , window_size, window_size)
    X_testing = X_testing.reshape(X_testing.shape[0], 1, window_size, window_size)
    X_training = X_training.astype('float32')
    X_testing = X_testing.astype('float32')
    X_training /= 255
    X_testing /= 255
    Y_training = np_utils.to_categorical(Y_training, nclasses)
    Y_testing = np_utils.to_categorical(Y_testing, nclasses)
    model = keras.models.load_model(modelname)
    #predict classes
    y_pred = model.predict_classes(X_testing)    
    y_pred = y_pred.astype('float32')
    
    y_pred = np_utils.to_categorical(y_pred, nclasses)
    print(classification_report(Y_testing, y_pred))#, target_names = target_names))
    y_pred = y_pred.argmax(1).astype('float32')

    Y_testing= Y_testing.argmax(1).astype('float32')
    X_testing = []
    y_testing_multi = []
    y_pred_scaled = []
    y_pred_scaled_cropped = []
    y_pred = np.reshape(y_pred, (len(test_slice), int(test_img_shape_padded[0][0]/patch_size), int(test_img_shape_padded[0][1]/patch_size)))* (255/(nclasses-1))
    for sl in range(0,len(test_slice)):
        y_pred_scaled.append(np.reshape(y_pred[sl].repeat(patch_size, axis =0).repeat(patch_size, axis =1), (1, test_img_shape_padded[0][0], test_img_shape_padded[0][1])))
    
    y_pred_scaled = np.reshape(y_pred_scaled,(len(test_slice), test_img_shape_padded[0][0], test_img_shape_padded[0][1]))    
    y_testing_multi = (np.reshape(Y_testing, (len(test_slice), int(test_img_shape_padded[0][0]/patch_size), int(test_img_shape_padded[0][1]/patch_size)))* (255/(nclasses-1)))

    for s in range(0, len(test_slice)):
        y_pred_scaled_cropped.append(np.reshape(y_pred_scaled[s][:-pads[0],:-pads[1]], (test_img_shape[0][1], test_img_shape[0][2])))
    
    y_pred_scaled_cropped = np.array(y_pred_scaled_cropped)

    if visualize == 1:
        for s in range(0, 77):
            plt.figure()
            plt.imshow(y_pred_scaled_cropped[s]) 
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(y_pred[s]) 
            plt.subplot(1,2,2)            
            plt.imshow(y_testing_multi[s]) 
 
    return y_pred_scaled_cropped, y_testing_multi 

def PatchMaker(patch_size, window_size, nclasses, pid, datapath):  
     
    patch_labels_training=[]
    patch_labels_testing=[]    
    window_intensities_training=[]
    window_intensities_testing=[]
    test_img_shape = []#np.empty((len(pid_test),2))
    test_img_shape_padded = []#np.empty((len(pid_test),2))
#    pads=[]
    
    LGE = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-LGE-cropped.mhd')
    scar = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-scar-cropped.mhd')
    myo = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-myo-cropped.mhd')
    
    #convert a SimpleITK object into an array
    LGE_3D = SimpleITK.GetArrayFromImage(LGE)
    scar_3D = SimpleITK.GetArrayFromImage(scar) 
    myo_3D = SimpleITK.GetArrayFromImage(myo) 
   #masking LGE with GT myo
    LGE_3D = np.multiply(LGE_3D,myo_3D)

    h_LGE = LGE_3D.shape[1]
    w_LGE = LGE_3D.shape[2] 
    d_LGE = LGE_3D.shape[0]
    test_slice = range(0,d_LGE)

    #make windows size and patch size evenly dvideble 
    if (window_size-patch_size)%2 != 0:
        window_size +=1    
    #calculate the amount of padding for heaght and width of a slice for patches
    rem_w = w_LGE%patch_size
    w_pad=patch_size-rem_w      
    rem_h = h_LGE%patch_size
    h_pad=patch_size-rem_h    
    pads = (h_pad,w_pad)
    
    for sl in test_slice:
        LGE_padded_slice=numpy.lib.pad(LGE_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
        scar_padded_slice=numpy.lib.pad(scar_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))  
        LGE_patches = view_as_blocks(scar_padded_slice, block_shape = (patch_size,patch_size))
        LGE_patches = numpy.reshape(LGE_patches,(LGE_patches.shape[0]*LGE_patches.shape[1],patch_size,patch_size)) 
        #re-pad your padded image before you make your windows
        padding = int((window_size - patch_size)/2)
        LGE_repadded_slice = numpy.lib.pad(LGE_padded_slice, ((padding,padding),(padding,padding)), 'constant', constant_values=(0,0))
        #done with the labels, now we will do our windows, 
        LGE_windows = view_as_windows(LGE_repadded_slice, (window_size,window_size), step=patch_size)
        LGE_windows = numpy.reshape(LGE_windows,(LGE_windows.shape[0]*LGE_windows.shape[1],window_size,window_size))        
        #for each patches: 
        for p in range(0,len(LGE_patches)):            
            label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(LGE_patches[p]),numpy.square(patch_size)),nclasses),1))
            label = numpy.reshape(label, (1,1))           
            if label==nclasses:
                label -=1 #mmake sure the range for the classes do not exceed the maximum
            #making your window  intensities a single row
            intensities = numpy.reshape(LGE_windows[p],(window_size*window_size))
            intensities = numpy.reshape(intensities, (1,window_size*window_size))
            patch_labels_testing.append(label)
            window_intensities_testing.append(intensities)
                            
    print(pid)
    test_img_shape_padded.append(LGE_padded_slice.shape)        #Now, intensities=windows and patches labels will be comboined
    test_img_shape.append(LGE_3D.shape)        #Now, intensities=windows and patches labels will be comboined
        
    training_data= list(zip(numpy.uint8(window_intensities_training),numpy.uint8(patch_labels_training)))
    testing_data= list(zip(numpy.uint8(window_intensities_testing),numpy.uint8(patch_labels_testing)))  
    return training_data, testing_data, test_img_shape, test_img_shape_padded, pads, test_slice

#Dice Calculation
def DiceIndex(BW1, BW2):
    BW1 = BW1.astype('float32')
    BW2 = BW2.astype('float32')
    #elementwise multiplication
    t= (np.multiply(BW1,BW2))
    total = np.sum(t)
    DI=2*total/(np.sum(BW1)+np.sum(BW2))
    DI=DI*100
    return DI

Dice_list = []
##MAIN SECTION    
for pid in pid_test:
    (dataset_training, dataset_testing, test_img_shape, test_img_shape_padded, pads, test_slice) = PatchMaker(patch_size, window_size, nclasses, pid, datapath)
    (y_pred_scaled_cropped, y_testing_multi) = runCNNModel(dataset_training, dataset_testing, test_img_shape, test_img_shape_padded, pads, epocs, patch_size, window_size, nclasses, pid,test_slice)
    scarGT = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-scar-cropped.mhd')
    scar3D = sitk.GetArrayFromImage(scarGT)
    BW2 = np.array(y_pred_scaled_cropped)
    BW2=BW2/255# = y_pred_scaled_cropped
    BW1 = scar3D
    Dice = DiceIndex(BW1, BW2)
    Dice_list.append(Dice)
    print('\nTraining Slices: %s Testing Slices: %s' %(train_slice, test_slice[pid_test.index(pid)]))
    print('\nDice for pid %s: %2.3f percent \n\n' %(pid, Dice))
     
print(pid_test)
print(Dice_list)