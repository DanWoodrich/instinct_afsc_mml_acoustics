
#v1-1: do away with 'offset', replace with what the concept evolved into, which is stride. 
#v1-2: update parameters to get rid of stuff that doesn't matter, add stuff that does. 
#v1-3: read filepaths instead of assume them 

#still experimental- this vers will attempt to save the
#file name and associated array for test/train and label
#as a dict, and reference that during mapping function.
#
#try to avoid tfrecord

#v1-7:
#go back to random offset in ingest instead of randomcrop. Use padding to make sure that the tensors stay the same size and so the dataset
#can be used as a repeatable epoch. 

#v1-8:
#change training behavior so that only one dataset is used (dropout based on random number generated)
#remove augmentation during inference

#v1-16: merge between v13 and v15.

#v1-17: make a parameter for spectrogram averaging on/off and augmentation on/off

#import pandas as pd
import tensorflow as tf
import sys
import numpy as np
#from matplotlib.image import imread
import os
import pathlib
import matplotlib.pyplot as plt
import csv
import gzip
import time
import pandas as pd
from time import time
#import math

#from tensorflow import keras
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

sys.path.append(os.getcwd())
from user.misc import arg_loader

#import sklearn
print(tf.version.VERSION)



#######import params###########

#args="C:/Apps/INSTINCT/Cache/491386/235062/FileGroupFormat.csv.gz C:/Apps/INSTINCT/Cache/491386/159690/295444 C:/Apps/INSTINCT/Cache/491386/756342/508664/182373 C:/Apps/INSTINCT/Cache/117419/673303/707058/700613 C:/Apps/INSTINCT/Cache/251579/121916/587840/248952/527200 C:/Apps/INSTINCT/Cache/251579/121916/587840/248952/527200/295543 C:/Apps/INSTINCT/Cache/251579/121916/587840/248952/527200/model.keras test RW EffecientNetB0 151 31 0.97 0.03 5 n 1.4834 2 224 248 train-win-slide-dl-v1-3 175 y 1"
#args=args.split()

args=arg_loader()

FGpath = args[1] #may not end up needing this. 
spec_path=args[2]
label_path=args[3]
split_path=args[4]
resultpath=args[5]
modelpath=args[6]
stage= args[7] #train, test, or inf

if stage !="train":
    augstr = args[8].split(",")
    brightness_high=float(augstr[0])
    brightness_low=float(augstr[1])
    contrast_high =float(augstr[2])
    GT_depth = args[9].count(",")+1
    model_name = args[10]
    native_img_height = int(args[11])
    native_pix_per_sec = int(args[12])
    only_labs = args[13]
    stride_pix_inf = int(args[14]) 
    #view_plots = args[15]
    model_win_size = int(args[15])
    win_size_native = int(args[16])
    
    #win_f_factor= float(args[19])
    #win_t_factor= float(args[20])

    #arguments: 
    batch_size = int(args[18])

    #dummy vars

    tp_weights=1.
    fp_perc = 1
    if only_labs=='y':
        tp_perc = float(input("tp_perc? #"))
    else:
        tp_perc = 1


    view_plots = 'n'

    prop_tp = 0.
    prop_fp = 0.
    
    addnoise_mean=0.
    addnoise_sttdev=0.

    do_aug = 'n'
    do_spec_averaging = 'n'

else:
    addnoise_mean = float(args[8])
    addnoise_stddev = float(args[9])
    augstr = args[10].split(",")
    brightness_high=float(augstr[0])
    brightness_low=float(augstr[1])
    contrast_high =float(augstr[2])
    do_aug = args[11]
    do_spec_averaging = args[12]
    GT_depth = args[13].count(",")+1
    learning_rate = float(args[14])
    model_name = args[15]
    native_img_height = int(args[16])
    native_pix_per_sec = int(args[17])
    fp_perc= float(args[18])
    tp_perc= float(args[19])
    prop_fp= float(args[20])
    prop_tp= float(args[21])
    shuf_size = float(args[22])
    stride_pix_train = int(args[23]) 
    tp_weights = float(args[24])
    view_plots = args[25]
    model_win_size = int(args[26])
    win_size_native = int(args[27])
    zero_drop = float(args[28])
    #arguments: 
    batch_size_train = int(args[30])
    batch_size = int(args[31])
    epochs = int(args[32])
    epoch_steps = int(args[33])

    only_labs = 'n'

#calculate this based on given dimensions
win_t_factor = model_win_size/win_size_native

data_augmentation = keras.Sequential(
[
    tf.keras.layers.RandomBrightness(factor=[-brightness_low,brightness_high]),
    tf.keras.layers.RandomContrast(factor=[0,contrast_high]), 
]
)
    
FGnames = []
FGdurs = []

with gzip.open(FGpath, mode="rt") as f:
    cols = next(f, None)
    cols = cols[:-1] #lose \n
    cols = cols.split(",")
    idx = cols.index("Name")
    
    for row in f:
        FGnames.append(row.split(",")[idx].replace('"', ''))
        FGdurs.append(float(row.split(",")[cols.index("SegDur")].replace('"', '')))


#import code
#code.interact(local=dict(globals(), **locals()))

usedFG = set()
uniqueFG = [x for x in FGnames if x not in usedFG and (usedFG.add(x) or True)]

#read spectrogram paths
bigfiles =[]
bigfileskey=[]
with open(spec_path + '/filepaths.csv') as f:
    next(f, None)
    for row in f:
        bigfiles.append(row.split(',')[0].replace('"', ''))
        bigfileskey.append(row.split(',')[1].replace('"', ''))
        
#total files:
bigfile = len(bigfiles)

bfinds = []
for i in range(len(uniqueFG)):
    bfinds.append([x for x, j in enumerate(bigfileskey) if j[:-1] == uniqueFG[i]])
bfinds=sum(bfinds, [])

#sort lab_files:
bigfiles = [bigfiles[i] for i in bfinds]
        
#read label paths
lab_files= []
lab_fileskey =[]
with open(label_path + '/filepaths.csv') as f:
    next(f, None)
    for row in f:
        lab_files.append(row.split(',')[0].replace('"', ''))
        lab_fileskey.append(row.split(',')[1].replace('"', ''))
        
#sort by uniqueFG
lfinds = []
for i in range(len(uniqueFG)):
    lfinds.append([x for x, j in enumerate(lab_fileskey) if j[:-1] == uniqueFG[i]])
lfinds=sum(lfinds, [])

#sort lab_files:
lab_files = [lab_files[i] for i in lfinds]

#read splits paths
split_files= []
split_fileskey =[]
with open(split_path + '/filepaths.csv') as f:
    next(f, None)
    for row in f:
        split_files.append(row.split(',')[0].replace('"', ''))
        split_fileskey.append(row.split(',')[1].replace('"', ''))

#sort by uniqueFG
sfinds = []
for i in range(len(uniqueFG)):
    sfinds.append([x for x, j in enumerate(split_fileskey) if j[:-1] == uniqueFG[i]])
sfinds=sum(sfinds, [])

#sort split_files:
split_files = [split_files[i] for i in sfinds]

#import code
#code.interact(local=dict(globals(), **locals()))
#for n in range(bigfile):

#    bigfiles.append(spec_path + '/bigfiles/bigfile' + str(n+1) + '.png')
#    lab_files.append(label_path + '/labeltensors/labeltensor' + str(n+1) + '.csv.gz')
#    split_files.append(split_path + '/splittensors/splittensor' + str(n+1) + '.csv.gz')

dataset1 = tf.data.Dataset.from_tensor_slices(bigfiles)
dataset2 = tf.data.Dataset.from_tensor_slices(lab_files)
dataset3 = tf.data.Dataset.from_tensor_slices(split_files)
dataset4 = tf.data.Dataset.from_tensor_slices(np.zeros((len(bigfiles),), dtype=int))

full_dataset = tf.data.Dataset.zip((dataset1,dataset2,dataset3,dataset4))

index = [i for i, x in enumerate(lab_files) if 'is_tp' in x]

lab_files_sub = [lab_files[i] for i in index]
bigfiles_sub = [bigfiles[i] for i in index]
split_files_sub = [split_files[i] for i in index]

#import code
#code.interact(local=dict(globals(), **locals()))

TP_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(bigfiles_sub),
                                  tf.data.Dataset.from_tensor_slices(lab_files_sub),
                                  tf.data.Dataset.from_tensor_slices(split_files_sub),
                                  tf.data.Dataset.from_tensor_slices(np.zeros((len(bigfiles),), dtype=int))))

def assessTP(y):

    lab = tf.io.read_file(y)
    lab = tf.io.decode_compressed(lab,compression_type='GZIP')
    lab = tf.strings.split(lab, sep="\n", maxsplit=-1, name=None)[:-1]
    lab = tf.strings.to_number(lab,out_type=tf.int32,name=None)

    is_tp = tf.reduce_any(lab==2)

    return is_tp

def ScanForTP(dataset):

    #load in the unshuffled dataset. input the label tensors, if there are any tps, export a one, otherwise a 0.
    #use this to subset full_dataset into tp_dataset.

    #advanced version of this might include reducing the tensor to eliminate FP sections and loading the full TP tensor into memory.

    dataset = dataset.map(lambda x,y,z,z2: (x,y,z,assessTP(y)))

    dataset = dataset.map(lambda x,y,z,z2: z2)
    
    return dataset

def MakeDataset(dataset,wh,wl,stride_pix,do_offset=False,split=None,batchsize=None,shuffle_size=None,drop_assignment=True,\
                addnoise=None,augment=False,filter_splits=True,label=None,repeat_perma=False,weights=True,is_rgb=True,zero_drop_=None):

    #this should give a random offset, each epoch!
    
    if shuffle_size!=None:
        dataset = dataset.shuffle(bigfile) #shuffle the whole thing

    if repeat_perma==True:
        dataset =dataset.repeat()

    dataset = dataset.map(lambda x,y,z,z2: (ingest(x,y,z,z2,wh,wl,stride_pix,do_offset)),num_parallel_calls = tf.data.AUTOTUNE).unbatch() #test, with offsets implemented in loop within ingest. 
    #accumulate label
    dataset = dataset.map(lambda x,y,z,z2: (accumulate_lab(x,y,z,z2)))#.cache()

    #filter
    if filter_splits==True:
        #dataset = dataset.filter(lambda x,y,z,z2: tf.reduce_all(y[:,2:3]==0)) #take records where all labels do not have ambiguity . 
        dataset = dataset.filter(lambda x,y,z,z2: tf.shape(tf.unique(tf.reshape(z,[-1])))[0]>1) #Take record where split assigment is unambiguous
    
    #reduce to single assignment
    dataset = dataset.map(lambda x,y,z,z2: (x,y,z[0,0],z2))

    #reduce based on split assignment
    if(split!=None):
        dataset = dataset.filter(lambda x,y,z,z2: z[0] == split) #1 = train, 2 = val, 3 = test

    #format labels. 
    dataset = dataset.map(lambda x,y,z,z2: (x,tf.where(tf.equal(y,1))[:,1],z,z2)) #turn this into a tf slice to also work for multi-class

    #populate weights: 
    dataset = dataset.map(lambda x,y,z,z2: (x,y,z,(z2[:,2]*tp_weights*10)+1))

    #need to eliminate this so that training will not break between epoch- this will cause different values of labels to come through.
    #if this looks like it is working at all, try to find a way so that we can adjust weights to penalize these. 
    if filter_splits==True:
        dataset = dataset.filter(lambda x,y,z,z2: tf.reduce_all(y!=2))

    if label!=None:
        dataset = dataset.filter(lambda x,y,z,z2: tf.reduce_any(y==label)) #0 or 1

    if zero_drop_!=None:
        dataset = dataset.filter(lambda x,y,z,z2: tf.reduce_any(float(y) > (tf.random.uniform(shape=[])-zero_drop_)))
        
    #shuffle
    if shuffle_size!=None and shuffle_size!='initial':
        dataset = dataset.shuffle(shuffle_size)

    #faster before converting out of grayscale
    if addnoise!=None:
       dataset = dataset.map(lambda x,y,z,z2: (add_noise(x,y,z,z2,addnoise)))  #Gaussian_val

    #convert features to expected image dims
    if is_rgb:
        dataset = dataset.map(lambda x,y,z,z2: (tf.image.grayscale_to_rgb(x),y,z,z2))

    #do augmentation on training or if not specified
    if augment==True:
        dataset = dataset.map(lambda x,y,z,z2: (data_augmentation(x),y,z,z2))

    #cast target to integer
    dataset = dataset.map(lambda x,y,z,z2: (x,tf.cast(y, tf.int32),z,z2))
    
    #drop assignment
    if drop_assignment==True and weights==True:
        dataset = dataset.map(lambda x,y,z,z2: (x,y,z2)) #no assignment yes weights
    elif drop_assignment==True: #weights false
        dataset = dataset.map(lambda x,y,z,z2: (x,y)) #no assignment no weights
    elif weights==False: #drop assign false
        dataset = dataset.map(lambda x,y,z,z2: (x,y,z)) #yes assignment no weights
        
    if batchsize != None:
        dataset = dataset.batch(batchsize)

    dataset = dataset.prefetch(tf.data.AUTOTUNE) #1
                          
    return dataset

#@tf.function
def add_noise2(ds1,ds2,addnoise_mean,addnoise_stddev):

    #this will stitch together a window with a window from another dataset and average results
    
    x,y,z2 = ds1
    
    xb,yb,z2b = ds2

    fact =  tf.cast((tf.abs(tf.random.normal([],addnoise_mean,addnoise_stddev))),tf.float32)

    #fact=0.1

    #scale the impact and weights by the random factor
    #not sure if or what modification to make to the weight. I want to make these instances important, but not hurt training
    return (x+xb*fact)/(1+1*fact),y,(z2+z2b*fact)/(1+1*fact)
    
#@tf.function
def add_noise(x,y,z,z2,ds):

    #this will stitch together a window with a window from another dataset and average results

    xb,yb,z2b = next(ds)

    fact =  tf.cast((tf.abs(tf.random.normal([],0.5,0.25))),tf.float32)

    #fact=0.1

    #scale the impact and weights by the random factor
    #not sure if or what modification to make to the weight. I want to make these instances important, but not hurt training
    return (x+xb*fact)/(1+1*fact),y,z,(z2+z2b*fact)/(1+1*fact)
    
def accumulate_lab(x,y,z,z2):

    #the manipulation looks right, but doesn't seem to be correctly accumulating- check the source data and the transformations. 

    #out_tens = tf.reshape(y,[2,-1]) #should this be depth GTdepth instead of '2'? check it... 
    out_tens = tf.reshape(y,[GT_depth,-1])

    width = tf.shape(out_tens)[1]

    out_tens = tf.math.bincount(out_tens,axis=-1,minlength=3) #always populates 0,1,2 - uk,fp,tp

    out_tens = out_tens/width #makes it the proportion

    out_tens_tp_prop = out_tens[:2]

    out_tens = tf.math.greater_equal(out_tens,[0,fp_perc,tp_perc])

    out_tens = tf.where(out_tens, 1, 0)

    out_tens = tf.reverse(out_tens,axis=[1]) #flip so that order is 2,1,0 (tp,fp,uk)

    out_tens = tf.argmax(out_tens,axis=-1) #this makes the integer order matter- prioritizes in order of correct label for tp, then fp, then uk

    out_tens = tf.one_hot(out_tens,3)

    out_tens = tf.gather(out_tens,[1,0,2],axis=1)


    return x,out_tens,z,tf.cast(out_tens_tp_prop,tf.float32)

#@tf.function
def ingest(x,y,z,z1,wh,wl,stride_pix,do_offset): #try out a predetermined offset

    if do_offset:
        offset = int(tf.random.uniform(shape=[],maxval=round(((wl/win_t_factor)-1)),dtype=tf.int32)) #offset is on the resolution of label
    else:
        offset = 0
    #if this approach appears to work, make into a general function (lot of copy and paste here)

    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)
    #some reason the images is read in reversed:
    image = tf.reverse(image,[0])
    #image_dims = image.shape.as_list()
    #image_dims = tf.shape(image)
    width = tf.cast(tf.round(tf.cast(tf.shape(image)[1],tf.float32)*win_t_factor),tf.int32)
    #import code
    #code.interact(local=dict(globals(), **locals()))

    #get rid of top border
    image=tf.slice(image,[0,0, 0], [tf.shape(image)[0]-1, tf.shape(image)[1], 1])

    #resize to model size. 
    image = tf.image.resize(image,[wh,width])
    
    #apply the offest and padding to the image
    img_offset = int(tf.round(tf.cast(offset,tf.float32)*win_t_factor))

    image = image[:,img_offset:,:]
    #image = tf.pad(image,tf.constant([[0,0], [0,img_offset],[0,0]]),"CONSTANT")
    image = tf.pad(image,[[0,0], [0,img_offset],[0,0]],"CONSTANT")

    image = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, wh, wl, 1],
           strides=[1, wh, round(stride_pix*win_t_factor), 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), (-1, wh, wl, 1))

    lab = tf.io.read_file(y)
    lab = tf.io.decode_compressed(lab,compression_type='GZIP')
    lab = tf.strings.split(lab, sep="\n", maxsplit=-1, name=None)[:-1]
    lab = tf.strings.to_number(lab,out_type=tf.int32,name=None)

    #now, try to reshape as 3d array!
    lab = tf.expand_dims(lab,-1)
    lab = tf.expand_dims(lab,-1)
    lab = tf.reshape(lab,[native_img_height,-1,GT_depth])

    lab = lab[:,offset:,:]
    lab = tf.pad(lab,[[0,0], [0,offset],[0,0]],"CONSTANT")

    lab = tf.reshape(tf.image.extract_patches(
        images=tf.expand_dims(lab, 0),
        sizes=[1, native_img_height, round(wl/win_t_factor), 1],
       strides=[1, native_img_height, stride_pix, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'), [-1, native_img_height, round(wl/win_t_factor), GT_depth])

    splt = tf.io.read_file(z)
    splt = tf.io.decode_compressed(splt,compression_type='GZIP')
    splt = tf.strings.split(splt, sep="\n", maxsplit=-1, name=None)[:-1]
    splt = tf.strings.to_number(splt,out_type=tf.int32,name=None)

    splt = tf.expand_dims(splt,-1)
    splt = tf.expand_dims(splt,-1)
    splt = tf.reshape(splt,[1,-1,1])

    #don't change the splits- close enough. Changing the splits will cause the # of examples given to training each epoch
    #to fluctuate, making this unsuitable for training.
    #V1-8: do enforce the splits change, since using an infinite dataset with # of steps
    splt = splt[:,offset:,:]
    splt = tf.pad(splt,[[0,0], [0,offset],[0,0]],"CONSTANT")

    splt = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(splt, 0),
            sizes=[1, 1, round(wl/win_t_factor), 1],
           strides=[1, 1, stride_pix, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), [-1, 1, round(wl/win_t_factor), 1])

    #assert tf.shape(splt)[0] == tf.shape(lab)[0]
    #assert tf.shape(image)[0] == tf.shape(lab)[0]
    img_shape = tf.shape(image)
    lab_shape = tf.shape(lab)
    splt_shape = tf.shape(splt)
    
    image =  tf.slice(image,[0,0,0,0],[splt_shape[0], img_shape[1], img_shape[2],1]) #take whole tensor except for last item of batch
    lab =  tf.slice(lab,[0,0,0,0],[splt_shape[0], lab_shape[1], lab_shape[2],GT_depth])
    
    return image,lab,splt,tf.zeros([splt_shape[0]])

#test2 = iter(LabelView(full_dataset))

#plt.gray()
#for f in range(1000):
    #plt.imshow(next(test2),interpolation='nearest')
    #plt.show()
    #print(next(test2))
#    values = input("press any key to continue, type ! to abort")
#    if values =='!':
#        exit()



#select keras model constructorbased on given name
if model_name == "ResNet50V2":
    model_con=tf.keras.applications.resnet_v2.ResNet50V2
    assert model_win_size == 224
elif model_name == "ResNet50":
    model_con=tf.keras.applications.resnet50.ResNet50
    assert model_win_size == 224
elif model_name == "EffecientNet" and model_win_size==224:
    model_con=tf.keras.applications.efficientnet.EfficientNetB0
elif model_name == "EffecientNet" and model_win_size==240:
    model_con=tf.keras.applications.efficientnet.EfficientNetB1
elif model_name == "EffecientNet" and model_win_size==260:
    model_con=tf.keras.applications.efficientnet.EfficientNetB2
elif model_name == "EffecientNet" and model_win_size==300:
    model_con=tf.keras.applications.efficientnet.EfficientNetB3
elif model_name == "EffecientNet" and model_win_size==380:
    model_con=tf.keras.applications.efficientnet.EfficientNetB4
elif model_name == "EffecientNet" and model_win_size==456:
    model_con=tf.keras.applications.efficientnet.EfficientNetB5
elif model_name == "EffecientNet" and model_win_size==528:
    model_con=tf.keras.applications.efficientnet.EfficientNetB6
elif model_name == "EffecientNet" and model_win_size==600:
    model_con=tf.keras.applications.efficientnet.EfficientNetB7
else:
    raise MyValidationError("Model not found")

#if stage!="train":
#    win_length = model_win_size #set this equal to model dimension
#    win_height = model_win_size




if stage=="train":


    #this means that random contast, other augmentations will be used if called during training. 
    tf.keras.backend.set_learning_phase(1)

    #calculate estimated bins in training to use for right sizing shuffle later.
    estbins =(sum(FGdurs)*native_pix_per_sec)/stride_pix_train

    print(sum(FGdurs))
    print(estbins)
    #for now, hardcode that shuffle buffer will be 1/20th the size of total dataset

    shuffle_size_all = round(estbins*shuf_size)
    #shuffle_size_all = round(estbins/len(bigfiles))

    stride_pix_default = stride_pix_train
    wh_default = model_win_size
    wl_default = model_win_size
    _split = 1

    #define datasets

    #do_shuffle = 'initial'
    do_shuffle = shuffle_size_all

    #import code
    #code.interact(local=dict(globals(), **locals()))

    #def MakeDataset(dataset,wh,wl,stride_pix,do_offset=False,split=None,batchsize=None,shuffle_size=None,drop_assignment=True,\
    #               addnoise=None,augment=False,filter_splits=True,label=None,repeat_perma=False,weights=True,is_rgb=True,zero_drop_=None):


    #this produces noise, in grayscale
    train_dataset_noise = MakeDataset(full_dataset,model_win_size,model_win_size,stride_pix_train,True,1,None,do_shuffle,True,None,False,None,0,True,True,False,zero_drop)

    #this produces signal, in grayscale
    train_dataset_signal = MakeDataset(TP_dataset,model_win_size,model_win_size,stride_pix_train,True,1,None,round(shuffle_size_all/20),\
                                       True,None,False,True,1,True,True,False)#round(shuffle_size_all/20) #iter(train_dataset_2gs)
    if do_spec_averaging=="y":
        #this averages signal with noise based on function protocol
        train_dataset_signal2 = tf.data.Dataset.zip((train_dataset_signal, train_dataset_noise)).map(lambda ds1,ds2: (add_noise2(ds1,ds2,addnoise_mean,addnoise_stddev))) #add averagethesefxn
        
        #this averages noise with noise based on function protocol
        train_dataset_noise_aug_w_noise = tf.data.Dataset.zip((train_dataset_noise, train_dataset_noise)).map(lambda ds1,ds2: (add_noise2(ds1,ds2,addnoise_mean,addnoise_stddev)))
        
        #this samples from noise datasets. 
        train_dataset_noise2= tf.data.Dataset.sample_from_datasets([train_dataset_noise, train_dataset_noise_aug_w_noise], weights=[0.75, 0.25])
        
    else:
        #this is without the spectrogram averaging
        train_dataset_noise2 = train_dataset_noise

        train_dataset_signal2 = train_dataset_signal
        
    #this samples from noise and signal datasets, and provides final augmentation.
    train_dataset = tf.data.Dataset.sample_from_datasets([train_dataset_signal2, train_dataset_noise2], weights=[prop_tp, prop_fp])\
                    .map(lambda x,y,z2: (tf.image.grayscale_to_rgb(x),y,z2))
    
    if do_aug == "y":
        train_dataset = train_dataset.map(lambda x,y,z2: (data_augmentation(x),y,z2))
        
    train_dataset = train_dataset.batch(batch_size_train)

    val_dataset = MakeDataset(full_dataset,model_win_size,model_win_size,stride_pix_train,False,2,batch_size_train,None,True,None,False,True,None,False,False) #different wh and wl avoid the random crop

elif stage=="test":
    _split = 3

    stride_pix_default = stride_pix_inf
    wh_default = model_win_size
    wl_default = model_win_size

    shuffle_size_all=1000 #hard code...

    batch_size_train = batch_size

if view_plots =='y':
    do_plot =True
else:
    do_plot =False

if do_plot:

    choice = input("view train/val? (y/n)")
    if choice!="y":
        shuffle_ = input("shuffle? (y/n)")
        if shuffle_ == 'y':
            shuffle_ = round(shuffle_size_all/20)
        else:
            shuffle_ = None
        if _split ==1:
            include_val = input("include val? (y/n)")
            if include_val:
                _split = None
        addnoise_ = input("Add noise? (y/n)")
        if addnoise_=="y":
            addnoise_="high_reduce"
        else:
            addnoise_ = None
        offset_ = input("Add offset? (y/n)")
        if offset_=='y':
            offset_=True
        else:
            offset_=False
        do_augment = input("augment? (y/n)")
        if do_augment=="y":
            do_augment=True
        else:
            do_augment=False
        do_filter = input("filter? (y/n)")
        if do_filter=="y":
            do_filter=True
        only_pos = input("only positives? (y/n)")
        if only_pos =='y':
            only_pos=1
        else:
            only_pos=None
        custom_stride = input("custom stride? (y/n)")
        if custom_stride =="y":
            custom_stride = int(input("enter integer value > 0 :"))
        else:
            custom_stride = stride_pix_default

    #def MakeDataset(dataset,wh,wl,stride_pix,do_offset=False,split=None,batchsize=None,shuffle_size=None,drop_assignment=True,addnoise=None,
    #            augment=False,filter_splits=True,label=None,repeat_perma=False,weights=True):

    #MakeDataset(dataset,wh,wl,stride_pix,do_offset=False,split=None,batchsize=None,shuffle_size=None,drop_assignment=True,\
    #            addnoise=None,augment=False,filter_splits=True,label=None,repeat_perma=False,weights=True,is_rgb=True,zero_drop_=None):

        iter_obj = iter(MakeDataset(full_dataset,wh_default,wl_default,custom_stride,offset_,_split,batch_size_train,shuffle_,False,addnoise_,do_augment,do_filter,only_pos))

        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        def seespec(obj):
            spectrogram_batch, label_batch, assn_batch, weight_batch = obj
            plots_rows = 4
            plots_cols = 5
            fig, axes = plt.subplots(plots_rows, plots_cols, figsize=(12, 9))
            indices = list(range(spectrogram_batch.shape[0]))

            #random.shuffle(indices)
            for i in range(plots_rows * plots_cols):
              batch_index = indices[i]
              class_index = label_batch[batch_index].numpy()
              assn_index = assn_batch[batch_index].numpy()
              weight_index = weight_batch[batch_index].numpy()
              ax = axes[i // plots_cols,i % plots_cols]
              ax.pcolormesh(spectrogram_batch[batch_index, :,:,1].numpy())
              ax.set_title(str(class_index) + ":" +  str(assn_index) + ":" + str(weight_index))
              ax.get_xaxis().set_ticks([])
              ax.get_yaxis().set_ticks([])
            fig.show()
    else:

        choice2 = input("view train? (y/n)")
        if choice2=="y":
            print("chose train:")
            iter_obj = iter(train_dataset)
        else:
            iter_obj = iter(val_dataset)
        

        #import code
        #code.interact(local=dict(globals(), **locals()))

        def seespec(obj):
            spectrogram_batch, label_batch, weight_batch = obj
            plots_rows = 4
            plots_cols = 5
            fig, axes = plt.subplots(plots_rows, plots_cols, figsize=(12, 9))
            indices = list(range(spectrogram_batch.shape[0]))

            #random.shuffle(indices)
            for i in range(plots_rows * plots_cols):
              batch_index = indices[i]
              class_index = label_batch[batch_index].numpy()
              #assn_index = assn_batch[batch_index].numpy()
              weight_index = weight_batch[batch_index].numpy()
              ax = axes[i // plots_cols,i % plots_cols]
              ax.pcolormesh(spectrogram_batch[batch_index, :,:,1].numpy())
              ax.set_title(str(class_index)+ ":" + str(weight_index))
              ax.get_xaxis().set_ticks([])
              ax.get_yaxis().set_ticks([])
            fig.show()

    #next(iter_obj)
    #import code
    #code.interact(local=dict(globals(), **locals()))
    

    for f in range(1000):
        seespec(next(iter_obj))
        values = input("press any key to continue, type ! to abort")
        if values =='!':
            exit()

    import code
    code.interact(local=dict(globals(), **locals()))

if stage=="train":

    #determine if model exists or needs to be compiled.
    #if log exists, assume model has been created an one epoch has been run.

    logpath = resultpath + "/model_history_log.csv"

    if not os.path.isfile(resultpath + "/model.keras"):

        if os.path.isfile(logpath):
            os.remove(logpath)
    
        #define and compile model

        #select correct loss function for multi or single class:

        if GT_depth >1:
            loss_fxn = "categorical_crossentropy"
            loss_metric ="categorical_accuracy"
        else:
            loss_fxn = "binary_crossentropy"
            loss_metric = "binary_accuracy" #accuracy bugged for some reason...?
            #loss_metric = "accuracy" 

        def KerasModel(constructor=model_con):
          return tf.keras.Sequential([
            tf.keras.Input(shape=(model_win_size, model_win_size, 3)),
            #tf.keras.layers.RandomBrightness(factor=[-brightness_low,brightness_high]), #turn these on or off depending if limiation is in CPU or GPU
            #tf.keras.layers.RandomContrast(factor=[0,contrast_high]), 
            constructor(include_top=False, weights=None, pooling="max"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(GT_depth),
            tf.keras.layers.Activation("sigmoid"),
          ])

        model = KerasModel(model_con)

        #import code
        #code.interact(local=dict(globals(), **locals()))

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #experimenting with these. Ideally, weights could be provided per label, and correspond with the % of label inclusion so more 'on target' windows are higher weighted!
        #if this is working well, make this into a parameter. 
        #weights = [1,tp_weights]

        model.compile(
            optimizer=opt,
            loss=loss_fxn,
            #loss_weights=weights,
            #weighted_metrics = "accuracy",
            metrics=[
                loss_metric,
                tf.keras.metrics.AUC(name="rocauc"),
                tf.keras.metrics.AUC(curve="pr", name="ap"),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ],
        )

    #def MakeDataset(dataset,wh,wl,stride_pix,do_offset=False,split=None,batchsize=None,shuffle_size=None,drop_assignment=True,addnoise=None,
    #            augment=False,filter_splits=True,label=None,repeat_perma=False,weights=True):
    

    #
    #    os.remove(logpath)

        cur_epoch = 0

        #loaded=False

    else:

        #determine current epoch by counting # of previous epochs
        log = pd.read_csv(logpath)

        cur_epoch = len(log.index)

        model = keras.models.load_model(resultpath + "/model.keras")

        print("On epoch " + str(cur_epoch) + " of " + str(epochs)) 

        #loaded=True
        
    csv_logger = CSVLogger(logpath, append=True)
    class TimingCallback(keras.callbacks.Callback):
        def __init__(self, cur_epoch, end_epoch,logs={}):
            self.logs=[]
            self.cur_epoch=cur_epoch
            self.end_epoch=end_epoch
        def on_epoch_begin(self, epoch, logs={}):
            self.starttime = time()

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(time()-self.starttime)

            #print(self.cur_epoch)
            #print(epoch)

            if self.cur_epoch == self.end_epoch:
                #if the epoch= the total , communicate to outer loop to stop next time. 
                with open(resultpath + '/stop.txt', 'w') as f:
                    f.write('stop')

                #also, stop the current training
                self.model.stop_training = True  

            self.cur_epoch = self.cur_epoch +1

            if len(self.logs)>1:
                #if self.logs[-1] / self.logs[0] > 0.5: #just for testing... 
                if self.logs[-1] / self.logs[0] > 1: #once exceeds original runtime (which is always slower than next few epochs)
                    #, cease training

                #for a shuffle buffer fill time of 12 minutes, starts to become faster to restart compared to running
                #another epoch
                #if self.logs[-1] - self.logs[0] > 720:

                    #import code
                    #code.interact(local=dict(globals(), **locals()))
                    self.model.stop_training = True                    

    tc = TimingCallback(cur_epoch,epochs-1) #-1 is cause python counts from 0

    #iter_obj  =iter(MakeDataset(full_dataset,win_height,win_length,split=1,batchsize=batch_size_train))
    #next(iter_obj)

    try:
        model.fit(
                    train_dataset,
                    steps_per_epoch=epoch_steps,
                    validation_data=val_dataset, #temporarily disable to make it faster
                    epochs=epochs,
                    callbacks=[csv_logger,tc]
                    )
    except KeyboardInterrupt:
        with open(resultpath + '/stop.txt', 'w') as f:
                    f.write('stop')
        #pass

    model.save(resultpath + "/model.keras")

    
if stage=="test":

    tf.keras.backend.set_learning_phase(0) #set to inference phase

    model = keras.models.load_model(modelpath)


    #set phase to test
    #so random augments will still happen

    #def MakeDataset(dataset,wh,wl,stride_pix,do_offset=False,split=None,batchsize=None,shuffle_size=None,drop_assignment=True,\
    #            addnoise=None,augment=False,filter_splits=True,label=None,repeat_perma=False,weights=True,is_rgb=True,zero_drop_=None):
    ds= MakeDataset(full_dataset,model_win_size,model_win_size,stride_pix_inf,False,None,batch_size,None,False,None,False,False,None,False,False) #experimental- augment in val

    #scores1 = []
    #for i in range(5):
    labels=[]
    if only_labs == 'y':
        
        #counter = 1
        for _,labs,_ in ds:
            labels.append(labs.numpy())
            #print(counter)
            #counter +=1 
            #if counter == 3:
            #    break

        #call it scores, but it is labels
        scores = np.vstack(labels)
            
        #import code
        #code.interact(local=dict(globals(), **locals()))
        
    else:
        scores = model.predict(ds)
    #scores1.append(scores)

    #take max of scores
    #scores2 = np.stack(scores1)
    #scores3 = np.amax(scores2,0)

    #import code
    #code.interact(local=dict(globals(), **locals()))

    with gzip.open(resultpath + '/scores.csv.gz', 'wt', newline='') as f:   
        write = csv.writer(f)
        write.writerows(scores)

#import code
#code.interact(local=dict(globals(), **locals()))
#write scores


