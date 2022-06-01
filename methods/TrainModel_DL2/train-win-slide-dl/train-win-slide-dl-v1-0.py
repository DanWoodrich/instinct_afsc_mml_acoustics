#MethodID="train-simple-dl-v1-1"

#still experimental- this vers will attempt to save the
#file name and associated array for test/train and label
#as a dict, and reference that during mapping function.
#
#try to avoid tfrecord

import pandas as pd
import tensorflow as tf
import sys
import numpy as np
import random
#from matplotlib.image import imread
import os
import pathlib


#import keras
#from keras.callbacks import CSVLogger
#import sklearn

#######import params###########

#args="first C:/Apps/INSTINCT/Cache/394448/527039 C:/Apps/INSTINCT/Cache/394448/628717/947435/DETx.csv.gz C:/Apps/INSTINCT/Cache/394448/628717/947435/909093/summary.png 0.80 train-simple-dl-v1-0"
#args=args.split()

args=sys.argv

FGpath = args[1]
spec_path=args[2]
label_path=args[3]
split_path=args[4]
resultpath=args[5]
EPOCH = int(args[6])
GT_depth = args[7].count(",")+1
lab_reduce_fact = int(args[8])
model_name = args[9]
spec_img_height = int(args[10])
spec_pix_per_sec = args[11]
win_height = int(args[12])
win_length = int(args[13])

tp_prop = 0.1
fp_prop = 0.9 #doesn't have to = 1... especially not when not i_neg

FG = pd.read_csv(FGpath,compression='gzip')

bigfile = FG.DiffTime.unique()

bigfiles = []
lab_files= []
split_files =[]

for n in range(len(bigfile)):

    bigfiles.append(spec_path + '/bigfiles/bigfile' + str(n+1) + '.png')
    lab_files.append(label_path + '/labeltensors/labeltensor' + str(n+1) + '.csv.gz')
    split_files.append(split_path + '/splittensors/splittensor' + str(n+1) + '.csv')

dataset1 = tf.data.Dataset.from_tensor_slices(bigfiles)
dataset2 = tf.data.Dataset.from_tensor_slices(lab_files)
dataset3 = tf.data.Dataset.from_tensor_slices(split_files)

#dataset just maps bigfiles, to label_tensor, to split tensor
full_dataset = tf.data.Dataset.zip((dataset1,dataset2,dataset3))

def MakeDataset(dataset,split=None,batchsize=20):

    dataset = dataset.shuffle(20) #how to figure out best shuffle batch size? 

    #ingest_data
    dataset = dataset.map(lambda x,y,z: ingest(x,y,z)).unbatch() #this will extract slices, and associate with assignment/labels (form x,y,z: data,label,assignment)

    #accumulate label
    dataset = dataset.map(lambda x,y,z: (x,accumulate_lab(y),z))

    #filter
    dataset = dataset.filter(lambda x,y,z: tf.reduce_all(y[:,2:3]==0)) #take records where all labels do not have ambiguity . 
    dataset = dataset.filter(lambda x,y,z: tf.shape(tf.unique(tf.reshape(z,[-1])))[0]>1) #Take record where split assigment is unambiguous

    #reduce to single assignment
    dataset = dataset.map(lambda x,y,z: (x,y,z[0,0]))

    #reduce based on split assignment
    if(split!=None):
        dataset = dataset.filter(lambda x,y,z: z[0] == split) #1 = train, 2 = val, 3 = test

    #drop assignment
    dataset = dataset.map(lambda x,y,z: (x,y))
    
    #labels to one-hot
    dataset = dataset.map(lambda x,y: (x,y[:,0])) #only take 1st column, flip to horizontal

    dataset = dataset.shuffle(150) #big number because why not?

    dataset = dataset.map(lambda x,y: (tf.image.grayscale_to_rgb(x/255),y)) #divide color vals by 255... may or may not need to...
        
    dataset = dataset.batch(batchsize)

    dataset = dataset.prefetch(1)
                          
    return dataset

def accumulate_lab(y):

    #the manipulation looks right, but doesn't seem to be correctly accumulating- check the source data and the transformations. 

    out_tens = tf.reshape(y,[2,-1])

    width = out_tens.get_shape().as_list()[1]

    out_tens = tf.math.bincount(out_tens,axis=-1,minlength=3) #always populates 0,1,2 - tp,fp,uk

    out_tens = out_tens/width #makes it the proportion

    out_tens = tf.math.greater_equal(out_tens,[tp_prop,fp_prop,0])

    out_tens = tf.where(out_tens, 1, 0)

    out_tens = tf.argmax(out_tens,axis=-1) #this makes the integer order matter- prioritizes in order of correct label for tp, then fp, then uk

    out_tens = tf.one_hot(out_tens,3)

    return(out_tens)

def ingest(x,y,z):

    #if this approach appears to work, make into a general function (lot of copy and paste here)

    wh_lab =int(win_height/lab_reduce_fact)
    wl_lab = int(win_length/lab_reduce_fact)

    #this should give a random offset, each epoch!
    offset = tf.random.uniform(shape=[],maxval=(wl_lab-1),dtype=tf.int32) #offset is on the resolution of label

    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)

    #resample by offset    
    image = image[:,(offset*lab_reduce_fact):,:]

    image = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, win_height, win_length, 1],
           strides=[1, win_height, win_length, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), (-1, win_height, win_length, 1))

    #import code
    #code.interact(local=dict(globals(), **locals()))

    lab = tf.io.read_file(y)
    lab = tf.io.decode_compressed(lab,compression_type='GZIP')
    lab = tf.strings.split(lab, sep="\n", maxsplit=-1, name=None)[:-1]
    lab = tf.strings.to_number(lab,out_type=tf.int32,name=None)

    #now, try to reshape as 3d array!
    lab = tf.expand_dims(lab,-1)
    lab = tf.expand_dims(lab,-1)
    lab = tf.reshape(lab,[wh_lab,-1,GT_depth]) #don't do height yet, since putting it through patches.

    lab = lab[:,offset:,:]
    #need to do: splice the width by

    lab = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(lab, 0),
            sizes=[1, wh_lab, wl_lab, 1],
           strides=[1, wh_lab, wl_lab, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), [-1, wh_lab, wl_lab, GT_depth])

    splt = tf.io.read_file(z)
    splt = tf.strings.split(splt, sep="\r\n", maxsplit=-1, name=None)[:-1]
    splt = tf.strings.to_number(splt,out_type=tf.int32,name=None)

    splt = tf.expand_dims(splt,-1)
    splt = tf.expand_dims(splt,-1)
    splt = tf.reshape(splt,[1,-1,1])

    splt = splt[:,(offset*lab_reduce_fact):,:]

    splt = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(splt, 0),
            sizes=[1, 1, win_length, 1],
           strides=[1, 1, win_length, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), [-1, 1, win_length, 1])
        
    return image,lab,splt

test2 = iter(MakeDataset(full_dataset))
test3 = iter(MakeDataset(full_dataset,1))

next(test2)
next(test3)

import code
code.interact(local=dict(globals(), **locals()))

def KerasApplicationsModel(constructor=tf.keras.applications.resnet_v2.ResNet50V2):
  return tf.keras.Sequential([
    tf.keras.Input(shape=(model_input_size, wl_mod, 3)),
    tf.keras.layers.RandomCrop(height = model_input_size, width = model_input_size),
    #tf.keras.layers.Lambda(fake_image, name="fake_image"),
    constructor(include_top=False, weights=None, pooling="max"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Activation("sigmoid"),
  ])

#Matt Harvey proposed smaller CNN. 
def SmallCNNModel():
  return tf.keras.Sequential([
    tf.keras.Input(shape=(model_input_size, wl_mod, 3)),
    tf.keras.layers.RandomCrop(height = model_input_size, width = model_input_size),
    tf.keras.layers.Conv2D(16, 7, use_bias=False, activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, use_bias=False, activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, use_bias=False, activation="relu"), 
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, use_bias=False, activation="relu"), 
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Activation("sigmoid"),
  ])

if model_name == "ResNet50":
    model = KerasApplicationsModel(tf.keras.applications.resnet_v2.ResNet50V2)
elif model_name =="SmallCNN":
    model = SmallCNNModel()
else:
    raise MyValidationError("Model not found")

#model = KerasApplicationsModel(tf.keras.applications.resnet_v2.ResNet50V2)

#import code
#code.interact(local=dict(globals(), **locals()))

model.compile(
    optimizer="adam",
    # binary_crossentropy and sigmoid are for independent classes, which is the
    # proper assumption for species detection. For spoken digits, it's more
    # proper to use categorical_cross_entropy and softmax, but this example
    # won't.
    loss="binary_crossentropy", #binary_crossentropy
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="rocauc"),
        tf.keras.metrics.AUC(curve="pr", name="ap"),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ],
)

#import code
#code.interact(local=dict(globals(), **locals()))

#might think about a try catch here- on GPU OOM error, iteratively decrease batch size to maximize it.


train_dataset = MakeDataset(full_dataset,step_mod,wl_mod,model_input_size,isTrain=True,batchsize = 256)
test_dataset = MakeDataset(full_dataset,step_mod,wl_mod,model_input_size,isTrain=False,batchsize = 256)

logpath = resultpath + "/model_history_log.csv"

if os.path.isfile(logpath):
    os.remove(logpath)
    
csv_logger = CSVLogger(logpath, append=True)

try:
  model.fit(
      train_dataset,
      validation_data=test_dataset,
      epochs=EPOCH,
      callbacks=[csv_logger]
  )
except KeyboardInterrupt:
  pass

#import code
#code.interact(local=dict(globals(), **locals()))

#two outputs:

#np.save(resultpath + "/history.npy",model.history)
model.save(resultpath + "/model.keras")


#working well. Now, we can move on to training. 

#next(train_iter)[0].shape
#next(test_iter)[0].shape
#next(all_iter)[0].shape




