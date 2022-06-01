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
lab_reduce_fact = float(args[7])
model_name = args[8]
spec_img_height = args[9]
spec_pix_per_sec = args[10]
win_height = args[11]
win_length = args[12]

FG = pd.read_csv(FGpath,compression='gzip')

bigfile = FG.DiffTime.unique()

bigfiles = []
lab_files= []
split_files =[]

for n in range(len(bigfile)):

    bigfiles.append(spec_path + '/bigfiles/bigfile' + str(n+1) + '.png')
    lab_files.append(label_path + '/labeltensors/labeltensor' + str(n+1) + '.csv')
    split_files.append(split_path + '/splittensors/splittensor' + str(n+1) + '.csv')

dataset1 = tf.data.Dataset.from_tensor_slices(bigfiles)
dataset2 = tf.data.Dataset.from_tensor_slices(lab_files)
dataset3 = tf.data.Dataset.from_tensor_slices(split_files)

full_dataset = tf.data.Dataset.zip((dataset1,dataset2,dataset3))



#dataset just maps bigfiles, to label_tensor, to split tensor

def drop_assn(x,y,z):

    return x,z

def TestMD(dataset):

    dataset = dataset.map(lambda x,y,z: (parse_fxn(x,y),z)) #this will extract slices, and associate with assignment/labels (form x,y,z: data,label,assignment)

    #dataset = dataset.batch(10)
                          
    return dataset

def MakeDataset(dataset,steplen,wl_len,mis,isTrain=True,getAllLabs = False,batchsize = 20):

    #Will produce either train or test set. 
    if isTrain:
        subsetval = 1 #tf.constant([0])
    else:
        subsetval = 0 #tf.constant([1])

    #shuffle order of sound files
    dataset = dataset.shuffle(500)
    

    dataset = dataset.map(lambda x,y,z: (parse_fxn(x,y,mis,steplen,wl_len),y,z)).unbatch() #this will extract slices, and associate with assignment/labels (form x,y,z: data,label,assignment)

    if not getAllLabs:
        
        dataset = dataset.filter(lambda x,y,z: y == subsetval)
        #drop assignment feature
        dataset = dataset.map(lambda x,y,z: drop_assn(x,y,z))


    #might need another map function here to increase dimension size from grayscale to rgb for resnet model. 
    
    dataset = dataset.shuffle(1000) #big number because why not?

    dataset = dataset.map(lambda x,y: (tf.image.grayscale_to_rgb(x/255),y)) #divide color vals by 255... may or may not need to...
        
    dataset= dataset.batch(batchsize)
    
    #I need to better understand how map, unbatch, filter, and batch play together... do this in my test environment next!
    dataset = dataset.prefetch(1)
    
    return dataset

def parse_fxn(x,y):

    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)

    #import code
    #code.interact(local=dict(globals(), **locals()))

    lab = tf.io.read_file(y)
    lab = tf.strings.split(lab, sep="\r\n", maxsplit=-1, name=None)[:-1]
    lab = tf.strings.to_number(lab,out_type=tf.int32,name=None)

    #now, try to reshape as 3d array!
    lab = tf.reshape(lab,[,,,])
    
    #lab = tf.TextLineReader(skip_header_lines=1)
    #lablines = lab.split('\n')[1:-1]
    #lab = tf.io.decode_csv(lab,record_defaults=[tf.constant([],dtype=tf.int32)])

    #splt = tf.io.read_file(y)
    #splt = tf.io.decode_csv(splt,record_defaults=[tf.constant([],dtype=tf.int32)])
    
    #splt = tf.io.decode_csv(z,record_defaults=[tf.constant([],dtype=tf.int32)])

    #result = tf.reshape(tf.image.extract_patches(
    #        images=tf.expand_dims(image, 0),
    #        sizes=[1, mis, wl_len, 1],
    #       strides=[1, mis, step, 1],
    #        rates=[1, 1, 1, 1],
    #        padding='VALID'), (endloop, mis, wl_len, 1))

    #result = tf.tile(

    #result = tf.reshape(result,[-1,mis,wl_len,3])
        
    return image,lab#,splt



#test = TestMD(full_dataset)



test2 = iter(TestMD(full_dataset))

next(test2)
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




