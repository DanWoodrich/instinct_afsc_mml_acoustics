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


import keras
from keras.callbacks import CSVLogger
#import sklearn

#######import params###########

#args="first C:/Apps/INSTINCT/Cache/394448/527039 C:/Apps/INSTINCT/Cache/394448/628717/947435/DETx.csv.gz C:/Apps/INSTINCT/Cache/394448/628717/947435/909093/summary.png 0.80 train-simple-dl-v1-0"
#args=args.split()

args=sys.argv

FGpath = args[1]
SPECPATH=args[2]
datapath=args[3]
resultpath=args[4]
EPOCH = int(args[5])
model_name = args[6]
model_input_size = 224 #this shuold be a parameter
#should be a parameter
train_val_split = float(args[7])
#small_window= args[4] #I might not even need to pass this one- I can get the crop width based on the height of the spectrogram images

print("hello world!")

FG = pd.read_csv(FGpath,compression='gzip')

metadata = pd.read_csv(datapath,compression='gzip')

replicas = len([ f.path for f in os.scandir(SPECPATH) if f.is_dir() ])



metadata["Assignment"] = " "

metadata = metadata.sample(frac=1).reset_index(drop=True)

cutoff = int(len(metadata)*train_val_split)

metadata.loc[:cutoff,"Assignment"] = "Train"
metadata.loc[cutoff:,"Assignment"] = "Validation"

GTids = metadata.GTid.unique()
GTids = GTids[np.logical_not(np.isnan(GTids))]

random.shuffle(GTids)

train_ids = GTids[:int(train_val_split * len(GTids))]
val_ids = GTids[len(train_ids):]

metadata.loc[metadata.GTid.isin(train_ids), 'Assignment'] = "Train"
metadata.loc[metadata.GTid.isin(val_ids), 'Assignment'] = "Validation"

bigfile = metadata.DiffTime.unique()

bigfile_dict_assign = {}
bigfile_dict_labs = {}
bigfiles = []

for n in range(len(bigfile)):

    #may want to split data into set of only the difftime
    meta_dt = metadata.loc[metadata['DiffTime'] == bigfile[n]]
    FG_dt = FG.loc[FG['DiffTime'] == bigfile[n]]
    
    

    meta_dt['StartFile'] = pd.Categorical(meta_dt['StartFile'], FG_dt.FileName.unique()) #preserves order

    meta_dt = meta_dt.sort_values(['StartFile', 'StartTime'], ascending=[True, True])




    #for values to reflect continuous:
    #reduce FG to start time and duration per file
    #make a column of cumsum
    #for each starttime and endtime (assess independently), subtract FG.starttime and add FG.cumsum

    FG_red = FG_dt.groupby('FileName', as_index=False).agg({'SegStart':'min', 'SegDur':'sum'})
    FG_red["mod"] = FG_red.SegDur.cumsum()-FG_red.SegDur[0]-FG_red.SegStart[0]


    FG_red_end = FG_red.copy()
    FG_red["StartFile"]=FG_red["FileName"]
    FG_red["mod_start"]=FG_red["mod"]
    FG_red=FG_red.drop(columns=['FileName', 'mod','SegStart','SegDur'])

    FG_red_end["EndFile"]=FG_red_end["FileName"]
    FG_red_end["mod_end"]=FG_red_end["mod"]
    FG_red_end=FG_red_end.drop(columns=['FileName', 'mod','SegStart','SegDur'])

    meta_dt =meta_dt.merge(FG_red)
    meta_dt =meta_dt.merge(FG_red_end)

    meta_dt["StartTime"]= meta_dt["StartTime"] + meta_dt["mod_start"]
    meta_dt["EndTime"]= meta_dt["EndTime"] + meta_dt["mod_end"]
    #meta_dt.to_csv("C:/Apps/INSTINCT/lib/user/methods/TrainModel_DL/train-simple-dl/test.csv")

    meta_dt["label"] =  meta_dt["label"].map({'FP':0,'TP':1})
    meta_dt["Assignment"] = meta_dt["Assignment"].map({'Train':1,'Validation':0})

    meta_dt = meta_dt.sort_values('StartTime') #make sure it is ordered by StartTime.
    
    for i in range(replicas):
        file = SPECPATH + '/replica_' + str(i+1) + '/bigfile' + str(bigfile[n]) + '.png'

        #save all files in character array
        bigfiles.append(file)

        #save dict of file name and assignment to train/test
        bigfile_dict_assign.update({file: meta_dt["Assignment"].to_numpy()})

        #save dict of file name and label
        bigfile_dict_labs.update({file: meta_dt["label"].to_numpy()})


#load in one file to determine reduction multiple.

image_string = tf.io.read_file(SPECPATH + '/replica_' + str(i+1) + '/bigfile' + str(bigfile[n]) + '.png')
image = tf.image.decode_png(image_string, channels=1)

wl_secs = max(meta_dt.EndTime-meta_dt.StartTime)
maxlen = image.shape[1]

wl = round((wl_secs*maxlen)/max(meta_dt.EndTime))

#import code
#code.interact(local=dict(globals(), **locals()))

step = wl/wl_secs

reduce_multiple = model_input_size/image.shape[0]

step_mod = int(round(step * reduce_multiple))

wl_mod = int(round(wl*reduce_multiple))

imagedim_4check=int(round(image.shape[1]*reduce_multiple))
#check that steps divide cleanly with image length. 
assert (imagedim_4check % step_mod) ==0

#and that the image length is a clean multiple of the large tile. 
assert (imagedim_4check % wl_mod) ==0

def drop_assn(x,y,z):

    return x,z


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

def parse_fxn(x,y,mis,step,wl_len):

    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)

    image = tf.image.resize(image,[mis,tf.shape(image)[1]],preserve_aspect_ratio=True)

    #image = tf.image.grayscale_to_rgb(image)

    endloop = tf.shape(y)[0]

    result = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, mis, wl_len, 1],
            strides=[1, mis, step, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), (endloop, mis, wl_len, 1))

    #result = tf.tile(

    #result = tf.reshape(result,[-1,mis,wl_len,3])
        
    return result

assignments = []
labels = []

for n in range(len(bigfiles)):
    assignments.append(bigfile_dict_assign[bigfiles[n]])
    labels.append(bigfile_dict_labs[bigfiles[n]])

dataset1 = tf.data.Dataset.from_tensor_slices(bigfiles)
dataset2 = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(assignments))
dataset3 = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(labels))

full_dataset = tf.data.Dataset.zip((dataset1,dataset2,dataset3))

#train_iter = iter(MakeDataset(full_dataset,step_mod,wl_mod,model_input_size,isTrain=True,batchsize = 1000))
#next(train_iter)
#test_iter = iter(MakeDataset(full_dataset,step_mod,wl_mod,model_input_size,isTrain=False,batchsize = 1000))
#all_iter = iter(MakeDataset(full_dataset,step_mod,wl_mod,model_input_size,getAllLabs=True,batchsize = 1000))

#import code
#code.interact(local=dict(globals(), **locals()))

print(model_input_size)

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




