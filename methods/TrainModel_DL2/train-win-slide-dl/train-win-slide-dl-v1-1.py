
#v1-1: do away with 'offset', replace with what the concept evolved into, which is stride. 


#still experimental- this vers will attempt to save the
#file name and associated array for test/train and label
#as a dict, and reference that during mapping function.
#
#try to avoid tfrecord

#import pandas as pd
import tensorflow as tf
import sys
import numpy as np
#from matplotlib.image import imread
import os
import pathlib
import matplotlib.pyplot as plt

#from tensorflow import keras
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

#import sklearn
print(tf.version.VERSION)

#import code
#code.interact(local=dict(globals(), **locals()))

#######import params###########

#args="C:/Apps/INSTINCT/Cache/394448/FileGroupFormat.csv.gz C:/Apps/INSTINCT/Cache/394448/516798 C:/Apps/INSTINCT/Cache/394448/756783/391491 C:/Apps/INSTINCT/Cache/394448/516798/399814 C:/Apps/INSTINCT/Cache/394448/516798/399814/249442 15 125 HB.s.p.2,HB.s.p.4 4 EffecientNet 300 15 300 300 train-win-slide-dl-v1-0"
#args=args.split()

args=sys.argv

FGpath = args[1] #may not end up needing this. 
spec_path=args[2]
label_path=args[3]
split_path=args[4]
resultpath=args[5]
brightness=args[6]
contrast =args[7]
GT_depth = args[8].count(",")+1
lab_reduce_fact = int(args[9])
model_name = args[10]
spec_img_height = int(args[11])
spec_pix_per_sec = int(args[12])
stride_pix = int(args[13]) #based on label steps
train_test_split = float(args[14]) #used to calculate steps in epoch
train_val_split = float(args[15])
view_plots = args[16]
win_height = int(args[17])
win_length = int(args[18])
#arguments: 
batch_size = int(args[20])
epochs = int(args[21])


#determine which to run based on whether input is factor or 'None'
brightness=float(args[6])
contrast =float(args[7])

data_augmentation = keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(factor=[-brightness,brightness]),
        tf.keras.layers.RandomContrast(factor=contrast), 
    ]
)


tp_prop = 0.1
fp_prop = 0.9 #doesn't have to = 1... especially not when not i_neg

#FG = pd.read_csv(FGpath,compression='gzip')

f = open(spec_path + '/receipt.txt', "r")
bigfile = int(f.read())

#intead of using pandas to find this, just read from bigfiles export file. 
#bigfile = FG.DiffTime.unique()

bigfiles = []
lab_files= []
split_files =[]

for n in range(bigfile):

    bigfiles.append(spec_path + '/bigfiles/bigfile' + str(n+1) + '.png')
    lab_files.append(label_path + '/labeltensors/labeltensor' + str(n+1) + '.csv.gz')
    split_files.append(split_path + '/splittensors/splittensor' + str(n+1) + '.csv.gz')

dataset1 = tf.data.Dataset.from_tensor_slices(bigfiles)
dataset2 = tf.data.Dataset.from_tensor_slices(lab_files)
dataset3 = tf.data.Dataset.from_tensor_slices(split_files)

wh_lab =int(win_height/lab_reduce_fact)
wl_lab = int(win_length/lab_reduce_fact)

#dataset4 = tf.data.Dataset.from_tensor_slices(offsets)
#dataset just maps bigfiles, to label_tensor, to split tensor
full_dataset = tf.data.Dataset.zip((dataset1,dataset2,dataset3))

def MakeDataset(dataset,split=None,batchsize=20,do_shuffle=True,drop_assignment=True):

    if split==3:
        do_shuffle = False

    if do_shuffle==True:
        dataset = dataset.shuffle(bigfile) #shuffle the whole thing

    #ingest_data
    #dataset = dataset.map(lambda x,y,z: ingest(x,y,z)).unbatch() #this will extract slices, and associate with assignment/labels (form x,y,z: data,label,assignment)

    dataset = dataset.map(lambda x,y,z: (ingest(x,y,z))).unbatch() #test, with offsets implemented in loop within ingest. 
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


    #labels to one-hot
    dataset = dataset.map(lambda x,y,z: (x,y[:,0],z)) #only take 1st column, flip to horizontal

    #experimental- see if making the label a string lets me use accuracy. 
    #dataset = dataset.map(lambda x,y: (x,tf.as_string(tf.cast(y,tf.int32)))) #it didn't work!

    if do_shuffle==True:
        dataset = dataset.shuffle(1000) #10000 #big number because why not?

    #convert to 0/1 (not needed for all models)
    #if model_name != "EffecientNet";
    #    dataset = dataset.map(lambda x,y: (x/255,y))

    #convert features to expected image dims
    dataset = dataset.map(lambda x,y,z: (tf.image.grayscale_to_rgb(x),y,z))

    #do augmentation on training or if not specified
    if split==1 or split==None:
    dataset = dataset.map(lambda x,y,z: (data_augmentation(x),y,z))

    #drop assignment
    if drop_assignment==True:
        dataset = dataset.map(lambda x,y,z: (x,y))
    
    dataset = dataset.batch(batchsize)

    dataset = dataset.prefetch(tf.data.AUTOTUNE) #1
                          
    return dataset

def accumulate_lab(y):

    #the manipulation looks right, but doesn't seem to be correctly accumulating- check the source data and the transformations. 

    #out_tens = tf.reshape(y,[2,-1]) #should this be depth GTdepth instead of '2'? check it... 
    out_tens = tf.reshape(y,[GT_depth,-1])

    width = tf.shape(out_tens)[1]

    out_tens = tf.math.bincount(out_tens,axis=-1,minlength=3) #always populates 0,1,2 - uk,fp,tp

    out_tens = out_tens/width #makes it the proportion

    out_tens = tf.math.greater_equal(out_tens,[0,fp_prop,tp_prop])

    out_tens = tf.where(out_tens, 1, 0)

    out_tens = tf.reverse(out_tens,axis=[1]) #flip so that order is 2,1,0 (tp,fp,uk)

    out_tens = tf.argmax(out_tens,axis=-1) #this makes the integer order matter- prioritizes in order of correct label for tp, then fp, then uk

    out_tens = tf.one_hot(out_tens,3)

    return(out_tens)

@tf.function
def ingest(x,y,z): #try out a predetermined offset

    #if this approach appears to work, make into a general function (lot of copy and paste here)

    #this should give a random offset, each epoch!
    #offset = tf.random.uniform(shape=[],maxval=(wl_lab-1),dtype=tf.int32) #offset is on the resolution of label

    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)
    #some reason the images is read in reversed:
    image = tf.reverse(image,[0]) #not yet tested... 

    image = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, win_height, win_length, 1],
           strides=[1, win_height, stride_pix*lab_reduce_fact, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), (-1, win_height, win_length, 1))

    lab = tf.io.read_file(y)
    lab = tf.io.decode_compressed(lab,compression_type='GZIP')
    lab = tf.strings.split(lab, sep="\n", maxsplit=-1, name=None)[:-1]
    lab = tf.strings.to_number(lab,out_type=tf.int32,name=None)

    #now, try to reshape as 3d array!
    lab = tf.expand_dims(lab,-1)
    lab = tf.expand_dims(lab,-1)
    lab = tf.reshape(lab,[wh_lab,-1,GT_depth]) #don't do height yet, since putting it through patches.

    lab = tf.reshape(tf.image.extract_patches(
        images=tf.expand_dims(lab, 0),
        sizes=[1, wh_lab, wl_lab, 1],
       strides=[1, wh_lab, stride_pix, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'), [-1, wh_lab, wl_lab, GT_depth])

    splt = tf.io.read_file(z)
    splt = tf.io.decode_compressed(splt,compression_type='GZIP')
    splt = tf.strings.split(splt, sep="\n", maxsplit=-1, name=None)[:-1]
    splt = tf.strings.to_number(splt,out_type=tf.int32,name=None)

    splt = tf.expand_dims(splt,-1)
    splt = tf.expand_dims(splt,-1)
    splt = tf.reshape(splt,[1,-1,1])

    splt = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(splt, 0),
            sizes=[1, 1, win_length, 1],
           strides=[1, 1, stride_pix*lab_reduce_fact, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'), [-1, 1, win_length, 1])

    
    return image,lab,splt

#dataset class testing


#test2 = iter(MakeDatasetTest(full_dataset,2))
#test3 = iter(MakeDataset(full_dataset,2))

#next(test2)
#next(test3)

#import random
if view_plots =='y':
    do_plot =True
else:
    do_plot =False

if do_plot:
    iter_obj = iter(MakeDataset(full_dataset,None,20,False,False))
    def seespec(obj):
        spectrogram_batch, label_batch, assn_batch = obj
        plots_rows = 4
        plots_cols = 5
        fig, axes = plt.subplots(plots_rows, plots_cols, figsize=(12, 9))
        indices = list(range(spectrogram_batch.shape[0]))

        #random.shuffle(indices)
        for i in range(plots_rows * plots_cols):
          batch_index = indices[i]
          class_index = label_batch[batch_index].numpy()
          assn_index = assn_batch[batch_index].numpy()
          ax = axes[i // plots_cols,i % plots_cols]
          ax.pcolormesh(spectrogram_batch[batch_index, :,:,1].numpy())
          ax.set_title(str(class_index) + ":" +  str(assn_index))
          ax.get_xaxis().set_ticks([])
          ax.get_yaxis().set_ticks([])
        fig.show()

    for f in range(1000):
        seespec(next(iter_obj))
        val = input("press any key to continue, type ! to abort")
        if val =='!':
            exit()

    import code
    code.interact(local=dict(globals(), **locals()))


#select keras model constructorbased on given name

#for effecientnet, determine correct model name based on the inputs.
if model_name == "ResNet50V2":
    model_con=tf.keras.applications.resnet_v2.ResNet50V2
    model_win_size = 224
elif model_name == "ResNet50":
    model_con=tf.keras.applications.resnet50.ResNet50
    model_win_size = 224
elif "EffecientNet" in model_name:

    #for this, calculate particular one based on model input size. 
    if win_height >= 600:
        model_con=tf.keras.applications.efficientnet.EfficientNetB7
        model_win_size = 600
    elif win_height >= 528:
        model_con=tf.keras.applications.efficientnet.EfficientNetB6
        model_win_size = 528
    elif win_height >= 456:
        model_con=tf.keras.applications.efficientnet.EfficientNetB5
        model_win_size = 456
    elif win_height >= 380:
        model_con=tf.keras.applications.efficientnet.EfficientNetB4
        model_win_size = 380
    elif win_height >= 300:
        model_con=tf.keras.applications.efficientnet.EfficientNetB3
        model_win_size = 300
    elif win_height >= 260:
        model_con=tf.keras.applications.efficientnet.EfficientNetB2
        model_win_size = 260
    elif win_height >= 240:
        model_con=tf.keras.applications.efficientnet.EfficientNetB1
        model_win_size = 240
    elif win_height >= 224:
        model_con=tf.keras.applications.efficientnet.EfficientNetB0
        model_win_size = 224
    else:
        raise MyValidationError("input size too small for EffecientNet")  
else:
    raise MyValidationError("Model not found")

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
    tf.keras.Input(shape=(win_height, win_length, 3)),
    tf.keras.layers.RandomCrop(height = model_win_size, width = model_win_size), #in case it is differently sized
    #tf.keras.layers.RandomBrightness(factor=[-1,1]),
    #tf.keras.layers.RandomContrast(factor=0.8), 
    constructor(include_top=False, weights=None, pooling="max"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(GT_depth),
    tf.keras.layers.Activation("sigmoid"),
  ])


#def KerasApplicationsModel(constructor=tf.keras.applications.resnet_v2.ResNet50V2):
#  return tf.keras.Sequential([
#    tf.keras.Input(shape=(model_input_size, wl_mod, 3)),
#    tf.keras.layers.RandomCrop(height = model_input_size, width = model_input_size),
    #tf.keras.layers.Lambda(fake_image, name="fake_image"),
#    constructor(include_top=False, weights=None, pooling="max"),
#    tf.keras.layers.ReLU(),
#    tf.keras.layers.Dense(128, activation="relu"),
#    tf.keras.layers.Dense(1),
#    tf.keras.layers.Activation("sigmoid"),
#  ])

#Matt Harvey proposed smaller CNN. 
#def SmallCNNModel():
#  return tf.keras.Sequential([
#    tf.keras.Input(shape=(model_input_size, win_length, 3)),
#    tf.keras.layers.RandomCrop(height = model_input_size, width = model_input_size),
#    tf.keras.layers.Conv2D(16, 7, use_bias=False, activation="relu"),
#    tf.keras.layers.MaxPooling2D(2, 2),
#    tf.keras.layers.Conv2D(64, 3, use_bias=False, activation="relu"),
#    tf.keras.layers.MaxPooling2D(2, 2),
#    tf.keras.layers.Conv2D(64, 3, use_bias=False, activation="relu"), 
#    tf.keras.layers.MaxPooling2D(2, 2),
#    tf.keras.layers.Conv2D(64, 3, use_bias=False, activation="relu"), 
#    tf.keras.layers.GlobalMaxPool2D(),
#    tf.keras.layers.Dense(128, activation="relu"),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(1),
#    tf.keras.layers.Activation("sigmoid"),
#  ])

model = KerasModel(model_con)

#import code
#code.interact(local=dict(globals(), **locals()))

model.compile(
    optimizer="adam",
    # binary_crossentropy and sigmoid are for independent classes, which is the
    # proper assumption for species detection. For spoken digits, it's more
    # proper to use categorical_cross_entropy and softmax, but this example
    # won't.
    loss=loss_fxn,
    metrics=[
        loss_metric,
        tf.keras.metrics.AUC(name="rocauc"),
        tf.keras.metrics.AUC(curve="pr", name="ap"),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ],
)

#test2 = iter(MakeDatasetTest(full_dataset))
#test3 = iter(MakeDataset(full_dataset,1))

#out = 

#plt.gray()
#plt.imshow(next(test2),interpolation='nearest')

#import code
#code.interact(local=dict(globals(), **locals()))
#next(test3)
#


#might think about a try catch here- on GPU OOM error, iteratively decrease batch size to maximize it.

train_dataset = MakeDataset(full_dataset,split=1,batchsize=batch_size)
val_dataset = MakeDataset(full_dataset,split=2,batchsize=batch_size)

logpath = resultpath + "/model_history_log.csv"

if os.path.isfile(logpath):
    os.remove(logpath)
    
csv_logger = CSVLogger(logpath, append=True)

try:
  model.fit(
      train_dataset,
      validation_data=val_dataset,
      #steps_per_epoch = 100, #int(tot_steps//batch_size), #prevent model training from running out of data which it doesn't like
      #validation_steps=225,
      epochs=epochs,
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




