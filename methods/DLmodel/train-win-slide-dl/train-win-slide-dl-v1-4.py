
#v1-1: do away with 'offset', replace with what the concept evolved into, which is stride. 
#v1-2: update parameters to get rid of stuff that doesn't matter, add stuff that does. 
#v1-3: read filepaths instead of assume them 

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
import csv
import gzip


#from tensorflow import keras
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

#import sklearn
print(tf.version.VERSION)

sys.path.append(os.getcwd())
from user.misc import arg_loader

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

    GT_depth = args[8].count(",")+1
    model_name = args[9]
    native_img_height = int(args[10])
    native_pix_per_sec = int(args[11])
    fp_perc= float(args[12])
    tp_perc= float(args[13])
    stride_pix = int(args[14]) 
    view_plots = args[15]
    win_f_factor= float(args[16])
    win_t_factor= float(args[17])
    win_height = int(args[18])
    win_length = int(args[19])
    #arguments: 
    batch_size = int(args[21])
    
    data_augmentation = "NULL"

else:
    brightness_high=float(args[8])
    brightness_low=float(args[9])
    contrast_high =float(args[10])
    contrast_low =float(args[11])
    GT_depth = args[12].count(",")+1
    learning_rate = float(args[13])
    model_name = args[14]
    native_img_height = int(args[15])
    native_pix_per_sec = int(args[16])
    fp_perc= float(args[17])
    tp_perc= float(args[18])
    stride_pix = int(args[19]) 
    stride_pix_inf = int(args[20]) 
    view_plots = args[21]
    win_f_factor= float(args[22])
    win_t_factor= float(args[23])
    win_height = int(args[24])
    win_length = int(args[25])
    #arguments: 
    batch_size_train = int(args[27])
    batch_size = int(args[28])
    epochs = int(args[29])

    data_augmentation = keras.Sequential(
    [
        tf.keras.layers.RandomBrightness(factor=[-brightness_low,brightness_high]),
        tf.keras.layers.RandomContrast(factor=[contrast_low,contrast_high]), 
    ]
    )
    
FGnames = []

with gzip.open(FGpath, mode="rt") as f:
    cols = next(f, None)
    cols = cols[:-1] #lose \n
    cols = cols.split(",")
    idx = cols.index("Name")
    
    for row in f:
        FGnames.append(row.split(",")[idx].replace('"', ''))
        
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

full_dataset = tf.data.Dataset.zip((dataset1,dataset2,dataset3))

def LabelView(dataset):

    dataset = dataset.map(lambda x,y,z: (ingest(x,y,z))).unbatch() #test, with offsets implemented in loop within ingest. 
    #accumulate label
    dataset = dataset.map(lambda x,y,z: (x,accumulate_lab(y),z))

    #filter
    dataset = dataset.filter(lambda x,y,z: tf.reduce_all(y[:,2:3]==0)) #take records where all labels do not have ambiguity . 
    dataset = dataset.filter(lambda x,y,z: tf.shape(tf.unique(tf.reshape(z,[-1])))[0]>1) #Take record where split assigment is unambiguous

    dataset=dataset.map(lambda x,y,z: y)
    
    return(dataset)

def accumulate_lab2(y):

    #the manipulation looks right, but doesn't seem to be correctly accumulating- check the source data and the transformations. 

    out_tens = tf.reshape(y,[GT_depth,-1])

    width = tf.shape(out_tens)[1]

    out_tens = tf.math.bincount(out_tens,axis=-1,minlength=3) #always populates 0,1,2 - uk,fp,tp

    out_tens = out_tens/width #makes it the proportion

    out_tens = tf.math.greater_equal(out_tens,[0,fp_perc,tp_perc])

    out_tens = tf.where(out_tens, 1, 0)

    out_tens = tf.reverse(out_tens,axis=[1]) #flip so that order is 2,1,0 (tp,fp,uk)

    out_tens = tf.argmax(out_tens,axis=-1) #this makes the integer order matter- prioritizes in order of correct label for tp, then fp, then uk

    out_tens = tf.one_hot(out_tens,3)

    return(out_tens)

def MakeDataset(dataset,wh,wl,split=None,batchsize=20,do_shuffle=True,drop_assignment=True,augment=False,filter_splits=True):

    if split==3:
        do_shuffle = False

    if do_shuffle==True:
        dataset = dataset.shuffle(bigfile) #shuffle the whole thing

    #ingest_data
    #dataset = dataset.map(lambda x,y,z: ingest(x,y,z)).unbatch() #this will extract slices, and associate with assignment/labels (form x,y,z: data,label,assignment)

    dataset = dataset.map(lambda x,y,z: (ingest(x,y,z,wh,wl))).unbatch() #test, with offsets implemented in loop within ingest. 
    #accumulate label
    dataset = dataset.map(lambda x,y,z: (x,accumulate_lab(y),z))

    #filter
    if filter_splits==True:
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
    if split==1 or augment==True:
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

    out_tens = tf.math.greater_equal(out_tens,[0,fp_perc,tp_perc])

    out_tens = tf.where(out_tens, 1, 0)

    out_tens = tf.reverse(out_tens,axis=[1]) #flip so that order is 2,1,0 (tp,fp,uk)

    out_tens = tf.argmax(out_tens,axis=-1) #this makes the integer order matter- prioritizes in order of correct label for tp, then fp, then uk

    out_tens = tf.one_hot(out_tens,3)

    return(out_tens)

#@tf.function
def ingest(x,y,z,wh,wl): #try out a predetermined offset

    #if this approach appears to work, make into a general function (lot of copy and paste here)

    #this should give a random offset, each epoch!
    #offset = tf.random.uniform(shape=[],maxval=(wl_lab-1),dtype=tf.int32) #offset is on the resolution of label

    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)
    #some reason the images is read in reversed:
    image = tf.reverse(image,[0])
    #image_dims = image.shape.as_list()
    #image_dims = tf.shape(image)
    width = tf.cast(tf.round(tf.cast(tf.shape(image)[1],tf.float32)*win_t_factor),tf.int32)
    #import code
    #code.interact(local=dict(globals(), **locals()))
    image = tf.image.resize(image,[round(native_img_height*win_f_factor),width])

    image = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, wh, wl, 1],
           strides=[1, wh, round(stride_pix*win_t_factor), 1],
            rates=[1, 1, 1, 1],
            padding='SAME'), (-1, wh, wl, 1))

    lab = tf.io.read_file(y)
    lab = tf.io.decode_compressed(lab,compression_type='GZIP')
    lab = tf.strings.split(lab, sep="\n", maxsplit=-1, name=None)[:-1]
    lab = tf.strings.to_number(lab,out_type=tf.int32,name=None)

    #now, try to reshape as 3d array!
    lab = tf.expand_dims(lab,-1)
    lab = tf.expand_dims(lab,-1)
    lab = tf.reshape(lab,[native_img_height,-1,GT_depth])

    lab = tf.reshape(tf.image.extract_patches(
        images=tf.expand_dims(lab, 0),
        sizes=[1, native_img_height, round(wl/win_t_factor), 1],
       strides=[1, native_img_height, stride_pix, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'), [-1, native_img_height, round(wl/win_t_factor), GT_depth])

    splt = tf.io.read_file(z)
    splt = tf.io.decode_compressed(splt,compression_type='GZIP')
    splt = tf.strings.split(splt, sep="\n", maxsplit=-1, name=None)[:-1]
    splt = tf.strings.to_number(splt,out_type=tf.int32,name=None)

    splt = tf.expand_dims(splt,-1)
    splt = tf.expand_dims(splt,-1)
    splt = tf.reshape(splt,[1,-1,1])

    splt = tf.reshape(tf.image.extract_patches(
            images=tf.expand_dims(splt, 0),
            sizes=[1, 1, round(wl/win_t_factor), 1],
           strides=[1, 1, stride_pix, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'), [-1, 1, round(wl/win_t_factor), 1])

    
    return image,lab,splt

#test2 = iter(LabelView(full_dataset))

#plt.gray()
#for f in range(1000):
    #plt.imshow(next(test2),interpolation='nearest')
    #plt.show()
    #print(next(test2))
#    values = input("press any key to continue, type ! to abort")
#    if values =='!':
#        exit()
#import code
#code.interact(local=dict(globals(), **locals()))


#select keras model constructorbased on given name
if model_name == "ResNet50V2":
    model_con=tf.keras.applications.resnet_v2.ResNet50V2
    model_win_size = 224
elif model_name == "ResNet50":
    model_con=tf.keras.applications.resnet50.ResNet50
    model_win_size = 224
elif model_name == "EffecientNetB0":
    model_con=tf.keras.applications.efficientnet.EfficientNetB0
    model_win_size = 224
elif model_name == "EffecientNetB1":
    model_con=tf.keras.applications.efficientnet.EfficientNetB1
    model_win_size = 240
elif model_name == "EffecientNetB2":
    model_con=tf.keras.applications.efficientnet.EfficientNetB2
    model_win_size = 260
elif model_name == "EffecientNetB3":
    model_con=tf.keras.applications.efficientnet.EfficientNetB3
    model_win_size = 300
elif model_name == "EffecientNetB4":
    model_con=tf.keras.applications.efficientnet.EfficientNetB4
    model_win_size = 380
elif model_name == "EffecientNetB5":
    model_con=tf.keras.applications.efficientnet.EfficientNetB5
    model_win_size = 456
elif model_name == "EffecientNetB6":
    model_con=tf.keras.applications.efficientnet.EfficientNetB6
    model_win_size = 528
elif model_name == "EffecientNetB7":
    model_con=tf.keras.applications.efficientnet.EfficientNetB7
    model_win_size = 600
    #for this, calculate particular one based on model input size. 
else:
    raise MyValidationError("Model not found")

#if stage!="train":
#    win_length = model_win_size #set this equal to model dimension
#    win_height = model_win_size
    
if stage=="train":
    _split = 1
elif stage=="test":
    _split = 3

if view_plots =='y':
    do_plot =True
else:
    do_plot =False

if do_plot:
    shuffle = input("shuffle? (y/n)")
    if _split ==1:
        include_val = input("include val? (y/n)")
        if include_val:
            _split = None
    iter_obj = iter(MakeDataset(full_dataset,win_height,win_length,_split,20,(shuffle=='y'),False,True))
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
        values = input("press any key to continue, type ! to abort")
        if values =='!':
            exit()

    import code
    code.interact(local=dict(globals(), **locals()))

assert model_win_size <= win_length
assert model_win_size <= win_height

if stage=="train":

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
        #tf.keras.Input(shape=(win_height, win_length, 3)),
        tf.keras.Input(shape=(None, None, 3)),
        tf.keras.layers.RandomCrop(height = model_win_size, width = model_win_size), #in case it is differently sized
        #tf.keras.layers.RandomBrightness(factor=[-1,1]),
        #tf.keras.layers.RandomContrast(factor=0.8), 
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
    weights = [1,1]

    model.compile(
        optimizer=opt,
        loss=loss_fxn,
        loss_weights=weights,
        metrics=[
            loss_metric,
            tf.keras.metrics.AUC(name="rocauc"),
            tf.keras.metrics.AUC(curve="pr", name="ap"),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ],
    )

    train_dataset = MakeDataset(full_dataset,win_height,win_length,split=1,batchsize=batch_size_train)
    val_dataset = MakeDataset(full_dataset,model_win_size,model_win_size,split=2,batchsize=batch_size_train) #different wh and wl avoid the random crop

    logpath = resultpath + "/model_history_log.csv"

    if os.path.isfile(logpath):
        os.remove(logpath)
        
    csv_logger = CSVLogger(logpath, append=True)

    try:
      model.fit(
          train_dataset,
          validation_data=val_dataset,
          epochs=epochs,
          callbacks=[csv_logger]
      )
    except KeyboardInterrupt:
      pass

    model.save(resultpath + "/model.keras")
elif stage == 'test': #maybe same behavior for all test/inference? 

    #load model

    model = keras.models.load_model(modelpath)


#run prediction on full FG data. 
#scores = []
#import code
#code.interact(local=dict(globals(), **locals()))

ds= MakeDataset(full_dataset,wh=model_win_size,wl=model_win_size,split=None,batchsize=batch_size,augment=False,do_shuffle=False,filter_splits=False)
scores = model.predict(ds)

#scores = np.vstack(scores)


with gzip.open(resultpath + '/scores.csv.gz', 'wt', newline='') as f:   
    write = csv.writer(f)
    write.writerows(scores)

#import code
#code.interact(local=dict(globals(), **locals()))
#write scores


