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
import glob
import os
from PIL import Image

#######import params###########

#args="first C:/Apps/INSTINCT/Cache/394448/527039 C:/Apps/INSTINCT/Cache/394448/628717/947435/DETx.csv.gz C:/Apps/INSTINCT/Cache/394448/628717/947435/909093/summary.png 0.80 train-simple-dl-v1-0"
#args=args.split()

args=sys.argv

FGpath = args[1]
SPECPATH=args[2]
datapath=args[3]
resultpath=args[4]
model_input_size = 224 #this shuold be a parameter
train_val_split = float(args[5])
#small_window= args[4] #I might not even need to pass this one- I can get the crop width based on the height of the spectrogram images

print("hello world!")

FG = pd.read_csv(FGpath,compression='gzip')

metadata = pd.read_csv(datapath,compression='gzip')

replicas = len([ f.path for f in os.scandir(SPECPATH) if f.is_dir() ])



#img = imread('abc.tiff')
#pseudo:
#assign each bin as train or val category


#I am going to reimagine this structure. Instead of doing this per difftime, I am going to just do it once for the full dataset.
#loop through difftime will happen in a loop after. 


#for i in range(len(metadata.DiffTime.unique())):

    #dt = metadata.DiffTime.unique()[i]
    #determine how many GTids there are in the FG, if any. If there are any, split them by id into train val splits

    #meta_dt = metadata.loc[metadata['DiffTime'] == dt]

    #by default, assign all bins to train or val, depending on the train_val_split parameter

    #meta_dt["Assignment"] = " "

    #meta_dt = meta_dt.sample(frac=1).reset_index(drop=True)

    #cutoff = int(len(meta_dt)*train_val_split)

    #meta_dt.loc[:cutoff,"Assignment"] = "Train"
    #meta_dt.loc[cutoff:,"Assignment"] = "Validation"

    #now do grouped random sort of the GTids. 
    
    #GTids = meta_dt.GTid.unique()

    #GTids = GTids[np.logical_not(np.isnan(GTids))]

    #random.shuffle(GTids)

    #train_ids = GTids[:int(train_val_split * len(GTids))]
    #val_ids = GTids[len(train_ids):]

    #meta_dt.loc[meta_dt.GTid.isin(train_ids), 'Assignment'] = "Train"
    #meta_dt.loc[meta_dt.GTid.isin(val_ids), 'Assignment'] = "Validation"

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

step = wl/wl_secs

reduce_multiple = model_input_size/image.shape[0]

step_mod = step * reduce_multiple

#dim1 = int(round(image.shape[1]*reduce_multiple))

#pseudeo for consuming images:

#tf dataset from file (bigfiles)
#map function
#parse function which

#resizes image - 


#uses bigfile_dict_assign

def MakeDataset(files,assn_dict,lab_dict,steplen,mis,isTrain=True,batchsize = 1):

    #Will produce either train or test set. 
    if isTrain:
        subsetval = 0
    else:
        subsetval = 1

    #for loop through the files in order to assemble tensors for assignment and label
    assignments = []
    labels = []
    
    for n in range(len(files)):
        assignments.append(assn_dict[files[n]])
        labels.append(lab_dict[files[n]])

    dataset1 = tf.data.Dataset.from_tensor_slices(files)
    dataset2 = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(assignments))
    dataset3 = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(labels))

    dataset = tf.data.Dataset.zip((dataset1,dataset2,dataset3))

    #datacompare = tfds.load('mnist', split='test', as_supervised=True)

    #shuffle/repeat?

    #for x,y,z in dataset:
    #    print(x,y,z)

    #dataset.batch(batchsize)

    dataset = dataset.map(lambda x,y,z: (parse_fxn(x,y,mis,steplen),y,z)) #this will extract slices, and associate with assignment/labels (form x,y,z: data,label,assignment)

    #I need to better understand how map, unbatch, filter, and batch play together... do this in my test environment next!

    dataset = dataset.unbatch()
    return dataset

    #temporarily comment these out
    # #this will convert the batched labels from image extract into standalone records
    #dataset = dataset.filter(lambda x,y,z: tf.reduce_all(tf.not_equal(z, [subsetval]))) #make sure to check this behavior
    #control batch size
    #dataset = dataset.batch(batchsize)

    #return(dataset)

#parse_fxn = lambda x, y , y: (tf.reshape(
#        tf.image.extract_patches(
#            images=tf.expand_dims(x, 0),
#            sizes=[1, 14, 14, 1],
#            strides=[1, 14, 14, 1],
#            rates=[1, 1, 1, 1],
#            padding='VALID'), (4, 14, 14, 1)), [y,3,4,y])


def parse_fxn(x,y,mis,float_num):

    image = tf.io.read_file(x)
    image = tf.image.decode_png(image, channels=1)

    image = tf.image.resize(image,[mis,tf.shape(image)[1]],preserve_aspect_ratio=True)

    endloop = tf.shape(y)[0]

    h =tf.zeros([1,mis, mis,1])

    i = tf.constant(0)

    i_out,result = tf.while_loop(lambda i,h: tfloop_condition(i,endloop),
                                 lambda i,h: (tfloop_body(i,image,float_num,h)),
                                 [i,h],
                                 shape_invariants=[i.get_shape(),tf.TensorShape([None, mis,mis,1])])

    result = result[1:,:,:,:]

    return result

def start_check(a,img_size):
    while tf.math.greater(a, img_size): 
        a -= 1
    return a

def tfloop_condition(i,endloop):
   return i < endloop 

def tfloop_body(i,b,c,prev):
    
    start = tf.round(tf.cast(i,tf.float32)*c)
    start = start_check(start + c,tf.cast(tf.shape(b)[1],tf.float32))

    start = tf.cast(start,tf.int32)

    image_slice = tf.slice(b, [0,start,0],[tf.shape(b)[0],tf.shape(b)[0],1], name=None)

    image_slice = tf.reshape(image_slice,[1,tf.shape(b)[0],tf.shape(b)[0],1])

    prev = tf.concat([prev,image_slice],0)

    i+=1
            
    return i,prev
    #for p in range(tf.shape(assn_array)[0]):

    #   print(p)

    #    one = tf.cast(steplen, dtype=tf.float32)
    #    two = tf.cast(size_modify, dtype=tf.float32)
    #    three = tf.cast(p, dtype=tf.float32) #I think this is the issue- don't worry about it yet, since need to refactor in tf while loop 

        
        #start = tf.Variable(steplen * size_modify*p, dtype=tf.float64)
    #    start = tf.constant(one * two*three, dtype=tf.float32)
        #start = tf.constant(1.5*1.32, tf.float32 ) 
    #    start = tf.cast(tf.math.round(start),tf.int32) #it is having an issue here! I don't know why!
     #   start = 16
        #TypeError: Expected int32, got 15.995536959553696 of type 'float' instead

     #   while (start+wl) > maxlen: #catch if rounding exceeds bounds and then round down
     #       start=start-1

      #  image_slice = tf.slice(image, [0,start,0],[mis,mis,1], name=None)

       # outlist.append(image_slice)

    #out_tensor = tf.stack(outlist) #this pattern does not work -for loop not valid in this fxn. 

    #return(image,test)
        #tf.concat([t1, t2], 0)
    
    #images1 = tf.image.extract_patches(images=tf.expand_dims(image, 0),
    #                                             sizes=[1, mis, mis, 1],
    #                                             strides=[1, mis, steplen2, 1],
    #                                             rates=[1, 1, 1, 1],
    #                                             padding='VALID')

    #images = tf.reshape(tf.image.extract_patches(images=tf.expand_dims(image, 0),
    #                                             sizes=[1, mis, mis, 1],
    #                                             strides=[1, mis, steplen, 1],
    #                                             rates=[1, 1, 1, 1],
    #                                             padding='VALID'),(len(assn_array),mis,mis,1))

    #,assn_array,lab_array
    #return out_tensor



#test = parse_fxn(file,assn_dict,lab_dict,steplen,model_input_size,reduce_multiple)

#dataset = tf.data.Dataset.from_tensor_slices(bigfiles).map(lambda x: parse_fxn(x,bigfile_dict_assign,bigfile_dict_labs,steplen,model_input_size,reduce_multiple))
#dataset = tf.data.Dataset.from_tensor_slices(bigfiles).map(lambda x,y,z: parse_fxn(x,y,z,steplen,model_input_size,reduce_multiple),y,z)

files = bigfiles[0:2]
#assn_dict =bigfile_dict_assign[file]
#lab_dict = bigfile_dict_labs[file]

#get this working with batch
#image_batch, assn_batch, label_batch= next(iter(MakeDataset(bigfiles,bigfile_dict_assign,bigfile_dict_labs,steplen,model_input_size)))



#currently doesn't seem like it's returning multiple reads from MakeDataset. 
test = next(iter(MakeDataset(files,bigfile_dict_assign,bigfile_dict_labs,step_mod,model_input_size)))

import code
code.interact(local=dict(globals(), **locals()))

#image_batch, assn_batch, label_batch= next(iter(MakeDataset(files,bigfile_dict_assign,bigfile_dict_labs,step_mod,model_input_size)))


plots_rows = 4
plots_cols = 5
fig, axes = plt.subplots(plots_rows, plots_cols, figsize=(12, 9))
indices = list(range(image_batch.shape[0]))
#random.shuffle(indices)
for i in range(plots_rows * plots_cols):
  batch_index = indices[i]
  class_index = label_batch[batch_index].numpy()
  ax = axes[i // plots_cols,i % plots_cols]
  ax.imshow(image_batch[batch_index])
  ax.set_title(str(class_index))
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
fig.show()


def parse_fn(record):
    example = tf.io.parse_single_example( #changed this from parse_example to parse_single_example
      record,
      {
          "image": tf.io.FixedLenFeature([], tf.string),
          "label": tf.io.FixedLenFeature([], tf.int64)
      })
    
    img = tf.io.decode_image(example["image"])
    #img = tf.cast(example["image"],tf.float32)
    img = tf.stack(img,[model_input_size, dim1, 1])#careful here... dim1 in other 'process'
    #img
    return example["image"],example["label"]




#def MakeDataset(tfrecord_filename, batch_size=128, repeat=None, shuffle=True):
#  dataset = tf.data.TFRecordDataset(tfrecord_filename)
#  if shuffle:
#    dataset = dataset.shuffle(1024)
#  if repeat:
#    dataset = dataset.repeat(repeat)
#  dataset = dataset.batch(batch_size)
#  dataset = dataset.map(parse_fn)
#  dataset = dataset.prefetch(1)  # important to allow parsing to happen concurrently with GPU work
#  return dataset
    #here, split image into component bins



image_batch, label_batch = next(iter(MakeDataset("D:/test/train.tfrecords", shuffle=False)))



class_index_batch = tf.math.argmax(label_batch, axis=1)
plots_rows = 4
plots_cols = 5
fig, axes = plt.subplots(plots_rows, plots_cols, figsize=(12, 9))
indices = list(range(image_batch.shape[0]))
random.shuffle(indices)
for i in range(plots_rows * plots_cols):
  batch_index = indices[i]
  class_index = class_index_batch[batch_index].numpy()
  ax = axes[i // plots_cols,i % plots_cols]
  ax.pcolormesh(spectrogram_batch[batch_index, :].numpy().T)
  ax.set_title(str(class_index))
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
fig.show()
    #write to tfrecords

    #tf.io.TFRecordWriter("train.tfrecords").write(MakeExample(meta_dt[meta_dt['Assignment'] == 'Train']).SerializeToString())
    #tf.io.TFRecordWriter("test.tfrecords").write(MakeExample(meta_dt[meta_dt['Assignment'] == 'Validation']).SerializeToString())

#        makeexample()










    

    #meta_dt.to_csv("C:/Apps/INSTINCT/lib/user/methods/TrainModel_DL/train-simple-dl/test.csv")

    
    #alright, each bin has been assigned to train or val. Now, create record writer function so I can write the train and val datasets.



    
#loop through images
#use bins (note these need to retain difftime to associate to image!) to assign labels and bin extent to images.
#loop through each image and write label and power values to tfrecord. 

#read
