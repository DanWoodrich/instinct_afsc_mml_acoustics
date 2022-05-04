#MethodID="train-simple-dl-v1-0"

import pandas as pd
import tensorflow as tf
import sys
import numpy as np
import random
#from matplotlib.image import imread
import glob
import os
from PIL import Image


def WriteRecord(image,label,replica):
    #read file: 
    #sample_rate, audio = wavfile.read(os.path.join(WAV_DIR, filename))

    return tf.train.Example(features=tf.train.Features(feature={
        "image":tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[image])),
        #float_list=tf.train.FloatList(value= image)),
        #_float_feature(image),
        "label":tf.train.Feature(
        int64_list=tf.train.Int64List(value=[label])),
        #_int64_feature(label),
        "replica": tf.train.Feature(
        int64_list=tf.train.Int64List(value=[replica]))
        }))

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


#calculate time relative to start of file.
dontdo = True
#1: reorder metadata according to FG

if dontdo == False:
    #think it will be easier to follow Matt guide if I make the loops external, like so:
    with tf.io.TFRecordWriter("D:/test/train.tfrecords") as writer1, tf.io.TFRecordWriter("D:/test/test.tfrecords") as writer2:    
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

            meta_dt["bin_label"] =  meta_dt["label"].map({'FP':0,'TP':1})
            meta_dt["tfwrite"] = meta_dt["Assignment"].map({'Train':'train.tfrecords','Validation':'test.tfrecords'}) #this can be deleted...
            
            for i in range(replicas):

                print(str(n) + str(i))

                image_string = tf.io.read_file(SPECPATH + '/replica_' + str(i+1) + '/bigfile' + str(bigfile[n]) + '.png')

                image_decoded = tf.image.decode_png(image_string, channels=1)
                #image = tf.cast(image_decoded, tf.float32)
                image = image_decoded

                maxlen = image.shape[1]
                
                #this will fail if wl is > file. For that, I should rework above to function on the entire df, and then have wl calcuated via max of entire df. 
                wl_secs=max(metadata.EndTime-metadata.StartTime)

                wl = round((wl_secs*maxlen)/max(meta_dt.EndTime))
                step = wl/wl_secs

                for p in range(len(meta_dt)):

                    #print(p)

                    start = round(step*p)

                    #catch in case of rounding errors:
                    while (start+wl) > maxlen:
                        start=start-1

                    
                    imageseg = tf.slice(image, [0,start,0],[image.shape[0],wl,1], name=None)

                    reduce_multiple = model_input_size/imageseg.shape[0] #224 is resnet 50 size

                    dim1 = int(round(imageseg.shape[1]*reduce_multiple))
                    imageseg = tf.image.resize(imageseg,[model_input_size,dim1],preserve_aspect_ratio=True)

                    imageseg =tf.io.serialize_tensor(imageseg).numpy()

                    #import code
                    #code.interact(local=dict(globals(), **locals()))

                    if meta_dt["Assignment"][p] =="Train":
                        writer1.write(WriteRecord(imageseg,meta_dt["bin_label"][p],(i+1)).SerializeToString())
                    elif meta_dt["Assignment"][p] =="Validation":
                        writer2.write(WriteRecord(imageseg,meta_dt["bin_label"][p],(i+1)).SerializeToString())


                #tf.io.TFRecordWriter(meta_dt["tfwrite"][p]).write(WriteRecord(imageseg,meta_dt["bin_label"][p],(i+1)).SerializeToString())
else:
    dim1 = 240

#currently the data are waaaay to big!
#strategies to try
#1: keeping png encoding (shouldn't help alot...)
#2: reducing image size before saving
#3: convert to jpeg encoding?

#the whole concept here is terrible, even if it works... should work on a pipeline which creates a dataset with file names and the in map parse function
#generates slices and labels.

#also, prior to this shuold be it's own process if it is kept (it shouldn't be). 
        
      

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




def MakeDataset(tfrecord_filename, batch_size=128, repeat=None, shuffle=True):
  dataset = tf.data.TFRecordDataset(tfrecord_filename)
  if shuffle:
    dataset = dataset.shuffle(1024)
  if repeat:
    dataset = dataset.repeat(repeat)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(parse_fn)
  dataset = dataset.prefetch(1)  # important to allow parsing to happen concurrently with GPU work
  return dataset
    #here, split image into component bins

import code
code.interact(local=dict(globals(), **locals()))

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
