#MethodID="train-simple-dl-v1-0"

import pandas as pd
import tensorflow as tf
import sys
import numpy as np
import random
from matplotlib.image import imread


#######import params###########

#args="C:/Apps/INSTINCT/Cache/394448/527039/receipt.txt C:/Apps/INSTINCT/Cache/394448/628717/901278/DETx.csv.gz C:/Apps/INSTINCT/Cache/394448/628717/901278/960971/summary.png 0.80 train-simple-dl-v1-0"
#args=args.split()

args=sys.argv

specpath=args[1]
datapath=args[2]
resultpath=args[3]
train_val_split = float(args[4])
#small_window= args[4] #I might not even need to pass this one- I can get the crop width based on the height of the spectrogram images

print("hello world!")

metadata = pd.read_csv(datapath,compression='gzip')



#img = imread('abc.tiff')
#pseudo:
#assign each bin as train or val category

for i in range(len(metadata.DiffTime.unique())):

    dt = metadata.DiffTime.unique()[i]
    #determine how many GTids there are in the FG, if any. If there are any, split them by id into train val splits

    meta_dt = metadata.loc[metadata['DiffTime'] == dt]

    meta_dt["Assignment"] = " " #use this later
    
    GTids = meta_dt.GTid.unique()

    GTids = GTids[np.logical_not(np.isnan(GTids))]

    random.shuffle(GTids)

    train_ids = GTids[:int(train_val_split * len(GTids))]
    val_ids = GTids[len(train_ids):]

    tv = np.concatenate((train_ids, val_ids))

    meta_dt_ids = meta_dt[meta_dt['GTid'].notnull()]

    import code
    code.interact(local=dict(globals(), **locals()))

    meta_dt_ids = meta_dt_ids.set_index('GTid')
    meta_dt_ids.loc[tv]

    #pick up here tomorrow! Trying to sort the data by the combination of train/val ids, then paste an equal length corresponding array of assignment onto this frame.
    #for the other one (nan frame), randomize, then assign with array of proportion.


    #now, seperate the df into two tables- one with ids and one with not.
    #



    
#loop through images
#use bins (note these need to retain difftime to associate to image!) to assign labels and bin extent to images.
#loop through each image and write label and power values to tfrecord. 

#read
