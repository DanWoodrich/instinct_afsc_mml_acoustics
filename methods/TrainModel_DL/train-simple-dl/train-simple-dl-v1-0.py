#MethodID="train-simple-dl-v1-0"

import pandas as pd
#import tensorflow as tf
import sys
import numpy as np
import random
from matplotlib.image import imread


#######import params###########

args="C:/Apps/INSTINCT/Cache/394448/527039/receipt.txt C:/Apps/INSTINCT/Cache/394448/628717/947435/DETx.csv.gz C:/Apps/INSTINCT/Cache/394448/628717/947435/909093/summary.png 0.80 train-simple-dl-v1-0"
args=args.split()

#args=sys.argv
    
specpath=args[0]
datapath=args[1]
resultpath=args[2]

train_val_split = float(args[3])
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

    #by default, assign all bins to train or val, depending on the train_val_split parameter

    meta_dt["Assignment"] = " "

    meta_dt = meta_dt.sample(frac=1).reset_index(drop=True)

    cutoff = int(len(meta_dt)*train_val_split)

    meta_dt.loc[:cutoff,"Assignment"] = "Train"
    meta_dt.loc[cutoff:,"Assignment"] = "Validation"

    #now do grouped random sort of the GTids. 
    
    GTids = meta_dt.GTid.unique()

    GTids = GTids[np.logical_not(np.isnan(GTids))]

    random.shuffle(GTids)

    train_ids = GTids[:int(train_val_split * len(GTids))]
    val_ids = GTids[len(train_ids):]

    meta_dt.loc[meta_dt.GTid.isin(train_ids), 'Assignment'] = "Train"
    meta_dt.loc[meta_dt.GTid.isin(val_ids), 'Assignment'] = "Validation"

    #meta_dt.to_csv("C:/Apps/INSTINCT/lib/user/methods/TrainModel_DL/train-simple-dl/test.csv")

    import code
    code.interact(local=dict(globals(), **locals()))

    #alright, each bin has been assigned to train or val. Now, create record writer function so I can write the train and val datasets. 

    
#loop through images
#use bins (note these need to retain difftime to associate to image!) to assign labels and bin extent to images.
#loop through each image and write label and power values to tfrecord. 

#read
