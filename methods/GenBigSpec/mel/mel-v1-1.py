import sys
import os
import pandas as pd
import librosa
import numpy as np
import multiprocessing as mp
from PIL import Image
from multiprocessing import Pool

sys.path.append(os.getcwd())
from user.misc import arg_loader

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def load_file(row):

    filename = row.SFroot + row.path_and_file
    
    return [*librosa.load(filename,sr=None,offset=row['min'],duration=(row['max']-row['min']))] 
    
def gen_spec(y,sr,difftime,native_img_height,native_pix_per_sec,windowLength,outlier_perc):

    #SFroot,native_img_height,native_pix_per_sec,windowLength
    
    #limit loudness of audio:
    #y = (y - y.mean()) / y.std()
    
    #high_thresh = np.percentile(y, 99) #limit to 99th percentile of energy values to 
    #eliminate outliers
    #y = np.where(y > high_thresh, high_thresh, y)
    #y = np.where(y < -high_thresh, -high_thresh, y)
    
    #import code
    #code.interact(local=dict(globals(), **locals()))

    xbins = round((len(y)/sr)*native_pix_per_sec)

    hl = round(len(y)/xbins)

    n_mels = native_img_height #play around with this
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=windowLength, hop_length=hl, n_mels=n_mels)
    
    high_thresh = np.percentile(S, outlier_perc) #limit to 99th percentile of energy values to 
    #eliminate outliers
    #low_thresh = np.percentile(S, 25)
    #low_thresh = np.percentile(S, 100-outlier_perc)
    S = np.where(S > high_thresh, high_thresh, S)
    #S = np.where(S < low_thresh, low_thresh, S)
    
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    #S_DB = librosa.pcen(np.abs(S), sr=sr, hop_length=hl)
    
    #####S= np.abs(librosa.stft(y, n_fft=windowLength,  hop_length=hl))
    ######S_DB = librosa.power_to_db(S**2, ref=np.max)
    
    #median = np.median(S_DB)
    #std_dev = np.std(S_DB)
    
    #S_DB = np.clip((S_DB-median)/(1* std_dev),-1,1) #hardcode 3 as std deviations
    
    #import code
    #code.interact(local=dict(globals(), **locals()))
    
    #S = np.log(S + 1e-9) # add small number to avoid log(0)
    
    #remove very high outliers:
    #S = np.percentile(a, 99.999)
    
    #S_DB = S_DB[:,:xbins] #this truncation was causing delaterious effects down the pipe
        
    ######S_DB = np.log(S_DB + 0.000001)
    
    #im = scale_minmax(S, 0, 255)
    #S_DB = ((S_DB+1)/2)*255

    #im = S_DB.astype(np.uint8)
    
    #high_thresh = np.percentile(S_DB, 90)
    #S_DB = np.where(S_DB > high_thresh, high_thresh, S_DB)
    
    im = scale_minmax(S_DB, 0, 255)
    #log and rescale to deal with outlier values
    #im = scale_minmax(np.log(im+1e-9),0,255)
    
    
    im = im.astype(np.uint8)
    im = np.flip(im, axis=0) # put low frequencies at the bottom in image
    im = 255-im # invert. make black==more energy

    #assert im.max() == 255
    #assert im.min() == 0

    im = Image.fromarray(im)
    #im = im.convert('RGB')
   
    im=im.resize((xbins,n_mels)) #see if this improves result vs truncation
    
    return im
    
    
    #import code
    #code.interact(local=dict(globals(), **locals()))

def process_chunk(data):

    #import code
    #code.interact(local=dict(globals(), **locals()))

    difftimes= data.DiffTime.unique()
   
    if data["view_mode"].iloc[0]=="y" or data["view_mode"].iloc[0]=="rand":
        if data["view_mode"].iloc[0]=="rand":
            np.random.shuffle(difftimes)
        view_mode = True
    else:
        view_mode = False
    for i in difftimes:
    
        data_in = data[data['DiffTime'].isin([i])]    

        data_in['SegEnd'] = data_in['SegStart'] + data_in['SegDur']
        data_in['path_and_file'] = data_in['FullPath'] + data_in['FileName']
        
        data_merged = pd.DataFrame(pd.concat([data_in.groupby('path_and_file')['SegStart'].agg(['min']), data_in.groupby('path_and_file')['SegEnd'].agg(['max'])], axis=1)).reset_index()
        
        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        data_merged['SFroot'] = data["SFroot"].iloc[0]  #could be slightly less memory intensive, do this for convenience. 

        #load in the soundfiles
        out = data_merged.apply(load_file, axis=1).tolist()
                
        #y = [j for i in [a[0] for a in out] for j in i] 
        y = np.concatenate([a[0] for a in out])

        sr = [a[1] for a in out]
        
        #ensure sampling rates are the same
        assert len(set(sr)) == 1
        
        sr = sr[0]
       
        im = gen_spec(y,sr,str(i),data["native_img_height"].iloc[0],data["native_pix_per_sec"].iloc[0],data["windowLength"].iloc[0],data["outlier_perc"].iloc[0])
        
        if view_mode:
            
            im.show() 
            
            values = input("press any key to continue, type ! to abort")
            if values =='!':
                exit()
            
        else:
            im.save(data["resultpath"].iloc[0] + "/bigfiles/bigfile" + str(i) + ".png")
        
if __name__ == "__main__":

    args=arg_loader()

    FG= pd.read_csv(args[1] + "/FileGroupFormat.csv.gz")
    resultpath = args[2]
    SFroot = args[3]
    FGname = args[4]
    native_img_height = int(args[5])
    native_pix_per_sec = int(args[6])
    outlier_perc       = float(args[7])
    view_mode     = args[8]
    windowLength    = int(args[9])
    
    #for convenience for multiprocessing, just stick on constants to FG 
    
    FG["SFroot"] = SFroot
    FG["native_img_height"] = native_img_height
    FG["native_pix_per_sec"] = native_pix_per_sec
    FG["windowLength"] = windowLength
    FG["resultpath"] = resultpath
    FG["view_mode"] = view_mode
    FG["outlier_perc"] = outlier_perc
    
    cpus = mp.cpu_count()

    unqvalues = FG.DiffTime.unique()
    #np.random.shuffle(unqvalues)
    chunksize = len(unqvalues)//cpus

    chunks = [i for i in unqvalues[::chunksize]] #would like to randomize. optimize later
    
    #make the last chunk cover the remainder
    chunks[len(chunks)-1]= len(unqvalues)+1

    chunk_margins = [(chunks[i-1],chunks[i]) for i in range(1, len(chunks))]

    FG_chunks = [FG.DiffTime[(FG.DiffTime >= x[0]) & (FG.DiffTime < x[1])] for x in chunk_margins]
       
    #import code
    #code.interact(local=dict(globals(), **locals()))
    
    FG_chunks_fullds = [FG[FG.DiffTime.isin(i)] for i in FG_chunks]
    
    if view_mode =="n":
        if not os.path.exists(resultpath + "/bigfiles"):
            os.mkdir(resultpath + "/bigfiles")
        with Pool(processes=cpus) as pool:
            pool.map(process_chunk,FG_chunks_fullds)
    elif view_mode == "y" or view_mode == "rand":
        process_chunk(FG_chunks_fullds[0])
    else:
        ValueError("view_mode must be n, y, or rand")
    #

    files_list = [[resultpath+"/bigfiles/bigfile"+str(i)+".png",FGname] for i in unqvalues]

    ds = pd.DataFrame(files_list,columns=['filename', 'FGname'])

    ds.to_csv(resultpath+"/filepaths.csv",index=False)
    

