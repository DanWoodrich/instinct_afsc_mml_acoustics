#v1-1: change the tp,fp,uk ints to 2,1,0 
#reason is that padding in image patching in dl fills in 0 for uk, which makes behaviors consistent. 
#v1-2:
#remove the 'reduce_factor', as it's irrelevant

#1-6: will shorten single label segments (0s) to a single zero instead of taking the time and 
#cpu to compress a large file only full of 0s. 

library(png)
library(signal)
library(dplyr)
library(doParallel)

args="D:/Cache/882048 D:/Cache/882048/143179 D:/Cache/882048/789477 D:/Cache/882048/789477/573484 AL18_AU_WT01 2 4096 0 240 80 tens-simple-label-v1-9"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
bf_path = paste(args[2],"bigfiles",sep="/") #one change I need to make- this actually needs to output 
#a file with necessary parameters to perform this process- in particular, it needs to save frequency
#start and end so that the current process knows how to populate the label tensors. 
GTref = read.csv(paste(args[3],"/DETx.csv.gz",sep=""))

#if(nrow(GTref)==0){
#  GTref$HighPix = numeric(0)
#  GTref$LowPix = numeric(0)
#}else{
#  GTref$HighPix = NA
#  GTref$LowPix  =  NA
#}


resultpath = args[4]

FGname = args[5]

#parameters:
dimensions = as.numeric(args[6]) #1 means that frequency of dets will expand to encompass full freq range. 
freq_size = as.numeric(args[7])
freq_start = as.numeric(args[8])
native_img_height = as.numeric(args[9])
native_pix_per_sec = as.numeric(args[10])

#find project location and route to R miscellaneous dependency. 
source(paste(getwd(),"/user/R_misc.R",sep=""))

bigfiles = unique(FG$DiffTime)

#v1-5, crude fix for different gt file
if(!any("SignalCode" %in% colnames(GTref))){
  
  colnames(GTref)[which(colnames(GTref)=='signal_code')]= "SignalCode"
  
}

GT_depth = length(unique(GTref$SignalCode))
#one, at minimum
if(GT_depth==0){
  GT_depth = 1
}

GT_unq = sort(unique(GTref$SignalCode)) #sort arranges alphabetically- do the same later when 'unpacking' results

dir.create(paste(resultpath,"/labeltensors",sep=""))

#stealth v1-9 edit: 
#optimize to not go into parallel loop if there is no GT to assign
if(nrow(GTref)==0){
  filenames = list()
  for(i in 1:length(bigfiles)){
    filename = paste(resultpath,"/labeltensors/labeltensor",i,"no_tp.csv.gz",sep="")
    write.table(1, gzfile(filename),sep=",",quote = FALSE,col.names=FALSE,row.names = FALSE)
    filenames[[i]]=filename
  }
}else{
  crs<-detectCores()
  
  startLocalPar(crs,"FG","bf_path","bigfiles","GTref","native_img_height","native_pix_per_sec","resultpath","dimensions","freq_size","freq_start","GT_depth","GT_unq")
  
  filenames = foreach(i=1:length(bigfiles),.packages=c("signal","dplyr","png")) %dopar% {
    #for(i in 1:length(bigfiles)){
    #combine GT with FG
    fileFG = FG[which(FG$DiffTime==bigfiles[i]),,drop = FALSE]
    
    fileFG =data.frame(
      fileFG %>%
        group_by(fileFG$FileName) %>%
        summarise(SegStart = min(SegStart), SegDur = sum(SegDur))
    )
    
    #v1-5... does it not work because length 1, or because needs to be c(0,cumsum(...)) instead?
    #if(nrow(fileFG)>1){
    fileFG$mod = cumsum(fileFG$SegDur)-fileFG$SegDur-fileFG$SegStart
    #}else{
    #  fileFG$mod = 0
    #}
    
    fileFG_start = data.frame(fileFG$fileFG.FileName,fileFG$mod)
    colnames(fileFG_start)=c("StartFile","mod_start")
    
    fileFG_end = data.frame(fileFG$fileFG.FileName,fileFG$mod)
    colnames(fileFG_end)=c("EndFile","mod_end")
    
    GT = merge(GTref,fileFG_start)
    GT = merge(GT,fileFG_end)
    
    #stealth change: if no GT within specific FG, skip file load step
    if(nrow(GT)==0){
      any_TP = "no_tp"
      lab_array=1
    }else{
      GT$StartTime = GT$StartTime+ GT$mod_start
      GT$EndTime = GT$EndTime+ GT$mod_end
      
      FGsecs = sum(fileFG$SegDur)
      #needs testing!
      
      #Revert to just reading in image- more resistant to rounding errors on odd sized sfs
      image = readPNG(paste(bf_path,"/bigfile",i,".png",sep=""))
      
      i_dims = dim(image)
      
      dim_x = native_img_height
      dim_y =i_dims[2] 
      
      pix_per_sec = dim_y/FGsecs
      
      total_pixels = dim_x * dim_y * GT_depth
      
      #correct total pixels to match that of the spectrogram. Use the ratio, and use that same
      #ratio to correct the GT timestamps. 
      
      if(dimensions==2){
        
        if(nrow(GT)>0){
          
          highfreq = freq_start+freq_size #figure these out from bigfile generation output
          lowfreq = freq_start#figure these out from bigfile generation output
          
          pix_per_freq = dim_x / (highfreq-lowfreq)
          
          #v1-8: fix apparent bug, possibly due to smaller freq than spec range or the high lowfreq. 
          GT$HighPix = (GT$HighFreq-lowfreq) * pix_per_freq
          GT$LowPix = (GT$LowFreq-lowfreq) * pix_per_freq
          
          #bound by freq:
          #v1-8 stealth edit- remove gt which exceed the high freq on their low boundary. 
          if(any(GT$LowPix>dim_x)){
            GT= GT[-which(GT$LowPix>dim_x),]
          }
        }
        #v1-8 stealth edit- uncomment this out. Not sure why it was commented out in the first place... 
        if(nrow(GT)>0){
          GT$HighPix[GT$HighPix>dim_x] = dim_x
          GT$LowPix[GT$LowPix<0] = 0
          
          GT$HighPix =round(GT$HighPix)
          GT$LowPix =round(GT$LowPix)
        }
      }else if(dimensions==1){  
        
        if(nrow(GT)>0){
          GT$HighPix = dim_x
          GT$LowPix = 0
        }
        
      }
      
      
      
      
      
      GT$StartPix = GT$StartTime * pix_per_sec
      GT$EndPix = GT$EndTime * pix_per_sec
      
      #bound by time:
      #v1-9: made more permissive
      GT = GT[which(GT$StartPix<dim_y & GT$EndPix>0),]
      
      GT$StartPix = round(GT$StartPix)
      GT$EndPix =round(GT$EndPix)
      
      #create template matrix
      
      
      
      GT$depth = match(GT$SignalCode,GT_unq)
      
      if(nrow(GT)>0){
        
        lab_array <- array(rep(1, total_pixels), dim=c(dim_x, dim_y, GT_depth)) #1 is now 'fp'
        #loop through and assign. 
        for(j in 1:length(GT_unq)){
          for(k in 1:nrow(GT)){
            
            end_ = min(dim_y,GT[k,"EndPix"])
            start_ = max(0,GT[k,"StartPix"])
            
            lab_array[GT[k,"LowPix"]:GT[k,"HighPix"],start_:end_,GT[k,"depth"]] = 2 #0 is now 'tp'
          }
          
        }
        
        #for(k in 1:GT_depth){
        #lab_array[,,k] = t(as.matrix(lab_array[,,k]))
        #}
        
        lab_array = aperm(lab_array,dim=c(dim_y, dim_x, GT_depth))
        
        any_TP = "is_tp"
      }else{
          any_TP = "no_tp"
          lab_array=1
      }
    }
    
    

    #now, unpack this by writing serializing it going downward by each column, per page of depth. 
    
    #lab_ser <- c(NA)
    #length(lab_ser) <- total_pixels
    #counter = 1
    
    #optimize this using apply? Or something. Very slow. 
    
    #for(j in 1:length(GT_unq)){
    #  for(k in 1:dim_y){
    #    print(k)
    #    lab_ser[counter:dim_x] = lab_array[dim_x:1,k,j]
    #    counter = counter + dim_x
    #  }
    #}
    
    #it appears that c() on the array will unpack it going top down, 
    #test = array(c(1,2,3,4,5,6,7,8),dim=c(2,2,2))
    #c(test)
    
    #write.csv(c(lab_array),lab_path,row.names = FALSE) 
    
    filename=paste(resultpath,"/labeltensors/labeltensor",i,any_TP,".csv.gz",sep="")
    
    write.table(c(lab_array), gzfile(filename),sep=",",quote = FALSE,col.names=FALSE,row.names = FALSE)
    
    return(filename)
  }
  
  stopCluster(cluz)
  
}

filenames=do.call("c",filenames)
outtab = data.frame(filenames,FGname)
colnames(outtab)=c("filename","FGname")

write.csv(outtab,paste(resultpath,'filepaths.csv',sep="/"),row.names=FALSE)

