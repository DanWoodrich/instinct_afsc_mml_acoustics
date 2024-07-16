#v1-1:
#save paths instead of receipt

#v1-3:
#read in image dimensions instead of naively calculating to avoid rounding errors
#later associated with unpredictable behavior in image generation

#v1-3 stealth change: parallelize

library(signal)
library(dplyr)
library(png)
library(doParallel)
library(foreach)

args="D:/Cache/882048 D:/Cache/882048/143179 D:/Cache/882048/143179/671277 AL18_AU_WT01 80 1 within_file n 0 0.75 y"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) #source("C:/Apps/INSTINCT/lib/user/R_misc.R") 
args<-commandIngest()

FG = read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
bigfilespath = args[2]
resultpath = args[3]
FGname = args[4]
native_pix_per_sec= as.numeric(args[5])
seed_val = as.integer(args[6])
split_protocol= args[7] 
test_split= as.numeric(args[8])
train_test_split= as.numeric(args[9])
train_val_split=as.numeric(args[10])

#pseudo: splits are along 1 dimenion. Perform at resolution of original image. 

#randomness is determined by seed to be reproducible. 

#splits should be randomly positioned in each bigfile (if within_file)
#if by_file, splits should be randomly assigned to each file proportional to splits. 

#splits should be consecutive to get the most out of labels (tf method will discard labels overlapping splits)

bigfiles = unique(FG$DiffTime)

dir.create(paste(resultpath,"/splittensors",sep=""))

#stealth change: if there isn't any training data, just write a single 3 and avoid reads, parallelization. 
if(train_test_split==0){
  for(i in 1:length(bigfiles)){
    write.table(3, gzfile(paste(resultpath,"/splittensors/splittensor",i,".csv.gz",sep="")),sep=",",quote = FALSE,col.names=FALSE,row.names = FALSE)
  }
}else{
  
  #set.seed(seed_val) #this doesn't get propogated correctly into parallelization. 
  #so, change is to declare inside loop. Seeds won't be compatible b/t this way
  #and non-parallelized way. 
  

  
  crs<-detectCores()
  
  startLocalPar(crs,"seed_val","FG","bigfilespath","resultpath","native_pix_per_sec","split_protocol","test_split","train_test_split","train_val_split")
  
  foreach(i=1:length(bigfiles),.packages=c("signal","dplyr","png")) %dopar% {
    
    set.seed(i+seed_val)
    
    #combine GT with FG
    fileFG = FG[which(FG$DiffTime==bigfiles[i]),,drop = FALSE]
    
    fileFG =data.frame(
      fileFG %>%
        group_by(fileFG$FileName) %>%
        summarise(SegStart = min(SegStart), SegDur = sum(SegDur))
    )
    
    fileFG$mod = cumsum(fileFG$SegDur)-fileFG$SegDur-fileFG$SegStart
    
    c = sum(fileFG$SegDur)
    
    if(split_protocol=="within_file"){
      
      image = readPNG(paste(bigfilespath,"/bigfiles/bigfile",i,".png",sep=""))
      
      i_dims = dim(image)
      
      #total_pix = FGsecs * native_pix_per_sec
      total_pix = i_dims[2]
      
      training_pix =  train_test_split *total_pix
      
      train_pix =training_pix*train_val_split
      val_pix = training_pix-train_pix
      
      test_pix = total_pix-training_pix
      
      #1 = train
      #2 = val 
      #3 = test
      order = sample(1:3)
      
      split = rep(order, times = c(train_pix, val_pix,test_pix)[order])
      
      write.table(split, gzfile(paste(resultpath,"/splittensors/splittensor",i,".csv.gz",sep="")),sep=",",quote = FALSE,col.names=FALSE,row.names = FALSE)
      
    }else{
      print("case not yet defined")
    }
    
  }
  
  stopCluster(cluz)
}


outtab = data.frame(paste(resultpath,"/splittensors/splittensor",1:length(bigfiles),".csv.gz",sep=""),FGname)
colnames(outtab)=c("filename","FGname")

write.csv(outtab,paste(resultpath,'filepaths.csv',sep="/"),row.names=FALSE)




