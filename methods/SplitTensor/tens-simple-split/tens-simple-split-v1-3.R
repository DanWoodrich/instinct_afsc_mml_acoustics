#v1-1:
#save paths instead of receipt

#v1-3:
#read in image dimensions instead of naively calculating to avoid rounding errors
#later associated with unpredictable behavior in image generation

library(signal)
library(dplyr)
library(png)

args="D:/Cache/449513 D:/Cache/449513/439550 D:/Cache/449513/439550/913601 fw1difficult 20 1 within_file y 0.80 0.75 y     tens-simple-split-v1-3"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
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

set.seed(seed_val)

dir.create(paste(resultpath,"/splittensors",sep=""))

for(i in 1:length(bigfiles)){
  
  #combine GT with FG
  fileFG = FG[which(FG$DiffTime==bigfiles[i]),,drop = FALSE]
  
  fileFG =data.frame(
    fileFG %>%
      group_by(fileFG$FileName) %>%
      summarise(SegStart = min(SegStart), SegDur = sum(SegDur))
  )
  
  fileFG$mod = cumsum(fileFG$SegDur)-fileFG$SegDur-fileFG$SegStart

  FGsecs = sum(fileFG$SegDur)
  
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
    
    if(training_pix!=0){
      split = rep(order, times = c(train_pix, val_pix,test_pix)[order])
    }else{
      split = 3
    }
  
    write.table(split, gzfile(paste(resultpath,"/splittensors/splittensor",i,".csv.gz",sep="")),sep=",",quote = FALSE,col.names=FALSE,row.names = FALSE)
  
  }else{
    print("case not yet defined")
  }
  
}

outtab = data.frame(paste(resultpath,"/splittensors/splittensor",1:length(bigfiles),".csv.gz",sep=""),FGname)
colnames(outtab)=c("filename","FGname")

write.csv(outtab,paste(resultpath,'filepaths.csv',sep="/"),row.names=FALSE)




