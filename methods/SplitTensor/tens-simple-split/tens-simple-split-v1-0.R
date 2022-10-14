#v1-1:
#change parameters 

library(signal)
library(dplyr)

args="C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/921636 C:/Apps/INSTINCT/Cache/394448/921636/130114/receipt.txt 1 40 by_file y 0.80 0.75 y tens-simple-split-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()


FG = read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
bigfilespath = args[2]
resultpath = args[3]
native_pix_per_sec= as.numeric(args[4])
seed_val = as.integer(args[5])
split_protocol= args[6] 
test_split= as.numeric(args[7])
train_test_split= as.numeric(args[8])
train_val_split=as.numeric(args[9])

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
  
  total_pix = FGsecs * native_pix_per_sec
  
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

writeLines("split tensors successfully written",paste(resultpath,'receipt.txt',sep="/"))




