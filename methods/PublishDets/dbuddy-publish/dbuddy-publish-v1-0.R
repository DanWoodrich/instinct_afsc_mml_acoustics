MethodID<-"dbuddy-publish-v1-0"

#v1-2: change to work with dbuddy behavior of automatically copying files from local path when doing DML. 

args="C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418 C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418/227655  dbuddy-publish-v1-0"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

EditDataPath <-args[1]
resultPath <- args[2]
#transferpath<-args[4]

EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

#add columns


EditData$Analysis_ID = args[3]
EditData$LastAnalyst = args[4]
EditData$VisibleHz = args[5]

EditData$Comments[is.na(EditData$Comments)]<-""

stop()

#make sure columns have correct names:
detcols<-c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","probs","VisibleHz",
                "label","Comments","SignalCode","Type","Analysis_ID","LastAnalyst")

filepath = paste(resultPath,"NewTemp.csv",sep="/")
logpath = paste(resultPath,"temp2.txt",sep="/")

write.csv(EditData,filepath,row.names = FALSE)

command = paste("dbuddy insert detections",filepath,"--log",logpath)
system(command)
operations=readLines(logpath)

writeLines(operations,paste(resultPath,'receipt.txt',sep="/"))


