MethodID<-"dbuddy-publish-v1-0"

#v1-2: change to work with dbuddy behavior of automatically copying files from local path when doing DML. 

args="C:/Apps/INSTINCT/Cache/587676/869751/419712/843629/733076/831400/914402/599940/744239/398246 C:/Apps/INSTINCT/Cache/587676/869751/419712/843629/733076/831400/914402/599940/744239/398246/187127 12 DFW GS 2048 dbuddy-publish-v1-0"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

EditDataPath <-args[1]
resultPath <- args[2]
#transferpath<-args[4]

EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

#add columns

EditData$Analysis_ID = as.integer(args[3])
EditData$LastAnalyst = args[4]
EditData$SignalCode = args[5]
EditData$VisibleHz = args[6]

EditData$Type = "DET"

EditData$Comments[is.na(EditData$Comments)]<-""

#do some quality checks. Right now, restrict labels to y/m/n
allowedlabels = c("y","m","n")

if(any(!unique(EditData$label) %in% allowedlabels)){
  message = paste("Forbidden label detected:'",paste(unique(EditData$label)[!unique(EditData$label) %in% allowedlabels],collapse=","),"'","in row:",
                  paste(which(EditData$label %in% unique(EditData$label)[!unique(EditData$label) %in% allowedlabels]),collapse=","))
  stop(message)
}

EditData[,1]<-as.numeric(EditData[,1])
EditData[,2]<-as.numeric(EditData[,2])
EditData[,3]<-as.numeric(EditData[,3])
EditData[,4]<-as.numeric(EditData[,4])

detcols<-c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","probs","VisibleHz",
           "label","Comments","SignalCode","Type","Analysis_ID","LastAnalyst")

if(any(!colnames(EditData) %in% detcols)){
  stop("row name mismatch")
}

#make sure columns have correct names:

filepath = paste(resultPath,"NewTemp.csv",sep="/")
logpath = paste(resultPath,"temp2.txt",sep="/")

write.csv(EditData,filepath,row.names = FALSE)

command = paste("dbuddy insert detections",filepath,"--log",logpath)
system(command)
operations=readLines(logpath)

writeLines(operations,paste(resultPath,'receipt.txt',sep="/"))

stop()


