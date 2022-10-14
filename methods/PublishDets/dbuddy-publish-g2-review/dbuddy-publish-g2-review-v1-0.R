MethodID<-"dbuddy-publish-v1-0"

#v1-2: change to work with dbuddy behavior of automatically copying files from local path when doing DML. 

args="//161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/587676/869751/419712/843629/733076/775797/647477/382499/968941/161137 //161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/587676/869751/419712/843629/733076/775797/647477/382499/968941/161137/121346 x DFW y GS 2048 dbuddy-publish-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

EditDataPath <-args[1]
resultPath <- args[2]
analyst<-args[4]
only_validate = args[5]

EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

#add columns

EditData$Analysis_ID = as.integer(args[3])
EditData$LastAnalyst = args[4]
EditData$SignalCode = args[5]
EditData$VisibleHz = args[6]

EditData$Type = "DET"

EditData$LastAnalyst =analyst #this is JLC for this one, but take it from parameters anyways. 

EditData$Comments[is.na(EditData$Comments)]<-""

#if sperm is detected, change label type to OBS, and signal code to SP

EditData[which(EditData$label=="s"),"SignalCode"] ="SP"
EditData[which(EditData$label=="s"),"Type"] ="OBS"
EditData[which(EditData$label=="s"),"label"]="y"

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
  stop("column name mismatch")
}

#make sure columns have correct names:

filepath = paste(resultPath,"NewTemp.csv",sep="/")
logpath = paste(resultPath,"temp2.txt",sep="/")

if(only_validate!="y"){

write.csv(EditData,filepath,row.names = FALSE)

stop('whoa there!')
  
command = paste("dbuddy insert detections",filepath,"--log",logpath)
system(command)
operations=readLines(logpath)

}else{
  
  operations = "File successfully check for errors, protocol performed correctly!"
  
}

writeLines(operations,paste(resultPath,'receipt.txt',sep="/"))



