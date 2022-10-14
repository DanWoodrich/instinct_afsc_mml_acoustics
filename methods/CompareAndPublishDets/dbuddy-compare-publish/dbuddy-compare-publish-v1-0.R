MethodID<-"dbuddy-compare-publish-v1-0"

args="C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418 C:/Apps/INSTINCT/Cache/809545/548038 C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418/227655  dbuddy-compare-publish-v1-0"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

EditDataPath <-args[1]
PriorDataPath <- args[2]
resultPath <- args[3]

PriorData<-read.csv(paste(PriorDataPath,"DETx.csv.gz",sep="/"))
EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

detcols<-c("id","StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","probs","VisibleHz",
                "label","Comments","SignalCode","Type","Analysis_ID","LastAnalyst")

#make sure comments which are na are instead set to blank: 
PriorData$Comments[is.na(PriorData$Comments)]<-""
EditData$Comments[is.na(EditData$Comments)]<-""

#make sure dfs are in the right order
EditData<-EditData[,detcols]
PriorData<-PriorData[,detcols]

if(any(!colnames(EditData)==colnames(PriorData))){
  stop("edited data columns do not match database standard")
}

#this will not necessarily put this in a logical order when there are multiple deployments, but it is at least standard as every det needs a timestamp (sort by id will fail where
#id not present)
PriorData<-PriorData[order(PriorData$StartFile,PriorData$StartTime),]
EditData<-EditData[order(EditData$StartFile,EditData$StartTime),]

#seperate the edited data into those with keys that match priordata, and keys that are novel (or missing). 
mod_keys = EditData$id[which(EditData$id %in% PriorData$id)]
new_data = EditData[-which(EditData$id %in% PriorData$id),]
del_keys= PriorData$id[-which(PriorData$id %in% EditData$id)]

operations = as.vector(rep("",3),mode='list')
names(operations)<-c("Modify","Insert","Delete")

if(length(mod_keys)>0){
  
  EditMod= EditData[which(EditData$id %in% mod_keys),]
  PriorMod= PriorData[which(PriorData$id %in% mod_keys),]
  
  if(nrow(EditMod)!=nrow(PriorMod)){
    stop("ERROR: redundant IDs present in df")
  }
  
  #reduce this set to only rows which were modified
  testdf = EditMod!=PriorMod
  sums = rowSums(testdf,na.rm=TRUE)
  
  EditMod = EditMod[sums>0,]
  
  if(nrow(EditMod)>0){
    
    #save the edits in a temp file. 
    
    filepath = paste(resultPath,"ModTemp.csv",sep="/")
    
    write.csv(EditMod,filepath,row.names = FALSE)
    
    #publish the edits to dbuddy
    command = paste("dbuddy modify detections",filepath)
    operations[[1]]=system(command, intern = T)
    
    file.remove(filepath)
    
  }
}

if(nrow(new_data)>0){
  
  filepath = paste(resultPath,"NewTemp.csv",sep="/")
  write.csv(new_data,filepath,row.names = FALSE)
  
  command = paste("dbuddy insert detections",filepath)
  operations[[2]]=system(command, intern = T)
  
  file.remove(filepath)
  
}

if(length(del_keys)>0){
  filepath = paste(resultPath,"DelKeysTemp.csv",sep="/")
  write.csv(PriorData$id[PriorData$id %in% del_keys],filepath,row.names = FALSE)
  
  command = paste("dbuddy delete detections",filepath)
  operations[[3]]=system(command,intern=T)
  
  file.remove(filepath)
  
}

writeLines(do.call("cbind",operations),paste(resultPath,'receipt.txt',sep="/"))



#attempt to find changes in data
#Attempt to submit a modification of the data with ids. Data w/o ids will be assumed to be new data. 
#check that no records with Type of Det or SC have modified timestamp. 

