#install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/pgpamdb/pgpamdb_0.1.0.tar.gz", source = TRUE, repos=NULL)
library(pgpamdb)
library(DBI)

args = "D:/Cache/697284/885086/DETx.csv.gz [queryfg2] query remove_proc_ovlp querybody n pampgdb-standard-pullgt-v1-2"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

#print(args)
#print(commandArgs())

argsLen<-length(3:length(args))-1
argsSep<-argsLen/2

#print(argsSep)

ParamNames<-args[3:(3+argsSep-1)]
ParamArgs<-args[(3+argsSep):(length(args)-1)]

#print(ParamArgs)
#print(ParamNames)

#param string holds the actual query value, FGname is the query name.

source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")#populates connection paths which contain connection variables.

con=pamdbConnect("poc_v2",keyscript,clientkey,clientcert)


#this script will pull from either a named filegroup (1st check), named mooring deployment (2nd check), or a dynamic query.
#argument will be the same for all 3.


#docker values
GTpath <- args[1]
FGname <- args[2]

#print(ParamNames)

if(grepl('SELECT ',ParamArgs[which(ParamNames=="query")])){

  #pull FG by query.

  query = ParamArgs[which(ParamNames=="query")]

  #substitute {FG} for the actual bin ids.
  #load in filegroup


  FG = read.csv(paste(dirname(dirname(GTpath)),"FileGroupFormat.csv.gz",sep="/"))
  
  #determine ids based on bin parameters.
  
  if(grepl('bins.id ',ParamArgs[which(ParamNames=="query")])){
    
    FGred = data.frame(FG$Deployment,as.POSIXct(FG$StartTime,tz='utc'),FG$SegStart,FG$SegDur+FG$SegStart)
    
    colnames(FGred)=c("data_collection.name","soundfiles.datetime","bins.seg_start","bins.seg_end")
    
    bins =table_dataset_lookup(con,
                               "SELECT DISTINCT ON (soundfiles.datetime,data_collection.name,bins.seg_start,bins.seg_end) bins.id FROM bins JOIN soundfiles ON soundfiles.id = bins.soundfiles_id JOIN data_collection ON soundfiles.data_collection_id = data_collection.id",
                               FGred,
                               c("character varying","timestamp","DOUBLE PRECISION","DOUBLE PRECISION"),return_anything = TRUE)
    if(nrow(bins)==0){
      
      #this means that perhaps, the FG is on soundfiles and this is being called to produce an empty set. 
      is_empty = TRUE
      
      query = "SELECT * FROM detections LIMIT 0"
      
    }else{
      
      is_empty = FALSE
      
      #print("1st query done")
      bins_format = paste("(",paste(as.integer(bins$id),collapse=",",sep=""),")",sep="")
      
      query = gsub("\\{FG\\}", bins_format, query)
    }
    
   
    
  #determine based on soundfile
  }else if(grepl('detections.start_file ',ParamArgs[which(ParamNames=="query")])){
    
    FGred = data.frame(FG$Deployment,as.POSIXct(FG$StartTime,tz='utc'))
    
    colnames(FGred)=c("data_collection.name","soundfiles.datetime")
    
    sfs = table_dataset_lookup(con,
                               "SELECT DISTINCT ON (soundfiles.datetime,data_collection.name) soundfiles.id FROM soundfiles JOIN data_collection ON soundfiles.data_collection_id = data_collection.id",
                               FGred,
                               c("character varying","timestamp"))
    
    sfs_format = paste("(",paste(as.integer(sfs$id),collapse=",",sep=""),")",sep="")
    
    query = gsub("\\{FG\\}", sfs_format, query)
    
  }
  
  GTdata = dbFetch(dbSendQuery(con,query))

 # print("2nd query done")

  GTdata$id_ = GTdata$id

  GTdata$id = NULL

  colnames(GTdata)[1:6]=c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile")
  colnames(GTdata)[length(GTdata)]="id"

  #trade in soundfile ids for soundfile names.
  
  #ovlpcheck
  
  if(ParamArgs[which(ParamNames=="remove_proc_ovlp")]=='y' & nrow(GTdata)>1){
    
    StartFileDur_lookup = dbFetch(dbSendQuery(con,paste("SELECT id,duration FROM soundfiles WHERE id IN (",paste(unique(GTdata$StartFile),collapse=",",sep=""),")",sep="")))
    
    GTdata$StartFileDur = StartFileDur_lookup$duration[match(GTdata$StartFile,StartFileDur_lookup$id)]
    
    #compare on temporal dimension only. go one signal code at a time. 
    GTdata_all_sc = list()
    
    for(p in 1:length(unique(GTdata$signal_code))){
      
      GTdataIn = GTdata[which(GTdata$signal_code == unique(GTdata$signal_code)[p]),]
      
      #midpoint is simple and should do the trick. 
      
      #calculate midpoint, use to compare. 
      
      GTdataIn$midpoint = 0
      
      GTdataIn$tempEnd = GTdataIn$EndTime + GTdataIn$StartFileDur
      
      GTdataIn$delete = 0
      
      GTdataIn = GTdataIn[sample(1:nrow(GTdataIn),nrow(GTdataIn),replace=FALSE),]
      
      GTdataIn$real_end = 0
      
      if(nrow(GTdataIn[which(GTdataIn$StartFile==GTdataIn$EndFile),])>0){
        
        GTdataInsamefile = GTdataIn[which(GTdataIn$StartFile==GTdataIn$EndFile),]
        GTdataIn[which(GTdataIn$StartFile==GTdataIn$EndFile),"midpoint"]= GTdataIn[which(GTdataIn$StartFile==GTdataIn$EndFile),"StartTime"] + (GTdataIn[which(GTdataIn$StartFile==GTdataIn$EndFile),"EndTime"] -GTdataIn[which(GTdataIn$StartFile==GTdataIn$EndFile),"StartTime"])/2
        
        GTdataIn$real_end[which(GTdataIn$StartFile==GTdataIn$EndFile)] = GTdataInsamefile$EndTime
      
      }
      
      if(nrow(GTdataIn[-which(GTdataIn$StartFile==GTdataIn$EndFile),])>0){
        GTdataIn[-which(GTdataIn$StartFile==GTdataIn$EndFile),"midpoint"]=GTdataIn[-which(GTdataIn$StartFile==GTdataIn$EndFile),"StartTime"] + (GTdataIn[-which(GTdataIn$StartFile==GTdataIn$EndFile),"tempEnd"] -GTdataIn[-which(GTdataIn$StartFile==GTdataIn$EndFile),"StartTime"])/2
        GTdataIn$real_end[-which(GTdataIn$StartFile==GTdataIn$EndFile)]= GTdataIn[-which(GTdataIn$StartFile==GTdataIn$EndFile),"tempEnd"]
      }
    

      #shuffle the data, and then go row by row- if there are any duplicates which aren't marked for deletion,
      #mark current one for deletion
      

      for(i in 1:nrow(GTdataIn)){
        
        mp = GTdataIn[i,"midpoint"]
        
        if(GTdataIn$delete[i]==0){
        
          for(n in 1:nrow(GTdataIn)){
            
            #this is not safe for cases where same detection but only one starts on the previous file.. not a big enough 
            #problem to fix for now
            if(GTdataIn$StartTime[n]<=mp & GTdataIn$real_end[n]>=mp & GTdataIn$delete[n]==0 & GTdataIn$procedure[n]!=GTdataIn$procedure[i] & GTdataIn$StartFile[i] == GTdataIn$StartFile[n]){
              
              #print(GTdataIn[i,])
              #print(GTdataIn[n,])
              GTdataIn[i,"delete"]=1
              break
            }
          }
        }
      
      }
      
      GTdataIn = GTdataIn[which(GTdataIn$delete==0),]
      GTdataIn$real_end=NULL
      GTdataIn$midpoint=NULL
      GTdataIn$delete=NULL
      GTdataIn$StartFileDur=NULL
      GTdataIn$tempEnd=NULL
      
      GTdata_all_sc[[p]] = GTdataIn
    }
    
    GTdata = do.call('rbind',GTdata_all_sc)
    
  }
  
  if(nrow(GTdata)>0){
    

  sf_names = dbFetch(dbSendQuery(con,paste("SELECT id,name FROM soundfiles WHERE id IN (",paste(unique(c(GTdata$StartFile,GTdata$EndFile)),collapse=",",sep=""),")",sep="")))

  GTdata$StartFile =sf_names$name[match(GTdata$StartFile,sf_names$id)]
  GTdata$EndFile = sf_names$name[match(GTdata$EndFile,sf_names$id)]
  
  }

  #print("3rd query done")
  #print(str(GTdata))

  #print(head(GTdata))

  dbDisconnect(con)
  

  write.csv(GTdata,gzfile(GTpath),row.names = FALSE)

}else{

  stop("other querying patterns to pgpamdb yet supported for this method")
}


