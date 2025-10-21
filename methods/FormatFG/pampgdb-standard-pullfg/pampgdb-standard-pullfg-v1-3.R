#install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/pgpamdb/pgpamdb_0.1.0.tar.gz", source = TRUE, repos=NULL)
library(pgpamdb)
library(DBI)

args = "C:/Cache/731263/tempFG.csv.gz [queryfg] decimate_data difftime_limit file_groupID methodID2m methodvers2m target_samp_rate y 3600 query matlabdecimate V1s0 4000 pampgdb-standard-pullfg-v1-3"

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

#V1-3
#if there is bucket root, add a prefex to the outpath representing the bucket root. 
if("storage_service" %in% ParamNames){
  if(ParamArgs[which(ParamNames=="storage_service")] =="gcp"){
    prefix = "gs://afsc-1/bottom_mounted/"
  }else if(ParamArgs[which(ParamNames=="storage_service")] =="local"){
    prefix = "//"
  }
  
}else{
  prefix = "/"
}



source(Sys.getenv('DBSECRETSPATH')) #populates connection paths which contain connection variables.

con=pamdbConnect(dbname,keyscript,clientkey,clientcert)


#this script will pull from either a named filegroup (1st check), named mooring deployment (2nd check), or a dynamic query.
#argument will be the same for all 3.


#docker values
FGpath <- args[1]
FGname <- args[2]

#print(ParamNames)

if(grepl('SELECT ',ParamArgs[which(ParamNames=="file_groupID")])){

  #pull FG by query.

  FGdata = dbFetch(dbSendQuery(con,ParamArgs[which(ParamNames=="file_groupID")]))
  
  if(all(c("name","name..2") %in% colnames(FGdata))){
    
    colnames(FGdata)[which(colnames(FGdata)=='name')] = 'dname'
    colnames(FGdata)[which(colnames(FGdata)=='name..2')] = 'name'
  }
  


}else{

  #determine if the name corresponds to an existing filegroup:

  FGnames =dbFetch(dbSendQuery(con,"SELECT name FROM effort"))$name
  
  #v1-3 stealth change: give a standard order most likely to result in consecutive difftime where present

  if(FGname %in% FGnames){
    #pull FG from effort
    query = paste("SELECT data_collection.name AS dname,soundfiles.name,soundfiles.datetime,soundfiles.duration,bins.* FROM bins JOIN soundfiles ON 
          bins.soundfiles_id = soundfiles.id JOIN data_collection ON data_collection.id = soundfiles.data_collection_id
          JOIN bins_effort ON bins.id = bins_effort.bins_id JOIN effort on bins_effort.effort_id = effort.id 
          WHERE effort.name = '",FGname,"' ORDER BY data_collection.name,soundfiles.datetime,bins.seg_start",sep="")
    
    query = gsub("[\r\n]", "", query)

    FGdata =dbFetch(dbSendQuery(con,query))

  }else{

    depnames_new =dbFetch(dbSendQuery(con,"SELECT name FROM data_collection"))$name
    depnames_old =dbFetch(dbSendQuery(con,"SELECT historic_name FROM data_collection"))$historic_name

    if(FGname %in% depnames_new){
      #pull FG from new deployment name

      query = paste("SELECT data_collection.name AS dname,soundfiles.name,soundfiles.datetime,soundfiles.duration,0 AS seg_start, soundfiles.duration AS seg_end
                    FROM soundfiles JOIN data_collection ON soundfiles.data_collection_id = data_collection.id 
                    WHERE data_collection.name='",FGname,"' ORDER BY soundfiles.datetime",sep="")
      
      query = gsub("[\r\n]", "", query)
      
      FGdata =dbFetch(dbSendQuery(con,query))


    }else if(FGname %in% depnames_old){
      #pull FG from old deployment name
      
      query = paste("SELECT data_collection.name as dname,soundfiles.name,soundfiles.datetime,soundfiles.duration,0 AS seg_start, soundfiles_duration AS seg_end
                    FROM soundfiles JOIN data_collection ON soundfiles.data_collection_id = data_collection.id 
                    WHERE data_collection.historic_name='",FGname,"' ORDER BY soundfiles.datetime",sep="")
      
      query = gsub("[\r\n]", "", query)
      
      FGdata =dbFetch(dbSendQuery(con,query))

    }else{
      stop("cannot locate underlying effort of named FG or query")
    }


  }



}

#print(colnames(FGdata))

#print(str(FGdata))

if(prefix!="/"){
  full_files = paste(prefix,FGdata$dname,"/",format(FGdata$datetime,"%m"),"_",format(FGdata$datetime,"%Y"),"/",FGdata$name,sep="")
}else{
  full_files =  paste("/",FGdata$dname,"/",format(FGdata$datetime,"%m"),"_",format(FGdata$datetime,"%Y"),"/",sep="")
}

FGdataout = data.frame(FGdata$name,full_files,format(FGdata$datetime,"%y%m%d-%H%M%S"),FGdata$duration,FGdata$dname,FGdata$seg_start,FGdata$seg_end-FGdata$seg_start,FGname)
colnames(FGdataout)=c("FileName","FullPath","StartTime","Duration","Deployment","SegStart","SegDur","Name")

dbDisconnect(con)

write.csv(FGdataout,gzfile(FGpath),row.names = FALSE)



