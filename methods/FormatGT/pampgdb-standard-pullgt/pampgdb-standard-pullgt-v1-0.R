#install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/pgpamdb/pgpamdb_0.1.0.tar.gz", source = TRUE, repos=NULL)
library(pgpamdb)
library(DBI)

args = "C:/Apps/INSTINCT/Cache/652882/tempFG.csv.gz BS16_AU_PM02-a_files_1-175_rw_hg.csv decimate_data file_groupID methodID2m methodvers2m target_samp_rate y BS16_AU_PM02-a_files_1-175_rw_hg.csv matlabdecimate V1s0 1024 dbuddy-pull-FG-wname-v1-0"

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

source(Sys.getenv('DBSECRETSPATH')) #populates connection paths which contain connection variables.

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


  FG = read.csv(paste(dirname(dirname("C:/Apps/INSTINCT/Cache/675821/2998/DETx.csv.gz")),"FileGroupFormat.csv.gz",sep="/"))

  #determine ids based on bin parameters.

  FGred = data.frame(FG$Deployment,as.POSIXct(FG$StartTime,tz='utc'),FG$SegStart,FG$SegDur+FG$SegStart)

  colnames(FGred)=c("data_collection.name","soundfiles.datetime","bins.seg_start","bins.seg_end")

  bins =table_dataset_lookup(con,
                       "SELECT DISTINCT ON (soundfiles.datetime,data_collection.name,bins.seg_start,bins.seg_end) bins.id FROM bins JOIN soundfiles ON soundfiles.id = bins.soundfiles_id JOIN data_collection ON soundfiles.data_collection_id = data_collection.id",
                       FGred,
                       c("character varying","timestamp","DOUBLE PRECISION","DOUBLE PRECISION"))

  #print("1st query done")
  bins_format = paste("(",paste(as.integer(bins$id),collapse=",",sep=""),")",sep="")

  query = gsub("\\{FG\\}", bins_format, query)

  GTdata = dbFetch(dbSendQuery(con,query))

  #print("2nd query done")

  #print(str(GTdata))

  #print(head(GTdata))




  #now just need to format it with INSTINCT naming (camel case and expected order)

  GTdata$id_ = GTdata$id

  GTdata$id = NULL

  colnames(GTdata)[1:6]=c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile")
  colnames(GTdata)[length(GTdata)]="id"

  #trade in soundfile ids for soundfile names.

  sf_names = dbFetch(dbSendQuery(con,paste("SELECT id,name FROM soundfiles WHERE id IN (",paste(unique(c(GTdata$StartFile,GTdata$EndFile)),collapse=",",sep=""),")",sep="")))

  GTdata$StartFile =sf_names$name[match(GTdata$StartFile,sf_names$id)]
  GTdata$EndFile = sf_names$name[match(GTdata$EndFile,sf_names$id)]

  #print("3rd query done")
  #print(str(GTdata))

  #print(head(GTdata))

  dbDisconnect(con)

  write.csv(GTdata,gzfile(GTpath),row.names = FALSE)

}else{

  stop("other querying patterns to pgpamdb yet supported for this method")
}


