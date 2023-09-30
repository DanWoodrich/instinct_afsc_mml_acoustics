#this script will publish based on character of data. contains logic 
#for adding bin detections where detections aren't represented in fg. 

library(pgpamdb)
library(DBI)

args="D:/Cache/962525/742867/265067/208115/883370 D:/Cache/962525/742867/265067/208115/883370/969254 LOW 25 FW loose pampgdb-standard-publish-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

FGpath <-args[1]
EditDataPath <-args[2]
resultPath <- args[3]
#transferpath<-args[4]
analysis_binsize = args[4] #Low / reg/ shi
procedureID = args[5]
signal_code = args[6]
strength= args[7] 

FGdata = read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"))
DetData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

if("Verification" %in% colnames(DetData)){
  DetData$label = DetData$Verification
  DetData$Verification= NULL
}

if(nrow(DetData)!=0){
  DetData$Type = "DET"
  DetData$procedure = procedureID
  DetData$strength = strength
  DetData$SignalCode = signal_code
  DetData$Comments = ""
  
  if(!"label" %in% colnames(DetData)){
    DetData$label = "uk"
  }
  
}


source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
con=pamdbConnect(dbname,keyscript,clientkey,clientcert)

#convert FG into db form: 
#cols needed: "soundfiles_id","seg_start","seg_end","duration"

lookup = lookup_from_match(con,'soundfiles',FGdata$FileName,'name')

colnames(lookup)[2] = 'FileName'

FGdata = merge(FGdata,lookup)

fg_db = data.frame(FGdata$id,FGdata$SegStart,FGdata$SegDur+FGdata$SegStart,FGdata$Duration)
colnames(fg_db) = c("soundfiles_id","seg_start","seg_end","duration")

if(nrow(DetData)!=0){
  #pseudo: convert data into db form. 
  DetData_db = detx_to_db(con,DetData)
  
  DetData_db$FGID = NULL
  DetData_db$splits=NULL
  
  #before continuing, do a check to assess if the detections already exist on the db to avoid double
  #submitting dets. 
  
  testds = DetData_db[,c(colnames(DetData_db)[1:10])]
  
  matchcount = table_dataset_lookup(con,"SELECT * FROM detections",testds,c("DOUBLE PRECISION","DOUBLE PRECISION",
                                                                            "DOUBLE PRECISION","DOUBLE PRECISION","BIGINT","BIGINT",
                                                                            "DOUBLE PRECISION","INTEGER","INTEGER","INTEGER"),
                                    return_anything = TRUE)
  
  #temporarily change the label to yes: that way, will not submit bin negatives for 
  #potentially correct detections. 
  DetData_db$label = 1
  DetData_db$analyst = 'temp' #this is needed by bin_negatives fxn (unfortunately)
  
  if(nrow(matchcount)>0){
    stop("database check indicates these detections may have already been submitted. Aborting...")
  }
  
  bin_negs = bin_negatives(DetData_db,fg_db,analysis_binsize)
  
  DetData_db$analyst = NULL
  bin_negs$analyst = NULL
  
  #change label back to unknown. 
  DetData_db$label = 99
  
  outdata = rbind(DetData_db,bin_negs)
  

}else{
  DetData_db = dbFetch(dbSendQuery(con,"SELECT * FROM detections LIMIT 0"))
  
  #outdata is just the fg tab, turned into 20 detections: 
  
  selection = which(c("LOW", "REG", "SHI") == analysis_binsize)
  
  #default high frequency for low 
  highfreq = c(512, 8192, 16384)[selection]
  
  signal_code_db = dbFetch(dbSendQuery(con,paste("SELECT id FROM signals WHERE code ='",signal_code,"'",sep="")))
  strength_db = dbFetch(dbSendQuery(con,paste("SELECT id FROM strength_codes WHERE name ='",strength,"'",sep="")))
  
  
  outdata = data.frame(fg_db$seg_start,fg_db$seg_end,0,highfreq,fg_db$soundfiles_id,fg_db$soundfiles_id,
                       NA,"",procedureID,20,as.integer(signal_code_db$id),strength_db$id)
  
  colnames(outdata) = colnames(DetData_db)[2:(length(outdata)+1)]
  
  testds = outdata[,c(colnames(outdata)[1:11])]
  testds = testds[,c(-7,-8)] #get rid of comments and probability
  
  matchcount = table_dataset_lookup(con,"SELECT * FROM detections",testds,c("DOUBLE PRECISION","DOUBLE PRECISION",
                                                                            "DOUBLE PRECISION","DOUBLE PRECISION","BIGINT","BIGINT",
                                                                            "DOUBLE PRECISION","INTEGER","INTEGER","INTEGER"),
                                    return_anything = TRUE)
  
  if(nrow(matchcount)>0){
    stop("database check indicates these detections may have already been submitted. Aborting...")
  }
    
}

#rbind tables and submit. 

#submit to db. 

dbAppendTable(con,'detections',outdata)

operations=as.vector(rep("",2),mode='list')

operations[[1]]=paste(nrow(DetData_db),"unverified detections submitted")
operations[[2]]=paste(nrow(outdata)-nrow(DetData_db),"bin negatives on",analysis_binsize,"timescale with protocol negative assumpion submitted")

writeLines(do.call("cbind",operations),paste(resultPath,'receipt.txt',sep="/"))






