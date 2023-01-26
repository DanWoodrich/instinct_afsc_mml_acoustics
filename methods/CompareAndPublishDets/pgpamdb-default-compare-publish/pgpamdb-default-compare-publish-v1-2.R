library(pgpamdb)
library(DBI)

args="D:/Cache/359411/119603/206411/469561/979600 D:/Cache/359411/119603/206411/55319 D:/Cache/359411/119603/206411/469561/979600/312741  pgpamdb-default-compare-publish-v1-2"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

source(Sys.getenv('DBSECRETSPATH')) #populates connection paths which contain connection variables.
con=pamdbConnect("poc_v2",keyscript,clientkey,clientcert)

EditDataPath <-args[1]
PriorDataPath <- args[2]
resultPath <- args[3]
#transferpath<-args[4]

PriorData<-read.csv(paste(PriorDataPath,"DETx.csv.gz",sep="/"))
EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

#make sure comments which are na are instead set to blank:
PriorData$comments[is.na(PriorData$comments)]<-""
EditData$comments[is.na(EditData$comments)]<-""

#make sure that all of the placeholders etc are cleared:
#search for 'na' label (cant exist)

PriorData = PriorData[which(!is.na(PriorData$label)),]
EditData = EditData[which(!is.na(EditData$label)),]

#check columns are the same.
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

#make sure this doesn't contain blank or na values
#mod_keys = mod_keys[which(!is.na(EditData$id))]

operations = as.vector(rep("",3),mode='list')
names(operations)<-c("Modify","Insert","Delete")

affected_ids_total = c()

if(length(mod_keys)>0){

  EditMod= EditData[which(EditData$id %in% mod_keys),]
  PriorMod= PriorData[which(PriorData$id %in% mod_keys),]

  if(nrow(EditMod)!=nrow(PriorMod)){
    stop("ERROR: redundant IDs present in df")
  }

  #compare based on diffs since raven rounding rules are unclear.

  diffs = EditMod[,1:4]-PriorMod[,1:4]

  #.1 precision on frequency, .01 precision on time
  diffs[,3:4]= abs(diffs[,3:4])>0.1
  diffs[,1:2]= abs(diffs[,1:2])>0.01

  #if diffs are false, and the file end time has been modified, revert to original.
  #fix for the behavior where raven uses exact samples and rewrites end file (I round to 2 decimal for sf on db)
  ids_rev_et =EditMod[which(rowSums(diffs,na.rm=TRUE)==0 & EditMod$EndFile != PriorMod$EndFile),"id"]
  EditMod[which(EditMod$id %in% ids_rev_et),"EndFile"] = PriorMod[which(PriorMod$id %in% ids_rev_et),"EndFile"]

  #reduce this set to only rows which were modified
  testdf = EditMod[,5:length(EditMod)]!=PriorMod[,5:length(PriorMod)]
  testdf = cbind(diffs,testdf)
  sums = rowSums(testdf,na.rm=TRUE)

  EditMod = EditMod[sums>0,]

  if(nrow(EditMod)>0){

    colnames(EditMod)[1:6]=c("start_time","end_time","low_freq","high_freq","start_file","end_file")

    #lookup file names
    filelookup =lookup_from_match(con,'soundfiles',unique(c(EditMod$start_file,EditMod$end_file)),"name")

    EditMod$start_file = filelookup$id[match(EditMod$start_file,filelookup$name)]
    EditMod$end_file = filelookup$id[match(EditMod$end_file,filelookup$name)]

    #remove modified field so that it defaults to when it is submitted.
    EditMod$modified=NULL
    
    #remove analyst field so that it defaults to current user when submitted.
    #stealth change, want this as normal behavior. 
    EditMod$analyst=as.integer(dbFetch(dbSendQuery(con,"SELECT id FROM personnel WHERE personnel.pg_name = current_user"))$id)

    table_update(con,'detections',EditMod)

    operations[[1]]=paste(nrow(EditMod),"rows UPDATED")

    affected_ids_total = c(affected_ids_total,EditMod$id)

  }else{
    operations[[1]]="0 records UPDATED"
  }
}

if(nrow(new_data)>0){

  colnames(new_data)[1:6]=c("start_time","end_time","low_freq","high_freq","start_file","end_file")

  #lookup file names
  filelookup =lookup_from_match(con,'soundfiles',unique(c(new_data$start_file,new_data$end_file)),"name")

  new_data$start_file = filelookup$id[match(new_data$start_file,filelookup$name)]
  new_data$end_file = filelookup$id[match(new_data$end_file,filelookup$name)]

  #remove modified and id fields so that it defaults to when it is submitted.
  new_data$modified=NULL
  new_data$id = NULL
  new_data$original_id=NULL
  new_data$status=NULL

  if(any(new_data$analyst=="",na.rm=TRUE)){

      new_data[which(new_data$analyst==""),"analyst"]=NA

  }

  dbAppendTable(con,'detections',new_data)

  operations[[2]]=paste(nrow(new_data),"rows INSERTED")
  
  #look up newly submitted detections to find ids. 
  
  ids = table_dataset_lookup(con,"SELECT id FROM detections",
                             new_data[,c("start_time","end_time","low_freq","high_freq","start_file","end_file","procedure","signal_code","label","strength")],
                                      c("DOUBLE PRECISION","DOUBLE PRECISION","DOUBLE PRECISION","DOUBLE PRECISION","integer","integer","integer","integer","integer","integer"))$id
  

  affected_ids_total = c(affected_ids_total,as.integer(ids))

}else{
  operations[[2]]="0 records INSERTED"
}

if(length(del_keys)>0){

  table_delete(con,'detections',del_keys)

  operations[[3]]=paste(length(del_keys),"rows DELETED")
  
  #when I delete an id, the id changes. so, need to pull the ids of the deleted detection, 
  #which I can find with original id + signal_code + procedure
  
  #this can return multiple detections, but it shouldn't matter since updated shouldn't violate
  query =paste("SELECT DISTINCT ON (original_id) id FROM detections WHERE original_id IN (",
               paste(PriorData$original_id[which(PriorData$id %in% del_keys)],collapse=",",sep=""),
               ") AND status = 2 ORDER BY original_id,modified DESC",sep="")
    
  query <- gsub("[\r\n]", "", query)
    
  del_keys_find = dbFetch(dbSendQuery(con,query))
  
  affected_ids_total = c(affected_ids_total,as.integer(del_keys_find$id))
  

}else{
  operations[[3]]="0 records DELETED"
}

#now, examine changes to determine if i_neg recalc is required. 

if(length(affected_ids_total)>0){
  
  query = paste("SELECT DISTINCT effort.name, detections.procedure, detections.signal_code
                FROM detections JOIN bins_detections ON bins_detections.detections_id = detections.id
                JOIN bins ON bins.id = bins_detections.bins_id JOIN bins_effort ON bins_effort.bins_id
                = bins.id JOIN effort ON effort.id = bins_effort.effort_id JOIN effort_procedures ON 
                (effort_procedures.effort_id = effort.id AND detections.procedure = effort_procedures.procedures_id 
                AND detections.signal_code = effort_procedures.signal_code) WHERE effproc_assumption = 'i_neg'
                AND detections.id IN (",paste(affected_ids_total,collapse=",",sep=""),")",sep="")
  
  query <- gsub("[\r\n]", "", query)
  
  to_redo = dbFetch(dbSendQuery(con,query))
  
  for(i in 1:nrow(to_redo)){
    
    i_neg_update(con,to_redo$name[i],as.integer(to_redo$procedure)[i],as.integer(to_redo$signal_code)[i])
    
  }
  
  operations$i_neg_message = "i_neg sections recalculated:" 
  operations$i_neg_table = paste(to_redo,collapse=",")

}
writeLines(do.call("cbind",operations),paste(resultPath,'receipt.txt',sep="/"))



#attempt to find changes in data
#Attempt to submit a modification of the data with ids. Data w/o ids will be assumed to be new data.
#check that no records with Type of Det or SC have modified timestamp.

