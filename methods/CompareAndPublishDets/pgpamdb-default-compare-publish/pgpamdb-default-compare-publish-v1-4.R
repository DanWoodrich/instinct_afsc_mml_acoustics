library(pgpamdb)
library(DBI)

args="D:/Cache/669745/161133/776385/204574/63390 D:/Cache/669745/161133/776385/986780 D:/Cache/669745/161133/776385/204574/63390/228534  pgpamdb-default-compare-publish-v1-4"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
con=pamdbConnect("poc_v2",keyscript,clientkey,clientcert)

EditDataPath <-args[1]
PriorDataPath <- args[2]
resultPath <- args[3]
#transferpath<-args[4]

PriorData<-read.csv(paste(PriorDataPath,"DETx.csv.gz",sep="/"))
EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

#bugfix: if raven conversion resulted in case where a detection did not have an endfile, remove from 
#editdata and remove id from priordata (ignore it)

if(any(EditData$EndFile=="")){
  ids = EditData[which(EditData$EndFile==""),"id"]
  EditData= EditData[-which(EditData$EndFile==""),]
  PriorData = PriorData[-which(PriorData$id==ids),]
}



#make sure comments which are na are instead set to blank:
PriorData$comments[is.na(PriorData$comments)]<-""
EditData$comments[is.na(EditData$comments)]<-""

#remove unneeded columns, if they exist
PriorData$Name.x= NULL
PriorData$Name.y= NULL
EditData$Name.x= NULL
EditData$Name.y= NULL

#make sure that all of the placeholders etc are cleared:
#search for 'na' label (cant exist)

PriorData = PriorData[which(!is.na(PriorData$label)),]
EditData = EditData[which(!is.na(EditData$label)),]

#keep a vector of all the procedures. Use this vector to check assumptions and make sure
#that modifications outside of allowed rules (such as, moving/resizing a machine generated detection,
#or adding/removing machine generated detections.)

allprocedures = unique(EditData$procedure)

procedure_assumption_lookup = dbFetch(dbSendQuery(con,paste("SELECT * FROM procedures WHERE id IN (",
                                                  paste(allprocedures,sep="",collapse=","),")",sep="")))

#check columns are the same.
if(any(!colnames(EditData)==colnames(PriorData))){
  stop("edited data columns do not match database standard")
}

#this will not necessarily put this in a logical order when there are multiple deployments, but it is at least standard as every det needs a timestamp (sort by id will fail where
#id not present)
PriorData<-PriorData[order(PriorData$StartFile,PriorData$StartTime),]
EditData<-EditData[order(EditData$StartFile,EditData$StartTime),]

#do some type checks. Figure out if label needs to be changed to prior format
priorlabtype = typeof(PriorData$label)
if(typeof(EditData$label)!="integer"){
  #sanitize a little
  EditData$label[which(EditData$label=="nn")]="n"
  EditData$label[which(EditData$label=="yy")]="n"
  EditData$label[which(EditData$label=="ny")]="uk"
  EditData$label[which(EditData$label=="yn")]="uk"
  EditData$label[which(EditData$label=="")]="uk"
  
  lablookup =lookup_from_match(con,'label_codes',unique(EditData$label),"alias")
  
  EditData$label = lablookup$id[match(EditData$label,lablookup$alias)]
}



#seperate the edited data into those with keys that match priordata, and keys that are novel (or missing).
mod_keys = EditData$id[which(EditData$id %in% PriorData$id)]
new_data = EditData[-which(EditData$id %in% PriorData$id),]
del_keys= PriorData$id[-which(PriorData$id %in% EditData$id)]

#do a check on new and delete data up here to make sure assumption errors are caught before data modification.

if(nrow(new_data)>0){
  
  if(any(!is.na(procedure_assumption_lookup[which(as.integer(procedure_assumption_lookup$id) %in% new_data$procedure),"negative_bin"]))){
    #this means that there user tried to submit a new detection which is part of a deployment procedure, a nono
    
    stop("Attempted to insert new detection for a detector deployment as part of labeling workflow. This is not allowed, fix and rerun INSTINCT. Aborting... ")
  }
}

if(length(del_keys)>0){
  
  deltab = PriorData[which(PriorData$id %in% del_keys),]
  
  if(any(!is.na(procedure_assumption_lookup[which(as.integer(procedure_assumption_lookup$id) %in% deltab$procedure),"negative_bin"]))){
    stop("Attempted to delete a detection for a detector deployment as part of labeling workflow. This is not allowed, fix and rerun INSTINCT. Aborting... ")
  }
  
}



#make sure this doesn't contain blank or na values
#mod_keys = mod_keys[which(!is.na(EditData$id))]

#transactions: 
transaction_data = list()
transaction_operation = list()

#summary of operations for receipt
operations = as.vector(rep("",3),mode='list')
names(operations)<-c("Modify","Insert","Delete")

affected_ids_total = c()

if(length(mod_keys)>0){

  EditMod= EditData[which(EditData$id %in% mod_keys),]
  PriorMod= PriorData[which(PriorData$id %in% mod_keys),]
  
  #stealth change v1-3: move up here. remove from both editmod and modified so changes are not
  #counted
  EditMod$modified=NULL
  PriorMod$modified = NULL

  if(nrow(EditMod)!=nrow(PriorMod)){
    stop("ERROR: redundant IDs present in df")
  }
  
  #stealth change: sort by id. bug occured where maintaining order couldn't be assumed: 
  EditMod = EditMod[order(EditMod$id),]
  PriorMod = PriorMod[order(PriorMod$id),]
  #compare based on diffs since raven rounding rules are unclear.
  
  colnames(EditMod)[1:6]=c("start_time","end_time","low_freq","high_freq","start_file","end_file")
  colnames(PriorMod)[1:6]=c("start_time","end_time","low_freq","high_freq","start_file","end_file") #add, test

  diffs = EditMod[,1:4]-PriorMod[,1:4]

  #.1 precision on frequency, .01 precision on time
  diffs[,3:4]= abs(diffs[,3:4])>0.1
  diffs[,1:2]= abs(diffs[,1:2])>0.01

  #if diffs are false, and the file end time has been modified, revert to original.
  #fix for the behavior where raven uses exact samples and rewrites end file (I round to 2 decimal for sf on db)
  ids_rev_et =EditMod[which(rowSums(diffs,na.rm=TRUE)==0 & EditMod$EndFile != PriorMod$EndFile),"id"]
  EditMod[which(EditMod$id %in% ids_rev_et),"end_file"] = PriorMod[which(PriorMod$id %in% ids_rev_et),"end_file"]

  #reduce this set to only rows which were modified
  testdf = EditMod[,5:length(EditMod)]!=PriorMod[,5:length(PriorMod)]
  testdf = cbind(diffs,testdf)

  if(nrow(EditMod)>0){
    
    #lookup file names
    filelookup =lookup_from_match(con,'soundfiles',unique(c(EditMod$start_file,EditMod$end_file)),"name")
    
    EditMod$start_file = filelookup$id[match(EditMod$start_file,filelookup$name)]
    EditMod$end_file = filelookup$id[match(EditMod$end_file,filelookup$name)]
    
    #remove analyst field so that it defaults to current user when submitted.
    #stealth change, want this as normal behavior. 
    EditMod$analyst=as.integer(dbFetch(dbSendQuery(con,"SELECT id FROM personnel WHERE personnel.pg_name = current_user"))$id)
    
    colsums= colSums(testdf,na.rm=TRUE)
    sums = rowSums(testdf,na.rm=TRUE)
    
    #subset to only changed rows /columns
    EditMod = EditMod[sums>0,c('id','procedure',names(colsums[which(colsums>0)]))]
    testdf_reduce = testdf[sums>0,(names(colsums)=="id" |names(colsums)=="procedure" | colsums>0)]
    
    #new in 1-3: also check to see what columns have been modified. remove those which haven't
    
    #check changed columns by procedures. If EditMod contains a procedure which is a detector deployment
    #but disallowed columns change, abort the operation. 
    bn_added = 0 #running count of bin negatives added
    bn_removed = 0 
    all_bn_proc = c()
    if(nrow(EditMod)>0){
      for(i in 1:length(unique(EditMod$procedure))){
        
        EditMod_proc = EditMod[which(EditMod$procedure==unique(EditMod$procedure)[i]),]
        
        testdf_reduce_proc = testdf_reduce[which(EditMod$procedure==unique(EditMod$procedure)[i]),]
        
        check_row = procedure_assumption_lookup[which(procedure_assumption_lookup$id == unique(EditMod$procedure)[i]),]
        
        if(nrow(check_row)==0){
          stop("Procedure not found in database. Aborting. Enter procedure on db to allow for assumption checking")
        }
        
        if(check_row$id %in% c(6,7,8,9)){
          stop("Editing tools not yet supported for original fin procedure due to more complex assumptions. Aborting")
        }
        
        
        #perform this if procedure fits assumption. 
        if(!is.null(check_row$negative_bin) & !is.na(check_row$negative_bin)){
          
          all_bn_proc = c(all_bn_proc,as.integer(check_row$id))
          #this means that this is a deployment. if this is the case, changes here will affect bin negatives. 
          
          #determine which of these specifically had the label changed. 
          
          EditMod_proc_labelchange =EditMod_proc[testdf_reduce_proc$label,]
          
          
          #pseudo: using the changed detections, load in all affected bins, all existing detections, and those
          #detection labels. 
          
          affected_bins = dbFetch(dbSendQuery(con,paste("SELECT detections.id,detections.label,bins.* FROM detections JOIN bins_detections ON detections.id = bins_detections.detections_id JOIN bins ON 
                                                        bins.id = bins_detections.bins_id WHERE bins.id IN 
                                                        (SELECT DISTINCT bins_detections.bins_id FROM bins JOIN bins_detections ON bins.id = bins_detections.bins_id 
                                                        WHERE bins_detections.detections_id IN (",
                                                        paste(EditMod_proc_labelchange$id,sep="",collapse = ","),
                                                        ") AND type = ",check_row$negative_bin,")",
                                                        " AND detections.procedure = ",unique(EditMod$procedure)[i],sep="")))
          
          #fill in the new labels for considered detections. using this table, make another table which 
          #represents each bin, and whether it contains a 1 detection or not. 
          
          affected_bins$id = as.integer(affected_bins$id)
          affected_bins$id..3 = as.integer(affected_bins$id..3)
          
          affected_bins$label= EditMod_proc_labelchange[match(affected_bins$id,EditMod_proc_labelchange$id),"label"]
          
          affected_bins_0bins_temp = affected_bins[which(affected_bins$label %in% c(0,1)),]
          affected_bins_0bins = aggregate(affected_bins_0bins_temp$label,by=list(affected_bins_0bins_temp$id..3),mean)
          
          #this indicates if any 1s are present > 0 or not, 
          
          affected_bins_0bins$over0 = affected_bins_0bins$x>0
          
          if(any(affected_bins$label==20,na.rm=TRUE)){
            
            stop("new case, write out the rest of this method before continuing")
            
            #I will need to compare here
            
            #if the 1 bins have a matching bin negative, delete it. If the 0 bins have a matching bin negative, do nothing. 
            #if the 0 bins don't have a matching bin negative, add one. 
            
          }else{
            
            #this is a simpler case (doesn't need to be distinct from initial condition after I write it out)
            
            #if no 20s exist, I just submit each bin which has a 0 
            if(any(!affected_bins_0bins$over0)){
              
              submitbins = affected_bins_0bins$Group.1[which(!affected_bins_0bins$over0)]
              submitbins_frame = affected_bins[which(affected_bins$id..3 %in% submitbins),]
              adddets = data.frame(submitbins_frame$seg_start,submitbins_frame$seg_end,0,max(PriorData[which(PriorData$procedure==unique(EditMod$procedure)[i]),"HighFreq"]),
                                   submitbins_frame$soundfiles_id,submitbins_frame$soundfiles_id,NA,"",unique(EditMod$procedure)[i],20,PriorData[which(PriorData$procedure==unique(EditMod$procedure)[i]),"signal_code"][1],
                                   2)
              
              colnames(adddets) = c("start_time","end_time","low_freq","high_freq","start_file","end_file",
                                    "probability","comments","procedure","label","signal_code","strength")
              
              #add the new bin negatives
              out = dbAppendTable(con,'detections',adddets)
              
              bn_added = bn_added + out
              
            }
            
  
            
          }
          
        }
  
        
      }
      
      table_update(con,'detections',EditMod)

    }
    
    print(paste(nrow(EditMod),"rows UPDATED. Bin negatives inserted:",bn_added,". Bin negatives deleted:",bn_removed,". Over the following procedures:",paste(all_bn_proc,sep="",collapse=",")))
    
    operations[[1]]=paste(nrow(EditMod),"rows UPDATED. Bin negatives inserted:",bn_added,". Bin negatives deleted:",bn_removed,". Over the following procedures:",paste(all_bn_proc,sep="",collapse=","))

    affected_ids_total = c(affected_ids_total,EditMod$id)

  }else{
    operations[[1]]="0 records UPDATED. No bin negatives modified."
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
  
  print(paste(nrow(new_data),"rows INSERTED"))

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
  
  print(paste(length(del_keys),"rows DELETED"))
  
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
                AND effort_procedures.completed = 'y'
                AND detections.id IN (",paste(affected_ids_total,collapse=",",sep=""),")",sep="")
  
  query <- gsub("[\r\n]", "", query)
  
  to_redo = dbFetch(dbSendQuery(con,query))
  
  #also, pull from prior data in case procedures or signal types changed. 
  if(any(PriorData$id %in% affected_ids_total)){
    
    priordata_affected = PriorData[PriorData$id %in% affected_ids_total,c("procedure","signal_code","id")]
    
    unq_pdaffected = unique(priordata_affected[,c("procedure","signal_code")])
    
    unq_pdaffected_vec = c(unq_pdaffected$procedure,unq_pdaffected$signal_code)
    
    comb_seq = c()
    for(f in 1:(length(unq_pdaffected_vec)/2)){
      comb_seq = c(comb_seq,paste("(",unq_pdaffected_vec[f],",",unq_pdaffected_vec[f+length(unq_pdaffected_vec)/2],")",sep=""))
    }
    
    prior_data_affected_query = paste("SELECT DISTINCT effort.name, detections.id
                FROM detections JOIN bins_detections ON bins_detections.detections_id = detections.id
                JOIN bins ON bins.id = bins_detections.bins_id JOIN bins_effort ON bins_effort.bins_id
                = bins.id JOIN effort ON effort.id = bins_effort.effort_id JOIN effort_procedures ON 
                effort_procedures.effort_id = effort.id  WHERE effproc_assumption = 'i_neg'
                AND effort_procedures.completed = 'y' AND (effort_procedures.procedures_id,effort_procedures.signal_code)
                IN (",paste(comb_seq,collapse=","),")
                AND detections.id IN (",paste(affected_ids_total,collapse=",",sep=""),")",sep="")
    
    prior_data_affected_query <- gsub("[\r\n]", "", prior_data_affected_query)
    
    prior_data_affected_res = dbFetch(dbSendQuery(con,prior_data_affected_query))
    
    prior_data_affected_res$id = as.integer(prior_data_affected_res$id)
    
    to_redo_prior_data = merge(priordata_affected,prior_data_affected_res)
    
    to_redo_prior_data$id = NULL
    
    to_redo_prior_data = unique(to_redo_prior_data)
    
    unq_pdaffected2 = unique(to_redo_prior_data[,c("procedure","signal_code",'name')])
    
    unq_pdaffected_vec2 = c(unq_pdaffected2$procedure,unq_pdaffected2$signal_code,unq_pdaffected2$name)
    
    comb_seq2 = c()
    for(f in 1:(length(unq_pdaffected_vec2)/3)){
      comb_seq2 = c(comb_seq2,paste("(",unq_pdaffected_vec2[f],",",unq_pdaffected_vec2[f+length(unq_pdaffected_vec2)/3],",'",unq_pdaffected_vec2[f+(length(unq_pdaffected_vec2)/3)*2],"')",sep=""))
    }
    
    #confirm that all of the new possible procedure/sigcode/fg combinations is actually present in 
    #effort_procedures
    
    query_check = paste("SELECT DISTINCT procedures_id,signal_code,name FROM effort JOIN effort_procedures
                        ON effort.id = effort_procedures.effort_id WHERE effproc_assumption = 'i_neg'
                        AND effort_procedures.completed = 'y' AND (effort_procedures.procedures_id,effort_procedures.signal_code,effort.name)
                        IN (",paste(comb_seq2,collapse=","),")")
    
    query_check_query <- gsub("[\r\n]", "", query_check)
    
    query_check_res = dbFetch(dbSendQuery(con,query_check_query))
    
    query_check_res$procedures_id = as.integer(query_check_res$procedures_id)
    query_check_res$signal_code = as.integer(query_check_res$signal_code)
    
    to_redo = unique(rbind(unique(query_check_res),to_redo))
  }
  
  
  if(nrow(to_redo)>0){
    
    for(i in 1:nrow(to_redo)){
      
      i_neg_update(con,to_redo$name[i],as.integer(to_redo$procedure)[i],as.integer(to_redo$signal_code)[i])
      
    }
  }

  
  operations$i_neg_message = "i_neg sections recalculated:" 
  operations$i_neg_table = paste(to_redo,collapse=",")

}

writeLines(do.call("cbind",operations),paste(resultPath,'receipt.txt',sep="/"))



#attempt to find changes in data
#Attempt to submit a modification of the data with ids. Data w/o ids will be assumed to be new data.
#check that no records with Type of Det or SC have modified timestamp.

