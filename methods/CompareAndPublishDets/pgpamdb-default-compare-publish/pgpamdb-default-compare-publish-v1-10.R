check_sf_dups = function(conn,sfs){
  
  dups = dbFetch(dbSendQuery(conn,paste("SELECT COUNT( DISTINCT soundfiles.data_collection_id),soundfiles.name FROM soundfiles WHERE soundfiles.name IN ('",paste(sfs,collapse="','",sep=""),"') GROUP BY soundfiles.name HAVING COUNT(*) > 1 ",sep="")))
  
  if(nrow(dups)>0){
    return(TRUE)
  }else{
    return(FALSE)
  }
  
  
  
}


library(pgpamdb)
library(DBI)

args="C:/Apps/INSTINCT/Cache/915396/426232/843987/507076/378231 C:/Apps/INSTINCT/Cache/915396/426232/843987 C:/Apps/INSTINCT/Cache/915396/426232/843987/507076/378231/157859 n 38 Eric pgpamdb-default-compare-publish-v1-10"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
con=pamdbConnect(dbname,keyscript,clientkey,clientcert)

EditDataPath <-args[1]
PriorDataPath <- args[2]
resultPath <- args[3]
assume_full_review<-args[4]
insert_ok<-as.integer(args[5])
on_behalf_of = args[6]

PriorData<-read.csv(paste(PriorDataPath,"DETx.csv.gz",sep="/"))
EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))

#self for current pg user
if(on_behalf_of!= "self"){
  
  analyst = as.integer(dbFetch(dbSendQuery(con,paste("SELECT id FROM personnel WHERE personnel.pg_name ='",on_behalf_of,"'",sep="")))$id)
  if(length(analyst)==0){
    stop("unable to match analyst from db pg_name")
  }
}else{
  analyst = as.integer(dbFetch(dbSendQuery(con,"SELECT id FROM personnel WHERE personnel.pg_name = current_user"))$id)
}

#v1-6 stealth change: if assuming full review, change the analyst on EditData to the analyst of the current session. 
if(assume_full_review=="y"){
  EditData$analyst = analyst
}

#bugfix: if raven conversion resulted in case where a detection did not have an endfile, remove from 
#editdata and remove id from priordata (ignore it)

if(any(EditData$EndFile=="")){
  ids = EditData[which(EditData$EndFile==""),"id"]
  EditData= EditData[-which(EditData$EndFile==""),]
  if(any(!is.na(ids))){
    ids = ids[!is.na(ids)]
    PriorData = PriorData[-which(PriorData$id==ids),]
  }
  
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

#EditData[which(is.na(EditData$label)),'label']=99 #need to also set type to integer

#keep a vector of all the procedures. Use this vector to check assumptions and make sure
#that modifications outside of allowed rules (such as, moving/resizing a machine generated detection,
#or adding/removing machine generated detections.)

allprocedures = unique(EditData$procedure)

procedure_assumption_lookup = dbFetch(dbSendQuery(con,paste("SELECT * FROM procedures WHERE id IN (",
                                                  paste(allprocedures,sep="",collapse=","),")",sep="")))

db_columns = names(dbFetch(dbSendQuery(con,"SELECT * FROM detections LIMIT 0")))

#append on names that haven't been converted yet from detx 
db_columns = c(db_columns,"LowFreq", "HighFreq","StartFile","EndFile","StartTime","EndTime","data_collection_id")

allcols = unique(c(colnames(PriorData),colnames(EditData)))
non_db_columns = allcols[-which(allcols %in% db_columns)]

#check columns are the same.
#remove columns which are irrelevant:
if(any(c("changed_by","uploaded_by",non_db_columns) %in% colnames(PriorData))){
  PriorData = PriorData[,colnames(PriorData)[-which(colnames(PriorData) %in% c("changed_by","uploaded_by",non_db_columns))]]
}
if(any(c("changed_by","uploaded_by",non_db_columns) %in% colnames(EditData))){
  EditData = EditData[,colnames(EditData)[-which(colnames(EditData) %in% c("changed_by","uploaded_by",non_db_columns))]]
}


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
#if no data in prior, skip check
if(nrow(PriorData)>0){
  new_data = EditData[-which(EditData$id %in% PriorData$id),]
}else{
  new_data = EditData
}
del_keys= PriorData$id[-which(PriorData$id %in% EditData$id)]

#do a check on new and delete data up here to make sure assumption errors are caught before data modification.

if(nrow(new_data)>0){
  
  #let's get more precise about what is and what is not allowed: 
  
  #1. can specify a different signal code for a given detection. It will create a new detection
  #which retains the original id. So, check that for any new detections, that for each procedure
  #match the original id is still represented. 
  #1. probably fine to just throw in a random detection as desired. Ignore cases where procedure is
  #not the same in a new detection. 
  
  if(any(!is.na(procedure_assumption_lookup[which(as.integer(procedure_assumption_lookup$id) %in% new_data$procedure),"negative_bin"]))){
    
    for(m in unique(new_data$procedure)){
      
      #if any original id in new (of procedure m) not in original id, that's a problem- reject case
      if(any(!(new_data[which(new_data$procedure==m),"original_id"] %in% PriorData$original_id)) & m!=insert_ok){
        stop(paste("Attempted to insert a new detection under current labeling workflow procedure:",m,". Case not allowed"))
      }
      
    }
    
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

#find the unique procedures present in prior_data. If none of them are i_neg, don't worry about
#relaculating i_neg later for edited detections (very relevant to avoid in detection review workflows)

i_neg_proc_prior = dbFetch(dbSendQuery(con,paste("SELECT * FROM effort_procedures WHERE procedures_id IN (",paste(unique(PriorData$procedure),collapse=",",sep=","),") AND effproc_assumption = 'i_neg'",sep="")))

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
  
  #stealth change- add diffs comparison for probability as well. 
  
  if('probability' %in% colnames(EditMod)){
    prob_diffs = EditMod$probability-PriorMod$probability
    #really, probability shouldn't ever be changed- would imply a different procedure. 
    #what we'll do, is scan for any large difference, and return an error, otherwise, set diffs to False. 
    prob_diffs= abs(prob_diffs)>0.01
    if(any(prob_diffs,na.rm=TRUE)){
      stop("modified probability detected- not valid for a consistent procedure. stopping...")
    }else{
      testdf$probability=prob_diffs
    }
  }
  

  if(nrow(EditMod)>0){
    
    #lookup file names
    
    if("data_collection_id" %in% colnames(EditMod)){
      
      FGred = rbind(setNames(data.frame(EditMod$data_collection_id,EditMod$start_file),c("dcid","sf")),setNames(data.frame(EditMod$data_collection_id,EditMod$end_file),c("dcid","sf")))
      FGred = FGred[which(!duplicated(FGred)),]
      
      colnames(FGred)=c("data_collection.id","soundfiles.name")
      
      print('debug:1')
      
      sfs = table_dataset_lookup(con,
                                 "SELECT DISTINCT ON (soundfiles.name,data_collection.id) soundfiles.id,soundfiles.name,data_collection.id FROM soundfiles JOIN data_collection ON soundfiles.data_collection_id = data_collection.id",
                                 FGred,
                                 c("integer","character varying"))
      
      colnames(sfs) = c("soundfiles_id","soundfiles_name","data_collection_id")
      sfs$soundfiles_id = as.integer(sfs$soundfiles_id)
      sfs$data_collection_id = as.integer(sfs$data_collection_id)
      
      EditMod$start_file = merge(setNames(EditMod[,c("start_file","data_collection_id")],c("soundfiles_name","data_collection_id")),sfs)$soundfiles_id
      EditMod$end_file = merge(setNames(EditMod[,c("end_file","data_collection_id")],c("soundfiles_name","data_collection_id")),sfs)$soundfiles_id
      
      EditMod$data_collection_id = NULL
    }else{
      
      if(check_sf_dups(con,unique(c(EditMod$start_file,EditMod$end_file)))){
        stop("duplicate soundfiles present in insert- need to revise process to correctly distinguish duplicate file names. aborting..")
      }
      
      filelookup =lookup_from_match(con,'soundfiles',unique(c(EditMod$start_file,EditMod$end_file)),"name")
      
      EditMod$start_file = filelookup$id[match(EditMod$start_file,filelookup$name)]
      EditMod$end_file = filelookup$id[match(EditMod$end_file,filelookup$name)]
      
    }
    
    #v1-7
    #EditMod$analyst=analyst
    
    colsums= colSums(testdf,na.rm=TRUE)
    sums = rowSums(testdf,na.rm=TRUE)
    
    #subset to only changed rows /columns
    #EditMod = EditMod[sums>0,c('id','procedure',names(colsums[which(colsums>0)]))]
    #testdf_reduce = testdf[sums>0,(names(colsums)=="id" |names(colsums)=="procedure" | colsums>0)]
    
    EditMod = EditMod[sums>0,]
    testdf_reduce = testdf[sums>0,]
    
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
          bin_length = dbFetch(dbSendQuery(con,paste("SELECT length_seconds FROM bin_type_codes WHERE id =",check_row$negative_bin)))$length_seconds
          #optimization: make a subset of EditMod_proc_labelchange which only contains 
          #one representative detection per bin. For this, just take ceiling of start time /  bin_length
          
          EditMod_proc_labelchange$bin_unqst = ceiling(EditMod_proc_labelchange$start_time/bin_length)
          EditMod_proc_labelchange$bin_unqend = ceiling(EditMod_proc_labelchange$end_time/bin_length)
          
          EditMod_proc_labelchange$bin_unq= (EditMod_proc_labelchange$bin_unqst + EditMod_proc_labelchange$bin_unqend) / 2
          
          #these are ids representing unique bins. edge ids included for convenience, some may technically be duplicated. 
          #may not even be necessary since we're including start and end file in duplicated
          unq_ids = EditMod_proc_labelchange[which(!duplicated(data.frame(EditMod_proc_labelchange$start_file,EditMod_proc_labelchange$end_file,EditMod_proc_labelchange$bin_unq))),"id"]
         #print('1')
          print('debug:2')
          affected_bins = dbFetch(dbSendQuery(con,paste("SELECT detections.id,detections.label,detections.probability,bins.* FROM detections JOIN bins_detections ON detections.id = bins_detections.detections_id JOIN bins ON 
                                                        bins.id = bins_detections.bins_id WHERE bins.id IN 
                                                        (SELECT DISTINCT bins_detections.bins_id FROM bins JOIN bins_detections ON bins.id = bins_detections.bins_id 
                                                        WHERE bins_detections.detections_id IN (",
                                                        paste(unq_ids,sep="",collapse = ","),
                                                        ") AND type = ",check_row$negative_bin,")",
                                                        " AND detections.procedure = ",unique(EditMod$procedure)[i],sep="")))
          print('debug:3')
          #print('2')
          #fill in the new labels for considered detections. using this table, make another table which 
          #represents each bin, and whether it contains a 1 detection or not. 
          
          affected_bins$id = as.integer(affected_bins$id)
          affected_bins$id..4 = as.integer(affected_bins$id..4)
          
          affected_bins$newlabel= EditMod_proc_labelchange[match(affected_bins$id,EditMod_proc_labelchange$id),"label"]
          
          affected_bins_0bins_temp = affected_bins[which(affected_bins$newlabel %in% c(0,20,1,21)),]
          affected_bins_0bins_temp[which(affected_bins_0bins_temp$newlabel==20),"newlabel"]=0
          affected_bins_0bins_temp[which(affected_bins_0bins_temp$newlabel==21),"newlabel"]=1
          if(nrow(affected_bins_0bins_temp)>0){
            affected_bins_0bins = aggregate(affected_bins_0bins_temp$newlabel,by=list(affected_bins_0bins_temp$id..4),mean)
            affected_bins_0bins$over0 = affected_bins_0bins$x>0
            
          }else{
            affected_bins_0bins=affected_bins_0bins_temp
            affected_bins_0bins$over0 = numeric(0)
          }
            
          
          #this indicates if any 1s are present > 0 or not, 
          
          
          if(any(affected_bins$label==20 & is.na(affected_bins$probability),na.rm=TRUE)){
            
            bn_to_rem  =c()
            
            #loop through each affected bin with '20'. match id to editmod_proc_labelchange. if going from 20 -> 1/21,
            #then need to remove 20. Otherwise, ignore. add case is covered below. 
            
            ab_bin_negs = affected_bins[which(affected_bins$label==20 & is.na(affected_bins$probability)),]
            
            for(p in 1:nrow(ab_bin_negs)){
              
              alldets_in_bin = affected_bins[which(affected_bins$id..4==ab_bin_negs[p,"id..4"]),]
              
              newlabs = unique(EditMod_proc_labelchange[which(EditMod_proc_labelchange$id %in% alldets_in_bin$id),"label"])
              
              if(any(newlabs %in% c(1,21))){
                
                bn_removed = bn_removed + 1
                
                #delete bin negative
                bn_to_rem = c(bn_to_rem,ab_bin_negs$id[p])
              }
              
            }
            
            #remove all bn_to_rem
            if(length(bn_to_rem)>0){
              print('debug:4')
              table_delete(con,'detections',bn_to_rem,hard_delete = TRUE)
            }
            
          }
            
            #this is a simpler case (doesn't need to be distinct from initial condition after I write it out)
            
            #if no 20s exist, I just submit each bin which has a 0 
     
            
          if(any(!affected_bins_0bins$over0)){
            
            submitbins = affected_bins_0bins$Group.1[which(!affected_bins_0bins$over0)]
            submitbins_frame = affected_bins[which(affected_bins$id..4 %in% submitbins),]
            adddets = data.frame(submitbins_frame$seg_start,submitbins_frame$seg_end,0,max(PriorData[which(PriorData$procedure==unique(EditMod$procedure)[i]),"HighFreq"]),
                                 submitbins_frame$soundfiles_id,submitbins_frame$soundfiles_id,NA,"",unique(EditMod$procedure)[i],20,PriorData[which(PriorData$procedure==unique(EditMod$procedure)[i]),"signal_code"][1],
                                 2)
            
            colnames(adddets) = c("start_time","end_time","low_freq","high_freq","start_file","end_file",
                                  "probability","comments","procedure","label","signal_code","strength")
            
            if(any(duplicated(adddets))){
              adddets = adddets[-which(duplicated(adddets)),]
            }
            
            if(check_sf_dups(con,unique(c(adddets$start_file,adddets$end_file)))){
              stop("duplicate soundfiles present in insert- need to revise process to correctly distinguish duplicate file names. aborting..")
            }
            
            #v1-9 stealth change
            #set analyst to analyst
            adddets$analyst = analyst
            
            print('debug:5')
            
            #add the new bin negatives
            out = dbAppendTable(con,'detections',adddets)
            
            bn_added = bn_added + out
            
          }
          
        }
  
        
      }
      
      #pare down edit mod to only necessary columns
      EditMod = EditMod[,c('id','procedure',names(colsums[which(colsums>0)]))]
      
      #v1-9 stealth change
      #set analyst to analyst
      EditMod$analyst = analyst
      
      print('debug:6')
      
      #update editmod
      #chunk if too big: currently breaks db shared memory over a certain size
      if(nrow(EditMod)>10000000){
        chunk_size = 8000000
        for(i in 1:ceiling(nrow(EditMod)/chunk_size)){
          start = (1+((i-1)*chunk_size))
          end = i*chunk_size
          if(end>nrow(EditMod)){
            end = nrow(EditMod)
          }
          
          print(start)
          print(end)
          #print(EditMod[start:end,])
          table_update(con,'detections',EditMod[start:end,])
        }
      
      }else{
        table_update(con,'detections',EditMod)
      }
      
      
      print('debug:7')

    }
    
    print(paste(nrow(EditMod),"rows UPDATED. Bin negatives inserted:",bn_added,". Bin negatives deleted:",bn_removed,". Over the following procedures:",paste(all_bn_proc,sep="",collapse=",")))
    
    operations[[1]]=paste(nrow(EditMod),"rows UPDATED. Bin negatives inserted:",bn_added,". Bin negatives deleted:",bn_removed,". Over the following procedures:",paste(all_bn_proc,sep="",collapse=","))

    
    #discard adding to affected ids total those from procedures not represented in i_neg- implies det_review and not worth
    #the query later. 
    
    #print this out so I can check types after 1st run
    print(unique(EditMod$procedures))
    typeof(unique(EditMod$procedures))
    print(i_neg_proc_prior$procedures_id)
    typeof(i_neg_proc_prior$procedures_id)
    
    affected_ids_total = c(affected_ids_total,EditMod[which(EditMod$procedure %in% i_neg_proc_prior$procedures_id),"id"])

  }else{
    operations[[1]]="0 records UPDATED. No bin negatives modified."
  }
}

if(nrow(new_data)>0){

  colnames(new_data)[1:6]=c("start_time","end_time","low_freq","high_freq","start_file","end_file")

  if("data_collection_id" %in% colnames(new_data)){
    
    FGred = rbind(setNames(data.frame(new_data$data_collection_id,new_data$start_file),c("data_collection.id","soundfiles.name")),setNames(data.frame(new_data$data_collection_id,new_data$end_file),c("data_collection.id","soundfiles.name")))
    
    FGred$data_collection.id = as.integer(FGred$data_collection.id)
    
    FGred = FGred[which(!duplicated(FGred)),]
    
    if(any(is.na(FGred$data_collection.id))){
      FGred = FGred[-which(is.na(FGred$data_collection.id)),]
    }
    
    colnames(FGred)=c("data_collection.id","soundfiles.name")
    
    sfs = table_dataset_lookup(con,
                               "SELECT DISTINCT ON (soundfiles.name,data_collection.id) soundfiles.id,soundfiles.name,data_collection.id FROM soundfiles JOIN data_collection ON soundfiles.data_collection_id = data_collection.id",
                               FGred,
                               c("integer","character varying"))
    
    colnames(sfs) = c("soundfiles_id","soundfiles_name","data_collection_id")
    sfs$soundfiles_id = as.integer(sfs$soundfiles_id)
    sfs$data_collection_id = as.integer(sfs$data_collection_id)
    
    new_data$start_file = merge(setNames(new_data[,c("start_file","data_collection_id")],c("soundfiles_name","data_collection_id")),sfs)$soundfiles_id
    new_data$end_file = merge(setNames(new_data[,c("end_file","data_collection_id")],c("soundfiles_name","data_collection_id")),sfs)$soundfiles_id
    
    new_data$data_collection_id = NULL
  }else{
    
    if(check_sf_dups(con,unique(c(new_data$start_file,new_data$end_file)))){
      stop("duplicate soundfiles present in insert- need to revise process to correctly distinguish duplicate file names. aborting..")
    }
    
    filelookup =lookup_from_match(con,'soundfiles',unique(c(new_data$start_file,new_data$end_file)),"name")
    
    new_data$start_file = filelookup$id[match(new_data$start_file,filelookup$name)]
    new_data$end_file = filelookup$id[match(new_data$end_file,filelookup$name)]
    
  }

  #remove modified and id fields so that it defaults to when it is submitted.
  new_data$modified=NULL
  new_data$id = NULL
  new_data$original_id=NULL
  new_data$status=NULL
  new_data$analyst = analyst

  dbAppendTable(con,'detections',new_data)
  
  print(paste(nrow(new_data),"rows INSERTED"))

  operations[[2]]=paste(nrow(new_data),"rows INSERTED")
  
  #look up newly submitted detections to find ids. 
  
  #put a check- if a procedure has been used for i_neg effproc, do the more expensive lookup below. If
  #not, just do not add to affected_ids_total
  
  query = paste("SELECT DISTINCT procedures_id FROM effort_procedures WHERE effproc_assumption = 'i_neg'")
  
  recalc_i_neg = dbFetch(dbSendQuery(con,query))
  
  newdata_sub = new_data[which(new_data$procedure %in% as.integer(recalc_i_neg$procedures_id)),]
  
  if(nrow(newdata_sub)>0){
    ids = table_dataset_lookup(con,"SELECT id FROM detections",
                               newdata_sub[,c("start_time","end_time","low_freq","high_freq","start_file","end_file","procedure","signal_code","label","strength")],
                               c("DOUBLE PRECISION","DOUBLE PRECISION","DOUBLE PRECISION","DOUBLE PRECISION","integer","integer","integer","integer","integer","integer"))$id
    
    
    affected_ids_total = c(affected_ids_total,as.integer(ids))
  }

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
  
  #change! don't use .id IN (), use VALUES
  query = paste("SELECT DISTINCT effort.name, detections.procedure, detections.signal_code
                FROM detections JOIN bins_detections ON bins_detections.detections_id = detections.id
                JOIN bins ON bins.id = bins_detections.bins_id JOIN bins_effort ON bins_effort.bins_id
                = bins.id JOIN effort ON effort.id = bins_effort.effort_id JOIN effort_procedures ON 
                (effort_procedures.effort_id = effort.id AND detections.procedure = effort_procedures.procedures_id 
                AND detections.signal_code = effort_procedures.signal_code) JOIN (VALUES (",paste(affected_ids_total,collapse="),(",sep=""),")) 
                as v(id) ON detections.id = v.id WHERE effproc_assumption = 'i_neg'
                AND effort_procedures.completed = 'y'",sep="")
  
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
    
    #change! don't use .id IN (), use VALUES
    prior_data_affected_query = paste("SELECT DISTINCT effort.name, detections.id
                FROM detections JOIN bins_detections ON bins_detections.detections_id = detections.id
                JOIN bins ON bins.id = bins_detections.bins_id JOIN bins_effort ON bins_effort.bins_id
                = bins.id JOIN effort ON effort.id = bins_effort.effort_id JOIN effort_procedures ON 
                effort_procedures.effort_id = effort.id JOIN (VALUES (",paste(affected_ids_total,collapse="),(",sep=""),")) 
                as v(id) ON detections.id = v.id WHERE effproc_assumption = 'i_neg'
                AND effort_procedures.completed = 'y' AND (effort_procedures.procedures_id,effort_procedures.signal_code)
                IN (",paste(comb_seq,collapse=","),")",sep="")
    
    prior_data_affected_query <- gsub("[\r\n]", "", prior_data_affected_query)
    
    prior_data_affected_res = dbFetch(dbSendQuery(con,prior_data_affected_query))
    
    #stealth fix 1-5: test if not relevant 
    if(nrow(prior_data_affected_res)>0){
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
      
      #stealth change- make the column names line up 
      colnames(query_check_res)[which(colnames(query_check_res)=="procedures_id")]="procedure"
      
      to_redo = unique(rbind(unique(query_check_res),to_redo))
    }
    
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

