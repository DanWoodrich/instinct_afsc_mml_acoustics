library(pgpamdb)
library(DBI)


args="D:/Cache/110569/767967/467090/936111/453729/772088 D:/Cache/110569/767967/467090 D:/Cache/110569/767967/467090/809666 y 33 peaks-assoc-simple-v1-1"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

#source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
#con=pamdbConnect("poc_v2",keyscript,clientkey,clientcert)

peak_data= read.csv(paste(args[1],"/DETx.csv.gz",sep=""))
assoc_data= read.csv(paste(args[2],"/DETx2.csv.gz",sep=""))
resultPath = args[3]
allow_ambiguous_label <-args[4]
new_procedure = args[5]

if(any(assoc_data$peak_assoc_id==-1,na.rm=TRUE)){
  is_review = TRUE
  held_out_data = assoc_data[which(assoc_data$peak_assoc_id==-1),]
  assoc_data = assoc_data[-which(assoc_data$peak_assoc_id==-1),]
}else{
  is_review=FALSE
}

present99 = any(is.na(peak_data$label))

if(allow_ambiguous_label=='n'& present99){
  stop("Ambiguous label (99) present in the verified dataset and allow_ambiguous_label is set to 'n'. Aborting..")
  
}else if(present99){
  
  peak_data$label[is.na(peak_data$label)]=99
}

#print(head(peak_data))
#print(head(assoc_data))

lab_lookup = data.frame(c(0,1,2,99),c(20,21,22,99))
colnames(lab_lookup)<- c("labels","prot_labels")

for(i in 1:nrow(peak_data)){
  
  #look for na value and make sure to change to 99
  
  
  #if signal code is changed, copy the affected data and treat it as new detections of another signal code.
  if(peak_data[i,"signal_code"]!=assoc_data$signal_code[1]){
    
    add_data = assoc_data[which(assoc_data$peak_assoc_id==peak_data[i,"id"]),] #copy the assoc data. 
    
    add_data$signal_code = peak_data[i,"signal_code"]
    add_data$label = lab_lookup[match(peak_data[i,"label"],lab_lookup$labels),"prot_labels"]
    add_data[which(add_data$id==peak_data[i,"id"]),"label"] = peak_data[i,"label"]
    
    add_data$id = NA #this will be interpreted as a new detection later
    
    assoc_data[which(assoc_data$peak_assoc_id==peak_data[i,"id"]),"label"]= 20
    
    #set focal label to human label
    assoc_data[which(assoc_data$id==peak_data[i,"id"]),"label"] = 0
    
    assoc_data = rbind(assoc_data,add_data)
    
  }else{
    
    #set assoc data label = to that of peak_data
    assoc_data[which(assoc_data$peak_assoc_id==peak_data[i,"id"]),"label"]= lab_lookup[match(peak_data[i,"label"],lab_lookup$labels),"prot_labels"]
    
    #set focal label to human label
    assoc_data[which(assoc_data$id==peak_data[i,"id"]),"label"] = peak_data[i,"label"]
  }

}



if(is_review){
  
  if(length(unique(held_out_data$procedure))>1){
    stop("ambiguous procedures present- only pull in procedures which you would like to update")
  }
  review_procedure = held_out_data$procedure[1]
  
  #recreate the original dets
  assoc_data_copy = assoc_data
  assoc_data_copy = assoc_data_copy[-which(is.na(assoc_data_copy$probability)),]
  assoc_data_copy$label = 99
  
  assoc_data_negs = assoc_data[which(assoc_data$label==20 & is.na(assoc_data$probability)),]
  
  if(new_procedure!=review_procedure){
    #data will be inserted instead of modifed- rbind back existing data. 
    #make sure original data is also present, so that it isn't deleted 
    #make sure negatives are present in the newly submitted data as well for that protocol. 
    
    assoc_data$procedure = new_procedure
    assoc_data$id = NA
    
    #since we are inserting new data instead of update, won't hit the part in compare and publish dets which automatically
    #provides new assumed negs for 0 bins. Should put it here
    
    #need:
    #fg #unfortunately, need to query db?
    #bin size (should be able to infer from previous)]
    
    source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
    con=pamdbConnect(dbname,keyscript,clientkey,clientcert)
    
    #how to find fg? Can use assoc data to pull in bins corresponding to each det. Since new bins negs
    #will only concern bins which contain dets, can use assoc_data_copy, and subset this to only
    #unq dets for bin lookup (similar to what's done in compare/publish)
    
    proc_details = dbFetch(dbSendQuery(con,paste("SELECT * FROM procedures WHERE id =",new_procedure)))
    
    bin_length = dbFetch(dbSendQuery(con,paste("SELECT length_seconds FROM bin_type_codes WHERE id =",proc_details$negative_bin)))$length_seconds
    
    assoc_data_copy$bin_unqst = ceiling(assoc_data_copy$StartTime/bin_length)
    assoc_data_copy$bin_unqend = ceiling(assoc_data_copy$EndTime/bin_length)
    
    assoc_data_copy$bin_unq= (assoc_data_copy$bin_unqst + assoc_data_copy$bin_unqend) / 2
    
    unq_ids = assoc_data_copy[which(!duplicated(data.frame(assoc_data_copy$StartFile,assoc_data_copy$EndFile,assoc_data_copy$bin_unq))),"id"]
    
    all_bins = dbFetch(dbSendQuery(con,paste("SELECT DISTINCT bins.*,soundfiles.name FROM bins JOIN bins_detections ON bins.id = bins_detections.bins_id JOIN detections ON detections.id = bins_detections.detections_id
                                              JOIN soundfiles ON soundfiles.id = bins.soundfiles_id WHERE detections.id IN (",
                                             paste(unq_ids,sep="",collapse = ","),
                                             ") AND bins.type = ",proc_details$negative_bin,
                                             " AND detections.procedure = ",review_procedure,sep="")))
    
    
    assoc_data_copy$bin_unqst=NULL
    assoc_data_copy$bin_unqend=NULL
    assoc_data_copy$bin_unq=NULL
    
    all_bins$remove = 0
    #go through all bins, if any don't contain a 1 or 21, mark for deletion
    #need to speed this up. 
    lastfile =  ""
    assoc_data_sub = assoc_data[which(assoc_data$label %in% c(1,21,2,22)),]
    for(m in 1:nrow(all_bins)){
      #print(paste(m,"of",nrow(all_bins)))
      #ignoring case where det longer than bin
      #for cases where det is on single file
      if(lastfile != all_bins$name[m]){
        comp_dets =assoc_data_sub[which(assoc_data_sub$StartFile==all_bins$name[m] & assoc_data_sub$EndFile==all_bins$name[m]),]
      }
      
      if(nrow(comp_dets[which(comp_dets$StartTime<=all_bins$seg_start[m] & 
                              comp_dets$EndTime>all_bins$seg_start[m]) | (
                                comp_dets$StartTime>=all_bins$seg_start[m] &
                                comp_dets$EndTime<=all_bins$seg_end[m]) | (
                                  comp_dets$EndTime>=all_bins$seg_end[m] &
                                  comp_dets$StartTime<all_bins$seg_end[m]
                          ) ,] > 0)){
        all_bins$remove[m]=1
      }
      
      lastfile = all_bins$name[m]
      
      #for cases where det is between two files
      #if(any(assoc_data[which(assoc_data$EndTime>all_bins$seg_start[m]
      #                        &
      #                        (assoc_data$EndFile==all_bins$name[m] & assoc_data$EndFile!=assoc_data$StartFile )) ,"label"] %in% c(1,21))){
      #  all_bins$remove[m]=1
      #}
      
      
    }
    
    overlap_sf = assoc_data[which(assoc_data$StartFile!=assoc_data$EndFile),]
    overlap_sf = overlap_sf[which(overlap_sf$label %in%  c(1,21,2,22))]
    
    for(m in 1:nrow(overlap_sf)){
      
      all_bins[which(all_bins$name==overlap_sf$EndFile[m] & all_bins$seg_start==0),"remove"]=1
      all_bins[which(all_bins$name==overlap_sf$StartFile[m] & all_bins$seg_end>overlap_sf$StartTime[m]),"remove"]=1
     
    }
    
    neg_bins = all_bins[which(all_bins$remove==0),]
    
    assoc_data$id = NA
    
    assoc_data_out = rbind(assoc_data_copy,assoc_data,held_out_data,assoc_data_negs)
    
    new_neg_data = data.frame(neg_bins$seg_start,neg_bins$seg_end,0,max(assoc_data_out$HighFreq),neg_bins$name,neg_bins$name,
                              NA,"",new_procedure,20,17,2,NA,assoc_data$analyst[1],1,NA,NA,NA,NA)
    
    colnames(new_neg_data)= colnames(assoc_data_negs)
    
    assoc_data_out = rbind(assoc_data_out,new_neg_data)
    
    
  }else{
     
    assoc_dat_no_neg = assoc_data[-which(assoc_data$label==20 & is.na(assoc_data$probability)),]
    #data will be updated. find the original ids from 
    last_data = held_out_data[which(held_out_data$status==1 & !is.na(held_out_data$probability)),] #get the previous 1 detections
    
    
    ###not sure I want to do this- will soft delete, whereas leaving in unaffected should
    #result in intended hard delete in compare/publish
    
    #remove the extra neg bins from the last analysis from held out data- these will be recomputed
    #held_out_data = held_out_data[-which(held_out_data$status==1 & is.na(held_out_data$probability)),]
    
    ###
    
    #remove the rest of last data from held out data
    held_out_data = held_out_data[-which(held_out_data$id %in% last_data$id),]
    
    #grab label from assoc_dat_no_neg
    last_data$label = assoc_dat_no_neg$label[match(assoc_dat_no_neg$original_id,last_data$original_id)]
    
    assoc_data_out = rbind(assoc_data_copy,assoc_data_negs,held_out_data,last_data)
    
  }
  
}else{
  assoc_data_out = assoc_data
  assoc_data_out$procedure = new_procedure
}

#write out in vanilla detx, drop assoc id. 
assoc_data_out$peak_assoc_id=NULL



write.csv(assoc_data_out,gzfile(paste(resultPath,"DETx.csv.gz",sep="/")),row.names = FALSE)
