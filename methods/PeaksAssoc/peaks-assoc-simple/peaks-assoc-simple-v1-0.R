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

#print(allow_ambiguous_label)

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

#write out in vanilla detx, drop assoc id. 
assoc_data$peak_assoc_id=NULL



write.csv(assoc_data,gzfile(paste(resultPath,"DETx.csv.gz",sep="/")),row.names = FALSE)
