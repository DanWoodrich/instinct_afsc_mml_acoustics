args="D:/Cache/622910/915330/385092/298369/756597/311127 D:/Cache/622910/915330/385092 D:/Cache/622910/915330/385092/170683"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

#source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
#con=pamdbConnect("poc_v2",keyscript,clientkey,clientcert)

peak_data= read.csv(paste(args[1],"/DETx.csv.gz",sep=""))
assoc_data= read.csv(paste(args[2],"/DETx2.csv.gz",sep=""))
resultPath = args[3]

#print(head(peak_data))
#print(head(assoc_data))

lab_lookup = data.frame(c(0,1,2),c(20,21,22))
colnames(lab_lookup)<- c("labels","prot_labels")

for(i in 1:nrow(peak_data)){
  
  #if signal code is changed
  if(peak_data[i,"signal_code"]!=assoc_data$signal_code[1]){
    
    stop()
    assoc_data[which(assoc_data$peak_assoc_id==peak_data[i,"id"]),"signal_code"]=peak_data[i,"signal_code"]
    
    #extract from original data, change signal code, and rbind back onto data. 
  }
  
  
  
  #set assoc data label = to that of peak_data
  assoc_data[which(assoc_data$peak_assoc_id==peak_data[i,"id"]),"label"]= lab_lookup[match(peak_data[i,"label"],lab_lookup$labels),"prot_labels"]
  
  #set focal label to human label
  assoc_data[which(assoc_data$id==peak_data[i,"id"]),"label"] = peak_data[i,"label"]
}

#write out in vanilla detx, drop assoc id. 
assoc_data$peak_assoc_id=NULL

write.csv(assoc_data,gzfile(paste(resultPath,"DETx.csv.gz",sep="/")),row.names = FALSE)
