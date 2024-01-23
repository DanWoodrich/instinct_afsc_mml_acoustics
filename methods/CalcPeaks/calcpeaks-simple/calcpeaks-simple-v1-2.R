library(pgpamdb)
library(DBI)
library(foreach)
#library(dplyr)


doIDvec<-function(x){
  #define peak as from the first minima, through the maxima, and up until the next minima. If peak is not true detection,
  #turn off this region.
  
  #ID each peak
  IDvec<-rep(0,length(x))
  ID<-1
  for(z in 1:length(x)){
    if(x[z]==2){
      IDvec[z]<-ID
      rightID<-FALSE
      p=z
      while(rightID==FALSE){
        if(z>1){
          p=p-1
          if(x[p]==1){
            IDvec[z]<-ID
            rightID<-TRUE
          }
        }else{
          IDvec[z]<-ID
          rightID<-TRUE
        }
        
      }
      IDvec[p:z]<-ID
      
      ID<-ID+1
    }
  }
  
  IDvec<-cummax(IDvec) #this had date boundaries for the peaks
  IDvec[which(IDvec==0)]<-1
  
  return(IDvec)
}

MinimaAndMaxima<-function(x){
  x=as.numeric(x)
  #CALCULATE MAXIMA AND MINIMA
  out1<-localMaxima(x)
  vec1<-rep(FALSE,length(x))
  vec1[out1]<-TRUE
  
  out2<-localMinima(x)
  vec2<-rep(FALSE,length(x))
  vec2[out2]<-TRUE
  
  out3<-localMinima2(x)
  vec3<-rep(FALSE,length(x))
  vec3[out3]<-TRUE
  vec3<-rev(vec3)
  
  #turn on to visualize
  #plot(x,type="l")
  #points(vec1,add=TRUE,col="red")
  #points(vec2,add=TRUE,col="green")
  #points(vec3,add=TRUE,col="blue")
  
  #combine minima
  vec2<-(vec2|vec3)
  
  vec<-rep(0,length(x))
  vec[vec1]<-2
  vec[vec2]<-1
  
  return(vec)
}

localMaxima <- function(x) {
  # Use -Inf instead if x is numeric (non-integer)
  y <- diff(c(-.Machine$integer.max, x)) > 0L
  rle(y)$lengths
  y <- cumsum(rle(y)$lengths)
  y <- y[seq.int(1L, length(y), 2L)]
  if (x[[1]] == x[[2]]) {
    y <- y[-1]
  }
  y
}

localMinima <- function(x) {
  # Use -Inf instead if x is numeric (non-integer)
  y <- diff(c(.Machine$integer.max, x)) < 0L
  rle(y)$lengths
  y <- cumsum(rle(y)$lengths)
  y <- y[seq.int(1L, length(y), 2L)]
  if (x[[1]] == x[[2]]) {
    y <- y[-1]
  }
  y
}

localMinima2 <- function(x) {
  # Use -Inf instead if x is numeric (non-integer)
  y <- diff(c(.Machine$integer.max, rev(x))) < 0L
  rle(y)$lengths
  y <- cumsum(rle(y)$lengths)
  y <- y[seq.int(1L, length(y), 2L)]
  if (rev(x)[[1]] == rev(x)[[2]]) {
    y <- y[-1]
  }
  y
}

#pseudo: this script loads in detx data. From it, it will infer peaks. 
#This version will assume that data from only 1 logical location is included. 
#if multiple are included, it will bin them together based on datetime. 

#this is an adaptation of the original algorithm used in FinReview.R (in network/detector/tools)

args="D:/Cache/838624/491352 D:/Cache/838624/491352/631885  calcpeaks-simple-v1-2"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
con=pamdbConnect(dbname,keyscript,clientkey,clientcert)

data= read.csv(paste(args[1],"/DETx.csv.gz",sep=""))
resultPath = args[2]

if(nrow(data[which(!is.na(data$probability)),])==0){
  
  stop("no returned detections to review.")
}

#assume the data has all been reviewed by the user running script
data[which(!is.na(data$probability)),"analyst"] = as.integer(dbFetch(dbSendQuery(con,"SELECT id FROM personnel WHERE personnel.pg_name = current_user"))$id)

sfs = data.frame(rbind(data[,c("StartFile","data_collection_id")],setNames(data[,c("EndFile","data_collection_id")],c("StartFile","data_collection_id"))))
sfs = sfs[!duplicated(sfs),]
#sfs = data.frame(unique(c(data$StartFile,data$EndFile)))
colnames(sfs)[1] = 'name'
sfs$data_collection_id= as.integer(sfs$data_collection_id)

metadata = table_dataset_lookup(con,"SELECT * FROM soundfiles",sfs,c("character varying","integer"))

colnames(metadata)[which(colnames(metadata)=="name")]="StartFile"

data = merge(data,metadata,"StartFile")

data$hourly<-format(data$datetime,"%y%m%d %H")

#query this... 

SecInPng=dbFetch(dbSendQuery(con,paste("SELECT length_seconds FROM bin_type_codes JOIN signals ON bin_type_codes.id = signals.native_bin WHERE signals.id =",data$signal_code[1])))$length_seconds

#fix this: aggregate by duration per hour, then calculate pngs per hour.
#data$PNGSperUnit<-ceiling(data$duration/SecInPng)
#total_pngs = data[which(!duplicated(data$id.y)),]
#total_pngs <-aggregate(PNGSperUnit ~ hourly,data=total_pngs,sum)

total_pngs = data[which(!duplicated(data$id.y)),]
total_pngs <-aggregate(duration ~ hourly,data=total_pngs,sum)
total_pngs$PNGSperUnit = ceiling(total_pngs$duration/SecInPng)

#now find total pngs with detections
total_neg = aggregate((EndTime - StartTime) ~ hourly,data = data[which(is.na(data$probability)),],sum)

colnames(total_neg)[2]="neg_total"
total_neg$neg_total = ceiling(total_neg$neg_total/SecInPng)

hourly_perc = merge(total_pngs,total_neg,all.x = TRUE)
hourly_perc$neg_total[which(is.na(hourly_perc$neg_total))]=0
hourly_perc$perc_pos = 1- (hourly_perc$neg_total/hourly_perc$PNGSperUnit)

vec<-MinimaAndMaxima(hourly_perc$perc_pos)
IDvec<-doIDvec(vec)

#now that we have maxima, can make dataset to perform review.
MaxHrs<-unique(hourly_perc$hourly)[vec==2]


#need to do two things here: create peaks, as well as associate the existing data with the id of the peak. 
data$assoc_id = NA

peakTimes<-foreach(f=1:length(MaxHrs)) %do% {
  
  #select detections from each peak
  Datedata<-data[which(data$hourly %in% hourly_perc$hourly[IDvec==f] & !is.na(data$probability)),]
  
  countsInPeak<-nrow(Datedata)            
  Peak<-Datedata[which.max(Datedata[,"probability"]),]
  
  data[which(data$hourly %in% hourly_perc$hourly[IDvec==f] & !is.na(data$probability)),"assoc_id"]=Peak$id.x

  return(cbind(Peak,countsInPeak))
}

peakTimes<-do.call("rbind",peakTimes)

#order consecutively
peakTimes = peakTimes[order(peakTimes$datetime),]

#save two datasets in detx format. The peak detections (include id and dets per peak for review),
#and the associated detections (include id of associated peak)

peaksout = data.frame(peakTimes$StartTime,peakTimes$EndTime,peakTimes$LowFreq,peakTimes$HighFreq,
                      peakTimes$StartFile,peakTimes$EndFile,peakTimes$probability,peakTimes$comments,
                      peakTimes$procedure,peakTimes$label,peakTimes$signal_code,peakTimes$strength,
                      peakTimes$modified,peakTimes$analyst,peakTimes$status,peakTimes$original_id,peakTimes$date_created,
                      peakTimes$id.x,peakTimes$countsInPeak)

colnames(peaksout)=c('StartTime','EndTime','LowFreq','HighFreq','StartFile',
                     'EndFile','probability','comments','procedure','label',
                     'signal_code','strength','modified','analyst','status',
                     'original_id','date_created','id',"countsInPeak")

#reset the labels
peaksout$label = ""

dataout = data.frame(data$StartTime,data$EndTime,data$LowFreq,data$HighFreq,
                     data$StartFile,data$EndFile,data$probability,data$comments,
                     data$procedure,data$label,data$signal_code,data$strength,
                     data$modified,data$analyst,data$status,data$original_id,data$date_created,
                     data$id.x,data$assoc_id)

colnames(dataout)=c('StartTime','EndTime','LowFreq','HighFreq','StartFile',
                     'EndFile','probability','comments','procedure','label',
                     'signal_code','strength','modified','analyst','status',
                     'original_id','date_created','id',"peak_assoc_id")

#dataout = dataout[which(!is.na(dataout$probability)),]

write.csv(peaksout,gzfile(paste(resultPath,"DETx.csv.gz",sep="/")),row.names = FALSE)
write.csv(dataout,gzfile(paste(resultPath,"DETx2.csv.gz",sep="/")),row.names = FALSE)

#at end: write peaks as DETx.csv.gz, write associated detections as DETx2.csv.gz. Add column to detx2 to associate
#to peak id. 
