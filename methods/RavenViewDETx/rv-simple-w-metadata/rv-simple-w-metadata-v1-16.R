library(foreach)

formatToDets<-function(data,data2){
  colnames(data)[1:6]<-reqCols
  colnames(data)[7]<-'label'
  colnames(data)[8]<-'SignalCode'
  colnames(data)[9]<-'Type'

  dropCols<-c("label","SignalCode","Type") #drops any that aren't present in Dets

  if(any(!colnames(Dets) %in% dropCols)){
    dropColsDo<-dropCols %in% colnames(Dets)
    data<-data[,which(!colnames(data) %in% dropCols[!dropColsDo])]
  }

  data$StartTime<-as.numeric(data$StartTime)
  data$EndTime<-as.numeric(data$EndTime)
  data$LowFreq<-as.numeric(data$LowFreq)
  data$HighFreq<-as.numeric(data$HighFreq)

  #add dummy cols to outNeg to match Dets
  if(length(colnames(data2))>length(colnames(data))){

    addCols<-colnames(data2)[!(colnames(data2) %in% colnames(data))]
    dummy<-setNames(data.frame(matrix(ncol = length(addCols), nrow = nrow(data))),addCols)
    dummy[1:nrow(data),1:length(addCols)]<-NA
    data<-cbind(data,dummy)

  }
  return(data)
}

args="C:/Cache/953148/258047 C:/Cache/953148 C:/Cache/953148/258047/149794 //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/8192 y y y 1 1 rv-simple-w-metadata-v1-15 //161.55.120.117/NMML_AcousticsData/Audio_Data"
args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

DETpath <- args[1]
FGpath <-args[2]
Resultpath <- args[3]
dataPath <- args[4]
fillDat <- args[5]
randomize_order <- args[9]

#test if supposed to ignore decimation, and then repath dataPath:
if(args[length(args)-4]=="y"){
  dataPath=paste(args[length(args)],"Waves",sep="/")
}

#transform into Raven formatted data, retain data in other columns besides mandated 6.

Dets<-read.csv(paste(DETpath,"DETx.csv.gz",sep="/"),stringsAsFactors = FALSE) #add this in 1.3 for better backwards compatability with R

FG<-read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"),stringsAsFactors = FALSE)

#add check to make sure that if there is duplicate soundfile name, it errors out since there is not
#easily accesible data in the dets to evaluate correctly

if(any(duplicated(FG$StartFile)&!duplicated(paste(FG$StartFile,FG$Deployment)))){
  stop("duplicate sound files names present- this method does not properly distinguish gt between identically named sf in this vers")
}

#v1-7: right off the bat- if the detection is not within the effort (both begin and end file), remove it!

Dets<-Dets[which(Dets$StartFile %in% FG$FileName & Dets$EndFile %in% FG$FileName),]

#v1-5: if commment column is present in Dets, replace NA values with ""

if("Comments" %in% colnames(Dets)){
  Dets$Comments[is.na(Dets$Comments)]<-""
}

#mandatory column names
reqCols<-c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile")

if(any(!reqCols %in% colnames(Dets))){
  stop("Not a valid DETx object")
}

allFiles<-unique(c(Dets$StartFile,Dets$EndFile))

FGfull<-FG

FG<-FG[which(!duplicated(FG$FileName)),]

#if true, populate Dets for every file in FG which is not already present
if(fillDat=="y"){
  if(any(!FG$FileName %in% allFiles)){
    files<-FG$FileName[!FG$FileName %in% allFiles]
    rows<-foreach(n=1:length(files)) %do% {
      row<-c(0,0.1,0,0,files[n],files[n],NA,"Placeholder",NA)
      return(row)
    }
    placeHolderRows<-data.frame(do.call("rbind",rows))

    if(nrow(placeHolderRows)>0){

      placeHolderRows<-formatToDets(placeHolderRows,Dets)
      Dets<-rbind(Dets,placeHolderRows)

    }

    allFiles<-unique(c(Dets$StartFile,Dets$EndFile))

  }
}

#need to do the following:
#make sure script still works with old FG
#add functionality that blacks out not considered GT data when viewing in Raven.




colnames(FG)[which(colnames(FG)=="FileName")]<-"StartFile"

FG$StartTime<-NULL

FG$cumsum=c(0,cumsum(FG$Duration)[1:(nrow(FG)-1)])

#stick the null space data onto dets, so it gets formatted the same way!
#calculate the empty spaces in each file.
#can't think of a more elegant way to do this, so do a big ugly loop

outNeg<-foreach(i=1:length(allFiles)) %do% {
  segs<-FGfull[which(FGfull$FileName==allFiles[i]),]
  segs = segs[order(segs$SegStart),]
  segVec<-c(segs$SegStart[1],segs$SegStart[1]+segs$SegDur[1])
  if(nrow(segs)>1){
    for(p in 2:nrow(segs)){
      segVec<-c(segVec,segs$SegStart[p],segs$SegStart[p]+segs$SegDur[p])
    }
  }
  segVec<-c(0,segVec,segs$Duration[1])
  segVec<-segVec[which(!(duplicated(segVec)|duplicated(segVec,fromLast = TRUE)))]

  if(length(segVec)>0){
    outs<-foreach(f=seq(1,length(segVec),2)) %do% {
      segsRow<-c(segVec[f],segVec[f+1],0,5000,segs$FileName[1],segs$FileName[1],NA,"Not Considered",NA)
      return(segsRow)
    }

    outs<-do.call("rbind",outs)
  }else{
    outs<-NULL
  }
  
  #if(is.na(segVec[f+1])){
    #print(i)
    #print(segs)
  #  stop()
  #}


  return(outs)
  #chop up by 2s, write as negative space and rbind to outputs.

}

outNeg<-do.call("rbind",outNeg)
outNeg<-data.frame(outNeg)

if(nrow(outNeg)>0){

  outNeg<-formatToDets(outNeg,Dets)
  Dets<-rbind(Dets,outNeg)

}



#test if Dets have labels, or not.

FG$order<-1:nrow(FG)

DetsFG<-merge(Dets,FG,by="StartFile")

#reorder to original order
DetsFG<-DetsFG[order(DetsFG$order),]

#DetsFG$order<-NULL

#calculate delta time for each detection
#process these seperately

DetsFGSameFile <-DetsFG[which(DetsFG$StartFile==DetsFG$EndFile),]
DetsFGdiffFile <-DetsFG[which(!DetsFG$StartFile==DetsFG$EndFile),]

#for same time, just end - start
DetsFGSameFile$DeltaTime<-DetsFGSameFile$EndTime-DetsFGSameFile$StartTime

DetsFGdiffFile$DeltaTime<-(DetsFGdiffFile$Duration-DetsFGdiffFile$StartTime)+DetsFGdiffFile$EndTime

DetsFG<-rbind(DetsFGSameFile,DetsFGdiffFile)

#reorder to original order
DetsFG<-DetsFG[order(DetsFG$order),]
DetsFG$order<-NULL

#DetsFG<-DetsFG[order(DetsFG$StartFile,DetsFG$StartTime),]

DetsFG$FileOffset<-DetsFG$StartTime

#Raven friendly start and end times.
DetsFG$StartTime<-DetsFG$FileOffset+DetsFG$cumsum
DetsFG$EndTime<-DetsFG$StartTime+DetsFG$DeltaTime

#calculate fullpath for the end file as well

colnames(FG)[which(colnames(FG)=="StartFile")]<-"EndFile"
EFFP<-merge(Dets,FG,by="EndFile")
#EFFP<-EFFP[order(EFFP$StartFile,EFFP$StartTime),]
EFFP<-EFFP[order(EFFP$order),]

if(nrow(DetsFG)>=1){
  DetsFG$StartFile<-paste(dataPath,DetsFG$FullPath,DetsFG$StartFile,sep="")
  DetsFG$EndFile<-paste(dataPath,EFFP$FullPath,EFFP$EndFile,sep="")
}

#strike several metadata fields
dropCols<-c("DiffTime","FullPath","Deployment","SiteID","cumsum","Duration","SegStart","SegDur")

DetsFG<-DetsFG[,which(!colnames(DetsFG) %in% dropCols)]

keepCols<-c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","FileOffset","DeltaTime")
DetsFGxtra<-data.frame(DetsFG[,which(!colnames(DetsFG) %in% keepCols)])

colnames(DetsFGxtra)<-colnames(DetsFG)[which(!colnames(DetsFG) %in% keepCols)]

if(nrow(DetsFG)>=1){

  out<-data.frame(1:nrow(DetsFG),"Spectrogram 1",1,DetsFG$StartTime,DetsFG$EndTime,DetsFG$LowFreq,DetsFG$HighFreq,DetsFG$StartFile,
                  DetsFG$EndFile,DetsFG$FileOffset,DetsFG$DeltaTime,DetsFGxtra)
  
  if(randomize_order!="n"){
  
    if(randomize_order=="y"){
      seed = sample(9999)
    }else{
      #assume you give an integer for the seed if not n/y. 
      seed = as.integer(randomize_order)
    }
    
    #BUGFIX stealth v1-14: make the sample be for just the acoustic files (unchanged between editing sessions)
    #instead of for the full dataset with labels
    FG_files = unique(c(DetsFG$StartFile,DetsFG$EndFile))
    FG_files = sort(FG_files)
    
    set.seed(seed)
    #tested: should work consistently even if decimated levels are changing. 
    FG_files_consistent_shuffle = FG_files[sample(1:length(FG_files))]
    
    out = out[order(match(out$DetsFG.StartFile,FG_files_consistent_shuffle),out$DetsFG.StartTime),]
    
    out[,1]=1:nrow(out)
    
  }
  
}else{
  out<-data.frame(matrix(ncol = 11+length(DetsFGxtra), nrow = 0))
  if(length(DetsFGxtra)>0){
    colnames(out)[12:(11+length(DetsFGxtra))]<-names(DetsFGxtra)
  }
}

colnames(out)[1:11]<-c("Selection","View","Channel","Begin Time (s)","End Time (s)","Low Freq (Hz)","High Freq (Hz)",
                      "Begin Path","End Path","File Offset (s)","Delta Time (s)")


write.table(out,paste(Resultpath,'/RAVENx.txt',sep=""),quote=FALSE,sep = "\t",row.names=FALSE)
