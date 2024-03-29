library(foreach) 
library(doParallel) #need
library(tuneR) #need
library(signal) #need

args<-"C:/Apps/INSTINCT/ C:/Apps/INSTINCT/Cache/368058 C:/Apps/INSTINCT/Cache/368058/61314/741260 //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/2048 C:/Apps/INSTINCT/Cache/368058/61314/741260 1_1 99 n method1 feat-ext-hough-light-source-v1-4 y 85 0.75 50 specgram 1024 96 1.2 70 0 feat-ext-hough-light-source-v1-4 channel_normalize img_thresh isoblur_sigma overlap spectrogram_func t_samp_rate tile_axis_size time_min_buffer window_length zero_padding"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

ProjectRoot<-args[1]
FGpath <- args[2]
DETpath <- args[3]
DataPath <- args[4]
resultPath <- args[5] 

splitNum<-args[6] 

ReadFile2<-paste('DETx',splitNum,'.csv.gz',sep="")

FG<-read.csv(paste(FGpath,'FileGroupFormat.csv.gz',sep="/"))

#Not sure if this will be a good fix? Trying it, but be careful... 
FG<-FG[which(!duplicated(FG$FileName)),]

colnames(FG)[3]<-"RealTime"

tmpPath<-paste(resultPath,splitNum,sep="/")
dir.create(tmpPath)

data<-read.csv(paste(DETpath,ReadFile2,sep="/"))

#add buffer times: pseudocode:
#calcute duration of each call, for those that are too short, subtract duration from buffer size and add difference/2 to each call
#subset to find calls with negative start times and end times over the file duration
#for each reassign if there is an eligible previous/next file in file group
#otherwise, 

crs<- as.integer(args[7])

verbose = args[8]

args<-args[-8] #delete this to preserve behavior from other methods using this wrapper before change

if(verbose!='y'){
  verbose = 'n'
}


MethodID<-args[9]

argsLen<-length(10:length(args))-1
argsSep<-argsLen/2

ParamArgs<-args[10:(10+argsSep)]
ParamNames<-args[(10+argsSep+1):length(args)]

targetSampRate<-as.integer(ParamArgs[which(ParamNames=="t_samp_rate")])
TMB<-as.numeric(ParamArgs[which(ParamNames=="time_min_buffer")])

mIDind<-gregexpr("-",MethodID)[[1]][length(gregexpr("-",MethodID)[[1]])-1]
MethodIDcut<-substr(MethodID,0,mIDind-1)

#populate with needed fxns for ED
SourcePath<-paste(ProjectRoot,"/lib/user/methods/FeatureExtraction/",MethodIDcut,"/",MethodID,".R",sep="")
source(SourcePath) 

#load FE libraries: 
if(length(libraries)>0){
  for(l in 1:length(libraries)){
    library(libraries[l],character.only=TRUE)
  }
}

#and general fxns
source(paste(ProjectRoot,"/lib/user/R_misc.R",sep="")) 

#Merge FG and data so data has full paths 
data<- merge(data, FG, by.x = "StartFile", by.y = "FileName")

#this doesn't work with segstart segdur etc. But shouldn't matter for the TMB calculation, and not used later. 
data$mapID<-1:nrow(data)

dataDETx<-data[,c(1:6)]
dataDETx$mapID<-data$mapID

data$calDur<-data$EndTime-data$StartTime
data$calDur[which(data$calDur<0)]<-data$calDur[which(data$calDur<0)]+data$Duration[which(data$calDur<0)]
data$StartTime[which(TMB-data$calDur>0)]<-data$StartTime[which(TMB-data$calDur>0)]-((TMB-data$calDur[which(TMB-data$calDur>0)])/2)
data$EndTime[which(TMB-data$calDur>0)]<-data$EndTime[which(TMB-data$calDur>0)]+((TMB-data$calDur[which(TMB-data$calDur>0)])/2)

if(any(data$StartTime<0)){
  for(i in which(data$StartTime<0)){
    difftime<-data[i,'DiffTime']
    filesInDiff <- FG$FileName[which(FG$DiffTime %in% difftime)]
    fileName<-as.character(data[i,'StartFile'])
    filePos<-which(filesInDiff==fileName)
    if(filePos!=1){
      data[i,'StartFile']<-filesInDiff[filePos-1]
      rowFind<-FG[which(FG$FileName==filesInDiff[filePos-1]),]
      data[i,'Duration']<-rowFind$Duration[1]
      data[i,'StartTime']<-data[i,'Duration']+data[i,'StartTime']
    }else{
      #assume since its at the start of a file the end time can handle it if added on the other end. This behavior keeps 
      #TMB consistent and moves window to accomodate- does not attempt to find a valid file not specified in FG to 
      #keep effort assumptions consistent
      data[i,'EndTime']<-TMB
      data[i,'StartTime']<-0
    }
  }
}

if(any(data$EndTime>data$Duration)){
  for(i in which(data$EndTime>data$Duration)){
    difftime<-data[i,'DiffTime']
    
    filesInDiff <- FG$FileName[which(FG$DiffTime %in% difftime)]
    fileName<-as.character(data[i,'StartFile'])
    filePos<-which(filesInDiff==fileName)
    if(filePos!=length(filesInDiff)){
      data[i,'EndFile']<-filesInDiff[filePos+1]
      data[i,'EndTime']<-data[i,'EndTime']-data[i,'Duration']
    }else{
      #assume since its at the start of a file the end time can handle it if added on the other end. This behavior keeps 
      #TMB consistent and moves window to accomodate- does not attempt to find a valid file not specified in FG to 
      #keep effort assumptions consistent
      data[i,'EndTime']<-data[i,'Duration']
      data[i,'StartTime']<-data[i,'Duration']-TMB
    }
  }
}

#correct start and end times

#recalculate
data$calDur<-data$EndTime-data$StartTime
data$calDur[which(data$calDur<0)]<-data$calDur[which(data$calDur<0)]+data$Duration[which(data$calDur<0)]

dataOg<-data
#drop unneeded columns
data<-data[,c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","FullPath","Duration","mapID")]

#sort by start time and file 
data<-data[order(data$StartFile,data$StartTime),]

if(crs>detectCores()){
  crs<-detectCores()
}

#go through data sequentially. 
itemz<-nrow(data)

#turn off parallel if low amounts of data 
if(itemz<=crs){
  crs<-1
}

chunkz_size<-ceiling(itemz/crs)
chunkzAssign<-rep(1:crs,each=chunkz_size)

realcrs<-unique(chunkzAssign)
#in case of rounding issues: 

chunkzAssign<-chunkzAssign[1:itemz]

crs<-length(unique(chunkzAssign)) #moved this in a stealth change: don't think it should break anything. 

#make a column to identify FG ID 
FG$ID<-1:nrow(FG)

#save chunks to temp files 
for(n in 1:crs){
  write.csv(data[chunkzAssign==n,],gzfile(paste(tmpPath,"/chunk",n,".csv.gz",sep="")),row.names=FALSE)
}
#divide up effort into consecutive chunks 

startLocalPar(crs,"crs","tmpPath","FG","targetSampRate","readWave2","decimateData","resampINST","decDo","prime.factor","ParamArgs","FeatureExtracteR","verbose",nameSpaceFxns)

out2<-foreach(f=1:crs,.packages=c("tuneR","doParallel","seewave","signal",librariesToLoad)) %dopar% {
  
  #for(f in 1:32){
  dataIn<-read.csv(paste(tmpPath,"/chunk",f,".csv.gz",sep=""))
  
  #attempt to use IO/readwav more effeciently by reusing wav objects between iterations
  StartNameL<-"NULL1"
  StartFileL<-"NULL1"
  EndNameL<-"NULL2"
  EndFileL<-"NULL2"
  
  FG$FileName<-as.character(FG$FileName)
  FG$FullPath<-as.character(FG$FullPath)
  #startTimes<-c()
  #endTimes<-c()
  
  
  out1<-foreach(r=1:nrow(dataIn)) %do% {
    #print(paste(f,r))
    #for(r in 1:16){
    #check if start file is correct file, try to use loaded end file if it is the new start file
    if(StartNameL!=dataIn$StartFile[r]){
      if(dataIn$StartFile[r]==EndNameL){
        StartFileL<-EndFileL
        StartNameL<-EndNameL
      }else{
        StartNameL<-as.character(dataIn$StartFile[r])
        StartFileL<-readWave2(paste(DataPath,FG$FullPath[which(FG$FileName==StartNameL)],StartNameL,sep=""))
      }
    }
    if(EndNameL!=dataIn$EndFile[r]){
      if(dataIn$EndFile[r]==StartNameL){
        EndFileL<-StartFileL
        EndNameL<-StartNameL
      }else{
        EndNameL<-as.character(dataIn$EndFile[r])
        EndFileL<-readWave2(paste(DataPath,FG$FullPath[which(FG$FileName==EndNameL)],EndNameL,sep=""))
      }
    }
    
    #feature vector can be numeric, pass start/end file as seperate vars. After going through sequentially attach it to 
    #file name data again to export. 
    
    featList<-as.numeric(dataIn[r,1:4])
    StartFileDur<-dataIn$Duration[r]
    #startTimes<-c(startTimes,Sys.time())
    #store reused calculations to avoid indexing 
    
    Start<-featList[1]
    End<-  featList[2]
    Low<-featList[3]
    High<-featList[4]
    
    if(StartNameL==EndNameL){
      wav<-extractWave(StartFileL, from = Start, to = End,interact = FALSE, xunit = "time")
    }else{
      SoundList <- vector(mode = "list", length = 2)
      SoundList[[1]]<-extractWave(StartFileL, from = Start, to = StartFileDur,interact = FALSE, xunit = "time")
      SoundList[[2]]<-extractWave(EndFileL, from = 0, to = End,interact = FALSE, xunit = "time")
      wav<-do.call(bind, SoundList)
    }
    
    wav<-decimateData(wav,targetSampRate)
    
    #could render spectrogram here 
    metadata=list(substr(StartNameL,1,nchar(StartNameL)-4),ProjectRoot)
    featList<-FeatureExtracteR(wav,spectrogram=NULL,featList,args=ParamArgs,verbose=verbose,metadata=metadata)
    #endTimes<-c(endTimes,Sys.time())
    
    featList<-c(dataIn[r,"mapID"],featList[5:length(featList)]) #this is a test line to see if it fixes bug
    
    
    featList
  }
  
  out1<-do.call("rbind",out1)
  return(out1)
  
}

parallel::stopCluster(cluz)

outName<-paste("DETx_int",splitNum,".csv.gz",sep="")
out2<-do.call("rbind",out2)

out2<-data.frame(out2)

colnames(out2)<-c("mapID",
                  "Rugosity","Crest","Temporal Entropy","Shannon Entropy","Roughness", "autoc mean", "autoc median","autoc se",
                  "dfreq mean","dfreq se","specprop mean","specprop sd","specprop se","specprop median","specprop mode","specprop q25",
                  "specprop q75","specprop IQR","specprop centroid","specprop skewness","specprop kurtosis","specprop sfm","specprop sh","specprop precision",
                  "Amp env med","total entropy","Modinx","Startdom","Enddom","Mindom",
                  "Maxdom","Dfrange","Dfslope","Meanpeakf","AreaX maxP","AreaX Max", "AreaX dom","AreaX std","AreaY maxP",
                  "AreaY max","AreaY dom","AreaY std","Area spread","AreaMax","AreaMax Dom","AreaTop3 Dom","Num Shapes",
                  "BestTheta Hough","BestRho Hough","BestScore Hough","BestSlope Hough","BestB Hough","MedTheta Hough","MeanTheta Hough","sdTheta Hough",
                  "MedRho Hough","MeanRho Hough","sdRho Hough","MedScore Hough","MeanScore Hough","sdScore Hough","MedSlope Hough","MeanSlope Hough",
                  "sdSlope Hough","MedB Hough","MeanB Hough","sdB Hough","num Goodlines","xavg","yavg","SwitchesX mean","SwitchesX se",
                  "SwitchesX max","SwitchesX min","SwitchesY mean","SwitchesY se","SwitchesY max","SwitchesY min","sCompared"
)

#out2<-cbind(out2[,c("StartTime","EndTime","LowFreq","HighFreq")],data[,c("StartFile","EndFile")],out2[,5:ncol(out2)])

#make FG back into character then factor type: 

out2<-out2[order(out2$mapID),]

out2<-data.frame(dataDETx[,c(2:5,1,6)],out2[,2:ncol(out2)])

#test if AL has happened, add back in labels and other metadata if so

if('label' %in% colnames(dataOg)){
  
  possiblefields<-c('label','SignalCode','Analyst',"Comments") #some possible column reserved keywords
  possiblefields <- possiblefields[which(possiblefields %in% colnames(dataOg))]
  datamerge<-data.frame(dataOg[order(dataOg$mapID),possiblefields])
  colnames(datamerge)<-possiblefields
  out2<-cbind(out2,datamerge)
}

for(n in 1:crs){
  file.remove(paste(tmpPath,"/chunk",n,".csv.gz",sep=""))
}

unlink(tmpPath,recursive=TRUE, force = TRUE)

write.csv(out2,gzfile(paste(resultPath,outName,sep="/")),row.names = FALSE)
