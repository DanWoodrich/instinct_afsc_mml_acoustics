MethodID<-"rd-simple-w-metadata-v1-5"

#1.1: make it so any extra metadata is retained
#1.2: include a filename arg

args="C:/Apps/INSTINCT/Cache/587676/869751/419712/843629/733076/831400/914402/599940/744239 C:/Apps/INSTINCT/Cache/737376 C:/Apps/INSTINCT/Cache/587676/869751/419712/843629/733076/831400/914402/599940/744239/812782 RAVENx.txt"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()


RAVpath <- args[1] #//akc0ss-n086/NMML_CAEP_Acoustics/Detector/LF moan project/Cole Summer 2021 materials and analysis/Review of INSTINCT positive sound files/Cole data product
FGpath <-args[2] #C:/Users/daniel.woodrich/Desktop/database/FileGroups/oneHRonePerc.csv
resultPath <- args[3] #C:/Users/daniel.woodrich/Desktop/database/GroundTruth/LM
fileName <- args[4]#RAVENx.txt

RavGT<-read.delim(paste(RAVpath,fileName,sep="/"))
FG<-read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"))

#throw out the segment info. 
FG<-FG[which(!duplicated(FG$FileName)),]

#reduce RavGT files to just names, not locations. 

RavGT<-RavGT[which(RavGT$View!="Waveform 1"),]

RavGT<-RavGT[,which(!colnames(RavGT) %in% c("Selection","View","Channel"))]

RavGT<-RavGT[order(RavGT$Begin.Time..s.),]

#get rid of not considered, and placeholder

#v1-5 bugfix
if('SignalCode' %in% colnames(RavGT)){
  if(any(RavGT$SignalCode!="Not Considered"&RavGT$SignalCode!="Placeholder")){
    RavGT<-RavGT[which(RavGT$SignalCode!="Not Considered"&RavGT$SignalCode!="Placeholder"),]
  }
}


#changed this from backslash to forward slash, but not sure why it is coming out different...
for(i in 1:nrow(RavGT)){
  slashes<-length(gregexpr("/|\\\\",RavGT$Begin.Path[i])[[1]])
  lastSlash<-gregexpr("/|\\\\",RavGT$Begin.Path[i])[[1]][slashes]
  RavGT$Begin.Path[i]<-substr(RavGT$Begin.Path[i],lastSlash+1,nchar(RavGT$Begin.Path[i]))
}

for(i in 1:nrow(RavGT)){
  slashes<-length(gregexpr("/|\\\\",RavGT$End.Path[i])[[1]])
  lastSlash<-gregexpr("/|\\\\",RavGT$End.Path[i])[[1]][slashes]
  RavGT$End.Path[i]<-substr(RavGT$End.Path[i],lastSlash+1,nchar(RavGT$End.Path[i]))
}

#convert RavGT names back to DETx standard
colnames(RavGT)[1:8]<-c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","FileOffset","DeltaTime")


colnames(FG)[which(colnames(FG)=="FileName")]<-"StartFile"
FG$StartTime<-NULL

#merge RavGT with FG
RaVGTFG<-merge(RavGT,FG,by="StartFile")

#start becomes file offset
RaVGTFG$StartTime<-RaVGTFG$FileOffset

#end becomes file offset + delta time
RaVGTFG$EndTime<-RaVGTFG$StartTime+RaVGTFG$DeltaTime

#at the end, do a check if any end times are > duration (from FG). If so, subtract them by FG duration. 
RaVGTFG$EndTime[which(RaVGTFG$EndTime>RaVGTFG$Duration)]<-RaVGTFG$EndTime[which(RaVGTFG$EndTime>RaVGTFG$Duration)]-RaVGTFG$Duration[which(RaVGTFG$EndTime>RaVGTFG$Duration)]

#remove unnecessary non metadata columns. 

#1-3: we define unnecessary columns by name, allowing for others to be retained. 

out<-data.frame(RaVGTFG[,2:5],RaVGTFG[,1],RaVGTFG[,6:ncol(RaVGTFG)])
colnames(out)[5]<-c("StartFile")

unneccessaryCols = c("FileOffset","DeltaTime","FullPath","Duration","Deployment","SegStart","SegDur","SiteID","DiffTime")

out = out[,colnames(out)[!(colnames(out) %in% unneccessaryCols)]]

write.csv(out,gzfile(paste(resultPath,"DETx.csv.gz",sep="/")),row.names = FALSE)


