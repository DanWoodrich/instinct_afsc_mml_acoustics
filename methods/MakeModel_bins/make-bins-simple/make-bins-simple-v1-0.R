#method: generate bins for DL training. 

library(foreach)

args="C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/test2 15 1 make-bins-simple-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
resultpath = args[2]
largeWindow = as.numeric(args[3]) #15
step =  as.numeric(args[4])

#generate bin start and end times: largeWindow long, with a step of step. 

binsall = foreach(i=1:length(unique(FG$DiffTime))) %do% {
  
  #These will be in DETx standard
  
  #reduce FG to just the individual files in difftime
  
  FGfiles = FG[which(FG$DiffTime==unique(FG$DiffTime)[i]),c("FileName","Duration"),drop=FALSE]
  FGfiles = FGfiles[which(!duplicated(FGfiles$FileName)),,drop=FALSE]

  #calculate the total time of the segment
  tottime = sum(FG[which(FG$DiffTime==unique(FG$DiffTime)[i]),"SegDur"])+FG[which(FG$DiffTime==unique(FG$DiffTime)[i]),"SegStart"][1]
  
  tottimestart = tottime-largeWindow
  
  binstart = seq(FG[which(FG$DiffTime==unique(FG$DiffTime)[i]),"SegStart"][1],tottimestart,by=step)
  binend = seq(FG[which(FG$DiffTime==unique(FG$DiffTime)[i]),"SegStart"][1]+largeWindow,tottime,by=step)
  
  binstartfiles=vector("character",length=length(binstart))
  binendfiles=vector("character",length=length(binstart))
  #convert to offset- there's got to be a better way to do this
  for(n in 1:nrow(FGfiles)){
    
    binstartfiles[which(binstart<FGfiles$Duration[n]&binstartfiles=="")] = FGfiles$FileName[n]
    binendfiles[which(binend<=FGfiles$Duration[n]&binendfiles=="")] = FGfiles$FileName[n]
    
    binstart[which(binstart>=FGfiles$Duration[n])] = binstart[which(binstart>=FGfiles$Duration[n])]-FGfiles$Duration[n]
    
    binend[which(binend>FGfiles$Duration[n])] = binend[which(binend>FGfiles$Duration[n])]-FGfiles$Duration[n]
    
  }
  
  #assemble data
  
  outdata = data.frame(binstart,binend,0,1024,binstartfiles,binendfiles,unique(FG$DiffTime)[i])
  
  outdata

}

binsall = do.call("rbind",binsall)

colnames(binsall) <-c('StartTime','EndTime','LowFreq','HighFreq','StartFile',"EndFile","DiffTime")

#dir.create(resultpath)

write.csv(binsall,gzfile(paste(resultpath,"DETx.csv.gz",sep="/")))

