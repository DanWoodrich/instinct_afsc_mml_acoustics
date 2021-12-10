MethodID<-"R-pull-dbuddy-anyparam-v1-0"

args="C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418 C:/Apps/INSTINCT/Cache/809545/548038 C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418/227655  dbuddy-compare-publish-v1-0"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

outpath <-args[1]
FG <- args[2]

numparams = floor((length(args)-2)/2)

#length

command = paste("dbuddy pull detections",outpath,"--FileGroup",FG,sep=" ")

if(numparams!=0){
  
  paramnames = c(args[3:(3+numparams-1)])
  paramvals = c(args[(3+numparams):(3+numparams+numparams-1)])
  
  for(i in 1:length(numparams)){
    
    command = paste(command," --",paramnames[i]," ",paramvals[i],sep="")
  }
  
}

print(command)

system(command)


