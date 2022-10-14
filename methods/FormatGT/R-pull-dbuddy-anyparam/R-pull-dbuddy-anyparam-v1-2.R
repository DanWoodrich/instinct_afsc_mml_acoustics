MethodID<-"R-pull-dbuddy-anyparam-v1-2"
#v1-1: bugfix loop where extra commands are added. 

args="C:/Apps/INSTINCT/Cache/770628/777608/DETx.csv.gz test.csv Comments SignalCode Type UseFG %LMB% LM DET n R-pull-dbuddy-anyparam-v1-2"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

outpath <-args[1]
FG <- args[2]

numparams = floor((length(args)-2)/2)

#length



if(numparams!=0){
  
  paramnames = c(args[3:(3+numparams-1)])
  paramvals = c(args[(3+numparams):(3+numparams+numparams-1)])
  
  if(paramvals[which(paramnames=="UseFG")]=="n"){
    command = paste("dbuddy pull detections",outpath,sep=" ")
  }else{
    command = paste("dbuddy pull detections",outpath,"--FileGroup",FG,sep=" ")
  }
  

  paramvals = paramvals[-which(paramnames=="UseFG")]
  paramnames = paramnames[-which(paramnames=="UseFG")]

  
  for(i in 1:(numparams-1)){
    command = paste(command," --",paramnames[i]," ",paramvals[i],sep="")
  }
  
}

print(command)

system(command)


