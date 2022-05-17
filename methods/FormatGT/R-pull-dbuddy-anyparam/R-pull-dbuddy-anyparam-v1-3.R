MethodID<-"R-pull-dbuddy-anyparam-v1-3"
#v1-1: bugfix loop where extra commands are added. 
#v1-3: supports multiple items per argument. 

args="C:/Apps/INSTINCT/Cache/394448/643567/DETx.csv.gz round1_pull1_reduce.csv Analysis_ID SignalCode UseFG 17 HB.s.p.2,HB.s.p.2 y R-pull-dbuddy-anyparam-v1-3"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

stop()

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
  
  
  #determine if parameters are single value or multiple. 
  multi_true = grepl(",",paramvals) #determined by presence of comma
  
  #commas can be allowed if 
  multi_true[which(paramnames=="Comments")]=FALSE
  
  for(i in 1:(numparams-1)){
    
    if(multi_true[i]){
      
      val= 
    }else{
      val=
    }
    
    command = paste(command," --",paramnames[i]," ",paramvals[i],sep="")
  }
  
}

print(command)

system(command)


