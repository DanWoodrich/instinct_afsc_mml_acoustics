MethodID<-"R-FG-from-query-v1-0"
#v1-1: bugfix loop where extra commands are added. 

args="C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418 C:/Apps/INSTINCT/Cache/809545/548038 C:/Apps/INSTINCT/Cache/809545/548038/950334/563707/279418/227655  dbuddy-compare-publish-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()
outpath <-paste(dirname(args[1]),"temp1.csv.gz",sep="/")

numparams = floor((length(args)-2)/2)

#length

command = paste("dbuddy pull detections",outpath,sep=" ")

#print(args)

if(numparams!=0){
  
  paramnames = c(args[3:(3+numparams-1)])
  paramvals = c(args[(3+numparams):(3+numparams+numparams-1)])
  
  #print(paramnames)
  #print(paramvals)
  #stop()
  
  for(i in 1:(numparams-5)){
    command = paste(command," --",paramnames[i]," ",paramvals[i],sep="")
  }
  
}

print(command)

system(command)

data = read.csv(outpath)

files = unique(c(data$StartFile,data$Endfile)) #a little inneffecient here to have brought in all the detection data, but w/e. This will all change
#when can use direct SQL. 

files = data.frame(files)
filesPathName = paste(dirname(args[1]),"temp2.csv",sep="/")
write.csv(files,filesPathName)

outpath <-paste(dirname(args[1]),"temp.csv.gz",sep="/")
#making a new condition for this in dbuddy: pull soundfiles metadata from soundfile data. 
command = paste('dbuddy pull_from_data soundfiles',filesPathName,outpath) 

print(command)

system(command)

data = read.csv(outpath)

data$DateTime = as.POSIXct(data$DateTime,tz='UTC')

outdata = data.frame(data$Name,paste("/",data$Name.1,"/",format(data$DateTime,"%m"),"_",format(data$DateTime,"%Y"),"/",sep=""),
                     format(data$DateTime,"%y%m%d-%H%M%S"),data$Duration,data$Name.1,0,data$Duration,data$MooringID)

colnames(outdata) = c("FileName","FullPath","StartTime","Duration","Deployment","SegStart","SegDur","SiteID")

#print(str(outdata))

write.csv(outdata,gzfile(args[1]))


#from this query, just extract


#stop()
