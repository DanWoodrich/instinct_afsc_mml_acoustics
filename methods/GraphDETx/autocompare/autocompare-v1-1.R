
library(ggplot2)
#set these variables in all containers:
MethodID<-"autocompare-v1-1"

#V1-1: Clean up names and plot

args<-"C:/Apps/INSTINCT_2/Cache/132288/FileGroupFormat.csv.gz C:/Apps/INSTINCT_2/Cache/132288/984289/494165/DETx.csv.gz C:/Apps/INSTINCT_2/Cache/132288/984289/494165/331506  autocompare-v1-0"

args<-strsplit(args,split=" ")[[1]]



#test folder
#FGpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/"
#GTpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/50ae7a/"
#DETpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/af5c26/3531e3/"
#resultPath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/af5c26/3531e3/8bbfbd"
#IoUThresh<-0.15
#SignalCode="LM"

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#docker values
FGpath <- args[1]
DETxpath <- args[2]
resultPath <- args[3]
FGID <- args[4]
ggtptsize <-as.numeric(args[5])
ggtextsize <-as.numeric(args[6])
timecalc <-args[7]

FGdata<-read.csv(FGpath)
DETxData<-read.csv(DETxpath)


colnames(DETxData)[which(colnames(DETxData)=='probs')] = "Probability"
colnames(DETxData)[which(colnames(DETxData)=='label')] = "Label"


#tolerant to dash or slash

DETxData$Probability[which(is.na(DETxData$Probability))] = sample(100,sum(is.na(DETxData$Probability)),replace = TRUE)/100

DETxData$newlab = ""
DETxData$newlab[which(DETxData$SignalCode=="out")]='Detector'
DETxData$newlab[which(DETxData$SignalCode!="out")]='Truth'


if(timecalc!="FG"){
  midchar = substr(DETxData$StartFile[1],nchar(DETxData$StartFile[1])-10,nchar(DETxData$StartFile[1])-10)
  formatPOSIXct = paste("%y%m%d","%H%M%S",sep=midchar)
  
  #for detx data vreat a column which signifies the time in posixct
  DETxData$StartTimePOSIXct = as.POSIXct(substr(DETxData$StartFile,nchar(DETxData$StartFile)-16,nchar(DETxData$StartFile)-4),format=formatPOSIXct) + DETxData$StartTime
  
  p <- ggplot(DETxData, aes(StartTimePOSIXct, Probability,colour = Label)) + geom_point(size = ggtptsize) + labs(title = FGID,x=NULL)+ theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))
  
  
}else{
  
  colnames(FGdata)[1]= "StartFile"
  
  FGdata$cumdur = cumsum(FGdata$SegDur)
  
  DETxData=merge(DETxData,FGdata,by="StartFile")
  
  DETxData$hours = DETxData$cumdur+DETxData$StartTime.x
  
  DETxData$hours = DETxData$hours/3600
  
  p <- ggplot(DETxData, aes(hours, Probability,colour = Label)) + geom_point(size = ggtptsize) + labs(title =FGID,x="Hours since FG start")
  
}


#DETxData$Time = DETxData$StartTimePOSIXct

#facet based on SignalCode, Color based on label, y axis = probability, x axis = time. 

p + facet_grid(rows = vars(newlab))+ theme(text=element_text(size=ggtextsize))+ guides(colour = guide_legend(override.aes = list(size=5)))
# Use vars() to supply variables from the dataset:
#p 

ggsave(resultPath)

