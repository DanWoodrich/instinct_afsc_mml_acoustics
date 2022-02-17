
library(ggplot2)
#set these variables in all containers:
MethodID<-"autocompare-v1-0"

args<-"C:/Apps/INSTINCT_2/Cache/132288/FileGroupFormat.csv.gz C:/Apps/INSTINCT_2/Cache/132288/984289/494165/DETx.csv.gz C:/Apps/INSTINCT_2/Cache/132288/984289/494165/331506  autocompare-v1-0"

args<-strsplit(args,split=" ")[[1]]

#test folder
#FGpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/"
#GTpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/50ae7a/"
#DETpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/af5c26/3531e3/"
#resultPath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/af5c26/3531e3/8bbfbd"
#IoUThresh<-0.15
#SignalCode="LM"

args<-commandArgs(trailingOnly = TRUE)

#docker values
FGpath <- args[1]
DETxpath <- args[2]
resultPath <- args[3]

FGdata<-read.csv(FGpath)
DETxData<-read.csv(DETxpath)


#tolerant to dash or slash
midchar = substr(DETxData$StartFile[1],nchar(DETxData$StartFile[1])-10,nchar(DETxData$StartFile[1])-10)
formatPOSIXct = paste("%y%m%d","%H%M%S",sep=midchar)

#for detx data vreat a column which signifies the time in posixct
DETxData$StartTimePOSIXct = as.POSIXct(substr(DETxData$StartFile,nchar(DETxData$StartFile)-16,nchar(DETxData$StartFile)-4),format=formatPOSIXct) + DETxData$StartTime

DETxData$probs[which(is.na(DETxData$probs))] = sample(100,sum(is.na(DETxData$probs)),replace = TRUE)/100

#facet based on SignalCode, Color based on label, y axis = probability, x axis = time. 

p <- ggplot(DETxData, aes(StartTimePOSIXct, probs,colour = label)) + geom_point()
p + facet_grid(rows = vars(SignalCode))
# Use vars() to supply variables from the dataset:
#p 

ggsave(resultPath)


stop()