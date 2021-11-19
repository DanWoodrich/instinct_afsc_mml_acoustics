MethodID<-"push-to-dbuddy-v1-0"

args="C:/Apps/INSTINCT/Cache/696361/979514/942467/219053/89417/609768/104619 C:/Apps/INSTINCT/Cache/696361/979514/299372 C:/Apps/INSTINCT/Cache/696361/979514/942467/219053/89417/609768/104619/552720  push-to-dbuddy-v1-0"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

PriorDataPath <- args[1]
EditDataPath <-args[2]
resultPath <- args[3]

PriorData<-read.csv(paste(PriorDataPath,"DETx.csv.gz",sep="/"))
EditData<-read.csv(paste(EditDataPath,"DETx.csv.gz",sep="/"))


#attempt to find changes in data
#Attempt to submit a modification of the data with ids. Data w/o ids will be assumed to be new data. 
#check that no records with Type of Det or SC have modified timestamp. 

