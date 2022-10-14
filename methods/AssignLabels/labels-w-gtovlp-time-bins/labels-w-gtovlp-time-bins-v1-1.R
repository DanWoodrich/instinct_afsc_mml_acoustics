#apply labels to detector outputs
#use GTovlp criteria: area of overlap over area of union 

#set these variables in all containers:
MethodID<-"labels-w-GTovlp-simple-v1-3"

#v1-1
#change how FG duration is calculated
#clean up condition to test for GTovlp (still could do a lot of work here to optimize)
#fix bug in determining if intersection is present.
#v1-2:
#add a parameter to control whether GT is written to the data. 
#v1-3: 
#fix a bug where wasn't properly reordering FG to match other data. 

args<-"C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/628717 C:/Apps/INSTINCT/Cache/394448/713795 C:/Apps/INSTINCT/Cache/394448/628717/121273 0.60 y y labels-w-GTovlp-simple-v1-3"

args<-strsplit(args,split=" ")[[1]]

#should be same container method that runs on probabalistic outputs when a cutoff is provided. 

#test folder
#FGpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/"
#GTpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/50ae7a/"
#DETpath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/af5c26/3531e3/"
#resultPath<-"C:/Apps/INSTINCT/Cache/2e77bc96796a/af5c26/3531e3/8bbfbd"
#GTovlpThresh<-0.15
#SignalCode="LM"

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#docker values
FGpath <- args[1]
GTpath <- args[2]
DETpath <- args[3]
resultPath <- args[4]
GTovlpThresh<-args[5]
onlyTime <- args[6]
WriteGT<-args[7]


GTdata<-read.csv(paste(GTpath,"DETx.csv.gz",sep="/"))
FGdata<-read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"))

#convert FG back to old format

#FGdata<-FGdata[which()]

outData<-read.csv(paste(DETpath,"DETx.csv.gz",sep="/"))

#Probabalistic<-FALSE
#introduce this in most sensible way once we get to 




#order datasets: v1-3, do it by FG order


FGdata$order<-1:nrow(FGdata)

FGdata$StartFile = FGdata$FileName

#just used for purposes of ordering: removes duplicated sfs
if(any(duplicated(FGdata$StartFile))){ #1-3 stealth change
  FGdataOrd = FGdata[-which(duplicated(FGdata$StartFile)),]
  
}else{
  FGdataOrd=FGdata
}

outData<-merge(outData,FGdataOrd[,c("order","StartFile")],by="StartFile")
GTdata<-merge(GTdata,FGdataOrd[,c("order","StartFile")],by="StartFile")

GTdata<-GTdata[order(GTdata$order,GTdata$StartFile,GTdata$StartTime),]
outData<-outData[order(outData$order,outData$StartFile,outData$StartTime),]

GTdata$order=NULL
outData$order=NULL

FGdata$FileName<-as.character(FGdata$FileName)
GTdata$StartFile<-as.character(GTdata$StartFile)
GTdata$EndFile<-as.character(GTdata$EndFile)


FGdata$cumsum<- c(0,cumsum(FGdata$Duration)[1:(nrow(FGdata)-1)])


#convert each dataset into time from FG start instead of offset 

GTlong<-GTdata

if(nrow(GTlong)>0){
#these steps are long... could make much faster with a merge. To do
for(i in 1:nrow(GTlong)){
  GTlong$StartTime[i]<-GTlong$StartTime[i]+FGdata$cumsum[which(FGdata$FileName==GTlong$StartFile[i])]
  GTlong$EndTime[i]<-GTlong$EndTime[i]+FGdata$cumsum[which(FGdata$FileName==GTlong$EndFile[i])]
}

GTdata$StartFile<-as.factor(GTdata$StartFile)
GTdata$EndFile<-as.factor(GTdata$EndFile)

GTlong$GTovlp<-0

}else{
  GTlong<-cbind(GTlong, data.frame(GTovlp=double()))
}

outLong<-outData

for(i in 1:nrow(outLong)){
  outLong$StartTime[i]<-outLong$StartTime[i]+FGdata$cumsum[which(FGdata$FileName==outLong$StartFile[i])]
  outLong$EndTime[i]<-outLong$EndTime[i]+FGdata$cumsum[which(FGdata$FileName==outLong$EndFile[i])]
}

outLong$GTovlp<-0

GTlong$Dur<-GTlong$EndTime-GTlong$StartTime

#order datasets: 
GTlong<-GTlong[order(GTlong$StartTime),]
outLong<-outLong[order(outLong$StartTime),]

outLong$id = NA

#calculate GTovlp GT
if(nrow(GTlong)>0){

for(i in 1:nrow(GTlong)){
  
  GTlongDur<-GTlong$EndTime[i]-GTlong$StartTime[i]
  if(any(outLong$EndTime<(GTlong$StartTime[i]-GTlongDur))){
    klow<-max(which((outLong$EndTime<(GTlong$StartTime[i]-GTlongDur)))) #give this a little buffer to avoid issues with small detections preventing longer det from 
  }else{
    klow=1
  }
  if(any((outLong$StartTime>(GTlong$EndTime[i]+GTlongDur)))){
    khigh<-min(which((outLong$StartTime>(GTlong$EndTime[i]+GTlongDur)))) #fitting the criteria. if wanted to get fancy, could base the buffer length on GTovlp
  }else{
    khigh=nrow(outLong)
  }
  
  k=klow
  for(k in klow:khigh){
  #while(GTlong$GTovlp[i]<GTovlpThresh&k<=khigh){
    #test for intersection
    if((((GTlong$StartTime[i]<outLong$EndTime[k] & GTlong$StartTime[i]>=outLong$StartTime[k])| #GTstart is less than det end, and GT start is after or at start of det
       (GTlong$EndTime[i]>outLong$StartTime[k] & GTlong$StartTime[i]<outLong$StartTime[k]))| #GTend is after det end, and GT start is before det start
       (GTlong$EndTime[i]>=outLong$EndTime[k] & GTlong$StartTime[i]<=outLong$StartTime[k])) & #gt end is after or at det end, and gt start is at or before det start
       (((GTlong$LowFreq[i]<outLong$HighFreq[k] & GTlong$LowFreq[i]>=outLong$LowFreq[k])|
        (GTlong$HighFreq[i]>outLong$LowFreq[k] & GTlong$LowFreq[i]<outLong$LowFreq[k])) |
        (GTlong$HighFreq[i]>=outLong$HighFreq[k] & GTlong$LowFreq[i]<=outLong$LowFreq[k])))
        {

      #test for GTovlp
      #x1,y1,x2,y2
      box1<-c(GTlong$StartTime[i],GTlong$LowFreq[i],GTlong$EndTime[i],GTlong$HighFreq[i])
      box2<-c(outLong$StartTime[k],outLong$LowFreq[k],outLong$EndTime[k],outLong$HighFreq[k])
      
      if(onlyTime=='y'){
      box1[2]=1 #replace high and low with 1 and 2 so only time dimension varies. 
      box2[2]=1
      box1[4]=2
      box2[4]=2
      }
      
      intBox<-c(max(box1[1],box2[1]),max(box1[2],box2[2]),min(box1[3],box2[3]),min(box1[4],box2[4]))
      
      intArea = abs(intBox[3]-intBox[1]) * abs(intBox[4]-intBox[2])
      
      box1Area = abs(box1[3] - box1[1]) * abs(box1[4] - box1[2])
      box2Area = abs(box2[3] - box2[1]) * abs(box2[4] - box2[2])
      
      totArea = box1Area + box2Area - intArea
      
      #iou = intArea / totArea
      
      #for comparing time bins, sum the amount of time from the prevGTovlps outLong!
      
      GTovlp = intArea/box1Area
      
      #give GT the best GTovlp
      if(GTlong$GTovlp[i]<GTovlp){
        #GTlong$GTovlp[i]<-GTovlp #this should be modified to % box area overlaps with GTovlp. 
        
        GTlong$GTovlp[i]<-GTovlp #this is better. It is not the same as GTovlp, so be careful. 
        
        #the GTovlp thresh will still be used, but interpreted in the context of overlap. 
      }
      
      if(outLong$GTovlp[k]<GTovlp){
        
        outLong$id[k] = GTlong$id[i]
        
        #will identify detection by the value of the GT which contributed most to positive label. 
      }

      outLong$GTovlp[k]<-GTovlp + outLong$GTovlp[k] #this one now sums GTovlp from prevGTovlps.Meaning, if the
      
      
        #out box overlaps multiple GTs, the overlap is summed. 
        
        #use the degree that the GT is ovlp'd, not the bin. 
        
        #this is weak to overlapping calls- whatever! Deal with that when I get there, if necessary. 
      
      #print(i)
      
      #plot(0,xlim=c(min(c(box1[1],box2[1]))-2,max(c(box1[3],box2[3]))+2),ylim=c(0,512),col="white")
      #rect(box1[1],box1[2],box1[3],box1[4],col="green")
      #Sys.sleep(0.25)
      #rect(box2[1],box2[2],box2[3],box2[4],col="gray")
      #Sys.sleep(0.25)
      #rect(intBox[1],intBox[2],intBox[3],intBox[4],col="red")
      #text(mean(c(min(c(box1[1],box2[1]))-2,max(c(box1[3],box2[3]))+2)),256,paste(GTovlp,k))
      #Sys.sleep(0.5)
      
    }
  }
  
  
}
}

GTlong$StartTime<-GTdata$StartTime
GTlong$EndTime<-GTdata$EndTime

outLong$StartTime<-outData$StartTime
outLong$EndTime<-outData$EndTime

GTlong$label<-ifelse(GTlong$GTovlp<GTovlpThresh,"FN","TP") #keep on eye on this behavior. Not sure it 
#fully makes sense...?
GTlong$label<-as.factor(GTlong$label)
outLong$label<-ifelse(outLong$GTovlp<GTovlpThresh,"FP","TP")
outLong$id[which(outLong$GTovlp<GTovlpThresh)]=NA
outLong$label<-as.factor(outLong$label)

GTlong$Dur<-NULL
GTlong$Label<-NULL #I may want this later? But relevant labels past this point should be TP FP FN... 
GTlong$GTovlp<-NULL
outLong$GTovlp<-NULL

outLong$SignalCode<-'out'

#add back in probs if present
if("probs" %in% colnames(outLong)){
  if(nrow(GTlong)>0){
    GTlong$probs<-NA
  }else{
    GTlong<-cbind(GTlong, data.frame(probs=double()))
  }
}

GTlong$DiffTime = NA #add this in so it matches. 

#make sure columns match (throw out other metdata not relevant here)
cols <- intersect(colnames(outLong), colnames(GTlong))

CombineTab<-rbind(GTlong[,cols],outLong[,cols])

CombineTab<-CombineTab[order(CombineTab$StartFile,CombineTab$StartTime),]

colnames(CombineTab)[which(colnames(CombineTab)=="id")]="GTid"

outName<-paste("DETx.csv.gz",sep="_")

#v1-2:
if(WriteGT=="n"){
  CombineTab<-CombineTab[which(CombineTab$SignalCode=='out'),]
  #remove GT rows
  CombineTab$SignalCode<-NULL
  #remove signal code column- assumed these are 'out' 
}

write.csv(CombineTab,gzfile(paste(resultPath,outName,sep="/")),row.names = FALSE)

