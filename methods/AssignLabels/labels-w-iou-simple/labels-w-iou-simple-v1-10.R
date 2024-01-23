library(foreach)
library(doParallel)


#apply labels to detector outputs
#use IoU criteria: area of overlap over area of union 

#set these variables in all containers:
MethodID<-"labels-w-iou-simple-v1-3"

#v1-1
#change how FG duration is calculated
#clean up condition to test for IOU (still could do a lot of work here to optimize)
#fix bug in determining if intersection is present.
#v1-2:
#add a parameter to control whether GT is written to the data. 
#v1-3: 
#fix a bug where wasn't properly reordering FG to match other data. 
#v1-4:
#retain splits if present. May be buggy if outlong is not dense. 
#v1-6: retain cutoff if present. Leapfrogged 1-5, which I am not sure if it is in use. 
#v1-10: parallelize long comparison step. 

args<-"D:/Cache/264078 D:/Cache/264078/421141 D:/Cache/865375/401113/370627/843649/777039/772363 D:/Cache/264078/421141/277887 0.001 y labels-w-iou-simple-v1-10"

args<-strsplit(args,split=" ")[[1]]

#should be same container method that runs on probabalistic outputs when a cutoff is provided. 

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
GTpath <- args[2]
DETpath <- args[3]
resultPath <- args[4]
IoUThresh<-args[5]
WriteGT<-args[6]


GTdata<-read.csv(paste(GTpath,"DETx.csv.gz",sep="/"))
FGdata<-read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"))

#if("fw1difficult" %in% FGdata$Name){
#  stop("fw1difficult. stop for testing")
#}

#just for testing: 

#if("round1_pull1_reduce.csv" %in% FGdata$Name){
#  stop()
#}



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

outData<-merge(outData,FGdataOrd[,c("order","StartFile","DiffTime")],by="StartFile")
#v-10: add difftime
GTdata<-merge(GTdata,FGdataOrd[,c("order","StartFile","DiffTime")],by="StartFile")

#get the relative gt per difftime, use for allocation to cores later. 
GT_dt_table = data.frame(table(GTdata$DiffTime))

GTdata<-GTdata[order(GTdata$order,GTdata$StartFile,GTdata$StartTime),]
outData<-outData[order(outData$order,outData$StartFile,outData$StartTime),]

GTdata$order=NULL
outData$order=NULL

FGdata$FileName<-as.character(FGdata$FileName)
GTdata$StartFile<-as.character(GTdata$StartFile)
GTdata$EndFile<-as.character(GTdata$EndFile)


FGdata$cumsum<- c(0,cumsum(FGdata$Duration)[1:(nrow(FGdata)-1)])
#stealth fix v1-8
FGdataOrd$cumsum<-c(0,cumsum(FGdataOrd$Duration)[1:(nrow(FGdataOrd)-1)])


#convert each dataset into time from FG start instead of offset 

GTlong<-GTdata

if(nrow(GTlong)>0){
  
  #v1-7: do with indexes instead
  #stealth fix v1-8
  
  GTlong$StartTime = GTlong$StartTime + FGdataOrd[match(GTlong$StartFile,FGdataOrd$FileName),"cumsum"]
  GTlong$EndTime = GTlong$EndTime + FGdataOrd[match(GTlong$EndFile,FGdataOrd$FileName),"cumsum"]
  
  #stealth fix 1-9
  #if there are cases where end of a GT box exceeds the FG, truncate it to end of FG. 
  if(any(is.na(GTlong$EndTime))){
    GTlong[is.na(GTlong$EndTime),"EndTime"]=FGdataOrd[match(GTlong[is.na(GTlong$EndTime),"StartFile"],FGdataOrd$FileName),"Duration"]
  }

  GTdata$StartFile<-as.factor(GTdata$StartFile)
  GTdata$EndFile<-as.factor(GTdata$EndFile)
  
  GTlong$iou<-0
  
  #v1-10 if there is case where end is less than start, fix. happens when fg is not necessarily
  #ordered consecutively. 
  
  if(any(GTlong$EndTime<GTlong$StartTime)){
    GTlong[which(GTlong$EndTime<GTlong$StartTime),"EndTime"]= FGdataOrd[match(GTlong[which(GTlong$EndTime<GTlong$StartTime),"StartFile"],FGdataOrd$FileName),"cumsum"]+FGdataOrd[match(GTlong[which(GTlong$EndTime<GTlong$StartTime),"StartFile"],FGdataOrd$FileName),"Duration"] -0.1
  }

}else{
  GTlong<-cbind(GTlong, data.frame(iou=double()))
}

outLong<-outData

#1-6 stealth change



if(nrow(outData)>0){
  
  #v1-7 do this based on matching indeces instead of in loop: 
  #stealth fix v1-8
  
  outLong$StartTime = outLong$StartTime + FGdataOrd[match(outLong$StartFile,FGdataOrd$FileName),"cumsum"]
  outLong$EndTime = outLong$EndTime + FGdataOrd[match(outLong$EndFile,FGdataOrd$FileName),"cumsum"]
  
  #for(i in 1:10000){
  #  print(i)
  #  outLong$StartTime[i]<-outLong$StartTime[i]+FGdata$cumsum[which(FGdata$FileName==outLong$StartFile[i])]
  #  outLong$EndTime[i]<-outLong$EndTime[i]+FGdata$cumsum[which(FGdata$FileName==outLong$EndFile[i])]
  #}
  
  outLong$iou<-0
  
  #v-10 bugfix: if there is case where end is less than start, fix. happens when fg is not necessarily
  #ordered consecutively. 
  
  if(any(outLong$EndTime<outLong$StartTime)){
    outLong[which(outLong$EndTime<outLong$StartTime),"EndTime"]= FGdataOrd[match(outLong[which(outLong$EndTime<outLong$StartTime),"StartFile"],FGdataOrd$FileName),"cumsum"]+FGdataOrd[match(outLong[which(outLong$EndTime<outLong$StartTime),"StartFile"],FGdataOrd$FileName),"Duration"] -0.1
  }
  
  
}else{
  
  outData<-cbind(outData, data.frame(StartTime=double()))
  outData<-cbind(outData, data.frame(EndTime=double()))
  
  
}

GTlong$Dur<-GTlong$EndTime-GTlong$StartTime

#order datasets: 
GTlong<-GTlong[order(GTlong$StartTime),]
outLong<-outLong[order(outLong$StartTime),]

#GTlong = GTlong[which(GTlong$StartFile=="AU-ALPM02_b-210228-211000.wav"),]
#outLong = outLong[which(outLong$StartFile=="AU-ALPM02_b-210228-211000.wav"),]
#GTdata = GTdata[which(GTdata$StartFile=="AU-ALPM02_b-210228-211000.wav"),]
#outData = outData[which(outData$StartFile=="AU-ALPM02_b-210228-211000.wav"),]

GTlongIn=GTlong[which(GTlong$StartFile=="AU-ALPM02_b-210228-210000.wav"),]
outLongIn=outLong[which(outLong$StartFile=="AU-ALPM02_b-210228-210000.wav"),]

#calculate iou GT
if(nrow(GTlong)>0){
  
  crs<-detectCores()
  
  #allocate based on available cores
  
  avg_gt = nrow(GTlong)/crs

  GT_dt_table = GT_dt_table[order(GT_dt_table$Freq),]
  GT_dt_table$core = 0
  GT_dt_table_ind = 1
  
  for(p in 1:crs){
    cumsum_ = 0
    while(cumsum_ < avg_gt){
      
      cumsum_ = cumsum_ + GT_dt_table[GT_dt_table_ind,"Freq"]
      if(GT_dt_table_ind<nrow(GT_dt_table)){
        GT_dt_table[GT_dt_table_ind,"core"]=p
        
        GT_dt_table_ind = GT_dt_table_ind + 1
      }else{
        GT_dt_table[GT_dt_table_ind,"core"]=crs
      }

    }
    
    #readjust on the fly (since above will overshoot):
    
    avg_gt = avg_gt - (cumsum_ - avg_gt)/crs
  }
  
  startLocalPar(crs,"GTlong","outLong","GT_dt_table")
  
  long_comp_out = foreach(p=1:crs) %dopar% {
    
    GTlongIn = GTlong[which(GTlong$DiffTime %in% as.numeric(as.character(GT_dt_table[which(GT_dt_table$core==p),"Var1"]))),]
    outLongIn = outLong[which(outLong$DiffTime %in% as.numeric(as.character(GT_dt_table[which(GT_dt_table$core==p),"Var1"]))),]
    
    for(i in 1:nrow(GTlongIn)){
      
      GTlongInDur<-GTlongIn$EndTime[i]-GTlongIn$StartTime[i]
      if(any(outLongIn$EndTime<(GTlongIn$StartTime[i]-GTlongInDur))){
        klow<-max(which((outLongIn$EndTime<(GTlongIn$StartTime[i]-GTlongInDur)))) #give this a little buffer to avoid issues with small detections preventing longer det from 
      }else{
        klow=1
      }
      if(any((outLongIn$StartTime>(GTlongIn$EndTime[i]+GTlongInDur)))){
        khigh<-min(which((outLongIn$StartTime>(GTlongIn$EndTime[i]+GTlongInDur)))) #fitting the criteria. if wanted to get fancy, could base the buffer length on IOU
      }else{
        khigh=nrow(outLongIn)
      }
      
      k=klow
      if(nrow(outLongIn)>0){
        
        if(klow > khigh){
          stop("bug in detection timestamps detected (klow> khigh), stopping.")
        }
        
        for(k in klow:khigh){
          #while(GTlongIn$iou[i]<IoUThresh&k<=khigh){
          #test for intersection
          if((((GTlongIn$StartTime[i]<outLongIn$EndTime[k] & GTlongIn$StartTime[i]>=outLongIn$StartTime[k])| #GTstart is less than det end, and GT start is after or at start of det
               (GTlongIn$EndTime[i]>outLongIn$StartTime[k] & GTlongIn$StartTime[i]<outLongIn$StartTime[k]))| #GTend is after det end, and GT start is before det start
              (GTlongIn$EndTime[i]>=outLongIn$EndTime[k] & GTlongIn$StartTime[i]<=outLongIn$StartTime[k])) & #gt end is after or at det end, and gt start is at or before det start
             (((GTlongIn$LowFreq[i]<outLongIn$HighFreq[k] & GTlongIn$LowFreq[i]>=outLongIn$LowFreq[k])|
               (GTlongIn$HighFreq[i]>outLongIn$LowFreq[k] & GTlongIn$LowFreq[i]<outLongIn$LowFreq[k])) |
              (GTlongIn$HighFreq[i]>=outLongIn$HighFreq[k] & GTlongIn$LowFreq[i]<=outLongIn$LowFreq[k])))
          {
            
            #test for IoU
            #x1,y1,x2,y2
            box1<-c(GTlongIn$StartTime[i],GTlongIn$LowFreq[i],GTlongIn$EndTime[i],GTlongIn$HighFreq[i])
            box2<-c(outLongIn$StartTime[k],outLongIn$LowFreq[k],outLongIn$EndTime[k],outLongIn$HighFreq[k])
            
            intBox<-c(max(box1[1],box2[1]),max(box1[2],box2[2]),min(box1[3],box2[3]),min(box1[4],box2[4]))
            
            intArea = abs(intBox[3]-intBox[1]) * abs(intBox[4]-intBox[2])
            
            box1Area = abs(box1[3] - box1[1]) * abs(box1[4] - box1[2])
            box2Area = abs(box2[3] - box2[1]) * abs(box2[4] - box2[2])
            
            totArea = box1Area + box2Area - intArea
            
            iou = intArea / totArea
            
            #give GT the best IOU
            if(GTlongIn$iou[i]<iou){
              GTlongIn$iou[i]<-iou
            }
            #v1-8: instead of replacing iou for outLongIn, sum it. 
            outLongIn$iou[k] = outLongIn$iou[k]+ iou
            #if(outLongIn$iou[k]<iou){
            #  outLongIn$iou[k]<-iou
            #}
        
      }
      
          
          #print(i)
          
          #plot(0,xlim=c(min(c(box1[1],box2[1]))-2,max(c(box1[3],box2[3]))+2),ylim=c(0,512),col="white")
          #rect(box1[1],box1[2],box1[3],box1[4],col="green")
          #Sys.sleep(0.25)
          #rect(box2[1],box2[2],box2[3],box2[4],col="gray")
          #Sys.sleep(0.25)
          #rect(intBox[1],intBox[2],intBox[3],intBox[4],col="red")
          #text(mean(c(min(c(box1[1],box2[1]))-2,max(c(box1[3],box2[3]))+2)),256,paste(iou,k))
          #Sys.sleep(0.5)
          
        }
      }
      
    }
    
    return(list(GTlongIn,outLongIn))
    
  }
  stopCluster(cluz)
}



#unpack output object 

gttemp = list()
oltemp = list()
if(nrow(GTdata)>0){
  for(i in 1:crs){
    gttemp[i] = long_comp_out[[i]][1]
    oltemp[i]= long_comp_out[[i]][2]
    
  }
  
  GTlong = do.call("rbind",gttemp)
  GTlong = GTlong[order(GTlong$StartTime),]
  
}


outLong_ =do.call("rbind",oltemp)

#add back in outlong not in gt: 

if(!all(unique(outLong$DiffTime) %in% unique(GTlong$DiffTime))){
  difftimes_ = unique(outLong$DiffTime)[!unique(outLong$DiffTime) %in% unique(GTlong$DiffTime)]
  outLong = rbind(outLong_,outLong[which(outLong$DiffTime %in% difftimes_),])
    
}else{
  outLong = outLong_
}

outLong = outLong[order(outLong$StartTime),]
   
GTlong$DiffTime=NULL
outLong$DiffTime = NULL

if(nrow(outLong)!=nrow(outData)){
  stop("bug detected, parallelizing not working properly with detections.")
}

if(nrow(GTdata)>0){
  if("splits" %in% colnames(outLong)){
    #populate splits for GT, using closest detection (should be close enough considering DL has uniform
    #detection results). 
    
    GTlong$splits = 0
    
    GTlong$FGID = FGdata$Name[1]
    
    if(nrow(outLong)>0){
      
      for(i in 1:nrow(GTlong)){
        GTlong$splits[i]=outLong[which.min(abs(GTlong$StartTime[i]-outLong$StartTime)),"splits"]
      } 
      
      #need to retain FG info too
      
      
      
    }
  }

  
}

outLong$FGID = FGdata$Name[1]

GTlong$StartTime<-GTdata$StartTime
GTlong$EndTime<-GTdata$EndTime

outLong$StartTime<-outData$StartTime
outLong$EndTime<-outData$EndTime

GTlong$label<-ifelse(GTlong$iou<IoUThresh,"FN","TP")
GTlong$label<-as.factor(GTlong$label)
outLong$label<-ifelse(outLong$iou<IoUThresh,"FP","TP")
outLong$label<-as.factor(outLong$label)

GTlong$Dur<-NULL
GTlong$Label<-NULL #I may want this later? But relevant labels past this point should be TP FP FN... 
GTlong$iou<-NULL
outLong$iou<-NULL

if(nrow(outLong)>0){
  if('signal_code' %in% colnames(GTlong)){
    outLong$signal_code<-'out'
  }else{
    outLong$SignalCode<-'out'
  }
}


#add back in probs if present
if("probs" %in% colnames(outLong)){
  if(nrow(GTdata)>0){
    if(nrow(GTlong)>0){
      GTlong$probs<-NA
    }else{
      GTlong<-cbind(GTlong, data.frame(probs=double()))
    }
  }
}

#same as above for alternative naming
if("probability" %in% colnames(outLong)){
  if(nrow(GTlong)>0){
    GTlong$probability<-NA
  }else{
    GTlong<-cbind(GTlong, data.frame(probability=double()))
  }
}

#v1-6 same with cutoff
if("cutoff" %in% colnames(outLong)){
  if(nrow(GTdata)>0){
    if(nrow(GTlong)>0){
      GTlong$cutoff<-outLong$cutoff[1]
    }else{
      GTlong<-cbind(GTlong, data.frame(probs=double()))
    }
  }
}

#make sure columns match (throw out other metdata not relevant here)


if(nrow(GTdata)>0){
  cols <- intersect(colnames(outLong), colnames(GTlong))
  CombineTab<-rbind(GTlong[,cols],outLong[,cols])
}else{
  CombineTab = outLong
}


CombineTab<-CombineTab[order(CombineTab$StartFile,CombineTab$StartTime),]

outName<-paste("DETx.csv.gz",sep="_")

#v1-2:
if(WriteGT=="n"){
  if("SignalCode" %in% colnames(CombineTab)){
    CombineTab<-CombineTab[which(CombineTab$SignalCode=='out'),]
    #remove GT rows
    CombineTab$SignalCode<-NULL
    #remove signal code column- assumed these are 'out' 
  }else if("signal_code" %in% colnames(CombineTab)){
    CombineTab<-CombineTab[which(CombineTab$signal_code=='out'),]
    #remove GT rows
    CombineTab$signal_code<-NULL
  }

}

write.csv(CombineTab,gzfile(paste(resultPath,outName,sep="/")),row.names = FALSE)

