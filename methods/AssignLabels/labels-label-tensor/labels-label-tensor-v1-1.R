library(foreach)
library(doParallel)

# Method ID: labels-w-purity-simple-v1-0
# Description: Evaluates detections based on "Purity" (Intersection / Detection Area) 
# instead of IoU, to match sliding window training density thresholds.

args="D:/Cache/264078 D:/Cache/264078/421141 D:/Cache/865375/401113/370627/843649/777039/772363 D:/Cache/264078/421141/277887 0.001 y labels-w-purity-simple-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

# --- INPUTS ---
FGpath <- args[1]
GTpath <- args[2]
DETpath <- args[3]
resultPath <- args[4]
PercTP <- as.numeric(args[5]) # Threshold for Purity (e.g., 0.5)
WriteGT <- args[6]

# --- LOAD DATA ---
GTdata<-read.csv(paste(GTpath,"DETx.csv.gz",sep="/"))
FGdata<-read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"))
outData<-read.csv(paste(DETpath,"DETx.csv.gz",sep="/"))

# --- PRE-PROCESSING & ORDERING ---
FGdata$order<-1:nrow(FGdata)
FGdata$StartFile = FGdata$FileName

# Remove duplicated start files for ordering purposes
if(any(duplicated(FGdata$StartFile))){ 
  FGdataOrd = FGdata[-which(duplicated(FGdata$StartFile)),]
}else{
  FGdataOrd=FGdata
}

# Merge sorting info
outData<-merge(outData,FGdataOrd[,c("order","StartFile","DiffTime")],by="StartFile")
GTdata<-merge(GTdata,FGdataOrd[,c("order","StartFile","DiffTime")],by="StartFile")

# Table for parallel allocation
GT_dt_table = data.frame(table(GTdata$DiffTime))

# Sort
GTdata<-GTdata[order(GTdata$order,GTdata$StartFile,GTdata$StartTime),]
outData<-outData[order(outData$order,outData$StartFile,outData$StartTime),]

GTdata$order=NULL
outData$order=NULL

FGdata$FileName<-as.character(FGdata$FileName)
GTdata$StartFile<-as.character(GTdata$StartFile)
GTdata$EndFile<-as.character(GTdata$EndFile)

# Calculate Cumulative Time offsets
FGdata$cumsum<- c(0,cumsum(FGdata$Duration)[1:(nrow(FGdata)-1)])
FGdataOrd$cumsum<-c(0,cumsum(FGdataOrd$Duration)[1:(nrow(FGdataOrd)-1)])

# --- TIME ADJUSTMENT (Relative to FG Start) ---

# Process Ground Truth
GTlong<-GTdata
if(nrow(GTlong)>0){
  GTlong$StartTime = GTlong$StartTime + FGdataOrd[match(GTlong$StartFile,FGdataOrd$FileName),"cumsum"]
  GTlong$EndTime = GTlong$EndTime + FGdataOrd[match(GTlong$EndFile,FGdataOrd$FileName),"cumsum"]
  
  # Truncate if extends beyond file
  if(any(is.na(GTlong$EndTime))){
    GTlong[is.na(GTlong$EndTime),"EndTime"]=FGdataOrd[match(GTlong[is.na(GTlong$EndTime),"StartFile"],FGdataOrd$FileName),"Duration"]
  }
  
  GTdata$StartFile<-as.factor(GTdata$StartFile)
  GTdata$EndFile<-as.factor(GTdata$EndFile)
  
  # Initialize Purity column
  GTlong$purity<-0 
  
  # Fix inverted timestamps
  if(any(GTlong$EndTime<GTlong$StartTime)){
    GTlong[which(GTlong$EndTime<GTlong$StartTime),"EndTime"]= FGdataOrd[match(GTlong[which(GTlong$EndTime<GTlong$StartTime),"StartFile"],FGdataOrd$FileName),"cumsum"]+FGdataOrd[match(GTlong[which(GTlong$EndTime<GTlong$StartTime),"StartFile"],FGdataOrd$FileName),"Duration"] - 0.001
  }
}else{
  GTlong<-cbind(GTlong, data.frame(purity=double()))
}

# Process Detections
outLong<-outData
if(nrow(outData)>0){
  outLong$StartTime = outLong$StartTime + FGdataOrd[match(outLong$StartFile,FGdataOrd$FileName),"cumsum"]
  outLong$EndTime = outLong$EndTime + FGdataOrd[match(outLong$EndFile,FGdataOrd$FileName),"cumsum"]
  
  # Initialize Purity column
  outLong$purity<-0
  
  # Fix inverted timestamps
  if(any(outLong$EndTime<outLong$StartTime)){
    outLong[which(outLong$EndTime<outLong$StartTime),"EndTime"]= FGdataOrd[match(outLong[which(outLong$EndTime<outLong$StartTime),"StartFile"],FGdataOrd$FileName),"cumsum"]+FGdataOrd[match(outLong[which(outLong$EndTime<outLong$StartTime),"StartFile"],FGdataOrd$FileName),"Duration"] - 0.001
  }
}else{
  outData<-cbind(outData, data.frame(StartTime=double()))
  outData<-cbind(outData, data.frame(EndTime=double()))
}

GTlong$Dur<-GTlong$EndTime-GTlong$StartTime

# Final Sort before Comparison
GTlong<-GTlong[order(GTlong$StartTime),]
outLong<-outLong[order(outLong$StartTime),]


# --- CORE PROCESSING: PURITY CALCULATION ---

if(nrow(GTlong)>0){
  
  crs<-detectCores()
  
  # Dynamic Load Balancing
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
    avg_gt = avg_gt - (cumsum_ - avg_gt)/crs
  }
  
  startLocalPar(crs,"GTlong","outLong","GT_dt_table")
  
  long_comp_out = foreach(p=1:crs) %dopar% {
    
    # Subset data for this core
    GTlongIn = GTlong[which(GTlong$DiffTime %in% as.numeric(as.character(GT_dt_table[which(GT_dt_table$core==p),"Var1"]))),]
    outLongIn = outLong[which(outLong$DiffTime %in% as.numeric(as.character(GT_dt_table[which(GT_dt_table$core==p),"Var1"]))),]
    
    for(i in 1:nrow(GTlongIn)){
      
      # Define search window based on GT duration
      GTlongInDur<-GTlongIn$EndTime[i]-GTlongIn$StartTime[i]
      
      # Find potential overlapping detections (Time only filter)
      if(any(outLongIn$EndTime<(GTlongIn$StartTime[i]-GTlongInDur))){
        klow<-max(which((outLongIn$EndTime<(GTlongIn$StartTime[i]-GTlongInDur)))) 
      }else{
        klow=1
      }
      if(any((outLongIn$StartTime>(GTlongIn$EndTime[i]+GTlongInDur)))){
        khigh<-min(which((outLongIn$StartTime>(GTlongIn$EndTime[i]+GTlongInDur)))) 
      }else{
        khigh=nrow(outLongIn)
      }
      
      if(nrow(outLongIn)>0){
        if(klow > khigh){ stop("Timestamp bug in indexing") }
        
        for(k in klow:khigh){
          # Detailed Intersection Check (Time + Frequency)
          if((((GTlongIn$StartTime[i]<outLongIn$EndTime[k] & GTlongIn$StartTime[i]>=outLongIn$StartTime[k])| 
               (GTlongIn$EndTime[i]>outLongIn$StartTime[k] & GTlongIn$StartTime[i]<outLongIn$StartTime[k]))| 
              (GTlongIn$EndTime[i]>=outLongIn$EndTime[k] & GTlongIn$StartTime[i]<=outLongIn$StartTime[k])) & 
             (((GTlongIn$LowFreq[i]<outLongIn$HighFreq[k] & GTlongIn$LowFreq[i]>=outLongIn$LowFreq[k])|
               (GTlongIn$HighFreq[i]>outLongIn$LowFreq[k] & GTlongIn$LowFreq[i]<outLongIn$LowFreq[k])) |
              (GTlongIn$HighFreq[i]>=outLongIn$HighFreq[k] & GTlongIn$LowFreq[i]<=outLongIn$LowFreq[k])))
          {
            
            # --- PURITY CALCULATION ---
            
            # Define Boxes
            box1<-c(GTlongIn$StartTime[i],GTlongIn$LowFreq[i],GTlongIn$EndTime[i],GTlongIn$HighFreq[i]) # GT
            box2<-c(outLongIn$StartTime[k],outLongIn$LowFreq[k],outLongIn$EndTime[k],outLongIn$HighFreq[k]) # Detection
            
            # Calculate Intersection
            intBox<-c(max(box1[1],box2[1]),max(box1[2],box2[2]),min(box1[3],box2[3]),min(box1[4],box2[4]))
            intArea = abs(intBox[3]-intBox[1]) * abs(intBox[4]-intBox[2])
            
            # Calculate Detection Area
            box2Area = abs(box2[3] - box2[1]) * abs(box2[4] - box2[2]) 
            
            # Calculate Purity (Intersection / Detection Area)
            purity = intArea / box2Area
            
            # Update GT: Record the highest purity of any detection that touched this GT
            if(GTlongIn$purity[i] < purity){
              GTlongIn$purity[i] <- purity
            }
            
            # Update Detection: Record the highest purity this detection achieved against any GT
            if(outLongIn$purity[k] < purity){
              outLongIn$purity[k] <- purity
            }
          }
        }
      }
    }
    return(list(GTlongIn,outLongIn))
  }
  stopCluster(cluz)
}

# --- RE-ASSEMBLY ---

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

# Add back in outlong not in gt (processed on cores): 
if(!all(unique(outLong$DiffTime) %in% unique(GTlong$DiffTime))){
  difftimes_ = unique(outLong$DiffTime)[!unique(outLong$DiffTime) %in% unique(GTlong$DiffTime)]
  outLong = rbind(outLong_,outLong[which(outLong$DiffTime %in% difftimes_),])
}else{
  outLong = outLong_
}

outLong = outLong[order(outLong$StartTime),]

# Clean up temporary time columns
GTlong$DiffTime=NULL
outLong$DiffTime = NULL

if(nrow(outLong)!=nrow(outData)){
  stop("Row mismatch detection - parallelization error.")
}

# --- METADATA MERGE (SPLITS) ---
if(nrow(GTdata)>0){
  if("splits" %in% colnames(outLong)){
    GTlong$splits = 0
    GTlong$FGID = FGdata$Name[1]
    if(nrow(outLong)>0){
      for(i in 1:nrow(GTlong)){
        # Assign closest detection's split to the GT (approximate)
        GTlong$splits[i]=outLong[which.min(abs(GTlong$StartTime[i]-outLong$StartTime)),"splits"]
      } 
    }
  }
}

outLong$FGID = FGdata$Name[1]

# Restore original timestamps (relative to file, not FG)
GTlong$StartTime<-GTdata$StartTime
GTlong$EndTime<-GTdata$EndTime
outLong$StartTime<-outData$StartTime
outLong$EndTime<-outData$EndTime

# --- LABEL ASSIGNMENT ---
# Strictly uses Purity vs PercTP
GTlong$label<-ifelse(GTlong$purity < PercTP, "FN", "TP")
GTlong$label<-as.factor(GTlong$label)
outLong$label<-ifelse(outLong$purity < PercTP, "FP", "TP")
outLong$label<-as.factor(outLong$label)

# --- CLEANUP ---
# Remove Purity column before export (optional, matching previous cleanup style)
GTlong$Dur<-NULL
GTlong$purity<-NULL 
outLong$purity<-NULL

if(nrow(outLong)>0){
  if('signal_code' %in% colnames(GTlong)){
    outLong$signal_code<-'out'
  }else{
    outLong$SignalCode<-'out'
  }
}

# Restore metadata (Probs/Cutoff)
if("probs" %in% colnames(outLong)){
  if(nrow(GTdata)>0){
    if(nrow(GTlong)>0){
      GTlong$probs<-NA
    }else{
      GTlong<-cbind(GTlong, data.frame(probs=double()))
    }
  }
}

if("probability" %in% colnames(outLong)){
  if(nrow(GTlong)>0){
    GTlong$probability<-NA
  }else{
    GTlong<-cbind(GTlong, data.frame(probability=double()))
  }
}

if("cutoff" %in% colnames(outLong)){
  if(nrow(GTdata)>0){
    if(nrow(GTlong)>0){
      GTlong$cutoff<-outLong$cutoff[1]
    }else{
      GTlong<-cbind(GTlong, data.frame(probs=double()))
    }
  }
}

# --- FINAL MERGE & WRITE ---

if(nrow(GTdata)>0){
  cols <- intersect(colnames(outLong), colnames(GTlong))
  CombineTab<-rbind(GTlong[,cols],outLong[,cols])
}else{
  CombineTab = outLong
}

CombineTab<-CombineTab[order(CombineTab$StartFile,CombineTab$StartTime),]

outName<-paste("DETx.csv.gz",sep="_")

if(WriteGT=="n"){
  if("SignalCode" %in% colnames(CombineTab)){
    CombineTab<-CombineTab[which(CombineTab$SignalCode=='out'),]
    CombineTab$SignalCode<-NULL
  }else if("signal_code" %in% colnames(CombineTab)){
    CombineTab<-CombineTab[which(CombineTab$signal_code=='out'),]
    CombineTab$signal_code<-NULL
  }
}

write.csv(CombineTab,gzfile(paste(resultPath,outName,sep="/")),row.names = FALSE)