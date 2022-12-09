#v1-1: change so that this has more normal behavior to act on a single FG or all, and not loop within the script. This allows
#for no unnecessary rerun when swapping out fg. 


args<-"C:/Apps/INSTINCT/Cache/226127 C:/Apps/INSTINCT/Cache/226127/894219/214644/165905 C:/Apps/INSTINCT/Cache/226127/894219/214644/165905/300803 n pe1dl-simple-v1-2"
args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

calc_stats = function(FGID,data,FG,rsp,cutoff_){
  
  detTotal<-nrow(data[which(data$SignalCode=="out"),])
  numTPtruth<-nrow(data[which(data$SignalCode!="out"),])
  numTP <- nrow(data[which(data$SignalCode!="out"&data$label=="TP"),])
  numTPout <- nrow(data[which(data$SignalCode=='out'&data$label=="TP"),])
  numFP <- nrow(data[which(data$label=="FP"),])
  numFN <- nrow(data[which(data$SignalCode!="out"&data$label=="FN"),])
  
  Recall <- numTP/(numTP+numFN)                   
  Precision <- numTPout/(detTotal) #changed this to numTP out, was getting inflated precision readings. 
  F1<- (2 * Precision * Recall) / (Precision + Recall)
  OMB<-(numTP/numTPout) #values over one indicate 
  
  #correlate the FN and TP sf bin totals. Values near 1 imply less time phenomena based and more random distribution of FN, values closer to 1 imply less random distributed negatives and more related to phenomena
  #make vector of sound file bins
  allFiles<-unique(data$StartFile)
  
  TPvec<-as.numeric(table(c(data[which(data$SignalCode!="out"&data$label=="TP"),"StartFile"],allFiles)))-1
  FNvec<-as.numeric(table(c(data[which(data$SignalCode!="out"&data$label=="FN"),"StartFile"],allFiles)))-1
  
  HitMissCor<-cor(TPvec,FNvec)
  
  #stats related to dispersion of calls over time 
  EffortHours<-sum(FG$SegDur)/3600 * as.numeric(rsp[i])
  TPperHour<-numTPtruth/((sum(FG$SegDur)/3600)* as.numeric(rsp[i]))
  TPdetperHour<-numTP/((sum(FG$SegDur)/3600)* as.numeric(rsp[i]))
  FPperHours<-numFP/((sum(FG$SegDur)/3600)* as.numeric(rsp[i]))
  numFNperHOur<-numFN/((sum(FG$SegDur)/3600)* as.numeric(rsp[i]))
  
  split = data$split[1]
  
 # pr_auc = integrate(approxfun(data$x, data$y), 0, 0.6)
  
  return(data.frame(cbind(FGID,split,detTotal,numTPtruth,numTP,numTPout,numFP,numFN,cutoff_,Recall,Precision,F1,OMB,HitMissCor,EffortHours,TPperHour,TPdetperHour,FPperHours,numFNperHOur)))
}

#dataPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20"
#resultPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/53d3bb"
#PE1s2path<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/03bc3c/c393b4/6c4546"

#
FGpath <- args[1]
dataPath<-args[2]
resultPath <- args[3]
suppress_test = args[4]


if(suppress_test != 'y' & suppress_test!= 'n'){
  
  suppress_test = 'n'
}

FG = read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"))
data = read.csv(paste(dataPath,"DETx.csv.gz",sep="/"))

#print(str(FG))
#print(str(data))

#stop()

if(length(unique(data$FGID))>1){
  FGname = "all"
}else{
  FGname =  data$FGID[1]
}

datanotp = data[which(data$SignalCode=="out"),]

#do this for the minumum cutoff. If a very low value is not supplied (preferably 0) this estimate could be off. 
#actually, just enforce that there is a 0 cutoff: 

if((! 0 %in% unique(data$cutoff))){
  stop("must provide 0  as a reference point to correctly calculate stats.")
}

zerocutdata = datanotp[which(datanotp$cutoff==0),]

rough_split_prop=table(round(zerocutdata$splits))/nrow(zerocutdata)
rough_split_prop=round(rough_split_prop,3) #round to 2 digits


#take the split information of the 0 cutoff TPs, and apply them to the other tp data
#pseudo: extract tp from 0. For each other cutoff, sort the GT so it's the same order, 
#and then replace GT split with that of the 0 cutoff. 

#then, add the GT back in with a cutoff of 1 and call them all FN. 

zerocutdata_tp = data[which(data$cutoff==0),]

zeroGT = zerocutdata_tp[which(zerocutdata_tp$SignalCode!="out"),]
zeroGT = zeroGT[order(zeroGT$StartTime),] #order by a double. Hopefully consistent in cases of ties but that's unlikely. 

cut_loop = unique(data$cutoff)[which(unique(data$cutoff)!=0)]

if(length(cut_loop)>0){
  for(i in 1:length(cut_loop)){
    
    #extract GT from cut and replace with 0
    
    GTin = data[which(data$SignalCode!="out" & data$cutoff==cut_loop[i]),]
    GTin = GTin[order(GTin$StartTime),]
    
    if(any(GTin$StartTime-zeroGT$StartTime)>0){
      stop("incorrect assumption")
    }
    
    GTin$splits=zeroGT$splits
    
    data = data[-which(data$SignalCode!="out" & data$cutoff==cut_loop[i]),]
    
    data=rbind(data,GTin)
    
  }

}

#I don't really even know if this makes sense- no data out is not a valid point on curve since you divide by 0. 

#GT_none = GTin
#GT_none$cutoff=0.9999999999999
#GT_none$label="FN"

#add this to the dataset
#data = rbind(data,GT_none)

#pseudo
#compute stats
#save as FG row or all and one row per split.
#and, per cutoff.
outtab = list()

if(FGname=='all'){
  
  #rough_split_effort = sum(FG$SegDur)*rough_split_prop
  
  for(i in 1:length(rough_split_prop)){
    
    data_split = data[which(data$splits==as.numeric(names(rough_split_prop)[i])),]
    
    #subset to split
    
    Intab = list()
    
    for(p in 1:length(unique(data_split$cutoff))){
      
      data_split_cutoff = data_split[which(data_split$cutoff==unique(data_split$cutoff)[p]),]
      
      #subset to cutoffs
      
      Intab[[p]]<- calc_stats(FGname,data_split_cutoff,FG,rough_split_prop,as.numeric(unique(data_split$cutoff)[p]))
      
      #write.csv(Stats,gzfile(paste(resultPath,"Stats.csv.gz",sep="/")),row.names = FALSE)
      
    }
    
    Intab=do.call("rbind",Intab)
    
    #not sure yet if I want to include this... see how points compare to pe2
    
    #pr_auc = integrate(approxfun(x=as.numeric(Intab$Recall), y=as.numeric(Intab$Precision)),min(as.numeric(Intab$Recall)),max(as.numeric(Intab$Recall)))
    
    #plot(as.numeric(Intab$Recall),as.numeric(Intab$Precision),xlim = c(0,1),ylim=c(0,1))
    
    #Intab$pr_auc = pr_auc
    
    outtab[[i]]=Intab
  }
    
}else{

count=length(outtab)+1
for(k in 1:length(unique(data$FGID))){
  
  focal_FG = FG[which(FG$Name==unique(data$FGID)[k]),] #v1-1 some of the naming and subsetting are unnecessary now, grandfather from v1-0 behavior
  
  #rough_split_effort = sum(focal_FG$SegDur)*rough_split_prop
  
  focal_data = data[which(data$FGID==unique(data$FGID)[k]),]
  
  focal_datanotp = data[which(data$SignalCode=="out"),]
  
  focal_zerocutdata = focal_datanotp[which(focal_datanotp$cutoff==min(focal_datanotp$cutoff)),] 
  
  focal_rough_split_prop=table(round(focal_zerocutdata$splits))/nrow(focal_zerocutdata)
  focal_rough_split_prop=round(focal_rough_split_prop,3) #round to 2 digits
  
  for(i in 1:length(focal_rough_split_prop)){
    
    focal_data_split = focal_data[which(focal_data$splits==as.numeric(names(focal_rough_split_prop)[i])),]
    
    #subset to split
    
    for(p in 1:length(unique(focal_data_split$cutoff))){
      
      focal_data_split_cutoff = focal_data_split[which(focal_data_split$cutoff==unique(focal_data_split$cutoff)[p]),]
      
      #subset to cutoffs
      
      outtab[[count]]<- calc_stats(unique(data$FGID)[k],focal_data_split_cutoff,focal_FG,focal_rough_split_prop,as.numeric(unique(focal_data_split$cutoff)[p]))
      
      count= count+1
      
      
    }
  }
  
  
}
}



outtab = do.call("rbind",outtab)

if(suppress_test =='y' & 3 %in% outtab$split){
  outtab = outtab[which(outtab$split!=3),]
}

#go back through, and for every fg/split

#determine baseline precision:
#where cutoff = 1:

#calculate auc: 

write.csv(outtab,gzfile(paste(resultPath,"Stats.csv.gz",sep="/")),row.names = FALSE)
