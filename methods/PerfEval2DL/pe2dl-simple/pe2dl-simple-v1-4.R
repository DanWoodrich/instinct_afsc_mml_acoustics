library(PRROC)
library(flux)
#library(tidyverse)
#v1-1: make sure FG information makes it into the outputs. 

args<-"D:/Cache/942573/199414/268525 D:/Cache/942573/199414/268525/761024 D:/Cache/942573/199414/330016/664458/840531 2 n pe2dl-simple-v1-4"
args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#dataPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20"
#resultPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/53d3bb"
#PE1s2path<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/03bc3c/c393b4/6c4546"

#stop()

dataPath<-args[1]
resultPath<-args[2]
PE1path<-args[3]
cexmod = as.numeric(args[4])
suppress_test = args[5]

if(suppress_test != 'y' & suppress_test!= 'n'){
  
  suppress_test = 'n'
}


data<-read.csv(paste(dataPath,"DETx.csv.gz",sep="/"))
PE1data<-read.csv(paste(PE1path,"Stats.csv.gz",sep="/"))

if(length(unique(data$FGID))>1){
  FGName = "All"
  PE1data = PE1data[which(PE1data$FGID=='all'),]
}else{
  FGName = data$FGID[1]
  PE1data = PE1data[which(PE1data$FGID==data$FGID[1]),]
}

#drop FN labels. 
data<-data[which(data$label %in% c("TP","FP")),]

#if there are no TP, or no FP, cannot compute the curve. Just print out a file that says no plot
if(length(unique(data$label))<2){
  
  png(paste(resultPath,"/PRcurve.png",sep=""),width=1000,height = 1000)
  
  par(mar = c(0, 0, 0, 0))
  
  plot(x = 0:10, y = 0:10, ann = F,bty = "n",type = "n",
       xaxt = "n", yaxt = "n")
  
  text(x = 5,y = 5,"Lack of TP or FP: cannot compute PR curve")
  
  dev.off()
  
  setwd(resultPath)
  tar('PE2ball.tgz',compression='gzip')
  
}else{
  
  #v5 also drop labels where probs = NA
  if(any(is.na(data$probs))){
    data<-data[-which(is.na(data$probs)),]
  }
  
  if(any(is.na(data$probability))){
    data<-data[-which(is.na(data$probability)),]
  }
  
  #if still has species code labels, drop them
  
  if("SignalCode" %in% colnames(data)){
    data<-data[which(data$SignalCode=="out"),]
  }
  if("signal_code" %in% colnames(data)){
    data<-data[which(data$signal_code=="out"),]
  }
  
  labelVec<-data$label=="TP"
  labelVec[labelVec]<-1
  
  data$label<-labelVec
  
  png(paste(resultPath,"/PRcurve.png",sep=""),width=1000,height = 1000)
  
  curves = list()
  pe1_ref_points = list()
  
  for(i in 1:3){
    
    datasub = data[which(data$splits==i),] #either subset
    
    statssub = PE1data[which(PE1data$split == i),]
    
    if(nrow(datasub)!=0){
      curve<-pr.curve(scores.class0=datasub$probs,weights.class0 = datasub$label,curve=TRUE)
    }else{
      curve = NA
    }
    
    curves[[i]]=curve
    pe1_ref_points[[i]]=data.frame(statssub$Recall,statssub$Precision,statssub$cutoff_)
    
  }
  
  #so I can compute the curve and plot. right now it plot above the PR curve, which I don't like. 
  #I also think that the probability cutoffs should be 
  par(mgp=c(2.4*cexmod,1*cexmod,0),mar=c(4*cexmod,4*cexmod,2*cexmod,4*cexmod))
  
  par(cex.main=2*cexmod,cex.lab=2*cexmod,cex.axis=1.5*cexmod)
  
  cols = c("Sky blue","Orange","Black")
  
  plot(curves[[unique(data$splits)[1]]], auc.main=FALSE, main =FGName,legend=FALSE,col=cols[unique(data$splits)[1]],cex.axis=1.8*cexmod,lwd = 3*cexmod)
  points(pe1_ref_points[[unique(data$splits)[1]]]$statssub.Recall,pe1_ref_points[[unique(data$splits)[1]]]$statssub.Precision, main ="",legend=FALSE,col=cols[[unique(data$splits)[1]]],cex.axis=1.8*cexmod,lwd = 3*cexmod)
  
  if(length(unique(data$splits))>1){
    for(i in unique(data$splits)[2:length(unique(data$splits))]){
      
      if(i != 3 | suppress_test == 'n'){
        
        plot(curves[[i]], auc.main=FALSE, main ="",legend=FALSE,col=cols[[i]],cex.axis=1.8*cexmod,lwd = 3*cexmod,add=TRUE)
        points(pe1_ref_points[[i]]$statssub.Recall,pe1_ref_points[[i]]$statssub.Precision, main ="",legend=FALSE,col=cols[[i]],cex.axis=1.8*cexmod,lwd = 3*cexmod)
        
      }
    }
  }
  
  
  AUCs = list()
  for(i in 1:3){
    if(!is.na(curves[[i]]) & (i != 3 | suppress_test == 'n')){
      if(!is.na(curves[[i]]$auc.davis.goadrich)){
        AUCs[[i]]=auc(curves[[i]]$curve[,1],curves[[i]]$curve[,2])
      }else{
        AUCs[[i]] = NA
      }
      
    }else{
      AUCs[[i]] = NA
    }
  }
  
  AUCs = do.call('c',AUCs)
  AUCs = round(AUCs,2)
  
  plotsplits = c("Train","Val","Test")[which(!is.na(AUCs))]
  plotsplitscols = cols[which(!is.na(AUCs))]
  
  legend(x=-0.125+cexmod/8,y=0.5, 
         legend = paste(plotsplits,"mAP",AUCs[which(!is.na(AUCs))]), #,round(minDist_CUT_Meanspec,digits=2)),round(CUTmeanspec,digits=2)
         col = plotsplitscols, 
         pch = c("-","-"), 
         bty = "n",
         pt.cex = 5*cexmod, 
         cex = 2*cexmod, 
         text.col = "black", 
         horiz = F , 
         inset = c(0.1))
  
  
  dev.off()
  
  #total artifacts: 
  
  #1. PR curve vector (allows for plotting different species against each other)
  
  #assemble dataframe
  
  outdata = list()
  for(i in unique(data$splits)){
    
    curvesub = curves[[i]]
    
    curvesub<-data.frame(curvesub$curve,i,FGName)
    colnames(curvesub)<-c("Recall","Precision","Cutoff","Split","FGID")
    
    outdata[[i]]=curvesub
    
  }
  
  outdata = do.call("rbind",outdata)
  outName<-"PRcurve.csv.gz"
  write.csv(outdata,gzfile(paste(resultPath,outName,sep="/")),row.names = FALSE)
  
  #2. pr curve
  #did above 
  #3. perf stats (for now, only PR auc)
  outName<-"PRcurve_auc.txt"
  AUCs2 = data.frame(t(AUCs))
  AUCs2 = cbind(FGName,AUCs2)
  
  colnames(AUCs2)=c("FGID",paste(c("Train","Val","Test"),"AUC"))
  write.table(AUCs2,paste(resultPath,outName,sep="/"),quote=FALSE,sep = "\t",row.names=FALSE)
  
  #retrieve later with: 
  #text = readLines(paste(resultPath,outName,sep="/"))
  
  #save all outputs in tarball
  setwd(resultPath)
  tar('PE2ball.tgz',compression='gzip')
  
}
