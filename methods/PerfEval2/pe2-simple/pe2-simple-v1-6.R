library(PRROC)
library(flux)

#v1-2: Add a FP/hour axis on the curve. Try to fix glitches
#v1-2: make graphs bigger, now that they contain more info 
#v1-4: add behavior to subset points of interest if not represented in curve

#v1-6: remove specification of All/FG, have it only be routed from single row or combined rows. Add curvename, since will be distinguishing these by context of plots 
#probably will edit this to be a title later, so more publication ready. Or, make a publication read vers of this later. 

args<-"C:/Apps/INSTINCT/Cache/809545/410308/30699/562853 C:/Apps/INSTINCT/Cache/809545/410308/30699/562853/918553 C:/Apps/INSTINCT/Cache/743615/937472/124470/678362/150214/895258"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#dataPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20"
#resultPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/53d3bb"
#PE1s2path<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/03bc3c/c393b4/6c4546"

dataPath<-args[1]
resultPath<-args[2]
PE1path<-args[3]
#DataType<-args[4]


data<-read.csv(paste(dataPath,"DETx.csv.gz",sep="/"))
PE1data<-read.csv(paste(PE1path,"Stats.csv.gz",sep="/"))

EDrecDiff<-1-PE1data$Recall[nrow(PE1data)]
Hours<-PE1data$EffortHours[nrow(PE1data)]
CurveName<-PE1data$FGID[nrow(PE1data)]

#drop FN labels. 
data<-data[which(data$label %in% c("TP","FP")),]

#v5 also drop labels where probs = NA
if(any(is.na(data$probs))){
  data<-data[-which(is.na(data$probs)),]
}

#if still has species code labels, drop them

if("SignalCode" %in% colnames(data)){
  data<-data[which(data$SignalCode=="out"),]
}

labelVec<-data$label=="TP"
labelVec[labelVec]<-1

data$label<-labelVec

png(paste(resultPath,"/PRcurve.png",sep=""),width=1000,height = 1000)

#compute PR curve
curve<-pr.curve(scores.class0=data$probs,weights.class0 = data$label,curve=TRUE)
curve$curve[,1]<-curve$curve[,1]-EDrecDiff

curve$curve[which(curve$curve[,1]<0),1]<-0 #set this to 0 instead of negative when applicable

#compute FP curve 

pointsOfInterest<-c(0.01,0.2,0.5,0.65,0.75,0.85,0.95,0.99)
pointsOfInterestChar<-c(".01",".2",".5",".65",".75",".85",".95",".99")

CurveRound<-round(curve$curve[,3],digits=2)


#pseudo: at each curve step, add a value corresponding to the FPs remaining at that cutoff divided by the total effort 
FPperHrs<-vector('numeric',length=nrow(curve$curve))
for(n in 1:nrow(curve$curve)){
  thresh<-curve$curve[n,3]
  datLabs<-data[which(data$probs>=thresh),"label"]
  FPperHr<-sum(datLabs==0)/Hours
  FPperHrs[n]<-FPperHr
}

#ok, the condition is if there are NA's in PR curve (no positives), we will instead
#plot a curve that just has FP/hr on the y, and cutoff on the X. Keep y axes on right side for consistency

par(cex.main=2,cex.lab=2,cex.axis=1.5)

if(sum(data$label==1)>0){
  
  xvals<-c()
  for(n in 1:length(pointsOfInterest)){
    x=curve$curve[median(which(CurveRound==pointsOfInterest[n])),1]
    xvals<-c(xvals,x)
  }
  
  xvalssUse<-!is.na(xvals)
  
  cutoffPos<-xvals[xvalssUse][length(xvals[xvalssUse])]-0.1
  cutoffVec<-c(xvals[xvalssUse],cutoffPos)
  labVec<-c(pointsOfInterestChar[xvalssUse],"Cutoffs:")
  if(cutoffPos<0){
    cutoffPos<-xvals[xvalssUse][1]+0.1
    cutoffVec<-c(cutoffPos,xvals[xvalssUse])
    labVec<-c("<-Cutoffs",pointsOfInterestChar[xvalssUse])
  }
  
  
par(mgp=c(2.4,1,0),mar=c(4,4,2,4))
  
plot(curve$curve[,1], log10(FPperHrs),col="white",axes=F, xlab=NA, ylab=NA,cex.axis=1.8,xlim=c(0,1))

for(n in xvals){
  abline(v=n,lty=3)
}

lines(curve$curve[,1], log10(FPperHrs),lwd=3,col="gray")
aty <- axTicks(2)
aty[which(aty%%1!=0)]<-""
labels <- sapply(aty,function(i)
  as.expression(bquote(10^ .(i)))
)

axis(side = 4,at=aty,labels=labels, cex.axis = 1.8, las = 3)
axis(side = 3,at=cutoffVec,labels=labVec, cex.axis = 0.75, tick = FALSE, padj = 1.9)
mtext(side = 4, line = 2.5, 'False Positives per Hour',cex=1.8)

#so I can compute the curve and plot. right now it plot above the PR curve, which I don't like. 
#I also think that the probability cutoffs should be 
par(new = T,cex.main=2,cex.lab=2,cex.axis=1.5)

plot(curve, auc.main=FALSE, main ="",legend=FALSE,col="black",cex.axis=1.8)


legend(x=-0.025,y=0.5, 
       legend = c("PR curve","FP/HR curve"), #,round(minDist_CUT_Meanspec,digits=2)),round(CUTmeanspec,digits=2)
       col = c("black","gray"), 
       pch = c("-","-"), 
       bty = "n",
       pt.cex = 5, 
       cex = 2, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.1))

PRauc<-auc(curve$curve[,1],curve$curve[,2])

text(0.33,0.20,paste("Average          ",CurveName,"\nprecision =",round(PRauc,digits=2)),cex=2)

}else{

  par(mgp=c(2.4,1,0),mar=c(4,4,0,0.5))
  plot(curve$curve[,1], log10(FPperHrs),col="white",axes=F, xlab=NA, ylab=NA,cex.axis=1.8,xlim=c(0,1))
  lines(curve$curve[,3], log10(FPperHrs),lwd=3,col="gray")
  box(col = "black")
  
  aty <- axTicks(2)
  aty[which(aty%%1!=0)]<-""
  labels <- sapply(aty,function(i)
    as.expression(bquote(10^ .(i)))
  )
  
  axis(side = 2,at=aty,labels=labels, cex.axis = 1.8, las = 3, padj = 0.25)
  mtext(side = 2, line = 2.5, 'False Positives per Hour',cex=1.8)
  
  axis(side = 1)
  mtext(side = 1, line = 2.5, 'Probability Cutoff',cex=1.8)
  #this is the plot if no PR curve is available (no TPs.. )
  PRauc=NA
}

dev.off()



#total artifacts: 

#1. PR curve vector (allows for plotting different species against each other)
PRcurve<-data.frame(curve$curve)
colnames(PRcurve)<-c("Recall","Precision","Cutoff")
outName<-"PRcurve.csv.gz"
write.csv(PRcurve,gzfile(paste(resultPath,outName,sep="/")),row.names = FALSE)

#2. pr curve
#did above 
#3. perf stats (for now, only PR auc)
outName<-"PRcurve_auc.txt"
write.table(PRauc,paste(resultPath,outName,sep="/"),quote=FALSE,sep = "\t",row.names=FALSE,col.names=FALSE)

#retrieve later with: 
#text = readLines(paste(resultPath,outName,sep="/"))
