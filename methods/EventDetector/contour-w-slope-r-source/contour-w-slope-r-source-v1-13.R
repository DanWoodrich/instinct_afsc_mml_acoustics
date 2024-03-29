libraries<-c("imager","pracma","caTools")
librariesToLoad<-c("imager","pracma")
nameSpaceFxns<-c("runquantile")

#V1-0: this mainly uses the contour algorithm from pracma for detection. Right now, it takes parameters that attempt to 
#weed out FP using slope and island size. Slope is crude, if I like it probably should use hough lines instead. 
#v1-2: this uses a simple sound window for noise reduction, instead of an image wide averaging.
#v1-3: fix bug from 1-2 (1-2 broken...)
#v1-4 default to only taking calls with positive slope and of certain size (no more pixthreshdiv)
#v1-5: This will be a larger experiment. Try a couple things: 
#instead of rolling median, try a rolling noise percentile. 
#try to isoblur prior to noise reduction to get smoother values. 
#if this works well, consider making the noise window and percentile into parameters. 

#v1-6: add hough determination of slope
#remove isoblur, which acts differently at different spectrogram time scales (unpredictable)

#v1-7: add blurring step to hough line measurement. Lower thresh to 0.65 for max score, and use median instead of mean. 

#v1-8: Have libraries load from this script. 

#v1-9: Make params loadable from INSTINCT. Make general to downsweeps also. 
#v1-10:

#v12- branch from v9. thresholds again after smoothing. 
EventDetectoR<-function(soundFile=NULL,spectrogram=NULL,dataMini=NULL,ParamArgs=NULL,verbose=NULL,metadata=NULL){
  
  CombineDets<-ParamArgs[1]
  CombInt<-as.numeric(ParamArgs[2])
  DesiredSlope<-ParamArgs[3]#"Upsweep"
  highFreq<-as.numeric(ParamArgs[4])
  houghSlopeMax<-as.numeric(ParamArgs[5])
  houghSlopeMin<-as.numeric(ParamArgs[6])
  ImgThresh1=paste(as.integer(ParamArgs[7]),"%",sep="")#"85"
  ImgThresh2=paste(as.integer(ParamArgs[8]),"%",sep="")#"50"
  IsoblurSigma1=as.numeric(ParamArgs[9])#  1.2
  IsoblurSigma2=as.numeric(ParamArgs[10])#  2
  lowFreq<-as.numeric(ParamArgs[11])
  noiseThresh<-as.numeric(ParamArgs[12]) #0.9
  noiseWinLength<-as.numeric(ParamArgs[13]) #2.5
  Overlap<-as.numeric(ParamArgs[14]) 
  pixThresh<-as.numeric(ParamArgs[15])#  100
  #t_samp_rate
  windowLength<-as.numeric(ParamArgs[17]) #
  
  #for this 
  
  #idea: find some noise value for segments. Normalize from high to low freq probably. (whiten)
  #then, just run the countour() algorithm to return detections. 

  if(!is.null(soundFile)){
    #optional spectrogram calculation. Only do if soundFile is passed
    spectrogram<- specgram(x = soundFile@left,
                           Fs = soundFile@samp.rate,
                           window=windowLength,
                           overlap=Overlap
    )
  }
  
  #use for testing upsweeps: inverts the matrix : 
  #spectrogram$S <- apply(spectrogram$S, 2, rev)
  
  #band limit spectrogram
  spectrogram$S<-spectrogram$S[which(spectrogram$f>=lowFreq&spectrogram$f<highFreq),]
  spectrogram$f<-spectrogram$f[which(spectrogram$f>=lowFreq&spectrogram$f<highFreq)]
  
  P = spectrogram$S
  P = abs(P)
  
  image1<-as.cimg(as.numeric(t(P)),x=dim(P)[2],y=dim(P)[1])
  #image1<-resize(image1,size_x=TileAxisSize,size_y=TileAxisSize) #this distorts the image, may or may not matter. Allows for standardized 
  #contour size thresholding
  
  #image1<-isoblur(image1,sigma=1)
  
  P=t(as.matrix(image1))
  
  #Pold=P
  
  #extent<-500:1200
  #call at ~308
  #call at 2100.28
  
  #P= Pold[,extent]
  
  #tempogram<-spectrogram
  #tempogram$S<-P
  #tempogram$t<-tempogram$t[extent]
  #plot(tempogram)

  tAdjust=length(soundFile@left)/soundFile@samp.rate/length(spectrogram$t)
  
  noiseWinLength = round(noiseWinLength/tAdjust) #figure out a way to round this to nearest odd number 
  
  #could replace this with a moving window for more precise calculation
  #or, with a 'smart' window that doesn't average between big swings, and instead averages within these (such as in the case of mooring noise)
  for(k in 1:nrow(P)){
    med<-runquantile(P[k,], noiseWinLength, noiseThresh,endrule = "quantile")
    #
    #test=P[k,]-med
    #test[which(test<0)]<-0
    #plot(P[k,],main=k)
    #lines(med,col="blue")
    #abline(v=315,col="red")
    P[k,]<-P[k,]-med
    
    P[k,][which(P[k,]<0)]<-0
    #plot(P[k,],main=k)
    #abline(v=315,col="red")
    
    #P[k,]<-abs(P[k,])
      #ksmooth(1:length(P[k,]),P[k,],bandwidth=20)$y
    
    #one thing to try: could mark mooring noise 
    
    
    #test<-P[k,]-ksmooth(1:length(P[k,]),P[k,],bandwidth=100)$y
    #test<-test-ksmooth(1:length(test),test,bandwidth=100)$y
    #test[which(test<0)]<-0
    #plot(test)
  }
  image1<-as.cimg(t(P))
  image1<-as.cimg(image1[,dim(image1)[2]:1,,])
  #plot(as.cimg(image1[4500:5500,,,]))
  
  #plot(image1)
  image1<-threshold(image1,ImgThresh1) #changed order of this and next line 
  image1<-isoblur(image1,sigma=IsoblurSigma1)
  image1<-threshold(image1,"90%")
  #plot(image1)
  #plot(as.cimg(image1[4500:5500,,,]))
  
  #image1<-clean(image1,ImgNoiseRedPower) %>% imager::fill(ImgFillPower) 
  
  #Black border so edge islands are respected 
  image1[1,1:dim(image1)[2],1,1]<-FALSE
  image1[1:dim(image1)[1],1,1,1]<-FALSE
  image1[dim(image1)[1],1:dim(image1)[2],1,1]<-FALSE #get rid of side border artifact 
  image1[1:dim(image1)[1],dim(image1)[2],1,1]<-FALSE 
  
  cont<-contours(image1)
  
  size<-vector(mode="numeric", length=length(cont))
  slope<-vector(mode="numeric", length=length(cont))

  for(i in 1:length(cont)){
    size[i]<-abs(polyarea(cont[[i]]$x,cont[[i]]$y))
    
    #don't bother with super small ones 
    if(size[i]>=pixThresh){ #& size[i]<750
      
      #if(any(cont[[i]]$x>4600)){
      #  stop()
      #}
    
      xs=round(min((cont[[i]]$x)-1):(max(cont[[i]]$x)+1))
      if(any(xs<0)){
        xs<-xs+1
      }
      ys=round((min(cont[[i]]$y)-1):(max(cont[[i]]$y)+1))
      if(any(ys>(length(soundFile@left)/soundFile@samp.rate))){
        ys<-ys-1
      }
      imgSub<-as.cimg(image1[xs,ys,1,1]) #it would be nice to just start from scratch here (create new binary image instead of cropping) 
      #since this would prevent inclusion of competing signals. But, I might need other packages to do this. 
      imgSub<-isoblur(imgSub,sigma=IsoblurSigma2)
      imgSub<-threshold(imgSub,ImgThresh2) 
      
      lines=hough_line(imgSub,data.frame = TRUE)
    
      #Bestline<-lines[which.max(lines$score),]
      Bestline2<-lines[which(lines$score>=max(lines$score)*.65),]
      
      vals<-(-(cos(Bestline2$theta)/sin(Bestline2$theta)))
      vals<-vals[is.finite(vals)]
      #avg slope
      slope[i]=-as.numeric(median(vals))
      #nfline(Bestline2$theta,Bestline2$rho,col="red")
      
      #if(size[i]>125& (slope[i]>slopeMost | slope[i]<slopeMin)){
      #  print(i)
      #  plot(imgSub)
      #  nfline(Bestline2$theta,Bestline2$rho,col="red")
      #  print(slope[i])
      #  Sys.sleep(1)
      #  }
    
    #slope[i]<-ifelse(Bestline[4]>0,1,0)
    }else{

      slope[i]=NA
    }
    
    #hough line slope
    #hough_line(image1,data.frame = TRUE) could put this in later to try, would be better after knowing who is considered in pixThreshDiv
  }
  
  if(DesiredSlope=="Stacked"){ #ignore slope for this option, in the future, can look at degrees of slope vs flat
    cont2<-cont[which(size>pixThresh & abs(slope)<=houghSlopeMax)] #ignore slopeMin for stacked. 
  }else if(DesiredSlope=="Upsweep"){
    cont2<-cont[which(size>pixThresh & slope<=houghSlopeMax & slope>=houghSlopeMin)]
  }else if(DesiredSlope=="Downsweep"){
    cont2<-cont[which(size>pixThresh & slope>=houghSlopeMax & slope<=houghSlopeMin)]
  }
  
 
  
  #cont2<-cont[which(slope==slopeTest)]

  #plot(image1)
  #purrr::walk(cont2,function(v) lines(v$x,v$y,col="red",lwd=4))
  
  #yeses=c(30.3,66.7,95,140,240.8,287.9,326.2,342.7,382.3,475.5,477.8,497.8,517.5,540.2,567.6,629.9) #for file AU-BSPM02_b-151115-173000.wav
  #yeses= c(243.991,761.4)
  #for(u in 1:length(yeses)){
  #  abline(v=yeses[u]/tAdjust,col="blue",lwd=5)
  #}
  fAdjust=(highFreq-lowFreq)/length(spectrogram$f)
  
  if(length(cont2)>0){
  Detections<-foreach(i=1:length(cont2)) %do% { #could expand size of detection, and then combine overlapping for better boxes 
    x1=min(cont2[[i]]$x)*tAdjust
    x2=max(cont2[[i]]$x)*tAdjust
    y1=highFreq-(max(cont2[[i]]$y)*fAdjust) #((length(spectrogram$f)-max(cont2[[i]]$y))*fAdjust)+lowFreq
    y2=highFreq-(min(cont2[[i]]$y)*fAdjust)
    return(c(x1,x2,y1,y2))
  }
  }else{
    Detections<-list()
  }
  
  Detections<-do.call("rbind",Detections)
  
  #combine detections. Since we are doing this for GS, first just do it based on time similarity. 
  #pseudo: 
  #sort by startime . Give integer ID to each detection
  #for each detection, if any following detections start within endtime +x, reassign them to the current ID. 
  #last, melt the df by ID, taking the lowest low, highest high, earliest start, latest end. 
  
  if(CombineDets=="y"&length(Detections)!=0){ #stealth change 7/29/21!!!
    Detections<-as.data.frame(Detections)
    Detections<-Detections[order(Detections$V1),]
    Detections$ID<-1:nrow(Detections)
    
    detslen<-nrow(Detections)
    
    for(n in 1:detslen){
      endtime<-Detections[n,2]
      p=n+1
      if(p<=detslen){
        while(Detections[p,1]<(endtime+CombInt) & Detections[n,5]!=Detections[p,5]){
          Detections[p,5]<-Detections[n,5]
          p=n+1
        }
      }
    }
    
    detsUnq<- unique(Detections[,5])
    Detections<-foreach(n=detsUnq) %do% {
      dat<-Detections[which(Detections[,5]==n),]
      return(c(min(dat[,1]),max(dat[,2]),min(dat[,3]),max(dat[,4])))
    }
    
    Detections<-do.call("rbind",Detections)

  }
  
  if(verbose=='y'&length(Detections)!=0){
    
    
    endpath1<-paste(metadata[2][[1]],"tmp",sep="")
    if(!file.exists(endpath1)){
      dir.create(endpath1)
    }
    
    endpath2<-paste(endpath1,"/ED/",sep="")
    if(!file.exists(endpath2)){
      dir.create(endpath2)
    }
    
    #kind of crude, but make an additional folder to divide by site/year
    
    endpath3<-paste(endpath2,substr(metadata[1][[1]],1,nchar(metadata[1][[1]])-11),sep="")
    if(!file.exists(endpath3)){
      dir.create(endpath3)
    }
    
    chunk_size=round(5/tAdjust)
    
    chunks = floor(dim(image1)[1]/chunk_size)
    
    for(p in 1:chunks){
      
      #for each time chunk, subset spectrogram, and image1, and add detection boxes to each. 
      
      if(sample(1:50,1)==1){ #randomly sample from available chunks. 
        
        chunkStart<-(p*chunk_size)-chunk_size
        chunkEnd<-chunkStart+chunk_size
        
        sec_start<-round(chunkStart*tAdjust)
        sec_end<-round(chunkEnd*tAdjust)
        
        spec_chunk<-spectrogram
        
        spec_chunk$S<-spec_chunk$S[,chunkStart:chunkEnd]
        spec_chunk$t<-spec_chunk$t[chunkStart:chunkEnd]
        
        jpeg(paste(endpath3,"/",metadata[1][[1]],"_",sec_start,"_to_",sec_end,"_spectrogram.jpg",sep=""),quality=100,height=240)
        
        par(#ann = FALSE,
          mai = c(0,0,0,0),
          mgp = c(0, 0, 0),
          oma = c(0,0,0,0),
          omd = c(0,1,0,1),
          omi = c(0,0,0,0),
          xaxs = 'i',
          xaxt = 'n',
          xpd = FALSE,
          yaxs = 'i',
          yaxt = 'n')
        
        plot(spec_chunk)
        
        dets<-which(Detections[,1]>sec_start&Detections[,1]<sec_end)
        
        if(length(dets)>0){
          for(h in 1:length(dets)){
            rect(Detections[dets[h],1],Detections[dets[h],3],Detections[dets[h],2],Detections[dets[h],4],border='red')
          }
        }
        
        dev.off()
        
        img_chunk<-image1[chunkStart:chunkEnd,,,]
        
        jpeg(paste(endpath3,"/",metadata[1][[1]],"_",sec_start,"_to_",sec_end,"_img.jpg",sep=""),quality=100,height=240)
        
        par(#ann = FALSE,
          mai = c(0,0,0,0),
          mgp = c(0, 0, 0),
          oma = c(0,0,0,0),
          omd = c(0,1,0,1),
          omi = c(0,0,0,0),
          xaxs = 'i',
          xaxt = 'n',
          xpd = FALSE,
          yaxs = 'i',
          yaxt = 'n')
        
        plot(as.cimg(img_chunk))
      
        if(length(dets)>0){
          for(h in 1:length(dets)){
            rect((Detections[dets[h],1]/tAdjust)-chunkStart,dim(img_chunk)[2]-(Detections[dets[h],3]/fAdjust)+min(spec_chunk$f)/fAdjust,(Detections[dets[h],2]/tAdjust)-chunkStart,dim(img_chunk)[2]-(Detections[dets[h],4]/fAdjust)+min(spec_chunk$f)/fAdjust,border='red')
          }
        }
        dev.off()
      }
      
    }
    
    
    
  }
  
  return(Detections)

}
