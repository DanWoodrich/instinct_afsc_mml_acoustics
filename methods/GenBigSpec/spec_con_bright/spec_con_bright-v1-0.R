#method: spectro w contrast and brightness

library(tuneR)
library(signal)
library(imager)

#spectrogram generation. 
gen_spec_image <-function(x,wl,ovlp,contrast,brightness){
  
  spectrogram = specgram(x = x@left,
           Fs = x@samp.rate,
           window=wl,
           overlap=ovlp
  )
  
  spectrogram$S = log(abs(spectrogram$S))
  
  #rescale to # of sd
  specmean = mean(spectrogram$S)
  spec_sd = sd(spectrogram$S)
  
  while((min(spectrogram$S)+(spec_sd*contrast))>specmean|(max(spectrogram$S)-(spec_sd*contrast))<specmean){
    contrast = contrast - 0.5
    print("Warning: high contrast reduced for safe plotting.")
  }
  
  spectrogram$S[which(spectrogram$S<(min(spectrogram$S)+(spec_sd*contrast)))]=min(spectrogram$S)+(spec_sd*contrast)
  spectrogram$S[which(spectrogram$S>(max(spectrogram$S)-(spec_sd*contrast)))]=max(spectrogram$S)-(spec_sd*contrast)
  
  #brightness can be thought of as raising the cutoff for the right tail
  if(brightness>0){
    spectrogram$S[which(spectrogram$S>(max(spectrogram$S)-(spec_sd*brightness)))]=max(spectrogram$S)-(spec_sd*brightness)
  }else if(brightness<0){
    spectrogram$S[which(spectrogram$S<(min(spectrogram$S)+(spec_sd*-brightness)))]=min(spectrogram$S)+(spec_sd*-brightness)
  } #cut off from left tail instead
  
  image1<-as.cimg(t(spectrogram$S))
  image1<-as.cimg(image1[,dim(image1)[2]:1,,])
  
  return(image1)
}

img_print <-function(object,xbins,pix_height,path){
  
  #calculate the total number of height pixels: 
  
  png(path,height=pix_height,width=xbins)
  
  par(#ann = FALSE,
    mai = c(0,0,0,0),
    mgp = c(0, 0, 0),
    oma = c(0,0,0,0),
    omd = c(0,0,0,0),
    omi = c(0,0,0,0),
    xaxs = 'i',
    xaxt = 'n',
    xpd = FALSE,
    yaxs = 'i',
    yaxt = 'n')
  
  plot(object)
  
  dev.off()
  
}

#needs PARAMSET_GLOBALS['SF_foc'] in process

args="C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/test //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/2048 2 2 2 -2 2 2 1 15 14 256 dbuddy-compare-publish-v1-2"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#if I am sharing these same parameters for multiple processes, I should probably pass the names and values to make it more robust to 
#changes in the different processes. 

FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
resultpath = args[2]
SFroot = args[3]
brightness_end= as.numeric(args[4])
brightness_levels= as.numeric(args[5])
brightness_start = as.numeric(args[6])
contrast_end = as.numeric(args[7])
contrast_levels = as.numeric(args[8])
contrast_start = as.numeric(args[9])
largeWindow = as.numeric(args[10]) #15
smallWindow = as.numeric(args[11])
windowLength = as.numeric(args[12])

contrast_vec = seq(from=contrast_start,to=contrast_end,length.out=contrast_levels)
brightness_vec = seq(from=brightness_start,to=brightness_end,length.out=brightness_levels)


#for each difftime, load in the sound data. 

bigfiles = unique(FG$DiffTime)

replicas= contrast_levels * brightness_levels

for(n in 1:replicas){
  dir.create(paste(resultpath,"/replica_",(1:replicas)[n],sep=""))
}


for(i in 1:length(bigfiles)){
  
  fileFG = FG[which(FG$DiffTime==bigfiles[i]),,drop = FALSE]
  
  sounddata = list()
  sounddataheader = readWave(paste(SFroot,fileFG$FullPath[1],fileFG$FileName[1],sep=""),from=fileFG$SegStart[1],to = fileFG$SegStart[1]+fileFG$SegDur[1],header=TRUE)
  
  
  #read in the files in chunks
  for(n in 1:nrow(fileFG)){
    
    sounddata[[n]]= readWave(paste(SFroot,fileFG$FullPath[n],fileFG$FileName[n],sep=""),from=fileFG$SegStart[n],to = fileFG$SegStart[n]+fileFG$SegDur[n],units = "seconds")@left
    
  }
  
  sounddata = do.call("c",sounddata)
  
  sounddata = Wave(sounddata,samp.rate=sounddataheader$sample.rate,bit = sounddataheader$bits)
  
  xbins = length(sounddata@left)/sounddataheader$sample.rate/largeWindow*windowLength
  
  #programatically generated so that final image pixel size = windowLength
  overlap = windowLength/xbins + windowLength- (length(sounddata@left)/xbins)  
  
  replica_counter = 0
  #loops for contrast and brightness here: 
  
  #this should not be two loops - should be one which passes an object that maps the replace to a contrast and brightness value. 
  #that way, I can parallize this level. Do this later 
  
  for(n in 1:contrast_levels){
    for(p in 1:brightness_levels){
  
    spec_img<- gen_spec_image(sounddata,windowLength,overlap,contrast_vec[n],brightness_vec[p])
	
	#go up to next nearest integer and find difference. 
	newWin = windowLength*smallWindow/largeWindow
	
	perc_diff = ceiling(newWin) / newWin
    
    #plot(as.cimg(spec_img[1:windowLength,1:128,,]))
    spec_img =resize(spec_img,size_x = round(perc_diff*xbins),size_y = ceiling(newWin)) #make height
    #sized to small window so that when it is cropped in network it won't randomize on the vertical
    #plot(as.cimg(spec_img[1:windowLength,1:windowLength,,]))
    
    replica_counter = replica_counter+1
    
    filename = paste(resultpath,"/replica_",replica_counter,"/bigfile",bigfiles[i],".png",sep="")
    
    img_print(spec_img,round(perc_diff*xbins),dim(spec_img)[2],filename) #inputHeight
    
    #im <- load.image(filename)
    #plot(as.cimg(im[1:windowLength,1:windowLength,,]))
    
    }
  }
  
}

#In future versions, a printout of the spectrogram power histograms and a sample plot would be nice.
#like in a two column multiplot.
writeLines("spectrograms successfully written",paste(resultpath,'receipt.txt',sep="/"))
#
