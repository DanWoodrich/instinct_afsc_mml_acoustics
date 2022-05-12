#method: spectro w contrast and brightness

library(tuneR)
library(signal)
library(imager)

#spectrogram generation. #sounddata,windowLength,overlap,contrast_mod,brightness_mod,crop_freq,crop_freq_start,crop_freq_size
gen_spec_image <-function(x,wl,ovlp,contrast,brightness,do_crop,crop_start,crop_size){
  
  spectrogram = specgram(x = x@left,
           Fs = x@samp.rate,
           window=wl,
           overlap=ovlp
  )
  
  #before doing any normalizing etc, apply frequency cropp
  if(do_crop=="y"){
    
    bins = dim(spectrogram$S)[1]
    
    freq_per_bin = (x@samp.rate/2)/bins
    
    bin_start = ceiling(crop_freq_start/freq_per_bin)
    
    bin_end = ceiling((crop_freq_start+crop_freq_size)/freq_per_bin)
    
    spectrogram$S = spectrogram$S[bin_start:bin_end,]
    spectrogram$f = spectrogram$f[bin_start:bin_end]
    
  }
  
  
  spectrogram$S = log(abs(spectrogram$S))
  
  #rescale to # of sd
  specmean = mean(spectrogram$S)
  spec_sd = sd(spectrogram$S)
  
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

args="C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/921636 //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/2048 -1 0 y 450 50 600 40 512 con_bright_no_rep-v1-0"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

#if I am sharing these same parameters for multiple processes, I should probably pass the names and values to make it more robust to 
#changes in the different processes. 

FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
resultpath = args[2]
SFroot = args[3]
brightness_mod = as.numeric(args[4])
contrast_mod = as.numeric(args[5])
crop_freq = args[6]
crop_freq_size = as.numeric(args[7])
crop_freq_start = as.numeric(args[8])
img_height = as.numeric(args[9])
pix_per_sec = as.numeric(args[10])
windowLength = as.numeric(args[11])

#for each difftime, load in the sound data. 

bigfiles = unique(FG$DiffTime)

dir.create(paste(resultpath,"/bigfiles",sep=""))

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
  
  xbins = (length(sounddata@left)/sounddataheader$sample.rate)*pix_per_sec
  
  overlap = windowLength/xbins + windowLength- (length(sounddata@left)/xbins)  
  
  spec_img<- gen_spec_image(sounddata,windowLength,overlap,contrast_mod,brightness_mod,crop_freq,crop_start,crop_size)

  spec_img =resize(spec_img,size_x = xbins,size_y = img_height) #make height

  filename = paste(resultpath,"/bigfiles/bigfile",bigfiles[i],".png",sep="")

  img_print(spec_img,xbins,img_height,filename) #inputHeight
  
}

#In future versions, a printout of the spectrogram power histograms and a sample plot would be nice.
#like in a two column multiplot.
writeLines("spectrograms successfully written",paste(resultpath,'receipt.txt',sep="/"))
#
