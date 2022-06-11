#method: spectro w contrast and brightness
#v1-1: implement a behavior where odd samples (ends in 99 instead of 100) create errors later in DL training. 
#v-2: remove brightness and contrast(implemented in tensorflow data augmentation instead

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

args="C:/Apps/INSTINCT/Cache/843873/665107 C:/Apps/INSTINCT/Cache/843873/665107/545866 //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/1024 -1 0 y 300 40 300 75 512 con_bright_no_rep-v1-1"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

#if I am sharing these same parameters for multiple processes, I should probably pass the names and values to make it more robust to 
#changes in the different processes. 

FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
resultpath = args[2]
SFroot = args[3]
crop_freq = args[6]
crop_freq_size = as.numeric(args[7])
crop_freq_start = as.numeric(args[8])
native_img_height = as.numeric(args[9])
native_pix_per_sec = as.numeric(args[10])
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
  
  #v1-1: try to round the division to protect against dropped samples
  xbins = round(length(sounddata@left)/sounddataheader$sample.rate)*pix_per_sec
  
  overlap = windowLength/xbins + windowLength- (length(sounddata@left)/xbins)  
  
  spec_img<- gen_spec_image(sounddata,windowLength,overlap,contrast_mod,brightness_mod,crop_freq,crop_freq_start,crop_freq_size)
  
  print("initial dimensions:")
  print(dim(spec_img)) #initial dimenions

  spec_img =resize(spec_img,size_x = xbins,size_y = native_img_height) #make height

  filename = paste(resultpath,"/bigfiles/bigfile",bigfiles[i],".png",sep="")

  img_print(spec_img,xbins,native_img_height,filename) #inputHeight
  
}

#In future versions, a printout of the spectrogram power histograms and a sample plot would be nice.
#like in a two column multiplot.
writeLines(as.character(length(bigfiles)),paste(resultpath,'receipt.txt',sep="/")) #write the number of bigfiles to expect
#
