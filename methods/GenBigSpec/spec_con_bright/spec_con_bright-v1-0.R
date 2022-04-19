#method: spectro w contrast and brightness

library(tuneR)
library(signal)
library(imager)

#spectrogram generation. 
gen_spec <-function(x,wl,ovlp){
  
  spec = specgram(x = x@left,
           Fs = x@samp.rate,
           window=wl,
           overlap=ovlp
  )
  
  return(spec)
}

#needs PARAMSET_GLOBALS['SF_foc'] in process

args="//161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/394448 //161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/394448/test //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/2048 512 90 7.5 dbuddy-compare-publish-v1-2"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
resultpath = args[2]
SFroot = args[3]
windowLength = as.numeric(args[4])
overlap = as.numeric(args[5])
contrast = as.numeric(args[6])


#for each difftime, load in the sound data. 

bigfiles = unique(FG$DiffTime)

for(i in 1:length(bigfiles)){
  
  fileFG = FG[which(FG$DiffTime==bigfiles[i]),,drop = FALSE]
  
  sounddata = list()
  sounddataheader = readWave(paste(SFroot,fileFG$FullPath[n],fileFG$FileName[n],sep=""),from=fileFG$SegStart[n],to = fileFG$SegStart[n]+fileFG$SegDur[n],header=TRUE)
  
  
  #read in the files in chunks
  for(n in 1:nrow(fileFG)){
    
    sounddata[[n]]= readWave(paste(SFroot,fileFG$FullPath[n],fileFG$FileName[n],sep=""),from=fileFG$SegStart[n],to = fileFG$SegStart[n]+fileFG$SegDur[n],units = "seconds")@left
    
  }
  
  sounddata = do.call("c",sounddata)
  
  sounddata = Wave(sounddata,samp.rate=sounddataheader$sample.rate,bit = sounddataheader$bits)
  
  #now, apply function that generates spectrogram
  spectrogram<- gen_spec(sounddata,windowLength,overlap)
  spectrogram$S = log(abs(spectrogram$S))
  tempogram = spectrogram
  
  #plot(spectrogram)
  
  #first, for now always going to log normalize data and discard phase
  
 # hist(spectrogram$S)
  
  #This or similar will decrease contrast- how to increase contrast though? 
  
  #dilute power differences: 
  #vals= (length(spectrogram$f)*length(spectrogram$t))
  #valsadd = rnorm(vals,mean = mean(spectrogram$S))
  
  #can loop this for more signal padding? 
  #tempogram$S = spectrogram$S+ valsadd
  #tempogram$S= tempogram$S+ valsadd
  #
  
  #rescale to # of sd
  #specmean = mean(spectrogram$S)
  #spec_sd = sd(spectrogram$S)
  #multiplier2=11-contrast #this makes the spectrogram higher contrast with smaller values
  #ideally will be 
  
  
  #tempogram$S[which(tempogram$S<(specmean-(spec_sd*multiplier2)))]=specmean-(spec_sd*multiplier2)
  #tempogram$S[which(tempogram$S>(specmean+(spec_sd*multiplier2)))]=specmean+(spec_sd*multiplier2)
  
  #instead of doing a cutoff, move the distribution to 0 and then just use a multiplier. 
  #multiplying affects all the values the same way... I need to affect the higher values more. 
  #so use log? I don't get as of yet to tune the log scaling to meaningfully change the distribution
  
  tempogram$S = spectrogram$S-mean(spectrogram$S)
  
  tempogram$S = abs(tempogram$S)
  
  hist(tempogram$S)
  
  tempogram$S = tempogram$S*contrast
  
  tempogram$S = log(tempogram$S)
  
  hist(tempogram$S)
  
  
  diff = (max(tempogram$S)*1.5)/10
  
  brigthness2 = brightness*diff
  
  
  
  tempogram$S[which.min(tempogram$S)]= min(tempogram$S)
  tempogram$S[which.max(tempogram$S)]= brightness
  #tempogram$S = rescale(tempogram$S,specmean-(spec_sd*multiplier),specmean+(spec_sd*multiplier))
  plot(tempogram)
  hist(tempogram$S)
  
  #hist(abs(spectrogram$S),xlim= c(0,5000),breaks = 4000)
  
  #hist(abs(spectrogram$S))
  #hist(log(abs(spectrogram$S)))
}

#now that the data are 


