#method: spectro w contrast and brightness
#v1-1: implement a behavior where odd samples (ends in 99 instead of 100) create errors later in DL training. 
#v-2: remove brightness and contrast(implemented in tensorflow data augmentation instead
#v-3: export filepaths instead of receipt. 
library(tuneR)
library(signal)
library(imager)
library(doParallel)

source(paste(getwd(),"/user/R_misc.R",sep=""))

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
    
    if(bin_end>dim(spectrogram$S)[1]){
      
      stop("frequency cutoff does not work with spectrogram parameters as specified.")
     
      #bin_end = dim(spectrogram$S)[1]
    }
    
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
  
  png(path,height=pix_height,width=xbins,bg="black") #my problem is that upon saving, there is a background in 
  #the png. Not sure how to resolve- EBImage might have a solutioin? 
  
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
  
  plot(object, axes = 0)

  dev.off()
  
}

#needs PARAMSET_GLOBALS['SF_foc'] in process

args="C:/Apps/INSTINCT/Cache/333397 C:/Apps/INSTINCT/Cache/333397/537889 //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/2048 AL16_AU_BS03_files_77-170_gs_hg.csv y 1000 40 76 48 256 con_bright_no_rep-v1-3"

args<-strsplit(args,split=" ")[[1]]
source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#if I am sharing these same parameters for multiple processes, I should probably pass the names and values to make it more robust to 
#changes in the different processes. 

FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
resultpath = args[2]
SFroot = args[3]
FGname = args[4]
crop_freq = args[5]
crop_freq_size = as.numeric(args[6])
crop_freq_start = as.numeric(args[7])
native_img_height = as.numeric(args[8])
native_pix_per_sec = as.numeric(args[9])
windowLength = as.numeric(args[10])

#for each difftime, load in the sound data. 

bigfiles = unique(FG$DiffTime)

dir.create(paste(resultpath,"/bigfiles",sep=""))

crs<-detectCores()

startLocalPar(crs,"gen_spec_image","img_print","bigfiles","FG","resultpath","SFroot","FGname","crop_freq","crop_freq_size","crop_freq_start","native_img_height","native_pix_per_sec","windowLength")

foreach(i=1:length(bigfiles),.packages=c("tuneR","signal","imager")) %dopar% {
  
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
  xbins = round(length(sounddata@left)/sounddataheader$sample.rate)*native_pix_per_sec
  
  overlap = windowLength/xbins + windowLength- (length(sounddata@left)/xbins)  
  
  spec_img<- gen_spec_image(sounddata,windowLength,overlap,contrast_mod,brightness_mod,crop_freq,crop_freq_start,crop_freq_size)
  if(i ==1){
  print("initial dimensions:")
  print(paste(dim(spec_img)[1]/round(length(sounddata@left)/sounddataheader$sample.rate),dim(spec_img)[2])) #initial dimenions
  }
  
  #spec_img = as.cimg(spec_img[,1:73,,])
  
  if(dim(spec_img)[1]!=xbins | dim(spec_img)[2]!=native_img_height){
    spec_img =resize(spec_img,size_x = xbins,size_y = native_img_height) #make height
  }

  filename = paste(resultpath,"/bigfiles/bigfile",bigfiles[i],".png",sep="")

  img_print(spec_img,xbins,native_img_height,filename) #inputHeight
  
}

stopCluster(cluz)

outtab = data.frame(paste(resultpath,"/bigfiles/bigfile",1:length(bigfiles),".png",sep=""),FGname)
colnames(outtab)=c("filename","FGname")

write.csv(outtab,paste(resultpath,'filepaths.csv',sep="/"),row.names=FALSE)
