library(signal)
library(imager)

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

args="C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/test //161.55.120.117/NMML_AcousticsData/Audio_Data/DecimatedWaves/2048 2 2 y 50 450 600 40 512 dbuddy-compare-publish-v1-2"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)


FG = read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
bigfilespath = args[2]
resultpath = args[3]
split_protocol= args[4] 
test_split= args[5]
train_test_split= args[6]
train_val_split=args[7]








