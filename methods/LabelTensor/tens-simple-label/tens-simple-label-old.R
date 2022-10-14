library(signal)
library(imager)
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

args="C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/921636 C:/Apps/INSTINCT/Cache/394448/628717 C:/Apps/INSTINCT/Cache/394448/628717/174209  tens-simple-label-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()


FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
bf_path = paste(args[2],"bigfiles",sep="/")
GT = read.csv(paste(args[3],"/DETx.csv.gz",sep=""))
bigfiles = args[3]
resultpath = args[4]

bigfiles = unique(FG$DiffTime)

for(i in 1:length(bigfiles)){
  
  #how we get dims- possibly there is a ligher way out there. 
  image = load.image(paste(bf_path,"/bigfile",i,".png",sep=""))
  
  fileFG = FG[which(FG$DiffTime==bigfiles[i]),,drop = FALSE]


}




