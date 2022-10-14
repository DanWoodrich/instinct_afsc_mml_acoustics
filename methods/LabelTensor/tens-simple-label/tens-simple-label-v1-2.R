#v1-1: change the tp,fp,uk ints to 2,1,0 
#reason is that padding in image patching in dl fills in 0 for uk, which makes behaviors consistent. 
#v1-2:
#remove the 'reduce_factor', as it's irrelevant

library(signal)
library(dplyr)

args="C:/Apps/INSTINCT/Cache/394448 C:/Apps/INSTINCT/Cache/394448/921636 C:/Apps/INSTINCT/Cache/394448/756783 C:/Apps/INSTINCT/Cache/394448/756783/878262 2 450 50 4 600 40 tens-simple-label-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()


FG= read.csv(paste(args[1],"/FileGroupFormat.csv.gz",sep=""))
bf_path = paste(args[2],"bigfiles",sep="/") #one change I need to make- this actually needs to output 
#a file with necessary parameters to perform this process- in particular, it needs to save frequency
#start and end so that the current process knows how to populate the label tensors. 
GTref = read.csv(paste(args[3],"/DETx.csv.gz",sep=""))
resultpath = args[4]

#parameters:
dimensions = args[5] #1 means that frequency of dets will expand to encompass full freq range. 
freq_size = as.numeric(args[6])
freq_start = as.numeric(args[7])
native_img_height = as.numeric(args[8])
native_pix_per_sec = as.numeric(args[9])

bigfiles = unique(FG$DiffTime)

GT_depth = length(unique(GTref$SignalCode))

GT_unq = sort(unique(GTref$SignalCode)) #sort arranges alphabetically- do the same later when 'unpacking' results

dir.create(paste(resultpath,"/labeltensors",sep=""))

for(i in 1:length(bigfiles)){
  
  #combine GT with FG
  fileFG = FG[which(FG$DiffTime==bigfiles[i]),,drop = FALSE]
  
  fileFG =data.frame(
  fileFG %>%
    group_by(fileFG$FileName) %>%
    summarise(SegStart = min(SegStart), SegDur = sum(SegDur))
  )
  
  fileFG$mod = cumsum(fileFG$SegDur)-fileFG$SegDur-fileFG$SegStart
  
  fileFG_start = data.frame(fileFG$fileFG.FileName,fileFG$mod)
  colnames(fileFG_start)=c("StartFile","mod_start")
  
  fileFG_end = data.frame(fileFG$fileFG.FileName,fileFG$mod)
  colnames(fileFG_end)=c("EndFile","mod_end")
  
  GT = merge(GTref,fileFG_start)
  GT = merge(GT,fileFG_end)
  
  GT$StartTime = GT$StartTime+ GT$mod_start
  GT$EndTime = GT$EndTime+ GT$mod_end
  
  FGsecs = sum(fileFG$SegDur)
  #needs testing!
  
  #how we get dims- possibly there is a ligher way out there. 
  #image = load.image(paste(bf_path,"/bigfile",i,".png",sep=""))
  
  dim_x = native_img_height
  dim_y =(FGsecs*native_pix_per_sec) #(750 second bigfile at 40 pix per seconds)
  
  pix_per_sec = dim_y/FGsecs
  
  total_pixels = dim_x * dim_y * GT_depth
  
  if(dimensions==2){
  
  highfreq = freq_start+freq_size #figure these out from bigfile generation output
  lowfreq = freq_start#figure these out from bigfile generation output
  
  pix_per_freq = dim_x / (highfreq-lowfreq)
  
  GT$HighPix = GT$HighFreq * pix_per_freq
  GT$LowPix = GT$LowFreq * pix_per_freq
  
  #bound by freq:
  GT$HighPix[GT$HighPix>dim_x] = dim_x
  GT$LowPix[GT$LowPix<0] = 0
  
  GT$HighPix =round(GT$HighPix)
  GT$LowPix =round(GT$LowPix)
  }else if(dimensions==1){  
    
    GT$HighPix = dim_x
    GT$LowPix = 0
  }
  
  

  
  
  GT$StartPix = GT$StartTime * pix_per_sec
  GT$EndPix = GT$EndTime * pix_per_sec
  
  #bound by time:
  GT = GT[which(GT$EndPix<dim_y & GT$StartPix>0),]
  
  GT$StartPix = round(GT$StartPix)
  GT$EndPix =round(GT$EndPix)

  #create template matrix
  
  lab_array <- array(rep(1, total_pixels), dim=c(dim_x, dim_y, GT_depth)) #1 is now 'fp'
  
  GT$depth = match(GT$SignalCode,GT_unq)
  
  if(nrow(GT)>0){
  #loop through and assign. 
  for(j in 1:length(GT_unq)){
    for(k in 1:nrow(GT)){
      
      lab_array[GT[k,"LowPix"]:GT[k,"HighPix"],GT[k,"StartPix"]:GT[k,"EndPix"],GT[k,"depth"]] = 2 #0 is now 'tp'
    }
    
  }
  
  #for(k in 1:GT_depth){
	#lab_array[,,k] = t(as.matrix(lab_array[,,k]))
  #}
  
  lab_array = aperm(lab_array,dim=c(dim_y, dim_x, GT_depth))
    
  }
  
  #now, unpack this by writing serializing it going downward by each column, per page of depth. 
    
  #lab_ser <- c(NA)
  #length(lab_ser) <- total_pixels
  #counter = 1
  
  #optimize this using apply? Or something. Very slow. 
  
  #for(j in 1:length(GT_unq)){
  #  for(k in 1:dim_y){
  #    print(k)
  #    lab_ser[counter:dim_x] = lab_array[dim_x:1,k,j]
  #    counter = counter + dim_x
  #  }
  #}
  
  #it appears that c() on the array will unpack it going top down, 
  #test = array(c(1,2,3,4,5,6,7,8),dim=c(2,2,2))
  #c(test)
  
 #write.csv(c(lab_array),lab_path,row.names = FALSE) 
  
  write.table(c(lab_array), gzfile(paste(resultpath,"/labeltensors/labeltensor",i,".csv.gz",sep="")),sep=",",quote = FALSE,col.names=FALSE,row.names = FALSE)
  
  
}

#need to save a receipt. 
writeLines("labels tensors successfully written",paste(resultpath,'receipt.txt',sep="/"))

