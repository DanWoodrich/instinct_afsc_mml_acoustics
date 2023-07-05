#pseudo: 

#load in FG, scores, split files, stride_pix, native pix per second, parameter 'stage' (train,val,test) , parameter group_pixels, parameter smooth method , parameter split method (global) 

#what if I just output a pixel start along with the scores? Then I could backtrack however I want? (would need some knowledge of the split protocol - if knowing splits were 'by file', would assume that each 
#break is the start of a new file. But if I used a different procedure, I would have to encorportate this. 

#v1-1:
#bugfix error where detections not properly mapping when files switch due to not encoding initial file offset.
#bugfix to make detections properly represent size of window

#v1-2:
#experiment with moving the scores so that they represent midpoint instead of start of detection
#seems to be more true to form, not sure why they are working that way from the model perspective. 

#v1-3:
#add FGID to column to it is retained in outputs. 

args = "D:/Cache/449513/FileGroupFormat.csv.gz D:/Cache/449513/439550/913601/197533/438071 D:/Cache/449513/439550 D:/Cache/449513/439550/913601 D:/Cache/449513/439550/913601/217256 3600 0 96 20 80 240 20 mean within_file 10 moving-smooth-v1-11 y n 1"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#docker values
FGpath <- args[1]
ScorePath <- args[2]
SpecPath <- args[3]
SplitPath <- args[4]
resultPath <- args[5]

difftime_cap = as.integer(args[6])
freq_low = as.integer(args[7])
freq_size = as.integer(args[8])
group_pix = as.integer(args[9])
win_size_native = as.integer(args[10])
model_win_size = as.integer(args[11])
native_pix_per_sec= as.integer(args[12])
smooth_method= eval(parse(text=args[13]))
split_protocol= args[14]
stride_pix= as.integer(args[15])

#argument for inference to remove images if no longer needed: 
remove_spec= args[18]

time_expand = model_win_size/win_size_native


#detx also requires frequency information. This will be freq size * modifier / model size, and using
#information on how windows are split vertically if at all. For single vertical pass, will only use
#freq information params [freq_start] and [freq_size]

FGs = read.csv(FGpath)

#to associate
Scores = read.csv(paste(ScorePath,"/scores.csv.gz",sep=""),header=F)
SpecTab = read.csv(paste(SpecPath,"/filepaths.csv",sep=""))
SplitTab = read.csv(paste(SplitPath,"/filepaths.csv",sep=""))

FGs_og_order= unique(FGs$Name)
#steps: 
#1. unpack scores serially to match filepaths. 
#2. recalculate difftime for each FG
#3. associate filepaths with FG difftimes. 
#4. convert scores to DETx

#recalculate difftime:
FGvec = list()

for(i in 1:length(unique(FGs$Name))){
  
  #this is currently bugged, adding 6 new difftime to XP14_UK_KO01_sample1.csv. Fix! 
  
  indFG = FGs[which(FGs$Name==unique(FGs$Name)[i]),]
  
  indFG$DiffTime=1
  
  iterator = 1
  
  indFG$startTime_posx = as.POSIXct(indFG$StartTime,tz="UTC")+ as.integer(indFG$SegStart)
  
  indFG = indFG[order(sort(indFG$startTime_posx)),]
  
  indFG$endTime_posx = indFG$startTime_posx + indFG$SegDur
  
  #bool = indFG$startTime_posx[2:nrow(indFG)]==indFG$endTime_posx[1:(nrow(indFG)-1)]
  
  #change this calculation to within 1 second like in formatFG process
  
  bool = abs(indFG$startTime_posx[2:nrow(indFG)]-indFG$endTime_posx[1:(nrow(indFG)-1)])<2
  
  for(m in 1:(nrow(indFG)-1)){
    #every time we hit a false, increase the iterator by 1
    if(!bool[m]){
      iterator = iterator + 1
    }
    
    indFG$DiffTime[m+1]=iterator
  }
  
  
  difftimes =unique(indFG$DiffTime)
  
  iterator=0
  
  chunks = list()
  
  for(p in 1:length(difftimes)){
    
    indFGdf = indFG[which(indFG$DiffTime==difftimes[p]),]
    
    if(nrow(indFGdf)!=1){
    #recalculate difftime
      
    iterator = iterator + 1
    
    indFGdf$DiffTime[1]=iterator
    
    #bugfix v1-9: shouldn't subtract 1st segdur 
    cums = cumsum(indFGdf$SegDur) #-indFGdf$SegDur[1]
    #bugfix v1-10: in case of exact match should not be new difftime. 
    breaks = floor((cums-0.000001) / (difftime_cap)) 
    
    time_bool = breaks[2:length(breaks)]==breaks[1:(length(breaks)-1)]
    
    for(m in 1:(nrow(indFGdf)-1)){
      #every time we hit a false, increase the iterator by 1
      if(!time_bool[m]){
        iterator = iterator + 1
      }
      
      indFGdf$DiffTime[m+1]=iterator
    }
    
    }else{
      iterator = iterator + 1
      indFGdf$DiffTime = iterator
    }
    
    chunks[[p]]=indFGdf
    
    
  }
  
 outFG = do.call("rbind",chunks)
  
 FGvec[[i]]=outFG
  
}

FGs = do.call('rbind',FGvec)

#now, add to SpecTab total seconds per file. 
durtab = aggregate(SegDur ~ DiffTime + Name, data = FGs, sum)

#order SpecTab and SplitTab- usually in order, not sure why not sometimes. 

SpecTab = SpecTab[order(SpecTab$filename),]
SplitTab= SplitTab[order(SplitTab$filename),]

filetabs = cbind(SpecTab,SplitTab)
filetabs = filetabs[,c(1,3,4)]
colnames(filetabs) = c("bigfile","splitfile","Name")

#now order back to original order: 
#filetabs = filetabs[match(filetabs$Name,FGs_og_order),]

indfiletabs =list() 


for(i in 1:length(unique(durtab$Name))){
  
  indfiletab = filetabs[which(filetabs$Name==unique(FGs$Name)[i]),]
  
  #this didn't take into account cache numbers which lead with a zero. fix to be relative to bigfiles instead
  #one_dig_len = nchar(indfiletab$bigfile[1])
  
  
  
  indfiletab$DiffTime = as.numeric(substr(indfiletab$bigfile,unlist(gregexpr('bigfiles/bigfile', indfiletab$bigfile))+16,(nchar(indfiletab$bigfile)-4)))
  
  indfiletabs[[i]] = indfiletab
}

filetabs = do.call('rbind',indfiletabs)

filetabs=merge(filetabs,durtab)

filetabs = filetabs[order(filetabs$Name,filetabs$DiffTime),]

filetabs = filetabs[order(factor(filetabs$Name, levels = FGs_og_order)), ]

#now that we have this, can use the stride information to determine which scores go to which files. 

filetabs$tot_pix = filetabs$SegDur *native_pix_per_sec
filetabs$tot_strides = ceiling(filetabs$tot_pix/stride_pix)



vertical_bins = round(nrow(Scores)/sum(filetabs$tot_strides))

#from here, for each difftime in the FG, create the scores vector and then determine the 
#

#right now, seems like test deployment of the 

model_s_size = model_win_size/(native_pix_per_sec*time_expand)

if(vertical_bins==1){
  
  scores_ind=1
  data_all = list()
  datachunk=1
  
  for(i in 1:length(unique(FGs$Name))){
    indFG = FGs[which(FGs$Name==unique(FGs$Name)[i]),]
    
    for(p in unique(indFG$DiffTime)){

      FGdt = indFG[which(indFG$DiffTime==p),]
      
      dur = sum(FGdt$SegDur)
      
      #v-4: need to find
      #stealth bugfix to try to fix offset seen in scores
      
      #v1-11: max rounding behavior same as spectogram generation so scores line up better. 
      end_ind = (scores_ind+floor((round(dur)-model_s_size)*native_pix_per_sec/stride_pix))
      
      scores= Scores$V1[scores_ind:end_ind]
      
      #this is the point I smooth.
      new_size = length(scores)*(stride_pix/group_pix)
      scores = approx(scores, n=new_size)$y
      
      scores_starts= seq(0,length(scores)*group_pix/native_pix_per_sec-group_pix/native_pix_per_sec,group_pix/native_pix_per_sec)
      
      #v1-2: shift so that the scores correspond instead to midpoint of signal: 
      #no longer needed after v4 change (change to model step behavior)
      #scores_starts = scores_starts-(model_win_size/(native_pix_per_sec*time_expand))/2
      #--
      
      scores_ends = scores_starts + model_s_size
      
      #v1-2: crop the initial value so that it is >0
      #scores_starts[which(scores_starts<0)]=0.1
      
      #crop the final value so that it is within difftime interval
      scores_ends[which(scores_ends>=dur)]=dur-0.1
      
      FGdt2 = aggregate(SegDur ~ FileName, data = FGdt, sum)
      FGdt2$cumdur = cumsum(FGdt2$SegDur)
      #now can assemble detx info. 
      if(nrow(FGdt2)>1){
        FGdt2$startcum =c(0,FGdt2$cumdur[1:(nrow(FGdt2)-1)])
        FGdt2$offset = c(FGdt$SegStart[1],rep(0,nrow(FGdt2)-1))
      }else{
        FGdt2$startcum=0
        FGdt2$offset = FGdt$SegStart[1]
      }
      
      startfiles = FGdt2$FileName[findInterval(scores_starts,c(0,FGdt2$cumdur))]
      endfiles = FGdt2$FileName[findInterval(scores_ends,c(0,FGdt2$cumdur))]
      
      startint = findInterval(scores_starts,c(0,FGdt2$cumdur))
      endint = findInterval(scores_ends,c(0,FGdt2$cumdur))
      
      scores_starts = scores_starts- FGdt2$startcum[startint]+FGdt2$offset[startint]
      scores_ends = scores_ends- FGdt2$startcum[endint]+FGdt2$offset[endint]
      
      detx = data.frame(scores_starts,scores_ends,freq_low,freq_low+freq_size,startfiles,endfiles,scores,FGdt$Name[1])

        
      colnames(detx) = c("StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","probs","FGID")
      
      #if(any(detx$StartFile=="AU-BSPM04-160815-085000.wav")){
      #  stop("temporary debug catch")
      #}
      
      
      if(any((detx$EndTime-detx$StartTime)>(model_s_size+1))){
        print(detx[which((detx$EndTime-detx$StartTime)>(model_s_size+1)),])
        stop("error found")
      }
      
      if(any(is.na(detx$EndFile)) | any(is.na(detx$StartFile)) | 
         any(is.na(detx$scores_starts)) | any(is.na(detx$scores_ends))){
        stop("Bug catch")
      }
      
      #read in split data
      
      splits = read.csv(filetabs[datachunk,"splitfile"])
      
      #v1-10 stealth bugfix: incorrectly assessed # of columns instead of # of rows
      if(length(splits$X1)>1){
        #downsample splits to the level of scores
        splits2 = approx(splits, n=new_size)$y
      }else{
        splits2 = 3
      }
      
      detx$splits = splits2
      
      data_all[[datachunk]]=detx
      
      datachunk = datachunk + 1
      scores_ind= end_ind+1
    }
  }
  
  data_all = do.call('rbind',data_all)
  
}

#finally, tie this back to splits. 

write.csv(data_all,gzfile(paste(resultPath,"/DETx.csv.gz",sep="")),row.names = FALSE)

#remove the spectrograms if argument is set: 

if(remove_spec=="y"){
  
  file.remove(SpecTab$filename)
  file.remove(paste(SpecPath,"/filepaths.csv",sep=""))
  
}

  