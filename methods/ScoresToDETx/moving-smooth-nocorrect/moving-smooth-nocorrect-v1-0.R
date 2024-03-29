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

args = "C:/Apps/INSTINCT/Cache/117592/772508/FileGroupFormat.csv.gz C:/Apps/INSTINCT/Cache/251579/121916/587840/248952/527200/836944 C:/Apps/INSTINCT/Cache/117592/273952/556187 C:/Apps/INSTINCT/Cache/397754/84782/748378/331477 C:/Apps/INSTINCT/Cache/397754/84782/748378/331477/325430 40 300 20 224 31 mean within_file 20 2 moving-smooth-v1-2"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#docker values
FGpath <- args[1]
ScorePath <- args[2]
SpecPath <- args[3]
SplitPath <- args[4]
resultPath <- args[5]

freq_low = as.integer(args[6])
freq_size = as.integer(args[7])
group_pix = as.integer(args[8])
model_win_size = as.integer(args[9])
native_pix_per_sec= as.integer(args[10])
smooth_method= eval(parse(text=args[11]))
split_protocol= args[12]
stride_pix= as.integer(args[13])
time_expand = as.integer(args[14])

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
  
  indFG = FGs[which(FGs$Name==unique(FGs$Name)[i]),]
  
  #recalculate difftime
  
  indFG$startTime_posx = as.POSIXct(indFG$StartTime,tz="UTC")+ as.integer(indFG$SegStart)
  
  indFG = indFG[order(sort(indFG$startTime_posx)),]
  
  indFG$endTime_posx = indFG$startTime_posx + indFG$SegDur
  
  bool = indFG$startTime_posx[2:nrow(indFG)]==indFG$endTime_posx[1:(nrow(indFG)-1)]
  
  
  #also, hard cap difftime at 40 minutes to match behavior of fxn in misc. 
  #Just noticed that original seems to be bugged- doesn't look for 40 minute breaks
  #within a difftime, just throughout the file. Match this behavior although it should be changed. 
  #note to fix it in both locations eventually. 
  
  #also, does it by duration, but should be doing it by segdur... also need to fix in both places
  cums = cumsum(indFG$Duration)-indFG$Duration[1]
  breaks = floor(cums / (40*60)) 

  time_bool = breaks[2:length(breaks)]==breaks[1:(length(breaks)-1)]
  
  bool = bool & time_bool
  
  indFG$DiffTime=1
  
  iterator = 1
  
  for(m in 1:(nrow(indFG)-1)){
    #every time we hit a false, increase the iterator by 1
    if(!bool[m]){
      iterator = iterator + 1
    }
    
    indFG$DiffTime[m+1]=iterator
  }
 
  
 FGvec[[i]]=indFG
  
}

FGs = do.call('rbind',FGvec)


#now, add to SpecTab total seconds per file. 
durtab = aggregate(SegDur ~ DiffTime + Name, data = FGs, sum)

filetabs = cbind(SpecTab,SplitTab)
filetabs = filetabs[,c(1,3,4)]
colnames(filetabs) = c("bigfile","splitfile","Name")

indfiletabs =list() 


for(i in 1:length(unique(durtab$Name))){
  
  indfiletab = filetabs[which(filetabs$Name==unique(FGs$Name)[i]),]
  
  one_dig_len = nchar(indfiletab$bigfile[1])
  
  indfiletab$DiffTime = as.numeric(substr(indfiletab$bigfile,rep(one_dig_len,nrow(indfiletab))-4,(nchar(indfiletab$bigfile)-4)))
  
  indfiletabs[[i]] = indfiletab
}

filetabs = do.call('rbind',indfiletabs)

filetabs=merge(filetabs,durtab)

filetabs = filetabs[order(filetabs$Name,filetabs$DiffTime),]

filetabs = filetabs[order(factor(filetabs$Name, levels = FGs_og_order)), ]

#now that we have this, can use the stride information to determine which scores go to which files. 

filetabs$tot_pix = filetabs$SegDur *native_pix_per_sec
filetabs$tot_strides = ceiling(filetabs$tot_pix/stride_pix)
#the model is outputting fewer than the calculated total number of strides- this likely has to do 
#with behavior which throws out strides between splits? 

vertical_bins = as.integer(nrow(Scores)/sum(filetabs$tot_strides))

#from here, for each difftime in the FG, create the scores vector and then determine the 
#

SplitTab = SplitTab[order(factor(SplitTab$FGname, levels = FGs_og_order)), ]

#right now, seems like test deployment of the 

if(vertical_bins==1){
  
  scores_ind=1
  data_all = list()
  datachunk=1
  
  for(i in 1:length(unique(FGs$Name))){
    indFG = FGs[which(FGs$Name==unique(FGs$Name)[i]),]
    
    for(p in unique(indFG$DiffTime)){

      FGdt = indFG[which(indFG$DiffTime==p),]
      
      dur = sum(FGdt$SegDur)
      
      end_ind = (scores_ind+ceiling(dur*native_pix_per_sec/stride_pix)-1)
      
      scores= Scores$V1[scores_ind:end_ind]
      
      #this is the point I smooth.
      new_size = length(scores)*(stride_pix/group_pix)
      scores = approx(scores, n=new_size)$y
      
      scores_starts= seq(0,length(scores)*group_pix/native_pix_per_sec-group_pix/native_pix_per_sec,group_pix/native_pix_per_sec)
      
      #: shift so that the scores correspond instead to midpoint of signal: 
      #nocorrect- disable this to assess effect. 
      #scores_starts = scores_starts-(model_win_size/(native_pix_per_sec*time_expand))/2
      #--
      
      scores_ends = scores_starts + (model_win_size/(native_pix_per_sec*time_expand))
      
      #v1-2: crop the initial value so that it is >0
      scores_starts[which(scores_starts<0)]=0.1
      
      #crop the final value so that it is within difftime interval
      scores_ends[which(scores_ends>dur)]=dur-0.1
      
      FGdt2 = aggregate(SegDur ~ FileName, data = FGdt, sum)
      FGdt2$cumdur = cumsum(FGdt2$SegDur)
      #now can assemble detx info. 
      if(nrow(FGdt2)>1){
        FGdt2$startcum =c(0,FGdt2$SegDur[1:(nrow(FGdt2)-1)])
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
      
      #read in split data
      
      splits = read.csv(SplitTab[datachunk,"filename"])
      
      #downsample splits to the level of scores
      splits2 = approx(splits, n=new_size)$y
      
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

  