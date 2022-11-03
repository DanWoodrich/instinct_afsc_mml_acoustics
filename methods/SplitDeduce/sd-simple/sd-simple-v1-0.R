

#pseudo: load in detections and determine time boundaries of splits. 

args<-"C:/Apps/INSTINCT/Cache/462670/93924 C:/Apps/INSTINCT/Cache/461006/316555/216778/514835 C:/Apps/INSTINCT/Cache/461006/316555/216778/514835/791293  sd-simple-v1-0"
args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

#dataPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20"
#resultPath<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/53d3bb"
#PE1s2path<-"C:/Apps/INSTINCT/Cache/2f38f7440b5b/a04a78/04f178/813e20/03bc3c/c393b4/6c4546"

#
FGpath <- args[1]
dataPath<-args[2]
#resultPath<-args[2]

FG = read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"))
data = read.csv(paste(dataPath,"DETx.csv.gz",sep="/"))

data=data[which(data$SignalCode=='out'),]

print(str(FG))
print(str(data))

data$sf_difftime=FG$DiffTime[match(data$StartFile,FG$FileName)]
data$ef_difftime=FG$DiffTime[match(data$EndFile,FG$FileName)]

data$sf_dur=FG$Duration[match(data$StartFile,FG$FileName)]


#make sure that difftime calculation is correct: any windows which overlap difftimes are assumed to be correct, and the FG difftime
#incorrect

while(nrow(data[which(data$sf_difftime!=data$ef_difftime),])>0){
  
  enddt = data[which(data$sf_difftime!=data$ef_difftime)[1],"ef_difftime"]
  newdt = data[which(data$sf_difftime!=data$ef_difftime)[1],"sf_difftime"]
  print("Warning: discrepancy in difftime calculation for FG and model")
  print(newdt)
  print(enddt)
  
  data[which(data$ef_difftime==enddt),"sf_difftime"] = newdt
  data[which(data$ef_difftime==enddt),"ef_difftime"] = newdt
}


#since this should work for single or multiple FG, test to see which stage it is which will inform output. 

#determine step size: 

step=as.numeric(names(table(data$StartTime[2:length(data$StartTime)]-data$StartTime[1:(length(data$StartTime)-1)])[which.max(table(data$StartTime[2:length(data$StartTime)]-data$StartTime[1:(length(data$StartTime)-1)]))]))
window_size = as.numeric(names(table(data$EndTime-data$StartTime))[which.max(table(data$EndTime-data$StartTime))])

if(length(unique(data$FGID))>1){
  FGname = "all"
}else{
  FGname =  data$FGID[1]
}
#loop through data, which (we assume!) comes ordered

new_dur=TRUE



dur = window_size-step

#now that I have FG, what is my condition? If difftime is different, or split is different, then that is a new segment
#if difftime is different, determine the length of the end segment, and subtract that from the step size

#so for first 1796 rows, the 1's (1347 rows) , 2s (449 rows)
#I feel like i am making an incorrect assumption about adding the flat dur at the start. This is a 3600 second segment, split perfectly
#at 75/25 splits. 

#here's why- the long segments overlap split boundaries, but get classified into one or the other. How to account for this? 


#statement- each difftime, the total value of the split will be the # of rows *step size, plus 4 (1st detection is (window - step)/2)

rough_split_prop=table(round(data$splits))/nrow(data)
rough_split_prop=round(rough_split_prop,3) #round to 2 digits

tab = data.frame(FGname,as.numeric(names(rough_split_prop)),0)

colnames(tab)=c("FG","split","duration")



#so instead of doing the below- loop through difftimes and just add up the counts

#this actually doesn't work, since files are used in multiple difftimes. 

#for(i in unique(data$sf_difftime)){
  
#  temptab = data.frame(FGname,as.numeric(names(rough_split_prop)),0)
#  colnames(temptab)=c("FG","split","duration")
  
#  datasub = data[which(data$sf_difftime==i),]
  
#  for(p in as.numeric(names(rough_split_prop))){
    
#    tab[which(tab$split==p),"duration"]=tab[which(tab$split==p),"duration"]+sum(datasub$split==p)*step + ((window_size-step)*unname(rough_split_prop[as.character(p)]))#*length(unique(c(datasub$StartFile,datasub$EndFile)))
#    temptab[which(temptab$split==p),"duration"]=temptab[which(temptab$split==p),"duration"]+sum(datasub$split==p)*step + ((window_size-step)*unname(rough_split_prop[as.character(p)]))#*length(unique(c(datasub$StartFile,datasub$EndFile)))
#  }
  

  #subtract extra values 
  
#  split = datasub[nrow(datasub),"splits"]
  
#  if(datasub[nrow(datasub),"StartFile"]==datasub[nrow(datasub),"EndFile"]){
#    tab[which(tab$split==split),"duration"]=tab[which(tab$split==split),"duration"]-(window_size-(datasub[nrow(datasub),"EndTime"]-datasub[nrow(datasub),"StartTime"]+0.1))
#    temptab[which(temptab$split==split),"duration"]=temptab[which(temptab$split==split),"duration"]-(window_size-(datasub[nrow(datasub),"EndTime"]-datasub[nrow(datasub),"StartTime"]+0.1))
#  }else{
#    tab[which(tab$split==split),"duration"]=tab[which(tab$split==split),"duration"]-(window_size-((datasub[nrow(datasub),"EndTime"]+datasub[nrow(datasub),"sf_dur"])-datasub[nrow(datasub),"StartTime"]+0.1))
#    temptab[which(temptab$split==split),"duration"]=temptab[which(temptab$split==split),"duration"]-(window_size-((datasub[nrow(datasub),"EndTime"]+datasub[nrow(datasub),"sf_dur"])-datasub[nrow(datasub),"StartTime"]+0.1))
#  }
  
#  print(temptab)
#  View(datasub)
#  readline(prompt="Press [enter] to continue")
  
#}

#data= data[order(data$sf_difftime),]

for(i in 1:nrow(data)){
  
  #if(i== 1348){
  #  stop()
  #}

  if(new_dur){
     #assume each segment as at least size of window_size. If not true, may be incorrect effort calculation. 
   # dur=0
    new_dur = FALSE
    split = round(data$splits[i])
    prevdt = data$StartTime[i]
    
    dur = ((window_size-step)*unname(rough_split_prop[as.character(split)]))
    
    print(paste(data$splits[i],i))
  }else if(((data$StartTime[i]-step== prevtime)&(data$StartFile[i]==data$StartFile[i-1])) & round(data$splits[i]) ==split){ #if detection has different start/end files, we can assume it is within a larger frame of effort.
    #won't be true in cases where soundfile is shorter than window, which will need to be addressed if ever present. 
    prevtime = data$sf_difftime[i]
    dur=dur + step
  }else{
    

    
    #dur = dur-step
    
    #need to account for segments which don't end in an even step
    
    if(round(data$splits[i]) ==split){ #if the split is not the same, assume it triggered based on end of sequence. 
      last_len = data$EndTime[i-1] -data$StartTime[i-1]
      #print(last_len)
      last_diff = window_size-last_len
      dur = dur - last_diff 
      
    }
    
    #if(data$EndTime[i]-data$StartTime[i]!=window_size){
      #if((window_size-(data$EndTime[i]-data$StartTime[i])) >10){
      #  stop()
      #}
    #  dur = dur - (window_size-(data$EndTime[i]-data$StartTime[i])) 
    #}
    
    
    #if(round(data$splits[i]) ==split)
    #need to account for final prevtime
    
    tab[which(tab$split==split),"duration"]=tab[which(tab$split==split),"duration"]+dur
    new_dur = TRUE
    #reset values

  }

}

#split_start 


stop()

