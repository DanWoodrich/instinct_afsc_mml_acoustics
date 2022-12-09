
#v1-1: bugfix loop where extra commands are added. 
#v1-3: supports multiple items per argument. 
#v1-5: truncates GT which occurs outside of FG. 

#install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/DbuddyTools_0.0.1.4.tar.gz", source = TRUE, repos=NULL)
library("DbuddyTools")

args="C:/Apps/INSTINCT/Cache/462670/252457/DETx.csv.gz round1_pull1_reduce.csv LastAnalyst SignalCode UseFG MW HB.s.p.4 y R-pull-dbuddy-anyparam-v1-5"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep="")) 
args<-commandIngest()

outpath <-args[1]
FG <- args[2]

numparams = floor((length(args)-2)/2)

#length

command1 = paste("dbuddy pull direct",outpath)


if(numparams!=0){
  
  paramnames = c(args[3:(3+numparams-1)])
  paramvals = c(args[(3+numparams):(3+numparams+numparams-1)])
  
  if(paramvals[which(paramnames=="UseFG")]=="y"){
    
    command = paste("SELECT DISTINCT detections.* FROM filegroups JOIN bins_filegroups ON filegroups.Name = bins_filegroups.FG_name JOIN bins ON bins.id = bins_filegroups.bins_id JOIN detections ON bins.FileName = detections.StartFile WHERE ")
    
    #if .csv is present, remove it from string
    if(grepl(".csv",FG)){
      FG = substr(FG,1,nchar(FG)-4)
    }
    
    command = paste(command,"filegroups.Name=","'",FG,"'",sep="") 
    
    if(length(paramvals)>1){
      #if there are more arguments add an and 
      command = paste(command," AND",sep="")
    }
    
  }else{
    
    command = "SELECT * FROM detections WHERE"
    
  }

  paramvals = paramvals[-which(paramnames=="UseFG")]
  paramnames = paramnames[-which(paramnames=="UseFG")]
  
  
  #determine if parameters are single value or multiple. 
  multi_true = grepl(",",paramvals) #determined by presence of comma
  
  #commas can be allowed if 
  multi_true[which(paramnames=="Comments")]=FALSE
  
  for(i in 1:(numparams-1)){
    
    if(paramnames[i]=="Comments"){
      
      val= paste("detections.Comments LIKE '",paramvals[i],"'",sep="")
      
    #v4- let analysis ID also have multiple options to pull from . Not sure why it didn't before, trying it out...
    }else if(multi_true[i]){
      string = strsplit(paramvals[i],",")[[1]]
      string = paste(string,collapse="','")
      val= paste("detections.",paramnames[i]," in ('",string,"')",sep="")
    }else{
      val=paste("detections.",paramnames[i],"='",paramvals[i],"'",sep="")
    }
    
    command = paste(command,val)
    
    if(i!=(numparams-1)){
      command = paste(command,"AND")
    }

  }
  
  command = paste(command,";",sep="")
  
}

paste(command1," \"",command,"\"",sep="")

#print(command)
#

fileConn<-file("command_temp.bat")
writeLines(paste(command1," \"",command,"\"",sep=""), fileConn)
close(fileConn)

#shell.exec("command_temp.bat")
system("command_temp.bat")

file.remove("command_temp.bat")

#since we only pass the name of the FG argument as input and assume that to be constant, we can get the FG from the db and not violate 
#assumptions. However, probably better in the future to actually pass the file itself, so that formatGT and formatFG can be independently
#sourced. 

FGdata = FG_pull(FG)

GT = read.csv(outpath)

#check that GT has any overlap with FG rows. 
GT$inFG = FALSE

for(i in 1:nrow(GT)){
  
  FGstart = FGdata[which(FGdata$FileName==GT[i,]$StartFile),]
  FGend = FGdata[which(FGdata$FileName==GT[i,]$EndFile),]
  if(any(GT[i,]$StartTime>=FGstart$SegStart & GT[i,]$StartTime<=FGstart$SegStart+FGstart$SegDur)){
    GT[i,"inFG"] = TRUE
  }else if(any(GT[i,]$EndTime>=FGend$SegStart & GT[i,]$EndTime<=FGend$SegStart+FGend$SegDur)){
    GT[i,"inFG"] = TRUE
  }
  
}

if(any(!GT$inFG)){
  print("Warning: appears that some GT of stated relation are not actually in FG.")
}

GT = GT[which(GT$inFG),]
GT$inFG=NULL
write.csv(GT,gzfile(outpath),row.names = FALSE)
