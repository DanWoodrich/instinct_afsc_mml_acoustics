#dbuddy pull filegroups %1 --FileGroup %2 

#install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/DbuddyTools_0.0.1.2.tar.gz", source = TRUE, repos=NULL)
library("DbuddyTools")

args = "C:/Apps/INSTINCT/Cache/834675/379128/FileGroupFormat.csv.gz NOPP6_EST_20090330_files_All.csv"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

#docker values
FGpath <- args[1]
FGname <- args[2]

FG = FG_pull(FGname)

#add FG name to FG

FG$Name = FGname

write.csv(FG,gzfile(FGpath),row.names = FALSE)
