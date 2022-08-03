#dbuddy pull filegroups %1 --FileGroup %2 

#install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/DbuddyTools_0.0.1.2.tar.gz", source = TRUE, repos=NULL)
library("DbuddyTools")

args = "C:/Apps/INSTINCT/Cache/652882/tempFG.csv.gz BS16_AU_PM02-a_files_1-175_rw_hg.csv decimate_data file_groupID methodID2m methodvers2m target_samp_rate y BS16_AU_PM02-a_files_1-175_rw_hg.csv matlabdecimate V1s0 1024 dbuddy-pull-FG-wname-v1-0"

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

#docker values
FGpath <- args[1]
FGname <- args[2]

FG = FG_pull(FGname)

#add FG name to FG

FG$Name = FGname

write.csv(FG,gzfile(FGpath),row.names = FALSE)
