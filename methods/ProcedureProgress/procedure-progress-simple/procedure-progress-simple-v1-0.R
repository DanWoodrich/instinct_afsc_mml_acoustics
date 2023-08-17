library(pgpamdb)
library(DBI)

args="D:/Cache/92439/159444/545091/818032/457752 D:/Cache/92439/159444/545091/818032/457752/805048 6,7,24 procedure-progress-simple-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
con=pamdbConnect("poc_v2",keyscript,clientkey,clientcert)

resultPath <- args[2]
procedures <- args[3]

png(paste(resultPath,"/progress_vis.png",sep=""),width=1000,height = 1000)

vis = procedure_prog(con,as.numeric(strsplit(procedures,",")[[1]]))

plot(vis[[1]])

write.csv(vis[[2]],paste(resultPath,"/progress_vis.png",sep=""))

dev.off()
