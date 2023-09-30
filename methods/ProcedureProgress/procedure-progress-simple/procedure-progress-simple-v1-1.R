library(pgpamdb)
library(DBI)
library(png)

args="D:/Cache/92439/159444/545091/818032/457752 D:/Cache/92439/159444/545091/818032/457752/805048 6,7,24 procedure-progress-simple-v1-0"

args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
con=pamdbConnect(dbname,keyscript,clientkey,clientcert)

resultPath <- args[2]
procedures <- args[3]

vis = procedure_prog(con,as.numeric(strsplit(procedures,",")[[1]]))

#print(paste(resultPath,"/progress_tab.csv",sep=""))

#write.csv(vis[[2]],paste(resultPath,"/progress_tab.csv",sep=""))

png(paste(resultPath,"/progress_vis.png",sep=""),width=1000,height = 1000)

plot(vis[[1]])

dev.off()

#png= readPNG(paste(resultPath,"/progress_vis.png",sep=""))

write.csv(vis[[2]],paste(resultPath,"/progress_tab.csv",sep=""))

#zip(paste(resultPath,"/artifacts.zip",sep=""), files = c(paste(resultPath,"/progress_vis.png",sep=""), resultPath,"/progress_tab.csv",sep=""))

tar(paste(resultPath,"/artifacts.tar",sep=""),c(paste(resultPath,"/progress_vis.png",sep=""), paste(resultPath,"/progress_tab.csv",sep="")))


#out = list(png,vis[[2]])


#names = c("progress_vis","progress_tab")
#for(i in seq_along(out)) {
#  write_tsv(out[[i]], file.path(resultPath, names[i]))
#}



