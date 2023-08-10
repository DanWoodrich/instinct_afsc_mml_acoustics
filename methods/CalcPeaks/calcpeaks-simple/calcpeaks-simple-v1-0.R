#library(pgpamdb)
#library(DBI)

#pseudo: this script loads in detx data. From it, it will infer peaks. 
#This version will assume that data from only 1 logical location is included. 
#if multiple are included, it will bin them together based on datetime. 

#this is an adaptation of the original algorithm used in FinReview.R (in network/detector/tools)

args=""

args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()

#source(Sys.getenv('DBSECRETSPATH')) #source("C:/Users/daniel.woodrich/Desktop/cloud_sql_db/paths.R")
#con=pamdbConnect("poc_v2",keyscript,clientkey,clientcert)

stop()
