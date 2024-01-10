options(timeout=1800)

Packages<-c("doParallel","dplyr","tuneR","signal","foreach","imager","oce","randomForest","seewave","plotrix","autoimage","pracma","PRROC","flux","stringi","caTools","sqldf","RPostgres","png") #"Rtools"?

for(n in Packages){
  if(require(n,character.only=TRUE)){
    print(paste(n,"is installed"))
  }else{
    print(paste("trying to install",n))
    install.packages(n, repos = "http://cran.us.r-project.org")
    if(require(n,character.only=TRUE)){
      print(paste(n,"installed"))
    }else{
      stop(paste("could not install",n))
    }

  }

}

install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/pgpamdb/pgpamdb_0.1.20.tar.gz", source = TRUE, repos=NULL)
