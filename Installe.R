options(timeout=1800)

#gh used to query the object we use to install the pgpamdb file later
#remove.packages("curl")
#install.packages("curl")

Packages<-c("gh","doParallel","dplyr","tuneR","signal","foreach","oce","randomForest","seewave","plotrix","autoimage","pracma","PRROC","flux","stringi","caTools","sqldf","RPostgres","png") #"Rtools"?

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

#if on windows, check for curl verion. Since this requires Rtools, will involve some manual setup in the case that curl 
if(.Platform$OS.type == "windows"){
  curl_version <- packageVersion("curl")
  
  # too early
  too_early_version <- "5.0.0"
  
  # Assert that the installed version is greater than the minimum
  stopifnot(
    "The 'curl' package version must be greater than 5.0.0. Please update it: requires Rtools for versions > 5.0.0" = 
      curl_version > too_early_version
  )
  
  
}

if(require("imager",character.only=TRUE)){
  print("imager is installed")
}else{
  #imager on linux tries to install X11 dependencies. Forbid it from doing so (should be harmless on windows) 
  install.packages('imager', configure.args='--without-X11', repos = "http://cran.us.r-project.org")
  if(require("imager",character.only=TRUE)){
    print("imager is installed")
  }else{
    stop("could not install imager")
  }
}

# 1. Define the repository owner and name
repo_owner <- "DanWoodrichNOAA"
repo_name <- "pgpamdb"

# Find the script's own directory
args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
script_dir <- dirname(script_path)

# Define the destination file path
dest_file <- file.path(script_dir, "pgpamdb_latest.tar.gz")

# 2. Use the gh package to query the GitHub API for the latest release
cat("Fetching latest release information for", paste0(repo_owner, "/", repo_name), "...\n")
latest_release <- gh::gh(
  "GET /repos/{owner}/{repo}/releases/latest",
  owner = repo_owner,
  repo = repo_name
)

asset_urls <- sapply(latest_release$assets, function(asset) asset$browser_download_url)
tarball_url <- asset_urls[endsWith(asset_urls, ".tar.gz")][1]

# 4. Check if a URL was found and download the file
if (!is.null(tarball_url) && !is.na(tarball_url)) {
  cat("Found .tar.gz asset URL:", tarball_url, "\n")
  cat("Downloading to:", dest_file, "\n")
  
  # Use download.file() to save the asset
  download.file(url = tarball_url, destfile = dest_file, mode = "wb")
  
  cat("Download complete!\n")
} else {
  cat("Could not find a .tar.gz asset in the latest release.\n")
}





install.packages(file.path(script_dir, "pgpamdb_latest.tar.gz"), source = TRUE, repos=NULL)
