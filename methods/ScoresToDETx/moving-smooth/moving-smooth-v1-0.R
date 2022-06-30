#pseudo: 

#load in FG, scores, split files, stride_pix, native pix per second, parameter 'stage' (train,val,test) , parameter group_pixels, parameter smooth method , parameter split method (global) 

#what if I just output a pixel start along with the scores? Then I could backtrack however I want? (would need some knowledge of the split protocol - if knowing splits were 'by file', would assume that each 
#break is the start of a new file. But if I used a different procedure, I would have to encorportate this. 


#args<-"//161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/bd89c5999c35/ //161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/bd89c5999c35/a21b11/ //161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/bd89c5999c35/d69c8f/ //161.55.120.117/NMML_AcousticsData/Working_Folders/INSTINCT_cache/Cache/bd89c5999c35/d69c8f/795dc2 n mean labels-w-GT-bins-v1-4"

#args<-strsplit(args,split=" ")[[1]]

args<-commandArgs(trailingOnly = TRUE)

#docker values
FGpath <- args[1]
ScorePath <- args[2]
SpecPath <- args[3]
SplitPath <- args[4]
resultPath <- args[5]

group_pix = args[6]
native_pix_per_sec= args[7]
smooth_method= eval(parse(text=args[8]))
split_protocol= args[9]
stride_pix= args[10]


stop()