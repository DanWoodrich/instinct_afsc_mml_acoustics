#WARNING: this is not going to be robust, as written, to soundfiles that have the same unique name. 
#todo: Need to grab SF ids within queries
#TODO: probably need splicing instead of unknown label- unknown helps classifier but boundaries are still 
#meaningful for detection and regression. Only copy/slice data that needs it, leave others in normal place. 
#TODO: ensure shuffling is done on the training dataset before writing out to regularize. 

#install.packages("//nmfs/akc-nmml/CAEP/Acoustics/Matlab Code/Other code/R/pgpamdb/pgpamdb_0.1.0.tar.gz", source = TRUE, repos=NULL)
library(pgpamdb)
library(DBI)


args="../cache/169763/801106 ../cache/399753/538819/307679 ../cache/399753/538819/307679/919206 ../temp/DecimatedWaves 1 y 0.80 0.75 y prune-associate-v1-1"
args<-strsplit(args,split=" ")[[1]]

source(paste(getwd(),"/user/R_misc.R",sep=""))
args<-commandIngest()


FGpath <-args[1]
DETpath <- args[2]
Resultpath <- args[3]
SF_foc = args[4] #this will get the prefix for absolute file path. 
sampling_rate = args[5]
seed_val = args[6]
use_seed <- args[7]
train_test_split <- as.numeric(args[8]) #= 1.00
train_val_split <-as.numeric(args[9]) #.66

test = 1 - train_test_split
train = train_val_split * (train_test_split)
val = 1 - test - train

#setwd("C:/Users/pam_user/INSTINCT/bin")

GT<-read.csv(paste(DETpath,"DETx.csv.gz",sep="/"),stringsAsFactors = FALSE) #add this in 1.3 for better backwards compatability with R

FG<-read.csv(paste(FGpath,"FileGroupFormat.csv.gz",sep="/"),stringsAsFactors = FALSE)

unique_combos <- unique(FG[, c("FileName", "Deployment")])

# 2. Count how many Deployments exist for each FileName
counts <- table(unique_combos$FileName)

# 3. Identify FileNames with > 1 Deployment
dupes <- names(counts[counts > 1])

# 4. Action: Stop and report if duplicates found
if (length(dupes) > 0) {
  # Extract the specific problem rows to show the user
  problem_rows <- unique_combos[unique_combos$FileName %in% dupes, ]
  problem_rows <- problem_rows[order(problem_rows$FileName), ]
  
  print("CRITICAL WARNING: The following files exist in multiple Deployments:")
  print(problem_rows)
  
  stop("Pipeline halted: Duplicate filenames detected across different deployments. Please resolve before proceeding.")
} else {
  message("Pre-check passed: No duplicate filenames across deployments found.")
}

#make a selection tables dir. 
dir.create(paste(Resultpath,"selection_tables",sep="/"))

#process: for each unique file in effort (FG): give random split assignment (have test = 15, val = 15, train = 70). Copy matching slice of annotations, format them into 
#the voxaboxen format. For my ground truth (GT), add 'positive' as label to voxaboxen format. Compare the bin start ends with the total duration
#of the soundfile, and for missing effort in the soundfile, add as a new label to voxaboxen format a 'out_of_effort' category spanning each missing
#bin. Write out selection table. save the manifests (mapping soundfile to selection table) in a memory list, then for a final step, 
#write out all into a zip file "VoxaboxenManifests.gz"

#here is the FG (effort) schema:
#FileName               FullPath           StartTime Duration   Deployment SegStart
#1 AU-AWNM01-150924-144000.wav /AW15_AU_NM01/09_2015/ 2015-09-24 14:40:00      600 AW15_AU_NM01      180
#2 AU-AWNM01-151220-115000.wav /AW15_AU_NM01/12_2015/ 2015-12-20 11:50:00      600 AW15_AU_NM01      540
#3 AU-AWNM01-151220-120000.wav /AW15_AU_NM01/12_2015/ 2015-12-20 12:00:00      600 AW15_AU_NM01        0
#4 AU-AWNM01-151220-120000.wav /AW15_AU_NM01/12_2015/ 2015-12-20 12:00:00      600 AW15_AU_NM01       90
#5 AU-AWNM01-151221-072000.wav /AW15_AU_NM01/12_2015/ 2015-12-21 07:20:00      600 AW15_AU_NM01        0
#6 AU-AWNM01-151221-072000.wav /AW15_AU_NM01/12_2015/ 2015-12-21 07:20:00      600 AW15_AU_NM01      270
#SegDur          Name DiffTime
#1     90 NM15_SC1_INS0        1
#2     60 NM15_SC1_INS0        2
#3     90 NM15_SC1_INS0        2
#4     90 NM15_SC1_INS0        2
#5     90 NM15_SC1_INS0        3
#6     90 NM15_SC1_INS0        4

#here is the GT (Ground truth) schema. Only id needs to be retained in addition to what voxaboxen expects. in memory df of
#head(GT)
#  StartTime EndTime LowFreq HighFreq                   StartFile                     EndFile probability
#1   270.155 272.556   869.6   1269.7 AU-AWNM01-160103-151000.wav AU-AWNM01-160103-151000.wav          NA
#2   114.628 116.882   469.6    765.3 AU-AWNM01-160125-122000.wav AU-AWNM01-160125-122000.wav          NA
#3   140.630 142.699  1008.8   1548.0 AU-AWNM01-160125-122000.wav AU-AWNM01-160125-122000.wav          NA
#4   179.350 181.382   765.3   1130.5 AU-AWNM01-160125-122000.wav AU-AWNM01-160125-122000.wav          NA
#5   183.975 185.859   643.5    939.2 AU-AWNM01-160220-130000.wav AU-AWNM01-160220-130000.wav          NA
#6   194.246 196.500  1043.6   1443.6 AU-AWNM01-160220-130000.wav AU-AWNM01-160220-130000.wav          NA
#  comments procedure label signal_code strength            modified analyst status original_id date_created
#1                 10     1          17        2 2024-01-25 19:01:49       2      1    75415607           NA
#2                 10     1          17        2 2024-01-25 19:01:49       2      1    75415608           NA
#3                 10     1          17        2 2024-01-25 19:01:49       2      1    75415609           NA
#4                 10     1          17        2 2024-01-25 19:01:49       2      1    75415610           NA
#5                 10     1          17        2 2024-01-25 19:01:49       2      1    75415611           NA
#6                 10     1          17        2 2024-01-25 19:01:49       2      1    75415612           NA
#  changed_by uploaded_by       id
#1          2           2 75415607
#2          2           2 75415608
#3          2           2 75415609
#4          2           2 75415610
#5          2           2 75415611
#6          2           2 75415612

#here is the output selection table schema. Text file of:

#Begin Time (s)	End Time (s)	Annotation	Low Freq (Hz)	High Freq (Hz)
#3.236750000000029	3.347750000000133	voc	100	3900.0
#17.508749999999964	17.606750000000147	voc	100	3900.0
#19.045749999999998	19.133749999999964	voc	100	3900.0
#21.864749999999958	21.984750000000076	voc	100	3900.0#

#here is the manifest schema. .csv. One per split (train/val/test), in a R memory list and zipped. 

#fn,audio_fp,selection_table_fp
##dcase_MK1_val,C:/Users/pam_user/Desktop/voxaboxen_test_env_config/voxaboxen-demo_formatted/formatted\audio/dcase_MK1_val.wav,C:/Users/pam_user/Desktop/voxaboxen_test_env_config/voxaboxen-demo_formatted/formatted\selection_tables/dcase_MK1_val.txt
#case_MK2_val,C:/Users/pam_user/Desktop/voxaboxen_test_env_config/voxaboxen-demo_formatted/formatted\audio/dcase_MK2_val.wav,C:/Users/pam_user/Desktop/voxaboxen_test_env_config/voxaboxen-demo_formatted/formatted\selection_tables/dcase_MK2_val.txt

sel_table_dir <- file.path(Resultpath, "selection_tables")
dir.create(sel_table_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Assign Splits (Train/Val/Test)
# Get unique files from the Effort (FG) table
unique_files <- unique(FG$FileName)
n_files <- length(unique_files)

# Set seed for reproducibility
if(use_seed=="true"){
  set.seed(as.integer(seed_val))
}


# Generate random assignments: 
assignments <- sample(1:3, n_files, replace = TRUE, prob = c(train, val, test))
file_map <- data.frame(FileName = unique_files, SplitGroup = assignments, stringsAsFactors = FALSE)

# Initialize list to store manifest rows
manifests <- list(train = list(), val = list(), test = list())

# 3. Process each file
for (i in 1:n_files) {
  
  # A. Identify File and Metadata
  f_name <- unique_files[i]
  
  # Extract file-level metadata from FG (using the first row found for this file)
  fg_sub <- FG[FG$FileName == f_name, ]
  #if(nrow(fg_sub) == 0) next # Skip if no effort data found
  
  file_meta <- fg_sub[1, ]
  total_dur <- file_meta$Duration
  
  # Construct full audio path (handling potential missing trailing slashes)
  audio_dir <- file_meta$FullPath
  if (!grepl("/$", audio_dir)) audio_dir <- paste0(audio_dir, "/")
  prefix = paste(SF_foc,"/",sampling_rate,"/",sep="")
  full_audio_path <- paste0(prefix,audio_dir, f_name)
  #expand relative path so raven is not sad
  full_audio_path = normalizePath(full_audio_path, winslash = "/", mustWork = FALSE)
  
  # B. Build Selection Table Data
  
  # --- Step B1: Ground Truth (Positive) ---
  gt_sub <- GT[GT$StartFile == f_name, ]
  
  # Initialize empty data frame for this file's selections
  # Using the specific column names required for the text output
  selections <- data.frame(
    begin = numeric(), end = numeric(), annot = character(),
    low_f = numeric(), high_f = numeric(), stringsAsFactors = FALSE
  )
  
  if (nrow(gt_sub) > 0) {
    pos_rows <- data.frame(
      begin = gt_sub$StartTime,
      end = gt_sub$EndTime,
      annot = "positive",
      low_f = gt_sub$LowFreq,
      high_f = gt_sub$HighFreq,
      stringsAsFactors = FALSE
    )
    selections <- rbind(selections, pos_rows)
  }
  
  # --- Step B2: Effort (Out of Effort) ---
  # Sort effort segments by start time
  effort_segs <- fg_sub[order(fg_sub$SegStart), ]
  
  cursor <- 0
  
  # Iterate through sorted segments to find gaps
  if (nrow(effort_segs) > 0) {
    for (r in 1:nrow(effort_segs)) {
      seg_start <- effort_segs$SegStart[r]
      seg_dur   <- effort_segs$SegDur[r]
      seg_end   <- seg_start + seg_dur
      
      # If there is a gap between current cursor and start of segment
      if (seg_start > cursor) {
        # Create 'out_of_effort' entry
        ooe_row <- data.frame(
          begin = cursor,
          end = seg_start,
          annot = "out_of_effort",
          low_f = 0,    # Defaulting to 0 as no specific freq requested for OOE
          high_f = 0,
          stringsAsFactors = FALSE
        )
        selections <- rbind(selections, ooe_row)
      }
      
      # Move cursor to the end of the current segment (if it pushes boundaries forward)
      if (seg_end > cursor) {
        cursor <- seg_end
      }
    }
  }
  
  # Check for final gap at the end of the file
  if (cursor < total_dur) {
    ooe_row <- data.frame(
      begin = cursor,
      end = total_dur,
      annot = "out_of_effort",
      low_f = 0,
      high_f = 0,
      stringsAsFactors = FALSE
    )
    selections <- rbind(selections, ooe_row)
  }
  
  # --- Step B3: Write Selection Table ---
  
  # Sort final table by start time
  if (nrow(selections) > 0) {
    selections <- selections[order(selections$begin), ]
  }
  
  # Define output path
  # Convention: replace extension with .txt
  out_txt_name <- sub("\\.[^.]+$", ".txt", f_name) 
  out_txt_path <- file.path(sel_table_dir, out_txt_name)
  
  # Rename columns to match the specific "Human Readable" format requested
  output_df <- selections
  colnames(output_df) <- c("Begin Time (s)", "End Time (s)", "Annotation", "Low Freq (Hz)", "High Freq (Hz)")
  
  # Write tab-delimited file
  write.table(output_df, file = out_txt_path, sep = "\t", 
              row.names = FALSE, quote = FALSE)
  
  # C. Add to Manifest List
  # Get split assignment
  split_id <- file_map$SplitGroup[file_map$FileName == f_name]
  
  # Create manifest row
  # fn: using filename without extension (common ID format)
  fn_id <- tools::file_path_sans_ext(f_name)
  
  # Using normalizePath for absolute paths if the files exist locally, 
  # otherwise using the constructed path strings.
  # Assuming we want the path string we constructed:
  
  man_entry <- data.frame(
    fn = fn_id,
    audio_fp = full_audio_path,
    selection_table_fp = normalizePath(out_txt_path, winslash = "/", mustWork = FALSE),
    stringsAsFactors = FALSE
  )
  
  if (split_id == 1) manifests$train[[length(manifests$train) + 1]] <- man_entry
  if (split_id == 2) manifests$val[[length(manifests$val) + 1]] <- man_entry
  if (split_id == 3) manifests$test[[length(manifests$test) + 1]] <- man_entry
}

# 4. Write Manifests and Zip
# Bind lists into dataframes
train_df <- do.call(rbind, manifests$train)
val_df   <- do.call(rbind, manifests$val)
test_df  <- do.call(rbind, manifests$test)

# Define manifest temp directory
man_dir <- file.path(Resultpath, "manifests_temp")
dir.create(man_dir, showWarnings = FALSE)

# Write CSVs
write.csv(train_df, file.path(man_dir, "train_info.csv"), row.names = FALSE, quote = FALSE)
write.csv(val_df, file.path(man_dir, "val_info.csv"), row.names = FALSE, quote = FALSE)
write.csv(test_df, file.path(man_dir, "test_info.csv"), row.names = FALSE, quote = FALSE)

# Zip them into VoxaboxenManifests.zip
# Note: Base R 'zip' creates a .zip archive. 
# If a specific .gz extension is strictly required for a multi-file archive, 
# it usually implies a tarball, but .zip is standard for Windows/Cross-platform lists.
cwd <- getwd()
setwd(man_dir)
zip(zipfile = "VoxaboxenManifests.zip", files = c("train_info.csv", "val_info.csv", "test_info.csv"))

# Move zip to Resultpath and clean up
setwd(cwd)
file.rename(paste0(Resultpath,"/manifests_temp/VoxaboxenManifests.zip"), file.path(Resultpath, "VoxaboxenManifests.zip"))

unlink(man_dir, recursive = TRUE)



