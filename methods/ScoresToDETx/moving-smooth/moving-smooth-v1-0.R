#pseudo: 

#load in FG, scores, split files, stride_pix, native pix per second, parameter 'stage' (train,val,test) , parameter group_pixels, parameter smooth method

#going to be tricky to reconstruct since have to use same assumptions used during model training to get to splits. Like, where does it start in 'splits'? Could be anywhere within the range, dependent on the size
#of what came before (anywhere within 1 window). 

#What I should do is export all of the scores, not just val. That way I can reuse the object and I can actually associate the split files. 

#next, think about how I want to handle the cases (train, test1, test2, and inference). can maybe do if train (run train), and then have standard script for outputing scores that can be run by both train and test. 