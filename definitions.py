jobs = {
"MPE":["PE2all","PE2_FG","ED_AM_perfEval"],
"TT":["PE2_FG_novel","PE2all","PerfEval_MPE_TT","PE2_FG","PE2all_novel"],
"TTplus":["PE2_FG_novel","PE2all","PerfEval_MPE_TT","PE2_FG","PE2all_novel","Graph_RFNwAL_AC","ViewRFN"],
"NewJob":["ViewGT","ViewED"],
"MPE_DL_SS":["DLModelTrain_Eval_SS","PE2DL_all_SS","PE2DL_FG_SS","ViewScoresDL_w_AL_SS","Model_Stats_SS"], #Training/test: assumes splits from same set of FGs. 
"MPE_DL_Train": ["DLModelTrain_Eval_Train","PE2DL_all_Train","PE2DL_FG_Train","ViewScoresDL_w_AL_Train","Model_Stats_Train","TrainModel_DL2"],#Training: assumes test split sourced from different data than FGs
"MPE_DL_Train_nopp": ["DLModelTrain_Eval_Train","PE2DL_all_Train","PE2DL_FG_Train","ViewScoresDL_w_AL_Train_allGT","Model_Stats_Train"],
"Test_DL_novel":["PE2DL_all_novel","PE2DL_FG_novel","Model_Stats_novel"],#,"ViewScoresDL_w_AL_novel"]#,""] #add MPE_DL_TT when complete #Test: assumes test split sourced from different data than FGs
"DL_TT":["Model_Stats_TT"] #"MPE_DL_Train","Test_DL_novel",
}

pipelines = {
    #testing!
'ViewGTxxxc':{'pipe':"pViewDETx",'pViewDETx':{'process':"RavenViewDETx",'GetDETx':{'pipe':"FormatGT.pipe",'loop_on':"FormatFG",'FormatGT.pipe':{'process':"FormatGT",'GetFG':"FormatFG"}},\
             "GetFG":{'pipe':"FormatFG.pipe",'loop_on':"FormatFG",'FormatFG.pipe':{'process':"FormatFG"}}}},
#'ViewGT':{'pipe':"pViewDETx",'loop_on':"FormatFG",'pViewDETx':{'process':"RavenViewDETx",'GetDETx':{'process':"FormatGT"},\
#             "GetFG":{'process':"FormatFG"}}}, #doesn't work, since loop is assessed on components not the product
'ViewED':{"pViewDETx":{'pipe':"RavenViewDETx","GetDETx":{'pipe':"EventDetector"},\
             "GetFG":{'pipe':"FormatFG"}}}
}
