jobs = {
"MPE":["PE2all","PE2_FG","ED_AM_perfEval"],
"TT":["PE2_FG_novel","PE2all","PerfEval_MPE_TT","PE2_FG","PE2all_novel"],
"TTplus":["PE2_FG_novel","PE2all","PerfEval_MPE_TT","PE2_FG","PE2all_novel","Graph_RFNwAL_AC","ViewRFN"],
"NewJob":["ViewGT","ViewED"],
"MPE_DL_SS":["DLModelTrain_Eval_SS","PE2DL_all_SS","PE2DL_FG_SS"], #Training/test: assumes splits from same set of FGs. 
"MPE_DL_Train": ["DLModelTrain_Eval_Train","PE2DL_all_Train","PE2DL_FG_Train"], #Training: assumes test split sourced from different data than FGs
"Test_DL_novel":["MPE_DL_Train","PE2DL_all_novel","PE2DL_FG_novel"] #add MPE_DL_TT when complete #Test: assumes test split sourced from different data than FGs
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
