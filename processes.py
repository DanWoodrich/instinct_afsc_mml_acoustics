#can choose to import in global namespace
from classes import INSTINCT_process,Split_process,SplitRun_process,Unify_process,INSTINCT_userprocess
from getglobals import PARAMSET_GLOBALS
from misc import get_param_names,file_peek,get_difftime

import hashlib
import pandas as pd
import os
import math
import subprocess

from pipe_shapes import *
from .pipe_shapes import *

#custom modification of process to hash files (I use this for FormatFG and FormatGT) 
 #############
#this will load in the attributes that are shared by both INSTINCT processes and jobs.
class HashableFile:

    def getfilehash(self):
        
        if self.__class__.__name__=="FormatFG":
            
            path = PARAMSET_GLOBALS['project_root']+ "lib/user/Data/FileGroups/" + self.parameters['file_groupID']
            
        elif self.__class__.__name__=="FormatGT":
            dirpath = PARAMSET_GLOBALS['project_root']+ "lib/user/Data/GroundTruth/"+self.parameters['signal_code']
            
            path = dirpath + "/"+self.parameters['signal_code']+"_" + self.ports[0].parameters['file_groupID']
            
            if not os.path.exists(path): #if GT file doesn't exist, create an empty file
                
                GT = pd.DataFrame(columns = ["StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","label","Type","SignalCode"])
                #import code
                #code.interact(local=locals())
                os.makedirs(dirpath,exist_ok=True)
                GT.to_csv(path,index=False)

       

def hashfile(path):
    buff_size = 65536
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            data = f.read(buff_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

#####

####custom function to find the right decimation level from format FG. Usually just the port, but if it is looped, it won't be.. 
def find_decimation_level(obj,portnum):

    if obj.ports[portnum].__class__.__name__ == "CombineExtLoop":
        return obj.ports[portnum].ports[0].parameters['target_samp_rate']
    else:
        return obj.ports[portnum].parameters['target_samp_rate']
    
class SplitED(Split_process,INSTINCT_process):
    outfile = 'FileGroupFormat.csv.gz'

    def run(self):
        inFile = self.ports[0].outfilegen() 
        
        FG_dict = file_peek(inFile,fn_type = object,fp_type = object,st_type = object,\
                              dur_type = 'float64',comp_type = 'gzip')
        FG = pd.read_csv(inFile, dtype=FG_dict,compression='gzip')

        if self.splits == 1:
            FG.to_csv(self.outfilegen(),index=False,compression='gzip')
        else:
            row_counts = len(FG['DiffTime'])
            breaks = int(row_counts/self.splits)
            blist = numpy.repeat(range(0,self.splits),breaks)
            bdiff = row_counts - len(blist)
            extra = numpy.repeat(self.splits-1,bdiff)
            flist = list(blist.tolist() + extra.tolist())
            FG.loc[[x==self.split_ID-1 for x in flist]].to_csv(self.outfilegen(),index=False,compression='gzip')

class RunED(Split_process,SplitRun_process,INSTINCT_process):

    #need to define an outpath that is based on splitED... default goes off of last port. 
    outfile = 'DETx.csv.gz'
    SplitInitial=SplitED
    
    def run(self):
        
        if 'verbose' in self.arguments:
            if self.arguments['verbose']=='y':
                verbose = 'y'
            else:
                verbose = 'n'
        else:
            verbose = 'n'

            
        #import code
        #code.interact(local=locals())
        #param_names grandfathered in, should just have R parse dictionaries as a standard
        self.cmd_args=[PARAMSET_GLOBALS['SF_foc'] + "/" + find_decimation_level(self,0),self.outpath(),self.outpath(),\
                        os.path.basename(self.input().path),'1',self.arguments['cpu'],self.arguments['file_chunk_size'],verbose,\
                       'method1',self.parameters['methodID'] + '-' + self.parameters['methodvers'],self.param_string2,get_param_names(self.parameters)] #params
        
        self.run_cmd()
        #do this manually instead of using run_cmd to be compatible with prvs method
        #rework ED wrapper to work with python dict before reworking run_cmd to work with wrapper
        
class EventDetector(Unify_process,INSTINCT_process):

    outfile = 'DETx.csv.gz'
    SplitRun = RunED

    def run(self):
        
        EDdict = {'StartTime': 'float64', 'EndTime': 'float64','LowFreq': 'float64', 'HighFreq': 'float64', 'StartFile': 'category','EndFile': 'category','ProcessTag': 'category'}
        
        dataframes = [None] * int(self.arguments['splits'])
        for k in range(int(self.arguments['splits'])):
            dataframes[k] = pd.read_csv(self.outpath() +'/DETx' + str(k+1) + "_" + self.arguments['splits'] + '.csv.gz',dtype=EDdict)
        ED = pd.concat(dataframes,ignore_index=True)
        ED['ProcessTag2']=ED.ProcessTag.str.split('_', 1).map(lambda x: x[0])
        #determin PT changes
        
        statustest=[None]*len(ED['ProcessTag'])
        for n in range(len(ED['StartTime'])-1):
            statustest[n]=(ED['ProcessTag'][n]==ED['ProcessTag'][n+1])
        #will need a catch in here for if this situation is not present
        chED= ED.loc[[x==False for x in statustest]]
        statustest2=[None]*len(chED['ProcessTag'])
        for n in range(len(chED['StartTime'])-1):
            statustest2[n]=(chED['ProcessTag2'].values[n]==chED['ProcessTag2'].values[n+1])
        chED2= chED.loc[[x==True for x in statustest2]]
        indecesED = chED2.index.append(chED2.index+1)
        #import code
        #code.interact(local=locals())
        
        if indecesED.empty:
            EDfin = ED[['StartTime','EndTime','LowFreq','HighFreq','StartFile','EndFile']]
            ED.to_csv(self.outfilegen(),index=False,compression='gzip')
        else:
            EDfin = ED.loc[indecesED._values]
            #reduce this to just file names to pass to Energy detector (FG style)

            FG_cols = ['FileName','FullPath','StartTime','Duration','DiffTime','Deployment','SegStart','SegDur']

            FG_dict = {'FileName': 'string','FullPath': 'category', 'StartTime': 'string','Duration': 'float','Deployment':'string','SegStart':'float','SegDur':'float','DiffTime':'int'}
            #import code
            #code.interact(local=locals())

            FG = pd.read_csv(self.ports[0].outpath() +'/FileGroupFormat.csv.gz', dtype=FG_dict, usecols=FG_cols)

            FG = FG[FG.DiffTime.isin(EDfin['ProcessTag2'].astype('int32'))&FG.FileName.isin(EDfin['StartFile'])] #subset based on both of these: if a long difftime, will only
            #take the relevant start files, but will also go shorter than two files in the case of longer segments.
            
            #recalculate difftime based on new files included. <- metacomment: not sure why we need to do this? 
            FG['StartTime'] = pd.to_datetime(FG['StartTime'], format='%Y-%m-%d %H:%M:%S')
            FG = get_difftime(FG)
            
            #save FG
            FG.to_csv(self.outpath() + '/EDoutCorrect.csv.gz',index=False,compression='gzip')

            if 'verbose' in self.arguments:
                if self.arguments['verbose']=='y':
                    verbose = 'y'
                else:
                    verbose = 'n'
            else:
                verbose = 'n'

            #run second stage of EventDetector method
            self.cmd_args=[PARAMSET_GLOBALS['SF_foc'] + "/" + find_decimation_level(self,0),self.outpath(),self.outpath(),\
                        'EDoutCorrect.csv.gz','2',self.arguments['cpu'],self.arguments['file_chunk_size'],verbose,\
                       'method1',self.parameters['methodID'] + '-' + self.parameters['methodvers'],self.param_string2,get_param_names(self.parameters)]

            self.process_ID = self.__class__.__name__ #needs to be specified here since it's a wrapper, otherwise assumed as class name
            
            self.run_cmd()

            ED = ED.drop(columns="ProcessTag")
            ED = ED.drop(columns="ProcessTag2")

            EDdict2 = {'StartTime': 'float64', 'EndTime': 'float64','LowFreq': 'float64', 'HighFreq': 'float64', 'StartFile': 'category','EndFile': 'category','DiffTime': 'int'}
            #now load in result,
            EDpatches = pd.read_csv(self.outpath()+'/DETx_int.csv.gz',dtype=EDdict2)
            PatchList = [None] * len(EDpatches['DiffTime'].unique().tolist())

            for n in range(len(EDpatches['DiffTime'].unique().tolist())):

                nPatch = [EDpatches['DiffTime'].unique().tolist()[n]]
                EDpatchN=EDpatches.loc[EDpatches['DiffTime'].isin(nPatch),]
                FGpatch = FG[FG['DiffTime']==(n+1)]
                FirstFile = EDpatchN.iloc[[0]]['StartFile'].astype('string').iloc[0]
                LastFile = EDpatchN.iloc[[-1]]['StartFile'].astype('string').iloc[0]

                BeginRangeStart= FGpatch.iloc[0]['SegStart']
                BeginRangeEnd = BeginRangeStart+FGpatch.iloc[0]['SegDur']/2

                LastRangeStart= FGpatch.iloc[-1]['SegStart']
                LastRangeEnd = LastRangeStart+FGpatch.iloc[-1]['SegDur']/2

                EDpatchN = EDpatchN[((EDpatchN['StartTime'] > BeginRangeEnd) & (EDpatchN['StartFile'] == FirstFile)) | (EDpatchN['StartFile'] != FirstFile)]
                EDpatchN = EDpatchN[((EDpatchN['StartTime'] < LastRangeEnd) & (EDpatchN['StartFile'] == LastFile)) | (EDpatchN['StartFile'] != LastFile)]

                EDpatchN=EDpatchN.drop(columns="DiffTime")

                ED1 = ED.copy()[(ED['StartTime'] <= BeginRangeEnd) & (ED['StartFile'] == FirstFile)] #get all before patch
                ED2 = ED.copy()[(ED['StartTime'] >= LastRangeEnd) & (ED['StartFile'] == LastFile)]         #get all after patch
                ED3 = ED.copy()[(ED['StartFile'] != FirstFile) & (ED['StartFile'] != LastFile)]

                ED = pd.concat([ED1,ED2,ED3],ignore_index=True)

                EDpNfiles = pd.Series(EDpatchN['StartFile'].append(EDpatchN['EndFile']).unique()) #switched to numpy array on an unknown condition, pd.Series forces it to stay this datatype. Needs testing

                FandLfile = [FirstFile,LastFile]
                internalFiles = EDpNfiles[EDpNfiles.isin(FandLfile)==False]

                if len(internalFiles)>0:
                    #subset to remove internal files from patch from ED
                    ED = ED[(ED.StartFile.isin(internalFiles)==False)&(ED.EndFile.isin(internalFiles)==False)]

                #here, subset all the detections within EDpatchN: find any sound files that are not start and end file, and remove them from ED
                #hint: isin to find files in EDpN, and isin to subset ED. 
                #ED = ED[(ED.StartFile.isin(EDpatchN['StartFile'])==False)

                #save ED patch
                PatchList[n]=EDpatchN

            EDpatchsub = pd.concat(PatchList,ignore_index=True)
            #combine ED and EDpatch
            
            ED = pd.concat([EDpatchsub,ED],ignore_index=True)
            ED = ED.sort_values(['StartFile','StartTime'], ascending=[True,True])

            os.remove(self.outpath() + '/DETx_int.csv.gz')
            os.remove(self.outpath() + '/EDoutCorrect.csv.gz')

            ED.to_csv(self.outfilegen(),index=False,compression='gzip')

class SplitFE(Split_process,INSTINCT_process):
    outfile = 'DETx.csv.gz'

    def run(self):

        inFile = self.ports[0].outfilegen() 

        DETdict = {'StartTime': 'float64', 'EndTime': 'float64','LowFreq': 'float64', 'HighFreq': 'float64', 'StartFile': 'category','EndFile': 'category'}
        DET = pd.read_csv(inFile, dtype=DETdict,compression='gzip')
        
        if self.splits == 1:
            DET.to_csv(self.outfilegen(),index=False,compression='gzip')
        #need to test this section to ensure forking works 
        else:
            row_counts = len(DET['StartTime'])
            breaks = int(row_counts/self.splits)
            blist = numpy.repeat(range(0,self.splits),breaks)
            bdiff = row_counts - len(blist)
            extra = numpy.repeat(self.splits-1,bdiff)
            flist = list(blist.tolist() + extra.tolist())
            DET.loc[[x==self.split_ID-1 for x in flist]].to_csv(self.outfilegen(),index=False,compression='gzip')
        
class RunFE(Split_process,SplitRun_process,INSTINCT_process):

    outfile = 'DETx_int.csv.gz'
    SplitInitial=SplitFE
    
    def run(self):

        if 'verbose' in self.arguments:
            if self.arguments['verbose']=='y':
                verbose = 'y'
            else:
                verbose = 'n'
        else:
            verbose = 'n'

        self.cmd_args= [self.ports[1].outpath(),os.path.dirname(self.input().path),PARAMSET_GLOBALS['SF_foc'] + "/" + find_decimation_level(self,1),\
                        self.outpath(),str(self.split_ID) + '_' + str(self.splits),str(self.arguments['cpu']),verbose,\
                        'method1',self.parameters['methodID'] + '-' + self.parameters['methodvers'],self.param_string2,get_param_names(self.parameters)]
        
        self.run_cmd()

class FeatureExtraction(Unify_process,INSTINCT_process):

    pipeshape = TwoUpstream
    upstreamdef = ['GetFG','GetDETx']

    outfile = 'DETx.csv.gz'
    SplitRun = RunFE

    def run(self):
        
        dataframes = [None] * int(self.arguments['splits'])
        for k in range(int(self.arguments['splits'])):
            dataframes[k] = pd.read_csv(self.outpath() + '/DETx_int' + str(k+1) + "_" + self.arguments['splits'] + '.csv.gz')
        FE = pd.concat(dataframes,ignore_index=True)

        
        FE.to_csv(self.outfilegen(),index=False,compression='gzip')

class RavenViewDETx(INSTINCT_process):
    outfile = 'RAVENx.txt'

    def run(self):
        #import code
        #code.interact(local=dict(globals(), **locals()))
        self.cmd_args=[self.ports[0].outpath(),self.ports[1].outpath(),self.outpath(),\
                       PARAMSET_GLOBALS['SF_foc'] + "/" + find_decimation_level(self,1),\
                       os.environ["INS_ARG_SEP"].join(self.arguments.values()).strip(),self.param_string2.strip(),\
                       PARAMSET_GLOBALS['SF_raw']]
        
        
        self.run_cmd()

class RavenToDETx(INSTINCT_process):
    outfile = 'DETx.csv.gz'

    def run(self):
        #import code
        #code.interact(local=locals())
        self.cmd_args=[self.ports[0].outpath(),self.ports[1].outpath(),self.outpath(),\
                       self.ports[0].outfile]
        
        self.run_cmd()

class ReduceByField(INSTINCT_process):

    outfile = 'DETx.csv.gz'

    def run(self):
        
        self.cmd_args=[self.ports[0].outpath(),self.ports[1].outpath(),self.outpath(),self.param_string2]
        
        self.run_cmd()

class ApplyCutoff(INSTINCT_process):

    pipeshape = OneUpstream
    upstreamdef = ["GetDETx"]

    outfile = 'DETx.csv.gz'

    def run(self):

        
        
        DETwProbs = pd.read_csv(self.ports[0].outpath() + '/DETx.csv.gz',compression='gzip')
        #print(self.parameters['cutoff'])
        #make it work for 'probability' too. 
        #import code
        #code.interact(local=locals())
        DwPcut = DETwProbs[DETwProbs.probs>=float(self.parameters['cutoff'])]

        if 'append_cutoff' in self.parameters:
            if self.parameters['append_cutoff']=='y':
                #import code
                #code.interact(local=locals())
                
                DwPcut["cutoff"]=float(self.parameters['cutoff'])
        
        DwPcut.to_csv(self.outfilegen(),index=False,compression='gzip')
    
class AssignLabels(INSTINCT_process):

    pipeshape =ThreeUpstream_bothUpTo1
    upstreamdef = ["GetFG","GetDETx","GetGT"]

    outfile = 'DETx.csv.gz'

    def run(self):
    
        #import code
        #code.interact(local=locals())
        
        self.cmd_args=[self.ports[2].outpath(),self.ports[0].outpath(),self.ports[1].outpath(),self.outpath(),self.param_string2]
        
        self.run_cmd()

class PerfEval1_s1(INSTINCT_process):

    pipeshape = TwoUpstream
    upstreamdef = ["GetFG","GetAL"]

    outfile = 'Stats.csv.gz'
    #FG,LAB,AC
    def run(self):

        #import code
        #code.interact(local=locals())
        
        self.cmd_args=[self.ports[1].outpath(),self.ports[0].outpath(),'NULL',self.outpath(),self.ports[1].parameters['file_groupID'],"FG"]
        
        self.run_cmd()

class PerfEval1_s2(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["GetFG","GetPE1_S1"]

    outfile = 'Stats.csv.gz'

    def run(self):

        #import code
        #code.interact(local=locals())
        
        self.cmd_args=['NULL','NULL',self.input()[0].path,self.outfilegen(),'NULL','All']
        
        self.run_cmd()

class PerfEval2(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["EDperfEval","GetModel_w_probs"]
    
    outfile = 'PRcurve.png'

    def run(self):
        
        self.cmd_args=[self.ports[0].outpath(),self.outpath(),self.ports[1].outpath(),self.param_string2]
        
        self.run_cmd()

class PerfEval2DL(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["GetStats","GetModel_w_probs"]
    
    outfile = 'PE2ball.tgz'

    def run(self):
        
        self.cmd_args=[self.ports[0].outpath(),self.outpath(),self.ports[1].outpath(),self.param_string2]
        
        self.run_cmd()

class TrainModel_RF_CV(INSTINCT_process):

    pipeshape = ThreeUpstream_noCon
    upstreamdef = ["GetFG","GetDETx_w_FE","GetDETx_w_AL"]

    outfile = 'DETx.csv.gz'

    def run(self):
        
        self.cmd_args=[self.input()[0].path,self.input()[1].path,self.input()[2].path,self.outpath(),"NULL","CV",self.arguments['cpu'],self.param_string2]
        
        self.run_cmd()

class TrainModel_RF_obj(INSTINCT_process):

    pipeshape = ThreeUpstream_noCon
    upstreamdef = ["GetFG","GetDETx_w_FE","GetDETx_w_AL"]
    
    outfile = 'RFmodel.rds'

    def run(self):

        self.cmd_args=[self.input()[0].path,self.input()[1].path,self.input()[2].path,self.outpath(),"NULL","train",self.arguments['cpu'],self.param_string2]
        
        self.run_cmd()

class TrainModel_RF_apply(INSTINCT_process):

    pipeshape = ThreeUpstream
    upstreamdef = ["GetFG","GetDETx_w_FE","GetModel_obj"]

    outfile = 'DETx.csv.gz'

    def run(self):
        #import code
        #code.interact(local=locals())
        self.cmd_args=["NULL",self.input()[1].path,self.input()[2].path,self.outpath(),self.input()[0].path,"apply",self.arguments['cpu'],self.param_string2]
        
        self.run_cmd()
        
class SplitForPE(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["GetFG","GetModel_w_probs"]
    
    outfile = "DETx.csv.gz"
    
    def run(self):
        DETwProbs = pd.read_csv(self.ports[0].outfilegen(),compression='gzip')
        
        #import code
        #code.interact(local=locals())
        
        DwPsubset = DETwProbs[DETwProbs.FGID == self.ports[1].parameters['file_groupID']]
        DwPsubset.to_csv(self.outfilegen(),index=False,compression='gzip')
        
class StatsTableCombine_ED_AM(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["EDperfEval","AMperfEval"]

    #this one's hardcoded, no elegant way I could find to extract the pipeline history in an elegant/readible way. 
    outfile = "Stats.csv.gz"

    stagename = 'ED/AM'
        
    def run(self):

        #import code
        #code.interact(local=locals())
    
        #import datasets, tag with new ID column, rbind them. 
        Stats1 = pd.read_csv(self.ports[0].outfilegen(),compression='gzip') #AM
        Stats2 = pd.read_csv(self.ports[1].outfilegen(),compression='gzip') #ED

        Stats1[self.stagename] = self.upstreamdef[1]
        Stats2[self.stagename] = self.upstreamdef[0]

        StatsOut = pd.concat([Stats2,Stats1])

        StatsOut.to_csv(self.outfilegen(),index=False,compression='gzip')

class StatsTableCombine_TT(StatsTableCombine_ED_AM):

    upstreamdef = ["MPE_perfEval","TT_perfEval"]

    stagename = 'MPE/TT'
    
class AddFGtoDETx(INSTINCT_process):

    pipeshape = TwoUpstream
    upstreamdef = ['GetFG','GetDETx']

    outfile = "DETx.csv.gz"
    
    def run(self):
        DETx = pd.read_csv(self.ports[0].outfilegen(),compression='gzip')
        DETx['FGID'] =self.ports[1].parameters['file_groupID']
        DETx.to_csv(self.outfilegen(),index=False,compression='gzip')

class QueryData(INSTINCT_process):

    pipeshape = NoUpstream
    upstreamdef = [None]

    outfile = 'table.csv.gz' #could be FG, detx, etc, who knows

    def run(self):

        self.cmd_args=[self.outpath(),PARAMSET_GLOBALS['SF_raw'],self.outfile,self.param_string2]
        
        self.run_cmd()


class GraphDETx(INSTINCT_process):

    pipeshape = TwoUpstream_noCon

    upstreamdef = ["GetFG","GetDETx"]

    outfile = 'DETxGraph.png'

    def run(self):

        self.cmd_args=[self.ports[1].outfilegen(),self.ports[0].outfilegen(),self.outfilegen(),self.ports[1].parameters['file_groupID'],self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class PeaksAssoc(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["GetPeaksDETx","GetAssocDETx"]

    outfile = 'DETx.csv.gz' #takes peak labels and associates them with other dets.

    def run(self):

        #import code
        #code.interact(local=locals())

        self.cmd_args=[self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class ProcedureProgress(INSTINCT_process):

    pipeshape = OneUpstream
    upstreamdef = ["DependsOn"] #doesn't actually use this, just needs for it be run prior. 

    outfile = 'artifacts.tar'#'progress_vis.png' #takes peak labels and associates them with other dets.

    def run(self):

        if self.ports[0]==None:
            depends_on = "NULL"
        else:
            depends_on = self.ports[0].outpath()

        self.cmd_args=[depends_on,self.outpath(),self.param_string2]#self.ports[0].outpath()

        self.run_cmd()
        

class CalcPeaks(INSTINCT_process):#

    pipeshape = OneUpstream
    upstreamdef = ["GetDETx"]

    outfile = 'DETx.csv.gz' #this is peaks - assoc dets saved as DETx2.csv.gz

    def run(self):

        #import code
        #code.interact(local=locals())

        self.cmd_args=[self.ports[0].outpath(),self.outpath(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class PublishDets(INSTINCT_process):#

    pipeshape = OneUpstream
    upstreamdef = ["GetData"]

    outfile = 'receipt.txt' #record of the database transaction

    def run(self):

        #import code
        #code.interact(local=locals())

        self.cmd_args=[self.ports[0].outpath(),self.outpath(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class PublishDetswFG(INSTINCT_process):#

    pipeshape = TwoUpstream
    upstreamdef = ['GetFG','GetData']

    outfile = 'receipt.txt' #record of the database transaction

    def run(self):

        #import code
        #code.interact(local=locals())

        self.cmd_args=[self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class CompareAndPublishDets(INSTINCT_process):#

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["GetPriorData","GetEditData"]

    outfile = 'receipt.txt' #record of the database transaction

    def run(self):

        #import code
        #code.interact(local=locals())

        self.cmd_args=[self.ports[0].outpath(),self.ports[1].outpath(),self.outpath(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class FormatGT(INSTINCT_process):

    pipeshape = OneUpstream
    upstreamdef = ['GetFG']
    
    outfile = 'DETx.csv.gz'

    def infile(self):
        _dir = PARAMSET_GLOBALS['project_root']+ "lib/user/Data/GroundTruth/"+self.parameters['signal_code']
        path = _dir + "/"+ self.parameters['signal_code']+"_" + self.ports[0].parameters['file_groupID']
        return _dir,path

    def __init__(self, *args, **kwargs):

        
        super().__init__(*args, **kwargs)

        #

        if self.descriptors["runtype"]=='no_method':

            #import code
            #code.interact(local=dict(globals(), **locals()))

            #add infile to hash
            _dir,path = self.infile()
            
            if not os.path.exists(path): #if GT file doesn't exist, create an empty file
                GT = pd.DataFrame(columns = ["StartTime","EndTime","LowFreq","HighFreq","StartFile","EndFile","label","Type","f"])
                #import code
                #code.interact(local=locals())
                os.makedirs(_dir,exist_ok=True)
                GT.to_csv(path,index=False)

            self._Task__hash = hash(str(self._Task__hash)+ hashfile(path))
            self._Task__hash = int(str(self._Task__hash)[1:(1+self.hash_length)])

    def run(self):
        
        if self.descriptors["runtype"]=='no_method':
            GT = pd.read_csv(self.infile()[1])

            GT.to_csv(self.outfilegen(),index=False,compression='gzip')
        elif self.descriptors["runtype"]=='lib':

            #import code
            #code.interact(local=locals())

            self.cmd_args=[self.outfilegen(),self.ports[0].parameters['file_groupID'],get_param_names(self.parameters),self.param_string2]

            self.run_cmd()

class FormatFG(INSTINCT_process):
    
    pipeshape = OneUpstream
    upstreamdef = ['GetFG']

    outfile = 'FileGroupFormat.csv.gz'
    
    def cloud_cp(self,src,dest):
        
        commands = []
        for src, dest in zip(src,dest):
            commands.append(f'"{src}""{dest}"')
        #working on it...
        #gsutil_command = ['gsutil','-m',"cp"] + sum(cmd.split() for cmd in commands], [])
        
        try:
            subprocess.run(gsutil_command,check=True,sdout=subprocess.PIPE,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error during upload:{e.stderr.decode()}")

    def infile(self):
        if self.ports[0]!=None:
            file =self.ports[0].outfilegen()
        else:
            file = PARAMSET_GLOBALS['project_root'] + "lib/user/Data/FileGroups/" + self.parameters['file_groupID']
        return file

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.descriptors["runtype"]=='no_method':
            #add infile to hash
            if self.ports[0]!=None:
                self._Task__hash = hash(str(self._Task__hash)+ hashfile(self.infile()))

    def run(self):

        #catch by looking for upstream
        if self.descriptors["runtype"]=='no_method':
            file = self.infile()
            FG_dict = file_peek(file,fn_type = object,fp_type = object,st_type = object,dur_type = 'float64')
            FG = pd.read_csv(file, dtype=FG_dict)
            
        elif self.descriptors["runtype"]=='lib':

            temppath = self.outpath()+"/tempFG.csv.gz"
           
    
            #self.cmd_args=[temppath,self.parameters['file_groupID'],get_param_names(self.parameters)+" storage_service",self.param_string2+PARAMSET_GLOBALS['storage_service']]
               
            self.cmd_args=[temppath,self.parameters['file_groupID'],get_param_names(self.parameters),self.param_string2]
            #import code
            #code.interact(local=dict(globals(), **locals()))
            
            self.run_cmd()

            FG_dict = file_peek(temppath,fn_type = object,fp_type = object,st_type = object,dur_type = 'float64')
            FG = pd.read_csv(temppath, dtype=FG_dict)

            os.remove(temppath)
            
        #supports additional metadata fields
        
        FG['StartTime'] = pd.to_datetime(FG['StartTime'], format='%y%m%d-%H%M%S')

        #import code
        #code.interact(local=dict(globals(), **locals()))

        if "difftime_limit" in self.parameters:
            difftime_lim = int(self.parameters["difftime_limit"])
        else:
            difftime_lim=None #default

        FG=get_difftime(FG,cap_consectutive=difftime_lim)

        #and self.parameters['methodID2m'] == 'matlabdecimate'
        if 'decimate_data' in self.parameters and self.parameters['decimate_data'] == 'y':
            
            FullFilePaths = FG['FullPath'].astype('str') + FG['FileName'].astype('str')

            #remove duplicates
            FullFilePaths=pd.Series(FullFilePaths.unique())

            ffpPath=self.outpath() + '/FullFilePaths.csv'
            
            #this is a little hacky- reset descriptors and methods vars to reflect the decimation method

            #'unfreeze' parameters and descriptors. Couldn't hurt anything right? 

        
            self.parameters = dict(self.parameters)
            self.descriptors = dict(self.descriptors)

            self.parameters['methodID']=self.parameters['methodID2m']
            self.parameters['methodvers']=self.parameters['methodvers2m']
            self.descriptors['runtype']=self.descriptors['runtype2m']
            self.descriptors['language']=self.descriptors['language2m']

                
            if self.parameters['methodID2m'] == 'matlabdecimate':
                #if decimating, run decimate. Check will matter in cases where MATLAB supporting library is not installed.
                #note that this can be pretty slow if not on a VM! Might want to figure out another way to perform this
                #to speed up if running on slow latency.
                
                FullFilePaths.to_csv(ffpPath,index=False,header = None) #don't do gz since don't want to deal with it in MATLAB!

                #import code
                #code.interact(local=dict(globals(), **locals()))

                self.cmd_args=[PARAMSET_GLOBALS['SF_raw'],ffpPath,self.parameters['target_samp_rate']]
                #wrap it into run cmd later.. will need to change it so that matlab recieves args in order of paths, args, parameters 
                
                self.run_cmd()

                
                os.remove(ffpPath)
            
            elif self.parameters['methodID2m'] == 'matlabdecimate_flexchunk':
            
                #this method relies on formatFG parameters to define relative paths, for more flexible pathing in/out of cloud,
                #ephemeral compute, and also allows for flexible chunking to

                chunksize = self.parameters['chunksize']
                #inpath = PARAMSET_GLOBALS['SF_raw'] #gs:// or local
                #outpath = PARAMSET_GLOBALS['SF_foc'] #gs:// or local

               #inpath_gs = True if inpath[0:5] == "gs://" else False
                #outpath_gs = True if outpath[0:5] == "gs://" else False
                
                src_service  = self.parameters['src_service'] #gcp or local
                dest_service = self.parameters['dest_service'] #gcp or local

                #test for cloud storage and 
                chunksize_in = chunksize if inpath_gs else FG.shape[0] #.nrows just a guess, replace with correct attribute
                chunksize_out = chunksize if outpath_gs else FG.shape[0]
                
                local_temp = "./tmp"
                
                os.makedirs(local_temp,exist_ok = True)

                #prestage data according to chunksize.
                #could parallelize this once I get it working. 
                for i in range(math.ceil(FG.shape[0]/chunksize_in)):
                
                    files = FullFilePaths[(chunksize_in*i):(chunksize_in*(i+1))] #need to make sure that the format
                    #fg method can assume cloud paths, given a bucket!
                    
                    #load files - assume gsutil is installed and default credentials are in place (will be in GKE autopilot)
                    if inpath_gs:
                        local_files = [local_temp + os.path.basename(i) for i in files]
                        cloud_cp(files,local_files)
                    else:
                        local_files = files
                    
                    #make ds of local files and desired output paths, pass to matlab exe
                    dec_files = [local_temp + "decimated_" + os.path.basename(i) for i in files]
                    FullFilePaths.to_csv(pd.Series(local_files),pd.Series(dec_files),index=False,header = None)
                    
                    self.cmd_args=[ffpPath,self.parameters['target_samp_rate']]
                    
                    self.run_cmd()
                    
                    import code
                    code.interact(local=dict(globals(), **locals()))
                                    
                    #if the decimated files are going back to cloud, call this here
                    #if outpath_gs:
                          
                    os.remove(ffpPath)
                    
                    
                    
                
                #going to make a different decision here, and include both the full paths of the decimated and original file
                #in FG.
                 
                #TODO 
                #FG.decimated_path = #dec files path
                
            
        #do it this way, so that task will not 'complete' if decimation is on and doesn't work
       
        
        FG.to_csv(self.outfilegen(),index=False,compression='gzip')
            

class EditRAVENx(INSTINCT_userprocess):
    #pipeshape = OneUpstream
    #upstreamdef = ['GetViewFile']
    
    outfile = 'RAVENx.txt'
    userfile_append = "_edit"

    def get_info(self): #this injects some formatFG info into the manifest
        if 'parameters' in dir(self.ports[1]):
            return self.ports[1].parameters['file_groupID']
        else:
            return 'multiple FDID'

class ReviewRAVENx(INSTINCT_userprocess):
    #pipeshape = OneUpstream
    #upstreamdef = ['GetViewFile']
    
    outfile = 'RAVENx.txt'
    userfile_append = "_edit"

    def get_info(self): #this injects some formatFG info into the manifest
        if 'parameters' in dir(self.ports[1]):
            return self.ports[1].parameters['file_groupID']
        else:
            return 'multiple FDID'
        #return self.ports[1].parameters['file_groupID']

    def file_modify(self,file):
        

        rav_tab = pd.read_csv(self.outfilegen(),sep="\t")
        rav_tab["label"] = ""
        rav_tab["Comments"] = ""

        rav_tab.to_csv(self.outfilegen(),index=False,sep="\t")
        
        #load in the file, add on label and Comments column.


class GenBigSpec(INSTINCT_process):#

    pipeshape = OneUpstream
    upstreamdef = ["GetFG"]

    outfile = 'filepaths.csv' 

    def run(self):

        #import code
        #code.interact(local=locals())

        self.cmd_args=[self.ports[0].outpath(),self.outpath(),PARAMSET_GLOBALS['SF_foc'] + "/" + find_decimation_level(self,0),
                       self.ports[0].parameters['file_groupID'],self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()
        
class MakeModel_bins(INSTINCT_process):#

    #pipeshape = OneUpstream
    #upstreamdef = ["GetFG"]

    outfile = 'DETx.csv.gz' 

    def run(self):

        #import code
        #code.interact(local=locals())

        self.cmd_args=[self.ports[0].outpath(),self.outpath(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class TrainModel_dl(INSTINCT_process):

    pipeshape = ThreeUpstream_noCon

    upstreamdef = ["GetFG","GetSpec","GetLabels"]

    outfile = 'model.keras'

    def run(self):
        
        self.cmd_args=[self.ports[2].outfilegen(),self.ports[1].outpath(),self.ports[0].outfilegen(),self.outpath(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class DLmodel_Train(INSTINCT_process):

    #pipeshape = dl2_special
    pipeshape = FourUpstream_noCon
    
    upstreamdef = ["GetFG","GetSpec","GetLabels","GetSplits"]

    outfile = 'model.keras'

    def run(self):

        arg_tlist= sorted(self.arguments.items())
        arg_vals = [arg_tlist[x][1] for x in range(len(arg_tlist))]
        arg_vals_sort = [sorted(arg_vals[x])[n] if isinstance(arg_vals[x],list) else arg_vals[x] for x in range(len(arg_vals))]
        arg_string = os.environ["INS_ARG_SEP"].join(arg_vals_sort)

        self.cmd_args=[self.ports[3].outfilegen(),self.ports[2].outpath(),self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),"NULL","train",self.param_string2,arg_string]#,self.arguments['transfer_loc']

        self.run_cmd()

class DLmodel_Test(INSTINCT_process):

    #pipeshape = dl2_special
    pipeshape = FiveUpstream_noCon
    
    upstreamdef = ["GetFG","GetSpec","GetLabels","GetSplits","GetModel"]

    outfile = 'scores.csv.gz'

    def run(self):

        arg_tlist= sorted(self.arguments.items())
        arg_vals = [arg_tlist[x][1] for x in range(len(arg_tlist))]
        arg_vals_sort = [sorted(arg_vals[x])[n] if isinstance(arg_vals[x],list) else arg_vals[x] for x in range(len(arg_vals))]
        arg_string = os.environ["INS_ARG_SEP"].join(arg_vals_sort)

        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        self.cmd_args=[self.ports[4].outfilegen(),self.ports[3].outpath(),self.ports[2].outpath(),self.ports[1].outpath(),self.outpath(),self.ports[0].outfilegen(),"test",self.param_string2,arg_string]#,self.arguments['transfer_loc']

        self.run_cmd()

class DLmodel_Inf(INSTINCT_process):

    #pipeshape = dl2_special
    pipeshape = FourUpstream_noCon
    
    upstreamdef = ["GetFG","GetSpec","GetSplits","GetModel"]

    outfile = 'model.keras'

    def run(self):

        arg_tlist= sorted(self.arguments.items())
        arg_vals = [arg_tlist[x][1] for x in range(len(arg_tlist))]
        arg_vals_sort = [sorted(arg_vals[x])[n] if isinstance(arg_vals[x],list) else arg_vals[x] for x in range(len(arg_vals))]
        arg_string = os.environ["INS_ARG_SEP"].join(arg_vals_sort)
        
        self.cmd_args=[self.ports[3].outfilegen(),self.ports[2].outpath(),self.ports[1].outpath(),"NULL",self.outpath(),self.ports[0].outpath(),"inf",self.param_string2,arg_string]#,self.arguments['transfer_loc']

        self.run_cmd()
        
class ScoresToDETx(INSTINCT_process):

    pipeshape = FourUpstream_noCon
    upstreamdef = ["GetFG","GetScores",'GetImg','GetSplits']
    
    outfile = "DETx.csv.gz"
    
    def run(self):
        
        arg_tlist= sorted(self.arguments.items())
        arg_vals = [arg_tlist[x][1] for x in range(len(arg_tlist))]
        arg_vals_sort = [sorted(arg_vals[x])[n] if isinstance(arg_vals[x],list) else arg_vals[x] for x in range(len(arg_vals))]
        arg_string = os.environ["INS_ARG_SEP"].join(arg_vals_sort)
        
        self.cmd_args=[self.ports[3].outfilegen(),self.ports[2].outpath(),self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),self.param_string2,arg_string]
        
        self.run_cmd()

class ModelEval_NN(INSTINCT_process):

    #pipeshape = TwoUpstream_noCon
    #upstreamdef = ["GetModel",'GetStats']
    pipeshape = OneUpstream
    upstreamdef = ["GetModel"]
    
    outfile = 'summary.png'
    
    def run(self):

        self.cmd_args=[self.ports[0].outpath(),self.outfilegen(),self.param_string2]#,self.arguments['transfer_loc']

        self.run_cmd()

class LabelTensor(INSTINCT_process):

    pipeshape =ThreeUpstream_bothUpTo1
    upstreamdef = ["GetFG","GetImg","GetGT"]

    outfile = 'filepaths.csv'

    def run(self):
    
        #import code
        #code.interact(local=locals())
        
        self.cmd_args=[self.ports[2].outpath(),self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),self.ports[2].parameters['file_groupID'],self.param_string2]
        
        self.run_cmd()

class SplitTensor(INSTINCT_process):

    pipeshape = TwoUpstream
    upstreamdef = ["GetFG","GetImg"]

    outfile = 'filepaths.csv'
    
    def run(self):

        self.cmd_args=[self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),self.ports[1].parameters['file_groupID'],self.param_string2]

        self.run_cmd()

class SplitDeduce(INSTINCT_process):
    #this is probably depreciated...
    pipeshape = TwoUpstream
    upstreamdef = ["GetFG","GetDets"]

    outfile = 'FGsplits.csv'

    def run(self):

        self.cmd_args=[self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),self.param_string2]

        self.run_cmd()

class PerfEval1DL(INSTINCT_process):

    pipeshape = TwoUpstream
    upstreamdef = ["GetFG","GetModel_w_probs"]
    
    outfile = 'Stats.csv.gz'

    def run(self):
        
        self.cmd_args=[self.ports[1].outpath(),self.ports[0].outpath(),self.outpath(),self.param_string2]
        
        self.run_cmd()

class StatsTableCombine_DL(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["PE1All","PE1FG"]

    #this one's hardcoded, no elegant way I could find to extract the pipeline history in an elegant/readible way. 
    outfile = "Stats.csv.gz"

    #stagename = 'ED/AM'
        
    def run(self):

        #import code
        #code.interact(local=locals())
    
        #import datasets, tag with new ID column, rbind them. 
        Stats1 = pd.read_csv(self.ports[0].outfilegen(),compression='gzip') 
        Stats2 = pd.read_csv(self.ports[1].outfilegen(),compression='gzip') 

        #Stats1[self.stagename] = self.upstreamdef[1]
        #Stats2[self.stagename] = self.upstreamdef[0]

        StatsOut = pd.concat([Stats2,Stats1])

        StatsOut.to_csv(self.outfilegen(),index=False,compression='gzip')

class Combine_DETx(INSTINCT_process):

    pipeshape = TwoUpstream_noCon
    upstreamdef = ["DETx1","DETx2"]

    #this one's hardcoded, no elegant way I could find to extract the pipeline history in an elegant/readible way. 
    outfile = "DETx.csv.gz"
        
    def run(self):

        #import code
        #code.interact(local=locals())
    
        #import datasets, tag with new ID column, rbind them. 
        Tab1 = pd.read_csv(self.ports[0].outfilegen(),compression='gzip') 
        Tab2 = pd.read_csv(self.ports[1].outfilegen(),compression='gzip') 

        #Stats1[self.stagename] = self.upstreamdef[1]
        #Stats2[self.stagename] = self.upstreamdef[0]

        #import code
        #code.interact(local=locals())

        #keep all unique columns
        TabOut = pd.concat([Tab1,Tab2],join='outer', ignore_index=True)

        TabOut.to_csv(self.outfilegen(),index=False,compression='gzip')
