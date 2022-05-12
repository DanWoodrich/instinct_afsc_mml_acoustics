from classes import INSTINCT_pipeline
from pipe_shapes import *


#have these in here to test idk
 
#class TrainModel_base(DoubleLoopAndRun):
#    comp0def = 'GetFG'
#    comp1def = 'GetDETx'
#    pass

class ViewFileToDETx(TwoUpstream):
    compdef = ['GetFG','GetViewFile']
    pass

class EditViewFile(TwoUpstream):
    compdef = ['GetFG','GetViewFile']
    pass

class ViewDETx(TwoUpstream):
    compdef = ['GetFG','GetDETx']
    pass

class dl2_special(INSTINCT_pipeline):

    def run(self):
        
        comp0 = self.run_component(self.compdef[0]) #formatFG
        comp1 = self.run_component(self.compdef[1],upstream = [comp0]) #genbigspec
        comp2 = self.run_component(self.compdef[2],upstream = [comp0]) #formatGT
        comp3 = self.run_component(self.compdef[3],upstream = [comp2,comp1,comp0]) #getlabels
        comp4 = self.run_component(self.compdef[4],upstream = [comp1,comp0]) #getsplits
        comp5 = self.run_component(final_process=True,upstream = [comp4,comp3,comp1,comp0]) 

        return comp5


###################################



##############




        
        
    
