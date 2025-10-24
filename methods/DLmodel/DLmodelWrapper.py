import subprocess
import sys
import os
import pandas as pd

sys.path.append(os.getcwd())
from user.misc import arg_loader

args=arg_loader()



#figure out how to get this to communicate with code inside subprocess
#
#while continue_loop=

#test to make sure it works!


result_path = args[6] #assume this stays constant for this process
            
logpath = result_path + "/model_history_log.csv"

stoppath = result_path + "/stop.txt"
if os.path.isfile(stoppath):
    os.remove(stoppath)

continue_loop = True

while continue_loop:
    arg_in= args[2:-4]
    subprocess.call('python ' + args[-1] + os.environ["INS_ARG_SEP"] + os.environ["INS_ARG_SEP"].join(arg_in))

    #if the 'stop' file exists, no longer continue loop:

    if os.path.isfile(stoppath):
        os.remove(stoppath)
        continue_loop=False

    #import code
    #code.interact(local=locals())

    #if len(log. index) >= 
    #
#import arguments: 
