import sys
import os

def arg_loader():
    
    args_=sys.argv
    
    sep = os.environ["INS_ARG_SEP"]
    
    if sep != " ":
        args_ = args_.split(sep)
    
    return args_
