#miscellaneous functions
import pandas as pd
import numpy


def get_difftime(data,cap_consectutive=True):

    data['TrueStart'] = data['StartTime']+pd.to_timedelta(data['SegStart'], unit='s')
    data['TrueEnd'] = data['TrueStart']+pd.to_timedelta(data['SegDur'], unit='s')
    data['DiffTime']=pd.to_timedelta(0)
    data['DiffTime'][0:(len(data)-1)] = pd.to_timedelta(abs(data['TrueEnd'][0:(len(data['TrueEnd'])-1)] - data['TrueStart'][1:len(data['TrueStart'])].values)) #changes 7/12/21, fix bug where difftime was assigned improperly
    data['DiffTime'] = (data['DiffTime']>pd.to_timedelta(2,unit='s'))==False #makes the assumption that if sound files are 1 second apart they are actually consecutive (deals with rounding differences)
    consecutive = numpy.empty(len(data['DiffTime']), dtype=int)
    consecutive[0] = 1
    iterator = 1

    #import code
    #code.interact(local=dict(globals(), **locals()))

    if cap_consectutive == True:
        #hardcoded as 40 minutes
        _cumsum = data["Duration"].cumsum()//(40*60) #hardcode this to be 40 minutes- reason is for backwards compatible. 
        _cumsum = pd.Series.tolist(_cumsum)
        indexes = [_cumsum.index(x) for x in set(_cumsum)]
        data.loc[indexes,"DiffTime"] = False #set first value to false. 
        data.loc[0,"DiffTime"] = True #except for the first one

    for n in range(0,(len(data['DiffTime'])-1)):
        if data['DiffTime'].values[n] != True:
            iterator = iterator+1
            consecutive[n+1] = iterator
        else:
            consecutive[n+1] = iterator

    data['DiffTime'] = consecutive
    data = data.drop(columns='TrueStart')
    data = data.drop(columns='TrueEnd')
    return(data)

def file_peek(file,fn_type,fp_type,st_type,dur_type,comp_type=0):#this is out of date- don't think I need to have fxn variables for how I load in the standard metadata.
        if comp_type != 0:
            heads = pd.read_csv(file, nrows=1,compression=comp_type)
        else:
            heads = pd.read_csv(file, nrows=1)
        heads = heads.columns.tolist()
        heads.remove('FileName')
        heads.remove('StartTime')
        heads.remove('Duration')
        heads.remove('SegStart')
        heads.remove('SegDur')
        hdict = {'FileName': fn_type, 'FullPath': fp_type, 'StartTime': st_type, 'Duration': dur_type, 'SegStart': 'float64', 'SegDur': 'float64'}
        if len(heads) != 0:
            metadict = dict.fromkeys(heads , 'category')
            hdict.update(metadict)
        return hdict


        
