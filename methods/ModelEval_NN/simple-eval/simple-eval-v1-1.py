#MethodID="train-simple-dl-v1-1"

#still experimental- this vers will attempt to save the
#file name and associated array for test/train and label
#as a dict, and reference that during mapping function.
#
#try to avoid tfrecord


import pandas as pd
from matplotlib import pyplot as plt
import sys

#import sklearn

#######import params###########

#args="first C:/Apps/INSTINCT/Cache/394448/527039 C:/Apps/INSTINCT/Cache/394448/628717/947435/DETx.csv.gz C:/Apps/INSTINCT/Cache/394448/628717/947435/909093/summary.png 0.80 train-simple-dl-v1-0"
#args=args.split()

args=sys.argv

model_path = args[1]
result_path = args[2]
gt_depth = args[3].count(",")+1
#small_window= args[4] #I might not even need to pass this one- I can get the crop width based on the height of the spectrogram images

model = pd.read_csv(model_path + "/model_history_log.csv")



fig, axs = plt.subplots(4)
fig.suptitle('Model evalution metrics')
#fig.xlabel("Epoch")
axs[0].plot(model.epoch, model.loss, marker="x")
axs[0].plot(model.epoch, model.val_loss, marker="x")
axs[0].set(ylabel='Loss')

if gt_depth== 1:

    axs[1].plot(model.epoch, model.binary_accuracy, marker="x")
    axs[1].plot(model.epoch, model.val_binary_accuracy, marker="x")
    axs[1].set(ylabel="Accuracy")

else:

    axs[1].plot(model.epoch, model.categorical_accuracy, marker="x")
    axs[1].plot(model.epoch, model.val_categorical_accuracy, marker="x")
    axs[1].set(ylabel="Accuracy")

axs[2].plot(model.epoch, model.rocauc, marker="x")
axs[2].plot(model.epoch, model.val_rocauc, marker="x")
axs[2].set(ylabel="ROCAUC")

axs[3].plot(model.epoch, model.ap, marker="x")
axs[3].plot(model.epoch, model.val_ap, marker="x")
axs[3].set(ylabel="prAUC")

for ax in axs.flat:
    ax.set(xlabel='Epoch')
    
for ax in axs.flat:
    ax.label_outer()

fig.savefig(result_path)

