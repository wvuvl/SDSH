import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('outFile',type=str)
parser.add_argument('infiles',nargs='+',type=str)

args = parser.parse_args()

##Ours

names_list = []
for ind in range(0,len(args.infiles),2):
    our_data = np.array(pickle.load(open(args.infiles[ind],'rb')))
    our_r = our_data[0:6000,1]
    our_p = our_data[0:6000,0]
    names_list.append(args.infiles[ind+1])
    plt.plot(our_r,our_p)

##create_plot


plt.legend(names_list)
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.grid()

plt.savefig(args.outFile,format='eps',dpi=1200)

