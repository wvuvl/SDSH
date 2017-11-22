import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('outFile',type=str)
parser.add_argument('--ourFile',type=str,default = None)
parser.add_argument('--matFile',type=str,default = None)
parser.add_argument('--dvsqFile',type=str,default = None)

args = parser.parse_args()

##DTSH
if args.matFile is not None:
    DTSH_data = loadmat(args.matFile)
    DTSHP = DTSH_data['p']
    DTSHR = DTSH_data['r']

    DTSHP = np.squeeze(DTSHP)
    DTSHR = np.squeeze(DTSHR)


##DVSQ

if args.dvsqFile is not None:
    DVSQ_data = np.load(args.dvsqFile)
    dvsq_r = DVSQ_data['r']
    dvsq_p = DVSQ_data['p']

##Ours

if args.ourFile is not None:
    our_data = np.array(pickle.load(open(args.ourFile,'rb')))
    our_r = our_data[:,1]
    our_p = our_data[:,0]

##create_plot

names_list = []

if args.ourFile is not None:
    plt.plot(our_r,our_p)
    names_list.append('BDSH')

if args.dvsqFile is not None:
    plt.plot(dvsq_r,dvsq_p)
    names_list.append('DVSQ')

if args.matFile is not None:
    plt.plot(DTSHR,DTSHP)
    names_list.append('DTSH')

plt.legend(names_list)
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.grid()

plt.savefig(args.outFile,format='eps',dpi=1200)

