import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import *

def reading_eta():
    for bits in [8,12,16,24]:
        with open('./pickles/'+str(bits)+'_loss_nus_A.pkl', 'rb') as f:
            All = pickle.load(f)


        print (All)

        Acc=All[1:]
        AccBase = [All[0]]

        t = np.arange(0, 1.05, .05)

        plt.plot(t, Acc)

        plt.xlabel(r"$\eta$", fontsize=20)
        plt.ylabel('MAP', fontsize=20)
        plt.title('Bits='+str(bits),fontsize=22)
        plt.grid(True)
        plt.plot(t, len(t) * AccBase,'r', label="Baseline")

        plt.legend(loc=4, fontsize=15)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # axis([0, 1,.771,.7750],fontsize=20)

        xlim(xmax=1) # adjust the max leaving min unchanged
        xlim(xmin=0)
        plt.savefig('./fig/'+str(bits)+'_loss_nus_A.eps',bbox_inches="tight")
        plt.savefig('./fig/'+str(bits)+'_loss_nus_A.pdf',bbox_inches="tight")
        # plt.show()

        plt.clf()
        plt.cla()
        plt.close()


def MAP_bits():

    aa=28
    aa2=18
    aa3=12
    aal=15

    t  = [0,1,2,3,4,5]
    t1 = [0,2,3,4,5]
    y1 = [0.641, 0.710, 0.723, 0.750, 0.765, 0.774]
    y2 = [0.715, 0.727, 0.730, 0.733, 0.764]
    y3 = [0.671, 0.742, 0.768, 0.786, 0.801, 0.807]
    y5 = [0.723, 0.752, 0.765, 0.780, 0.788, 0.795]




    plot(t, y1, 'r', t1, y2, 'b',t,y3, 'm', t, y5, 'c',t, y1, 'rs',t1, y2, 'bs',t,y3, 'ms', t, y5, 'cs',linewidth=4)
    xlabel('Number of Bits',fontsize=24)
    ylabel('MAP',fontsize=24)
    axis([0, 5, 0.6, .85],fontsize=20)
    legend(['DTSH','DVSQ', 'BDSH-LL','BDSH-S'],loc=4,fontsize=18)
    # legend([r"$Fine-tuning$",r"$CS$", r"$CSA$",r"$CCSA$"],loc=4,fontsize=18)

    labels = [8, 12, 16, 24,32,48]
    x=range(6)
    plt.xticks(x,labels,fontsize=aa2)
    plt.yticks(fontsize=aa2)
    grid(True)
    # tight_layout()
    title('CIFAR-10 Reduced setting:',fontsize=aa)
    savefig('./fig/CIFARreduced.pdf')
    savefig('./fig/CIFARreduced.eps')
    show()
    plt.clf()
    plt.cla()
    plt.close()


    t  = [0,1,2,3,4,5]
    t1 = [0,2,3,4,5]
    y1 = [0.814, 0.859, 0.915, 0.923, 0.925, 0.926]
    y2 = [0.839, 0.839, 0.843, 0.840, 0.842]
    y3 = [0.766, 0.854, 0.940, 0.941, 0.940, 0.941]
    y5 = [0.879, 0.933, 0.939, 0.939, 0.937, 0.933]




    plot(t, y1, 'r', t1, y2, 'b',t,y3, 'm', t, y5, 'c',t, y1, 'rs',t1, y2, 'bs',t,y3, 'ms', t, y5, 'cs',linewidth=4)
    xlabel('Number of Bits',fontsize=24)
    ylabel('MAP',fontsize=24)
    axis([0, 5, 0.7, 1],fontsize=20)
    legend(['DTSH','DVSQ', 'BDSH-LL','BDSH-S'],loc=4,fontsize=18)
    # legend([r"$Fine-tuning$",r"$CS$", r"$CSA$",r"$CCSA$"],loc=4,fontsize=18)

    labels = [8, 12, 16, 24,32,48]
    x=range(6)
    plt.xticks(x,labels,fontsize=aa2)
    plt.yticks(fontsize=aa2)
    grid(True)
    # tight_layout()
    title('CIFAR-10 Full setting',fontsize=aa)
    savefig('./fig/CIFARfull.pdf')
    savefig('./fig/CIFARfull.eps')
    show()


    plt.clf()
    plt.cla()
    plt.close()
