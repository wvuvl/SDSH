import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import *
from math import log, exp, sqrt

aa=28
aa2=14
aa3=12    
aal=15    
    
d = [ 4.0 * x / 100.0 - 2.0 for x in range(100)]

alpha = 1.

y1 = [ max(0.0, (4.0 * x / 100.0 - 2.0) + alpha) for x in range(100)]

def X(x):
	return ((4.0 * x / 100.0 - 2.0)  + alpha)

y2 = [ X(x) + log(1.0 + exp(-X(x * 6.0) / 6.0 ) ) for x in range(100)]

def X2(x):
	return ((4.0 * x / 100.0 - 2.0))

y3 = [ (2.0 - sqrt(2.0 - X2(x))) * (2.0 - sqrt(2.0 - X2(x))) for x in range(100)]

plot(d, y1, 'r', d, y2, 'b', d, y3, 'm', d, linewidth=4)    

xlabel('d',fontsize=24)    
ylabel('loss',fontsize=24)    
axis([-2.0, 2.0, -0.5, 3.0],fontsize=20)    
legend(['Margin Loss','Label Likelihood Loss', 'Spring Loss'],loc=4,fontsize=18)    
# legend([r"$Fine-tuning$",r"$CS$", r"$CSA$",r"$CCSA$"],loc=4,fontsize=18)    

labels = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
x=labels
  
plt.xticks(x,labels,fontsize=aa2)    
plt.yticks(fontsize=aa2)    
grid(True)
# tight_layout()
title('Loss functions comparison',fontsize=aa)
savefig('./fig/losses.pdf')
savefig('./fig/losses.eps')
show()


plt.clf()
plt.cla()
plt.close()