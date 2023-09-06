# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:01:46 2023

@author: rajpr
"""

# Importing the Libraries
import scipy.io
import numpy as np
from sklearn import metrics
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from matplotlib import pyplot as plt

# Cluster Purity function
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# Read Ground truth
gt_mat = scipy.io.loadmat('Salinas_gt.mat');
gt_mat = gt_mat['salinas_gt']
w, h = (gt_mat.shape)
gt_array = np.reshape(gt_mat, (w * h),)
idx = gt_array!=0
navaibleidx = gt_array ==0
lbl_true = gt_array[idx]

# Read HSI
mat = scipy.io.loadmat('SAl_KM_30  k Seg_ 6 L 13 run 1.mat');
mat = mat['lbl_img']
# np.loadtxt('TM_Seg_CFSFDP_Sal_k24_run9.txt', dtype="str", delimiter=",")
# mat = mat['lbl_img']
# convert data into float
mat = np.asfarray(mat, dtype='float')



Lbl = np.reshape(mat, (w * h))
lbl_pred = Lbl[idx]
ss = np.unique(lbl_pred)
nmi=normalized_mutual_info_score(lbl_true, lbl_pred)
print("Proposed entropy")
print('NMI = ', nmi)
purity = purity_score(lbl_true, lbl_pred)

print('Purity =', purity)
cm = metrics.cluster.contingency_matrix(lbl_true, lbl_pred)
# obtain cotingency matrix
#Solve the linear sum assignment problem. Select optimal clusters equal to 
# number of classes (ground reference labels).
row_ind, col_ind = linear_sum_assignment(-cm)
# Extract all those columns which result in optimum selection.
cm_2 = cm[:, col_ind]
# Calculate Overall Accuracy
OA = (cm_2.trace()/np.sum(cm_2))*100
print('OA = ', OA)

newTM = np.zeros((w*h,1))
uI = np.unique(lbl_true) 

#Label Cluster map based on the GT map
#a cluster is assigned to that physical class for which it is having 
#the maximum physical class labels


for i in range(len(uI)):
    idx2 = Lbl == col_ind[i]
    newTM[idx2] = i 

    
newTM = newTM + 1 # Removing 0 index. Zero is considered as background.
    
# Whereever there is no GT info availiable put 0.
newTM[navaibleidx] = 0 
# Obtain TMap in image dimensions.  
newTM = np.reshape(newTM, (w, h)) 
# Display result | clustered image | Thematic map 
clrs = [(1, 0, 0.5), (1, 0.5, 0), (0.5, 0, 1), (1, 0.6, 0.4), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0.7, 0.2), (0.7, 0, 1), (1, 0.5, 0.5), (0.6, 0.5, 0.9), (0.75, 0.6, 0.5), (0.4, 0.4, 0.4) ]
# Select color Map

#print('cmap', cmap.colors)
plt.imshow(label2rgb(newTM, colors=clrs, bg_label=0))
plt.axis('off')
plt.savefig('TM_Proposed gini.png', dpi=600,  bbox_inches='tight')


plt.show()
