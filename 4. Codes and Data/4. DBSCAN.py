# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 23:28:49 2022

@author: rajpr
"""

# Importing the Libraries
import scipy.io
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from skimage.color import label2rgb
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Cluster Purity function
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)



# Read HSI
mat = scipy.io.loadmat('Salinas_corrected.mat');
mat = mat['salinas_corrected']
# convert data into float
mat = np.asfarray(mat, dtype='float')

# Read Ground truth
gt_mat = scipy.io.loadmat('Salinas_gt.mat');
gt_mat = gt_mat['salinas_gt']
w, h = (gt_mat.shape)
gt_array = np.reshape(gt_mat, (w * h),)
idx = gt_array!=0
lbl_true = gt_array[idx]


# Plot image
from matplotlib import pyplot as plt
plt.imshow(mat[:,:,15])
#plt.show()

# Reshape image
w, h, d = original_shape = tuple(mat.shape)
image_array = np.reshape(mat, (w * h, d))
mat_01  = image_array
for m in range(500,600,100):
    for n in range(40,50,10):
        db = DBSCAN(eps=m, min_samples=n).fit(mat_01)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        # reshape whole image for display purpose
        lbl = labels
        lbl_img = np.reshape(lbl, (w, h))
        
        # display image
        plt.imshow(label2rgb(lbl_img))
        plt.show()
        
        # Calculate the Overall accurracy
        from sklearn.metrics.cluster import normalized_mutual_info_score
        
        lbl_pred = lbl[idx]
        NMI = normalized_mutual_info_score(lbl_true, lbl_pred)
        print("Eps = ", m)
        print("min sample = ", n)
        print("cluster =", n_clusters_)
        print('OA = ',NMI)
        
        purity = purity_score(lbl_true, lbl_pred)
        
        # write to text file
        filename = 'PU01_kmeans_16_'+ '.txt'
        np.savetxt(filename, lbl_img, delimiter=',',fmt='%d')
        
        # Save thematic Map \ cluster map - Matlab format
        mdic = {"lbl_img": lbl_img, "label": "thematic map"}
        scipy.io.savemat("SAl_KM_16_PCA_run" +'.mat', mdic)