# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:09:09 2018

@author: Anand
"""


# Importing the Libraries
import scipy.io
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from skimage.color import label2rgb
from skimage.measure import label, regionprops

# Cluster Purity function
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


for n in range(1,2):
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
    
    #feature Reduction
    mat_02 = np.transpose(mat_01)
    
    #apply KMeans
    from sklearn.cluster import KMeans
    k_fr = 8
    kmeans = KMeans(n_clusters = k_fr)
    
    # predict value for whole image
    kmeans_labels = kmeans.fit_predict(mat_02)
    
    
    # reshape whole image for display purpose
    Fr = kmeans.cluster_centers_
    mat_03 = np.transpose(Fr)
    
    
    #--------------Clustering_----------------
    #apply KMeans
    
    k = 16
    kmeans = KMeans(n_clusters = k)
    
    # predict value for whole image
    kmeans_labels = kmeans.fit_predict(mat_03)
    
    # reshape whole image for display purpose
    lbl = kmeans_labels
    lbl_img = np.reshape(lbl, (w, h))
    
    # display image
    plt.imshow(label2rgb(lbl_img))
    plt.show()
    
    
    # Calculate the Overall accurracy
    from sklearn.metrics.cluster import normalized_mutual_info_score
    
    lbl_pred = lbl[idx]
    NMI = normalized_mutual_info_score(lbl_true, lbl_pred)
    print('NMI = ',NMI)
    purity = purity_score(lbl_true, lbl_pred)
    print('Purity =', purity)
    
    # write to text file
    #filename = 'PU01_km_16_fr_8_run_' + str(n)+ '.txt'
    #np.savetxt(filename, lbl_img, delimiter=',',fmt='%d')
    
    # Save thematic Map \ cluster map - Matlab format
   # mdic = {"lbl_img": lbl_img, "label": "thematic map"}
   # scipy.io.savemat("SAl_KM_16_FR_8_run" + str(n) + '.mat', mdic)