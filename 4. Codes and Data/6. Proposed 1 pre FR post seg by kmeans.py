# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:09:09 2018

@author: Anand
"""


# Importing the Libraries
import scipy.io
import numpy as np
from sklearn import metrics
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# Cluster Purity function
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

for m in range(10,11):
    for n in range(1, 2):
        
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
        
        k_fr = 8
        kmeans = KMeans(n_clusters = k_fr)
        
        # predict value for whole image
        kmeans_labels = kmeans.fit_predict(mat_02)
        
        
        # reshape whole image for display purpose
        Fr = kmeans.cluster_centers_
        mat_03 = np.transpose(Fr)
        
        # Segmentation
        
        # convert data into 0-1 range
        #from sklearn import preprocessing
        #min_max_scaler = preprocessing.MinMaxScaler()
        #mat_01 = min_max_scaler.fit_transform(image_array)
        
        
        k_seg = 14
        # Segmentation using KMeans
        kmeans = KMeans(n_clusters=k_seg).fit(mat_03)
        lbl = kmeans.labels_
        
        #Taking care of 0 labels
        lbl = lbl + 1
        
        # Obtain Cluster Map
        lbl_img = np.reshape(lbl, (w, h))
        
        # Cluster Map to Segmentation Map
        labeled, numRegns = label(lbl_img,connectivity=1,return_num=True)
        
        #reshaping mat_03
        w, h, d = original_shape = tuple(mat.shape)
        mat_04 = np.reshape(mat_03, (w,h, k_fr))
        
        
        
        #Arrange segments in nxk form
        #For each segment, mean of all the pixels in each band, is the new pixel vector
        regions = regionprops(labeled)
        seg_nxk = np.zeros((numRegns,k_fr));
        i = 0
        for prop in regions:
            pxIdLst = prop.coords
            for j in range(k_fr):
                band = mat_04[pxIdLst[:,0],pxIdLst[:,1],j];
                seg_nxk[i,j] = np.mean(band)
            i = i + 1        
        
        
        
        #--------------Clustering_----------------
        #apply KMeans
        
        k_cl = 16
        kmeans = KMeans(n_clusters = k_cl)
        
        # predict value for whole image
        kmeans_labels = kmeans.fit_predict(seg_nxk)
        lbl = kmeans.labels_
        
        #Rearrange to form cluster map
        LbldImg = labeled;
        i = 0
        for prop in regions:
            pxIdLst = prop.coords
            LbldImg[pxIdLst[:,0],pxIdLst[:,1]] = lbl[i]
            i = i + 1
            
            
        # display image
        plt.imshow(label2rgb(LbldImg))
        plt.show()
        
        # write to text file
        #filename = "SA_Pre_FR_post_k_seg_"+ str(m) + 'run_' + str(n), '.txt'
        #np.savetxt(filename, LbldImg, delimiter=',',fmt='%d')
        
        
        
        
        # Calculate the Overall accurracy
        from sklearn.metrics.cluster import normalized_mutual_info_score
        
        Lbl = np.reshape(LbldImg, (w * h))
        lbl_pred = Lbl[idx]
        NMI = normalized_mutual_info_score(lbl_true, lbl_pred)
        
        print("K SEG = ", m, " run ",n)
        print('NMI = ',NMI)
        purity = purity_score(lbl_true, lbl_pred)
        print('Purity =', purity)
        
        # Save thematic Map \ cluster map - Matlab format
        #mdic = {"lbl_img": LbldImg, "label": "thematic map"}
        #scipy.io.savemat("SAl_KM_16_Pre_FR_8_Post_Seg_"+ str(m) +"_run_"+ str(n) + '.mat', mdic)