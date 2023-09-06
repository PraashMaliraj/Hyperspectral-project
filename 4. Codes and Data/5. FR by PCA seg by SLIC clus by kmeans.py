
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
from skimage.segmentation import slic
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
#from skimage import io


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

# Cluster Purity function
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# Reshape image
w, h, d = original_shape = tuple(mat.shape)
mat_01 = np.reshape(mat, (w * h, d))

# convert data into 0-1 range
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#mat_01 = min_max_scaler.fit_transform(image_array)

      # Apply PCA
pca = PCA(n_components = 0.99)
mat_02 = pca.fit_transform(mat_01)
      
      
a, d = original_shape = tuple(mat_02.shape)
      
      
      # # reshape whole image for display purpose
      # Fr = kmeans.cluster_centers_
      # mat_03 = np.transpose(Fr)
      
      
      # Segmentation
      
      #reshaping mat_03
      
mat_03 = np.reshape(mat_02, (w,h, d))


      
seg = slic(mat_03, n_segments=50, compactness= 10)
# Cluster Map to Segmentation Map
labeled, numRegns = label(seg,connectivity=1,return_num=True)
        
#reshaping mat_03
w, h, d = original_shape = tuple(mat_03.shape)


#Arrange segments in nxk form
#For each segment, mean of all the pixels in each band, is the new pixel vector
regions = regionprops(labeled)
seg_nxk = np.zeros((numRegns,d));
i = 0
for prop in regions:
    pxIdLst = prop.coords
    for j in range(d):
        band = mat[pxIdLst[:,0],pxIdLst[:,1],j];
        seg_nxk[i,j] = np.mean(band)
    i = i + 1        

   
# Perform Clustering
k = 16
kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100).fit(seg_nxk)
lbl = kmeans.labels_

#Rearrange to form cluster map
LbldImg = labeled;
i = 0
for prop in regions:
    pxIdLst = prop.coords
    LbldImg[pxIdLst[:,0],pxIdLst[:,1]] = lbl[i]
    i = i + 1

# Display
plt.imshow(label2rgb(LbldImg))
plt.show()

#io.imshow(label2rgb(LbldImg))
#plt.show()

# write to text file
filename = 'PU01_k_'+ str(k)+ '_run_' + str(1)+ '.txt'
np.savetxt(filename, LbldImg, delimiter=',',fmt='%d')


# Validation
Lbl = np.reshape(LbldImg, (w * h))
lbl_pre = Lbl[idx]
from sklearn.metrics.cluster import normalized_mutual_info_score
        
Lbl = np.reshape(LbldImg, (w * h))
lbl_pred = Lbl[idx]
NMI = normalized_mutual_info_score(lbl_true, lbl_pred)
        
print("SEG = 50", " run1 ",)
print('NMI = ',NMI)
purity = purity_score(lbl_true, lbl_pred)
print('Purity =', purity)