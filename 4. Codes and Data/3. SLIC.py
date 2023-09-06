# Importing the Libraries
import scipy.io
import numpy as np
from sklearn import metrics
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from sklearn.metrics import pairwise_distances
from skimage.segmentation import mark_boundaries

# Cluster Purity function
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

#define function to calculate Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))

for ab in range(300,4000,50):
    for cd in range(1,2):
        
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
        
       

        
        
        # # reshape whole image for display purpose
        # Fr = kmeans.cluster_centers_
        # mat_03 = np.transpose(Fr)
        
        # Reshape image
        w, h, d = original_shape = tuple(mat.shape)
        image_array = np.reshape(mat, (w * h, d))
        mat_01  = image_array
        
        
        # Segmentation
        
        
        seg = slic(mat, n_segments=100, compactness=10)
        
        
        # # Obtain Cluster Map
        # lbl_img = np.reshape(seg, (w*h))
        
        # Cluster Map to Segmentation Map
        labeled, numRegns = label(seg,connectivity=1,return_num=True)
        
        
        w, h, d = original_shape = tuple(mat.shape)
       
 
    
        
        #Arrange segments in nxk form
        #For each segment, mean of all the pixels in each band, is the new pixel vector
        regions = regionprops(labeled)
        
        seg_nxk_0 = np.zeros((numRegns,d+2));
        
        #Matrix of segment
        
        seg_nxk = np.zeros((numRegns,d));
        i = 0
        for prop in regions:
            pxIdLst = prop.coords
            for j in range(d):
                band = mat[pxIdLst[:,0],pxIdLst[:,1],j];
                seg_nxk[i,j] = np.mean(band)
            i = i + 1        
        
        #Assigning this value to previous matrix
        for m in range(numRegns):
            for n in range(d):
                seg_nxk_0[m,n]=seg_nxk[m,n]
        
        for u in range(numRegns):
            seg_nxk_0[u,d]= u
        
        #Calculating ginni values
        i = 0
        for prop in regions:
            pxIdLst = prop.coords
            for j in range(d):
                band = mat[pxIdLst[:,0],pxIdLst[:,1],j];
                seg_nxk_0[i,d+1] = gini(band)
            i = i + 1      



        #Sorting array in ascending order
        for x in range(numRegns):
            for y in range(x + 1, numRegns):
                if (seg_nxk_0[x,d+1] > seg_nxk_0[y,d+1]):
                    for c in range(d+2):
                        temp = seg_nxk_0[x,c]
                        seg_nxk_0[x,c] = seg_nxk_0[y,c]
                        seg_nxk_0[y,c] = temp
         
            
        k_cl = 16
        #Dividing matrices
        split_1 = np.zeros((k_cl,d+1))
        split_2 = np.zeros((numRegns,d+1))
        
        for x in range(numRegns):
            for y in range(d+1):
                if(x<k_cl):
                    split_1[x,y]=seg_nxk_0[x,y]
                else:
                    split_2[x,y]=seg_nxk_0[x,y]
               
                
        new_mat = pairwise_distances(split_1,split_2)
                
        #--------------Clustering_----------------
        
        #Extracting best k classes
        
        
        new_seg=np.zeros((k_cl,2))
        for s in range(k_cl):
            new_seg[s,0]=seg_nxk_0[s,0]
            new_seg[s,1]=seg_nxk_0[s,1]
        
        #Calculating distance between centroid of 16 segments and other segments
        
    
        lbl= np.zeros((numRegns,1))
        for i in range(k_cl):
            lbl[i]=i
            
        
        for z in range(k_cl,numRegns):
            min=new_mat[0,z]
            for x in range(k_cl):
                if(new_mat[x,z]<min):
                    min=new_mat[x,z]
                    lbl[z]=x
        
                          
        
        #Rearrange to form cluster map
        LbldImg = labeled;
        i = 0
        for prop in regions:
            pxIdLst = prop.coords
            LbldImg[pxIdLst[:,0],pxIdLst[:,1]] = lbl[i]
            i = i + 1
            
        plt.imshow(mark_boundaries(labeled, seg))
        plt.show()
        
        # display image
        plt.imshow(label2rgb(LbldImg))
        plt.show()
        
        # write to text file
        #filename = "SA_Pre_FR_post_k_seg_"+ str(ab) + 'run_' + str(cd), '.txt'
        #np.savetxt(filename, LbldImg, delimiter=',',fmt='%d')
        
        
        
        
        # Calculate the Overall accurracy
        from sklearn.metrics.cluster import normalized_mutual_info_score
        
        Lbl = np.reshape(LbldImg, (w * h))
        lbl_pred = Lbl[idx]
        NMI = normalized_mutual_info_score(lbl_true, lbl_pred)
        
        print("SEG = ", ab, " run ",cd)
        print('NMI = ',NMI)
        purity = purity_score(lbl_true, lbl_pred)
        print('Purity =', purity)
        
        # Save thematic Map \ cluster map - Matlab format
        #mdic = {"lbl_img": LbldImg, "label": "thematic map"}
        #scipy.io.savemat("SAl_KM_16_Pre_FR_8_Post_Seg_"+ str(ab) +"run"+ str(cd) + '.mat', mdic)