# -*- coding: utf-8 -*-

# Importing the Libraries
import scipy.io
import numpy as np
from sklearn import metrics
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skimage.measure import shannon_entropy

for ab in range(19,20):
    for cd in range(1,5):
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
        
        # convert data into float
        mat_01 = np.asfarray(image_array, dtype='float')
        
        # Segmentation
        k_seg = 7
        # Segmentation using KMeans
        kmeans = KMeans(n_clusters=k_seg).fit(mat_01)
        lbl = kmeans.labels_
        
        #Taking care of 0 labels
        lbl = lbl + 1
        
        # Obtain Cluster Map
        lbl_img = np.reshape(lbl, (w, h))
        
        # Cluster Map to Segmentation Map
        labeled, numRegns = label(lbl_img,connectivity=2,return_num=True)
        
        
        regions = regionprops(labeled)
        seg_nxk = np.zeros((numRegns,d));
        i = 0
        for prop in regions:
            pxIdLst = prop.coords
            for j in range(d):
                band = mat[pxIdLst[:,0],pxIdLst[:,1],j];
                seg_nxk[i,j] = np.mean(band)
            i = i + 1     
            
        #Arrange segments in nxk form
        #For each segment, mean of all the pixels in each band, is the new pixel vector
        regions = regionprops(labeled)
        
        #Matrix of segment
        entropy_band = np.zeros((numRegns,d))
        
        entropy_mat = np.zeros((numRegns,1)) #Shannon Entropy for all segment
        
        n_pix = np.zeros((numRegns,1))
        k_cl = 24
        i = 0
        L = 13 # Top bands
        L_entropy_band = np.zeros((k_cl,L))
        
        
        temp_array = np.zeros((k_cl,d)) #temp matrix to store band number  
        #for significant segments
        for prop in regions:
            pxIdLst = prop.coords
            matrix = np.zeros((len(pxIdLst),d))
            for j in range(d):
                band = mat[pxIdLst[:,0],pxIdLst[:,1],j];
                for aa in range(len(band)):
                    matrix[aa,j] = band[aa] #matrix for each segment
            entropy_mat[i] = shannon_entropy(matrix) #Shannon entropy for each segment                         
            n_pix[i] = len(pxIdLst)
                       
            i = i + 1 
        
        sig_entropy = np.zeros(numRegns)
        for i in range(numRegns):
            sig_entropy[i] = 50
            if(n_pix[i]> 5):
                sig_entropy[i] = entropy_mat[i]
        
        sig_entropy = np.argsort(sig_entropy)
        
        # for band selection
        i = 0
        for prop in regions:
            pxIdLst = prop.coords
            for j in range(d):
                band = mat[pxIdLst[:,0],pxIdLst[:,1],j];                    
                entropy_band[i,j] = shannon_entropy(band) # matrix for entropy of segment and band
                          
            i = i + 1   
        
        
        for i in range(k_cl):
            t = int(sig_entropy[i])
            b_dist = np.zeros((d,d))  # to store euclidean distance among all band for a particular segment
            for x in range(d):
                for y in range(d):
                    b_dist[x,y] = np.linalg.norm(seg_nxk[t,x] - seg_nxk[t,y])
                    
            for j in range(d):
                max = 0
                for k in range(d):
                    if entropy_band[t,k] > entropy_band[t,j] :  #band whose entropy greater than jth band    
                        if b_dist[j,k] > max :   #max distance among band whose entropy is greater than jth band
                            max = b_dist[j,k]
                temp_array[i,j] = max 
                
        scaler = MinMaxScaler()
        m = scaler.fit(temp_array)
        new_dist = m.transform(temp_array)
        
        score = np.zeros((k_cl,d))     #creating a parameter for sorting by multiplying entropy and max distance.      
        for i in range(k_cl):
            t = int(sig_entropy[i])
            for j in range(d):
                score[i,j] = entropy_band[t,j] * new_dist[i,j]                  
                
        score = np.argsort(-score) #sorting in decreasing order
        
        for i in range(k_cl): 
            for q in range(L):
                L_entropy_band[i,q] = score[i,q]  # Extracting top L bands   
         
        #Dividing matrices
        split_1 = np.zeros((k_cl,L+1))
        split_2 = np.zeros((numRegns,1))
               
        for x in range(numRegns):
            if (x<k_cl):
                split_1[x,0]=sig_entropy[x]
                for y in range(1,L+1):
                    c = int(split_1[x,0])
                    split_1[x,y] = L_entropy_band[x,y-1]
            else:
                split_2[x,0]=sig_entropy[x]
                             
        a = np.zeros((2,L))
        new_mat = np.zeros((k_cl,numRegns))
        for i in range(k_cl):
            for j in range(k_cl,numRegns):
                for y in range(L):
                    a[0,y] = int(split_1[i,y+1]) #assigning band number of significant cluster
                    ff = int(a[0,y])
                    u = int(split_1[i,0])  #extracting segment number of significant segments
                    a[0,y] = seg_nxk[u,ff] #assigning band value corresponding to band numbers
                    d = int(split_2[j,0]) #extracting segment number of non signifigant segment
                    a[1,y] = seg_nxk[d,ff] #assigning band value of nonsignificant segment 
                                            #corresponding significants segment's band number
                new_mat[i,j] = np.linalg.norm(a[0,:] - a[1,:]) #calculating and assigning 
                    #multidimensional euclidian distance between significant and non significant segments
                        
                        
                #  #--------------Clustering_----------------
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
           
        # display image
        plt.imshow(label2rgb(LbldImg))
        plt.show()           
            
        # Calculate the Overall accurracy
        from sklearn.metrics.cluster import normalized_mutual_info_score
        from sklearn.metrics.cluster import adjusted_mutual_info_score
        
        Lbl = np.reshape(LbldImg, (w * h))
        lbl_pred = Lbl[idx]
        nmi=normalized_mutual_info_score(lbl_true, lbl_pred)
        ami = adjusted_mutual_info_score(lbl_true, lbl_pred)
        print("k_seg = 7 ", "k_cl= 24  L = 13  run ",cd)
        purity = purity_score(lbl_true, lbl_pred)
        print('Purity =', purity)
                
       # Save thematic Map \ cluster map - Matlab format
        mdic = {"lbl_img": LbldImg, "label": "thematic map"}
        scipy.io.savemat("SAl_K seg 7 kcl 24 L 13  run"+ str(cd) + '.mat', mdic)