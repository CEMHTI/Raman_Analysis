#!/usr/bin/env python
# coding: utf-8

# Import all modules you'll need. if you get an error like : no module "module_name" found; just run "pip install module_name" in empty cell.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'Qt')
get_ipython().run_line_magic('load_ext', 'autoreload')
# to automaticaly reload all the modules that have been modified
get_ipython().run_line_magic('autoreload', '2')
import os
from glob import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn import svm, kernel_ridge, metrics, preprocessing
from PIL import Image, ExifTags, ImageFile
import visualize as vis
import preprocessing as pp
import processing as proc
import calculate as cc
import correction as corr
from read_WDF import read_WDF, get_exif
import warnings
warnings.filterwarnings('ignore')


# ### Reading
# 
# In os.chdir, put the enplacement of your .wdf file and run the cell to get all your wdf files.

# In[2]:


os.chdir('C:/Users/raoul.missodey/Desktop/Code_python')
files = glob('*.wdf')
for i, j in enumerate(files):
    print(i, ') ', j)


# In[3]:


filename = files[33] # choose the number of .wdf file you want to use.
# load the file
da, img = read_WDF(filename, verbose=0) # verbose = 1 or true to have all information about your file.


# In[4]:





# In[9]:


da.data.shape


# ### Visualisation & Preprocessing

# In[5]:


vis.show_grid(da, img) # show your Cartography and the scanning part


# In[6]:


# show all your spectra and scroll trough them using the slide bar
#you can also enter a number of specific spectra 
show_all_spectra = vis.ShowSpectra(da); 


# In[28]:


# after seing all your spectra, if you want to truncate run this cell. you can also add right values.
# if not, pass.
da = pp.select_zone(da, left=200)
after_cutting_check = vis.ShowSpectra(da);


# In[10]:


# show all the intensity map depending on wavenumber.
show_all_map = vis.AllMaps(da);


# #### Baseline Substraction

# In[11]:


#  substract baseline : you have to modify lambda_value(smoothing parameter), p_value(the penalizing weighting factor)
# and lambda1_value(the smoothing parameter for the first derivative of the residual).
check_bline = vis.FindBaseline(da,sigma=da.shifts.data)


# In[12]:


# run to see chosen values
print(f"p: {check_bline.p_val:.2e}, lam: {check_bline.lam_val:.2e},lam1: {check_bline.lam1_val:.2e} ")


# In[13]:


# calculate the basline for all spectrum
baseline = cc.baseline_ials(da, p=check_bline.p_val, lam=check_bline.lam_val, lam1= check_bline.lam1_val)


# In[14]:


# substraction
db = da.copy()
db.values = db.values - baseline


# In[15]:


# show result
after_baseline = vis.ShowSpectra(db)


# #### Cosmic Ray Removal

# In[16]:


# window_size : elements within shape [1,window_size,1] will get passed to the filter function.
# you can also change the mode{'reflect', 'constant', 'nearest', 'mirror', 'wrap'}. I use mirror by default
# if you see that the algorithm detect a spike but it's not really a spike, you can put the number of this spectrum in not_spike
# and run the cell again
dm = pp.remove_outliers(db,window_size = 5,visualize=1,not_spike=[])


# In[143]:


vis.ShowSpectra(dm)


# #### RPCA (Robust PCA)

# In[17]:


# run this cell to see what RPCA does.
# choose noise image and replace it by noise2. if you have .png image, you don't need to divide it by 255 
im = plt.imread("C:/Users/raoul.missodey/Desktop/Code_python/" + "noise5.png")
if im.ndim == 2 : 
    imgray = im.copy()
else :
    imgray = np.mean(im,-1)
A , E = pp.ialm_rpca(imgray)

fig,ax = plt.subplots(nrows = 1,ncols = 3,figsize=(15,8))
ax[0].imshow(imgray,cmap='gray')
ax[0].set_title('X')
ax[1].imshow(A,cmap='gray')
ax[1].set_title('A')
ax[2].imshow(E,cmap='gray')
ax[2].set_title('E')


# In[ ]:





# In[26]:


# let's try it on our data.
A, E = pp.ialm_rpca(da.data)
drpca = db.copy()
drpca.values = A
showrpca_spectra = vis.ShowSpectra(drpca) 


# In[30]:


# show all maps after rpca
showrpca_map = vis.AllMaps(drpca)


# In[62]:


showrpca_spectra = vis.ShowSpectra(db) 


# In[29]:


drpca1 = db.copy()
drpca1.values = E
showrpca_spectra = vis.ShowSpectra(drpca1) 
showrpca_map = vis.AllMaps(drpca1)


# #### Normalization

# In[18]:


# choose normalize method you want to apply : min_max, l1, l2, max, area, robust_scale 
#dm.values -= np.min(dm.values, axis=-1, keepdims=True)
dn = pp.normalize(dm, method="min_max")
after_normalize = vis.ShowSpectra(dn)


# In[ ]:





# ### Processing

# #### PCA

# In[28]:


# compute pca using svd decomposition
dsvd ,var,scores,loadings = pp.svd_decomp(dn, n_components=11, visualize_clean=1, visualize_components=1,visualize_err=1)


# In[21]:


# in var we have "explained variance" of each components
var


# In[23]:


# choose number of component you want to keep
dsvd ,var,score ,loadings = pp.svd_decomp(dn, n_components=7, visualize_clean=1, visualize_components=1,visualize_err=1)


# In[ ]:





# In[ ]:





# #### NMF Decomposition

# In[24]:


# Run this cell to see pure spectrum and pure contribution
dnm = proc.deconvolve_nmf(dn, n_components=10, visualize_components=1, visualize_compare=1, visualize_err=1)


# #### HCA (clustering)

# In[25]:


# run this cell will show the dendrogramm :
# you can change the method you want to use to calculate the distance between your classes : average, single, complete,ward...
# also you can change the metric you want to use to calculate the distance between your spectrum : 
#'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean'...
dhca,spectra_hca = proc.hca(dsvd,method='ward',metric ='euclidean',visualize=1)


# In[27]:


# after you have visualized your dendrogramm choose where you want to cute it : (t)
# then replace t by this value and run the cell to visualize the result
# criterion : inconsistent, distance, maxclust...
#right click on the map to see the corresponding spectrum in cluster
showhca = vis.ShowHca(dhca, dsvd,spectra_hca,t=40,criterion='distance')


# #### KMeans (cluster analysis in WiRE)

# In[86]:


# run this cell to see inertia curve. you can also changed the maximalumber of your cluster ; warn : big number can make
# the algorithm slow
inertia = vis.ShowElbpt(30, dsvd)


# In[87]:


# see the number of your maximal cluster given by Elbow method , warn : use the same maximal number of cluster
# curve : convex or concave depending on the look of your curve
# direction : decreasing or increasing
vis.print_elbowpt(30, inertia, curve="convex", direction="decreasing")


# In[88]:


# enter the number of cluster given by the Elbow method
# again right click on the map to see the corresponding spectrum in cluster
showkmeans = vis.ShowKMeans(dsvd,nb_clas= 4) 


# ### Postprocessing

# #### Curve fitting

# In[ ]:


# run this cell to fit all your spectra. here we use pseudo-voigt as model.
# left click to draw the model, left click on the top of a model to remove it,
# right click to draw the result of cumulative sum of all your models
# after right clicking, if the sum is not good, just remove your indesirable models et restart.
# scroll your mouse to adjust the width of the models

# Important : double rigth click to end.
#important : your data should have correct baseline et must be normalized.
fiallspectra = vis.FitAllSpectra(dn)


# In[ ]:


# after adding your model to the peaks position, run this cell to see the results.
# this may take few minutes because I stock the peaks parameters in array which you can access through "fitparams.pic_h"
# the table in the bottom of the plot doesn't change because it's just the parameters of the first fitting.
# by now, it will take time to change the table when updating the plot.
# "fitparams.pic_h" can be used to plot the peak ratio
from time import*
start = time()
fitparams = vis.FitParams1(dn, fiallspectra.x_size, fiallspectra.peaks_counter,
          fiallspectra.pic, fiallspectra.sum_peak)
end = time()
print((end - start)/60,"min")

