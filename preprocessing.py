#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:53:55 2021

@author: dejan
@co-author : Raoul
"""
import inspect
from warnings import warn
import numpy as np
import xarray as xr
from skimage import morphology, filters
from sklearn.neighbors import LocalOutlierFactor
from scipy.ndimage import median_filter
from scipy import signal
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, decomposition
import calculate as cc
import visualize as vis
import matplotlib.pyplot as plt
from numpy.linalg import norm
    
from matplotlib.patches import Ellipse
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse
# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn()
# except ModuleNotFoundError:
#     pass


def as_series(da:xr.DataArray, axis:str="shorter"):
    """Transform a map scan into a series.

    Takes the mean value along the given axis.
    The new `MeasurementType` attribute will become `"Series"`

    Parameters:
    -----------
    da: xr.DataArray
        your datarray containing spectra as constructed with read_WDF
    axis: str
        one of ["shorter", "longer"]. The axis along which you want to reduce the scan.
    Returns:
    --------
    The updated input object with initial values along the given axis
    replaced with mean values along that axis.
    """

    if axis=="shorter":
        ax = np.argmin(da.ScanShape)
    elif axis == "longer":
        ax = np.argmax(da.ScanShape)
    shape = da.ScanShape + (da.shape[-1], )
    new_shape = (da.ScanShape[~ax], 1)
    new_data = np.mean(da.data.reshape(shape), axis=ax)
    da.MeasuermentType = "Series"
    return da


def gimme_spectra(input_object):
    """Retreive the spectra and ndims"""

    if isinstance(input_object, xr.DataArray):
        spectra = input_object.data
    elif isinstance(input_object, xr.Dataset):
        spectra = input_object.Measured.data
    else:
        spectra = input_object
    return spectra, spectra.ndim

def giveback_same(input_object, output_spectra):

    if isinstance(input_object, xr.DataArray):
        output_object = xr.DataArray(output_spectra,
                                     dims=input_object.dims,
                                     coords=input_object.coords,
                                     attrs=input_object.attrs)
    elif isinstance(input_object, xr.Dataset):
        input_object.Measured.data = output_spectra
        f_name = inspect.stack()[1].function

        return input_object.rename_vars({"Measured":f_name})

    else:
        output_object = output_spectra
    return output_object

def line_up_spectra(spectra):
    old_shape = spectra.shape
    return spectra.reshape(-1, old_shape[-1]), old_shape

def pca_clean(da, n_components='mle', array="Measured", assign=False,
              visualize_clean=False, visualize_components=False,
              visualize_err=False, **kwargs):
    """Clean (smooth) the spectra using PCA.

    Parameters:
    -----------
    da: xr.DataArray or a 3D np.ndarray of shape=(ny,nx,shifts)
        The object containing your input spectra
    n_components: "mle", float, int or None
        "mle":
            The number of components is determined by "mle" algorithm.
        float:
            The variance rate covered.
            The number of components is choosen so to cover the variance rate.
        int:
            The number of components to use for pca decomposition
        None:
            Choses the minimum between the n_features and n_samples
        see more in scikit-learn docs for PCA
    visualize_clean: bool
        Wheather to visualize the result of cleaning
    visualize_components: bool
        Wheather to visualize the decomposition and the components
    assign: bool
        Wheather to assign the results of pca decomposition to the returned
        xr.DataArray.
        If True, the resulting xr.DataArray will be attributed new coordinates:
            `pca_comp[N]` along the `"RamanShifts"` dimension, and
            `pca_comp[N]_coeffs` along the `"points"` dimension
    Returns:
    --------
        updated object with cleaned spectra as .spectra
        spectra_reduced: numpy array
            it is the the attribute added to the WDF object
    """

    spectra = da.data
    #spectra = (da.data - da.data.mean())/da.data.std()
    #scaler = preprocessing.StandardScaler()
    #spectra = scaler.fit_transform(da.data)
    shape = da.attrs["ScanShape"] + (-1, )

    pca = decomposition.PCA(n_components,svd_solver='full', **kwargs)
    pca_fit = pca.fit(spectra)
    spectra_reduced = pca_fit.transform(spectra)
    spectra_cleaned = pca_fit.inverse_transform(spectra_reduced)
    n_components = int(pca_fit.n_components_)
    if visualize_components:
        visualize_components = vis.AllMaps(spectra_reduced.reshape(shape),
                                           components=pca_fit.components_,
                                           components_sigma=da.shifts.data)
        da.attrs["PCA_Components_visu"] = visualize_components

    if visualize_err:
        plt.figure()
        sqerr = np.sum((spectra - spectra_cleaned)**2, axis=-1)
        plt.imshow(sqerr.reshape(da.ScanShape))
        
 
    if visualize_clean:
        _s = np.stack((spectra, spectra_cleaned), axis=-1)
        label = ["original spectra", "pca cleaned"]
        visualize_result = vis.ShowSpectra(_s, da.shifts.data,
                                           label=label)
        da.attrs["PCA_Clean_visu"] = visualize_result

    if assign:
        da = da.expand_dims({"components_pca": n_components}, axis=1)
        da = da.assign_coords(pca_components=(("components_pca", "RamanShifts"),
                              pca_fit.components_))
        da = da.assign_coords(pca_mixture_coeffs = (("points", "components_pca"),
                              spectra_reduced))
        return da
    else:
        return giveback_same(da, spectra_cleaned)


    
def ica_decomp(da, n_components=10, array="Measured", assign=False,
              visualize_clean=False, visualize_components=False,
              visualize_err=False,scanshape=None,components_sigma=None,
               **kwargs):
    """Decompose (clean) the spectra using sklearn's PCA.

    Parameters:
    -----------
    da: xr.DataArray or a 3D np.ndarray of shape=(ny,nx,shifts)
        The object containing your input spectra
    n_components: "mle", float, int or None
        "mle":
            The number of components is determined by "mle" algorithm.
        float:
            The variance rate covered.
            The number of components is chosent so to cover the variance rate.
        int:
            The number of components to use for pca decomposition
        None:
            Choses the minimum between the n_features and n_samples
        see more in scikit-learn docs for PCA
    visualize_compare: bool
        Wheather to visualize the result of cleaning
    visualize_components: bool
        Wheather to visualize the decomposition and the components
    assign: bool
        Wheather to assign the results of pca decomposition to the returned
        xr.DataArray.
        If True, the resulting xr.DataArray will be attributed new coordinates:
            `pca_comp[N]` along the `"RamanShifts"` dimension, and
            `pca_comp[N]_coeffs` along the `"points"` dimension
    Returns:
    --------
        updated object with cleaned spectra as .spectra
        spectra_reduced: numpy array
            it is the the attribute added to the WDF object
    """
    if isinstance(da, xr.DataArray) :
        
        spectra = da.data# - da.data.mean())#/da.data.std()
    
        shape = da.attrs["ScanShape"] + (-1, )
        components_sigma = da.shifts.data
        
    else :
        spectra = da
        shape = scanshape + (-1,)
        components_sigma = components_sigma

    ica = decomposition.FastICA(n_components,whiten='warn', max_iter=500, **kwargs)
    ica_fit = ica.fit(spectra)
    spectra_reduced = ica_fit.transform(spectra)
    spectra_cleaned = ica_fit.inverse_transform(spectra_reduced)
    #n_components = int(ica_fit.n_components_)
    if visualize_components:
        visualize_components = vis.AllMaps(spectra_reduced.reshape(shape),
                                           components=ica_fit.components_,
                                           components_sigma=components_sigma)
        if hasattr(da, 'attrs'):
            da.attrs["score_Components_visu"] = visualize_components

    if visualize_err:
        plt.figure()
        sqerr = np.sum((spectra - spectra_cleaned)**2, axis=-1)
        plt.imshow(sqerr.reshape(da.ScanShape))

    if visualize_clean:
        _s = np.stack((spectra, spectra_cleaned), axis=-1)
        label = ["original spectra", "ica cleaned"]
        visualize_result = vis.ShowSpectra(_s, components_sigma,
                                           label=label)
        if hasattr(da, 'attrs'):
            da.attrs["svd_reconstruction_visu"] = visualize_result



    if assign:
        da = da.expand_dims({"components_ica": n_components}, axis=1)
        da = da.assign_coords(ica_components=(("components_ica", "RamanShifts"),
                              ica_fit.components_))
        da = da.assign_coords(ica_mixture_coeffs = (("points", "components_ica"),
                              spectra_reduced))
        return da
    else:
        return giveback_same(da, spectra_cleaned)


    
    
def svd_decomp(da, n_components=10,
              visualize_clean=False, visualize_components=False,
               col_lim = None,scanshape=None,components_sigma=None,
               visualize_var=False,
               **kwargs):
    """Clean (smooth) the spectra using PCA.

    Parameters:
    -----------
    da: xr.DataArray or a 3D np.ndarray of shape=(ny,nx,shifts)
        The object containing your input spectra
    n_components: "mle", float, int or None
        "mle":
            The number of components is determined by "mle" algorithm.
        float:
            The variance rate covered.
            The number of components is choosen so to cover the variance rate.
        int:
            The number of components to use for pca decomposition
        None:
            Choses the minimum between the n_features and n_samples
        see more in scikit-learn docs for PCA
    visualize_clean: bool
        Wheather to visualize the result of cleaning
    visualize_components: bool
        Wheather to visualize the decomposition and the components
    assign: bool
        Wheather to assign the results of pca decomposition to the returned
        xr.DataArray.
        If True, the resulting xr.DataArray will be attributed new coordinates:
            `pca_comp[N]` along the `"RamanShifts"` dimension, and
            `pca_comp[N]_coeffs` along the `"points"` dimension
    Returns:
    --------
        updated object with cleaned spectra as .spectra
        spectra_reduced: numpy array
            it is the the attribute added to the WDF object
    """
    #scaler = preprocessing.StandardScaler().fit(da.data)
    #spectra = scaler.transform(da.data)
    if isinstance(da, xr.DataArray) :
        
        spectra = da.data# - da.data.mean())#/da.data.std()
    
        shape = da.attrs["ScanShape"] + (-1, )
        components_sigma = da.shifts.data
        
    else :
        spectra = da
        shape = scanshape + (-1,)
        components_sigma = components_sigma
    
    U, A1, P = np.linalg.svd(spectra, full_matrices=True)
    A = np.zeros((spectra.shape[0],spectra.shape[1]),dtype=float)
    #A[:spectra.shape[1], :spectra.shape[1]] = np.diag(A1)
    np.fill_diagonal(A,A1)
    R = np.dot(U, A) # to obtain score array A*U
    #pca_fit = pca.fit(spectra)
    score = R[:,:n_components]
    spectra_recons = np.dot(R[:,:n_components],P[:n_components,])
    # remarquer le nombre de variance expliqué par chaque CP
    var_expl = A1**2/(spectra.shape[0] - 1)
    tot_var_expl = np.sum(var_expl)
    each_cp_var = np.round((var_expl / tot_var_expl),3) # arrondi à 3 chiffre
    #n_components = int(pca_fit.n_components_)
    if visualize_components:
        visualize_components = vis.AllMaps(score.reshape(shape),
                                           components=P[:n_components,],
                                           var = each_cp_var[0:n_components]*100,
                                           components_sigma=components_sigma,col_lim = col_lim)
        if hasattr(da, 'attrs'):
            da.attrs["score_Components_visu"] = visualize_components

   # if visualize_err:
        #plt.figure()
        #sqerr = np.sum((spectra - spectra_cleaned)**2, axis=-1)
        #plt.imshow(sqerr.reshape(da.ScanShape))

    if visualize_clean:
        _s = np.stack((spectra, spectra_recons), axis=-1)
        label = ["original spectra", "svd reconstruction"]
        visualize_result = vis.ShowSpectra(_s, components_sigma,
                                           label=label)
        if hasattr(da, 'attrs'):
            da.attrs["svd_reconstruction_visu"] = visualize_result
            
    if visualize_var :
        plt.figure()
        ncomp = np.arange(n_components) + 1
        plt.plot(ncomp, each_cp_var[0:n_components], 'o-',linewidth=3,color='green')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')


    #if assign:
       # da = da.expand_dims({"components_pca": n_components}, axis=1)
       # da = da.assign_coords(pca_components=(("components_pca", "RamanShifts"),
         #                     pca_fit.components_))
       # da = da.assign_coords(pca_mixture_coeffs = (("points", "components_pca"),
                        #      spectra_reduced))
        #return da
    #else:
    return giveback_same(da, spectra_recons) ,each_cp_var[0:n_components]*100,score,P[:n_components,]


    
class Score_Plot(object) : 
    
    """ this class is used to plot score i  vs score j .
    
    Parameters :  
    
        input_spectra : initial data or data after PCA
        scores : array of all scores.
        score_i : score on x axis
        score_j : score on y axis
        
    return : 
    
        score plot and maximum intensity image
        
    """
    
    def __init__(self,input_spectra,scores,score_i,score_j,**kwargs) : 
        
        self.da = input_spectra.copy()
        self.scores = scores.copy()
        self.scores1 = scores.copy()
        self.xmin = self.da.shifts.data.min()
        self.xmax = self.da.shifts.data.max()
        self.nshifts = self.da.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da.attrs["ScanShape"]
        self.scan_type = self.da.attrs["MeasurementType"]
        self.cmap = kwargs.pop("cmap", "viridis")
        figsize = kwargs.pop("figsize",(14,8))
        self.fig = plt.figure(figsize = figsize,**kwargs)
        #self.fig = plt.figure(**kwargs)
        self.vline = None
        self.axscoreplot = self.fig.add_axes([0.05, 0.25, 0.4, 0.6])
        self.aximg = self.fig.add_axes([0.5, 0.25, 0.4, 0.6])
                #---------------------------- about labels ---------------------------#
        try:
            xlabel = self.da.attrs["ColCoord"]
            ylabel = self.da.attrs["RowCoord"]
            if (self.scan_type == "Map") and (self.da.MapAreaType != "Slice"):
                self.xlabel = f"{xlabel} [{input_spectra[xlabel].units}]"
                self.ylabel = f"{ylabel} [{input_spectra[ylabel].units}]"
            else: # Not a map scan
                self.xlabel = xlabel
                self.ylabel = ylabel
        except:
            self.xlabel = "X"
            self.ylabel = "Y"
        #---------------------------------------------------------------------#
        
        self.X = score_i
        self.Y = score_j
        self.scoreplot = self.axscoreplot.scatter(self.X,self.Y)
        self.axscoreplot.set_title('Score Plot')
        self.elps = EllipseSelector(self.axscoreplot, self.oneselect,interactive=True)
        self.func = "max"
        if self.scan_type == "Map":
            self.imup = self.aximg.imshow(cc.calculate_ss(
                                        self.func,
                                        self.da),
                                        interpolation="gaussian",
                                        aspect=self.nx/self.ny,
                                        cmap=self.cmap)
            self.aximg.set_xlabel(f"{self.xlabel}")
            self.aximg.xaxis.set_label_position('top')
            self.aximg.set_ylabel(f"{self.ylabel}")
            try:
                set_img_coordinates(self.da, self.aximg, unit="")
            except:
                pass
            self.cbar = self.fig.colorbar(self.imup, ax=self.aximg)

        elif self.scan_type == 'Single':
            self.aximg.axis('off')
            self.imup = self.aximg.annotate('calculation result', (.4, .8),
                                        style='italic', fontsize=14,
                                        xycoords='axes fraction',
                                        bbox={'facecolor': 'lightcoral',
                                        'alpha': 0.3, 'pad': 10})
        else:
            _length = np.max((self.ny, self.nx))
            self.imup, = self.aximg.plot(np.arange(_length),
                                         np.zeros(_length), '--o', alpha=.5)
        
        
    def oneselect(self, eclick, erelease) :
        
        if self.vline :
            self.axscoreplot.lines.remove(self.vline)
            self.vline = None
        absc ,ordc = self.elps.center
        exmin, exmax, eymin, eymax = self.elps.extents
        width = exmax - exmin
        height = eymax - eymin
            
        ellipse = Ellipse((absc,ordc),width,height)
            
        XY = np.asarray((self.X, self.Y)).T
        inellipse = ellipse.contains_points(XY)
        X_in = self.scores[:,0][inellipse]
        self.scores[:,0][inellipse] = 89*np.ones((X_in.shape[0]))
        img = self.scores[:,0].reshape((self.ny,self.nx))
        self.scores = self.scores1.copy()
        if self.scan_type == "Map":
            self.imup.set_data(img)
            limits = np.percentile(img, [0, 100])                    
            self.imup.set_clim(limits)
            self.cbar.mappable.set_clim(*limits)
        elif self.scan_type == 'Single':
            self.imup.set_text(img[0][0])
        else:
            self.imup.set_ydata(img.squeeze())
            self.aximg.relim()
            self.aximg.autoscale_view(None, False, True)
        
        self.fig.canvas.draw_idle()
   
    

def select_zone(input_spectra, on_map=False, **kwargs):
    """Isolate the zone of interest in the input spectra.

    Parameters:
    -----------
        input_spectra: xr.DataArray or np.ndarray of spectra
        x_values: if input_spectra is not xarray, this shoud be given,
                otherwise, a simple np.arrange is used
        left, right : float
            The start and the end of the zone of interest in x_values
            (Ramans shifts)
        if on_map == True:
            left, right, top, bottom: int
                Now those keywords correspond to the zone to be selected
                on the map!
                 left: "from" column, right: "to" column,
                 bottom: "from" row, top: "to" row (or the other way around)
    Returns:
    --------
        spectra: the same type of object as `input_spectra`
            Updated object, without anything outside of the zone of interest."""

    spectra, nedim = gimme_spectra(input_spectra)

    if isinstance(input_spectra, xr.DataArray):
        x_values = input_spectra.shifts.data
    else:
        x_values = kwargs.get("x_values", np.arange(spectra.shape[-1]))
    if on_map==False:
        left = kwargs.get('left', x_values.min())
        right = kwargs.get('right', x_values.max())
        left_ind = np.argmax(x_values >= left)
        right_ind = np.where(x_values <= right)[0][-1]
        if isinstance(input_spectra, xr.DataArray):
            input_spectra.attrs["PointsPerSpectrum"] = right_ind - left_ind
            return input_spectra.sel({"RamanShifts": slice(left_ind, right_ind)})
        else:
            condition = (x_values >= left) & (x_values <= right)
            x_values = x_values[condition]
            spectra = spectra[..., condition]
    elif on_map==True:
        left = kwargs.get('left', 0)
        right = kwargs.get('right', spectra.n_x)
        top = kwargs.get('top', 0)
        bottom = kwargs.get('bottom', spectra.n_y)
        top, bottom = np.sort([top, bottom])
        if isinstance(input_spectra, xr.DataArray):
            n_y = bottom - top
            n_x = right - left
            input_spectra.attrs['NbSteps'][
                                input_spectra.attrs['NbSteps']>1] =\
                                n_x, n_y
            indices = np.arange(input_spectra.attrs["Capacity"], dtype=int
                               ).reshape(input_spectra.attrs["ScanShape"]
                               )[top:bottom, left:right].ravel()
            return input_spectra.sel({"points": indices})

    return giveback_same(input_spectra, spectra)

def normalize(inputspectra, method="l2", **kwargs):
    """
    scale the spectra

    Parameters
    ----------
    inputspectra : xr.DataArray or ndarray
    x_values: should be set if inputspectra is just an array
    method: str
        one of ["l1", "l2", "max", "min_max", "wave_number", "robust_scale", "area"]
        default is "area"

    if method == "robust_scale": the scaling with respect to given quantile range
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html
        quantile : tuple
            default = (5, 95)
        centering: bool
            default = False
    if method == "wave_number":
        wave_number: float
        Sets the intensity at the given wavenumber as 1 an the rest is scaled
        accordingly.

    Returns
    -------
    xr.DataArray with scaled spectra.

    """
    if isinstance(inputspectra, xr.DataArray):
        spectra = inputspectra.data.copy()
        x_values = inputspectra.shifts.data
    else:
        x_values = kwargs.get("x_values", np.arange(inputspectra.shape[-1]))
        spectra = inputspectra

    if method in ["l1", "l2", "max"]:
        normalized_spectra = preprocessing.normalize(spectra, axis=1,
                                                     norm=method, copy=False)
    elif method == "min_max":
        normalized_spectra = preprocessing.minmax_scale(spectra, axis=-1,
                                                        copy=False)
    elif method == "area":
        normalized_spectra = spectra / np.expand_dims(np.trapz(spectra,
                                                      x_values), -1)
    elif method == "wave_number":
        wave_number = kwargs.get("wave_number", x_values.min())
        idx = np.nanargmin(np.abs(x_values - wave_number))
        normalized_spectra = spectra / spectra[..., idx][:, np.newaxis]
    elif method == "robust_scale":
        normalized_spectra =  preprocessing.robust_scale(spectra, axis=-1,
                                                      with_centering=False,
                                                      quantile_range=(5,95))
    else:
        warn('"method" must be one of '
             '["l1", "l2", "max", "min_max", "wave_number", "robust_scale", "area"]')
    normalized_spectra -= np.min(normalized_spectra, axis=-1, keepdims=True)
    return giveback_same(inputspectra, normalized_spectra)



def remove_CRs(inputspectra, nx=0, ny=0, sensitivity=0.01, width=0.05,
               visualize=False, **initialization):
    """Remove the spikes using the similarity of neighbouring spectra.
    ATTENTION: Returns normalized spectra.
    
    Parameters:
    -----------
        inputspectra: xr.DataArray
            your input spectra
        nx, ny : int
            The number of columns / rows in the map spectra
        sensitivity: float from 0 to 1
            Adjusts the sensitivity (high sensitivity detects weaker spikes)
        width: float from 0 to 1
            How wide you expect your spikes to be

    Returns:
    --------
        outputspectra: xr.DataArray
            Normalized input spectra with (hopefully) no spikes.

    """
    if isinstance(inputspectra, xr.DataArray):
        mock_sp3 = inputspectra.data
        ny, nx = inputspectra.attrs["ScanShape"]
    else:
        mock_sp3 = inputspectra
    # NORMALIZATION:
    mock_sp3 -= np.min(mock_sp3, axis=-1, keepdims=True)
    area = np.trapz(mock_sp3)[:, np.newaxis]
    mock_sp3 /= area
    # Define the neighbourhood:
    neighbourhood = morphology.ball(3)
    # construct array so that each pixel has the median value of the hood:
    median_spectra3 = filters.median(mock_sp3.reshape(ny, nx, -1),
                                     neighbourhood).reshape(ny*nx, -1)
    # I will only take into account the positive values (CR):
    coarsing_diff = (mock_sp3 - median_spectra3)
    # To find the bad neighbours :
    bad_neighbours = np.nonzero(coarsing_diff > (1/sensitivity)*\
                               np.std(coarsing_diff))
    
    if len(bad_neighbours[0]) > 0:
        print(len(bad_neighbours[0]))
        # =====================================================================
        #             We want to extend the "bad neighbour" label
        #               to adjecent points in each such spectra:
        # =====================================================================
        mask = np.zeros_like(mock_sp3)
        mask[bad_neighbours]=1
        wn_window_size = int(width*mock_sp3.shape[-1])
        window = np.ones((1, wn_window_size))
        mask = morphology.binary_dilation(mask,window)
        mock_sp3[mask] = median_spectra3[mask]
        if visualize:
            _s = np.stack((inputspectra[bad_neighbours[0]],
                           median_spectra3[bad_neighbours[0]]), axis=-1)
            
            v = vis.ShowSpectra(_s, labels=["original", "corrected"])
    else:
         #print("No Cosmic Rays found!")
        pass
    
    return giveback_same(inputspectra, mock_sp3*area),v

def remove_outliers(inputspectra, nx=0,ny=0,window_size = 5,mode ="mirror",visualize=False,not_spike=[],
                   ScanShape=None,sigma=None) :
    
    """ find outliers and apply median_filter"""
    
    
    global show_result
    if isinstance(inputspectra, xr.DataArray):
        spectra = inputspectra.data.copy()
        ny, nx = inputspectra.attrs["ScanShape"]
        sigma = inputspectra.shifts.data
    else:
        spectra = inputspectra.copy()
        ny, nx = ScanShape
        sigma = sigma
    out = LocalOutlierFactor(n_neighbors =5,n_jobs=-1)
    pred = out.fit_predict(spectra)
    outliers = np.where(pred==-1)[0]
    mask = np.ones(np.ndim(spectra.reshape(ny,nx,-1)),dtype=int)
    mask[1] = window_size
    outliers = np.delete(outliers,not_spike)
    if len(outliers) > 0 :
        print(len(outliers))
        spectra_med = median_filter(spectra.reshape(ny,nx,-1),size = mask,mode = mode).reshape(ny*nx,-1)
        spectra[outliers] = spectra_med[outliers]
    else:
         #print("No Cosmic Rays found!")
        pass
    if visualize:
        _s = np.stack((inputspectra[outliers],spectra_med[outliers]), axis=-1)
        show_result = vis.ShowSpectra(_s, labels=["original", "corrected"],sigma = sigma)
    
         
    return giveback_same(inputspectra, spectra),outliers
 

    
def shrinkage(X,eps):
    
    return np.sign(X)*np.maximum(np.abs(X) - eps,0)

def ialm_rpca(X) :
    
    lbd = 1/np.sqrt(np.max(X.shape))
    E = np.zeros_like(X)
    E_1 = np.zeros_like(X)
    A = np.zeros_like(X)
    mu = 1.25 / norm(X,2)
    rho = 1.6
    sv = 10
    Y = X / np.max([norm(X,2),norm(X,np.inf)/lbd])
    
   # k = 0
     
    while (norm(X - A - E,'fro')/norm(X,'fro') > 1e-7) or (norm(E - E_1,'fro')/norm(X,'fro') > 1e-5) :
        
        E_1 = E
        E = shrinkage(X - A + (Y/mu),lbd/mu)
        U, S, V = np.linalg.svd(X - E + (Y/mu), full_matrices=False)
        svp = np.count_nonzero(S > 1/mu)
        
        if svp < sv :
            sv = np.min([svp + 1,np.min(X.shape)])
        else :
            sv = np.min([svp + int(np.round(0.05*np.min(X.shape))),np.min(X.shape)])
        
        A = np.dot(U[:,:sv],np.dot(np.diag(shrinkage(S[:sv],1/mu)),V[:sv,:]))
        
        Y = Y + mu*(X - A - E)
        mu = np.min([mu*rho, mu*1e7])
        
       
                   
    return A, E         
                   
                   
        
    
    
        

def despike_test(spectre) : 
    
    """ remove spike by grouping spectra in pairs using covariance
    Thus for each spectra corresponds a similar. Then calculate the standard daviation 
    of the noise in the spectra and define a threshold.
    It's better to substract baseline before. """
    
    #if isinstance(inputspectra, xr.DataArray):
        #spectre = inputspectra.data.copy()
        #ny, nx = inputspectra.attrs["ScanShape"]
    #else:
        #spectre = inputspectra.copy()
    nrows, ncols = spectre.shape
# median filtering
    medf_sp = np.zeros((nrows,ncols))
    for i in range(nrows) :
        medf_sp[i,:] = signal.medfilt(spectre[i,:],11)
# normalized covariance of spectra
    norcov = np.zeros(nrows)
    posmax = np.zeros(nrows)
#posmax = []
    nb = 0
    for j in range(nrows):
        for k in range(nrows):
            num = (np.dot(medf_sp[j,:],medf_sp[k,:])) ** 2
            deno = (np.dot(medf_sp[k,:],medf_sp[k,:])) * (np.dot(medf_sp[j,:],medf_sp[j,:]))
            norcov[k] = num / deno
            if j == k :
                norcov[k] = 0
    # Pair spectra , max(norcov)
        nb += 1
        posmax[j] = np.argmax(norcov) 
#estimation of the standard deviation of the noise
#posmax = np.array(posmax)
    correct = []
    for l in range(nrows) :
        c = 1;
        ecart = spectre[l,:] - signal.savgol_filter(spectre[l,:],11,5) # bruit
        eff , bins = np.histogram(ecart,6)
        for m in range(len(ecart)) :
            if ecart[m] < bins[4] :
                correct.insert(c,ecart[m])
                c += 1
        sigma = np.std(correct)
        auto = spectre[l,:] - spectre[int(posmax[l]),:]
    # seuillage
        for n in range(ncols) :
            if auto[n] > 5*sigma :
                spectre[l,n] = spectre[int(posmax[l]),n]
                for o in range(-1,1) :
                    if n + o  < 1 and n + o  > ncols :
                        continue
                    else : 
                        if auto[n+o] > 2*sigma :
                            spectre[l,n+o] = spectre[int(posmax[l]),n+o]
            
    return spectre


def remove_spike(inputspectra,visualize=False):
    
    if isinstance(inputspectra, xr.DataArray):
        spectre = inputspectra.data.copy()
        #ny, nx = inputspectra.attrs["ScanShape"]
    else:
        spectre = inputspectra.copy()
    pas = 2000
    saut = 0
    for i in range(1,(spectre.shape[0]//pas) + 1) :
        despike_test(spectre[saut:pas*i,])
        saut = pas*i
    despike_test(spectre[saut:spectre.shape[0],])
    
    if visualize:
            _sc = np.stack((inputspectra,
                           spectre), axis=-1)
            vc = vis.ShowSpectra(_sc, labels=["original", "corrected"])
    
    return giveback_same(inputspectra, spectre), vc