#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:33:21 2022

@author: dejan
@co-author : Raoul
"""
from warnings import warn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from skimage import morphology, filters
from sklearn import preprocessing, decomposition
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS, OLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
import calculate as cc
import visualize as vis
import preprocessing as pp
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
#from sklearn.preprocessing import normalize

def clean_pca(da, n_components='mle', assign=False,
              visualize_compare=False, visualize_components=False,
              visualize_err=False, **kwargs):
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
    spectra = da.data
    #spectra = scale(da.data, with_mean=True, with_std=False)
    #spectra[spectra<0] = 0
    shape = da.attrs["ScanShape"] + (-1, )

    pca = decomposition.PCA(n_components, **kwargs)
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

    if visualize_compare:
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
        return pp.giveback_same(da, spectra_cleaned)



def simple_deconvolve_nmf(da, n_components, assign=False,
                  visualize_compare=False, visualize_components=False,
                  visualize_err=False, **kwargs):
    """Deconvolve the spectra using sklearn's NMF.

    Parameters:
    -----------
    da: xr.DataArray or a 3D np.ndarray of shape=(ny,nx,shifts)
        The object containing your input spectra
    n_components: int
        The number of components to use for pca decomposition
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

    spectra = da.data
    warn("Setting the min of each spectra to zero...")

    spectra -= np.min(spectra, axis=-1, keepdims=True)
    shape = da.attrs["ScanShape"] + (-1, )

    nmf_model = decomposition.NMF(n_components=n_components, init='nndsvda',
                                  max_iter=1000)#, alpha_W=0.01, alpha_H=0.1)
    # _start = time()
    # print('starting nmf... (be patient, this may take some time...)')
    mix = nmf_model.fit_transform(spectra)
    # mix /= np.sum(mix, axis=-1, keepdims=True)
    # print('renormalizing components and coefficients...')
    # Note constraint order matters
    #%%
    # components_norma = np.sum(components, axis=-1, keepdims=True)
    components = nmf_model.components_
    reconstructed_spectra = nmf_model.inverse_transform(mix)
    if visualize_components:
        visualize_components = vis.AllMaps(mix.reshape(shape),
                                           components=components,
                                           components_sigma=da.shifts.data)
        da.attrs["NMF_Components_visu"] = visualize_components

    if visualize_err:
        plt.figure()
        sqerr = np.sum((spectra - reconstructed_spectra)**2, axis=-1)
        plt.imshow(sqerr.reshape(da.ScanShape))

    if visualize_compare:
        _s = np.stack((spectra, reconstructed_spectra), axis=-1)
        label = ["original spectra", "reconstructed spectra"]
        visualize_result = vis.ShowSpectra(_s, da.shifts.data,
                                           label=label)
        da.attrs["NMF_deconvolution_visu"] = visualize_result

    if assign:
        da = da.expand_dims({"components_pca": n_components}, axis=1)
        da = da.assign_coords(pca_components=(("components_pca", "RamanShifts"),
                              components))
        da = da.assign_coords(pca_mixture_coeffs = (("points", "components_pca"),
                              reconstructed_spectra))
        return da
    else:
        return pp.giveback_same(da, reconstructed_spectra)
    

    

def deconvolve_nmf(da, n_components, assign=False,
                  visualize_compare=False, visualize_components=False,
                  visualize_err=False,col_lim = None,scanshape=None,components_sigma=None, **kwargs):
    """Deconvolve the spectra using sklearn's NMF.

    Parameters:
    -----------
    da: xr.DataArray or a 3D np.ndarray of shape=(ny,nx,shifts)
        The object containing your input spectra
    n_components: int
        The number of components to use for pca decomposition
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
    #spectra = da.data
    warn("Setting the min of each spectra to zero...")

    spectra -= np.min(spectra, axis=-1, keepdims=True)
    #shape = da.attrs["ScanShape"] + (-1, )

    nmf_model = decomposition.NMF(n_components=n_components, init='nndsvda',
                                  max_iter=7)#, alpha_W=0.01, alpha_H=0.1)
    # _start = time()
    # print('starting nmf... (be patient, this may take some time...)')
    mix = nmf_model.fit_transform(spectra)
    # mix /= np.sum(mix, axis=-1, keepdims=True)
    # print('renormalizing components and coefficients...')
    # Note constraint order matters
    mcrar = McrAR(max_iter=120, st_regr='NNLS', c_regr='OLS',
                  c_constraints=[ConstraintNorm(), ConstraintNonneg()],
                  st_constraints=[ConstraintNonneg()])

    mcrar.fit(spectra, C=mix)
    #%%
    #components_norma = np.sum(components, axis=-1, keepdims=True)
    components = mcrar.ST_opt_
    mix = mcrar.C_opt_
    reconstructed_spectra = mcrar.D_opt_
    if visualize_components:
        visualize_components = vis.AllMaps(mix.reshape(shape),
                                           components=components,
                                           components_sigma=components_sigma,col_lim=col_lim)
        if hasattr(da, 'attrs'):
            da.attrs["score_Components_visu"] = visualize_components

    if visualize_err:
        plt.figure()
        sqerr = np.sum((spectra - reconstructed_spectra)**2, axis=-1)
        if hasattr(da, 'attrs'):
            plt.imshow(sqerr.reshape(da.ScanShape))
        else : 
            plt.imshow(sqerr.reshape(scanshape))

    if visualize_compare:
        _s = np.stack((spectra, reconstructed_spectra), axis=-1)
        label = ["original spectra", "reconstructed spectra"]
        visualize_result = vis.ShowSpectra(_s, components_sigma,
                                           label=label)
        if hasattr(da, 'attrs'):
            da.attrs["NMF_deconvolution_visu"] = visualize_result

    if assign:
        da = da.expand_dims({"components_pca": n_components}, axis=1)
        da = da.assign_coords(pca_components=(("components_pca", "RamanShifts"),
                              components))
        da = da.assign_coords(pca_mixture_coeffs = (("points", "components_pca"),
                              reconstructed_spectra))
        return da
    else:
        return pp.giveback_same(da, reconstructed_spectra) , components , mix

    
    

def deconvolve_mcr(da, n_components, assign=False,
                  visualize_compare=False, visualize_components=False,
                  visualize_err=False,col_lim = None,scanshape=None,components_sigma=None, **kwargs):
    """Deconvolve the spectra using sklearn's NMF.

    Parameters:
    -----------
    da: xr.DataArray or a 3D np.ndarray of shape=(ny,nx,shifts)
        The object containing your input spectra
    n_components: int
        The number of components to use for pca decomposition
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
    #spectra = da.data
    warn("Setting the min of each spectra to zero...")

   # spectra -= np.min(spectra, axis=-1, keepdims=True)
    #shape = da.attrs["ScanShape"] + (-1, )

    #nmf_model = decomposition.NMF(n_components=n_components, init='nndsvda',
    #                              max_iter=7)#, alpha_W=0.01, alpha_H=0.1)
    # _start = time()
    # print('starting nmf... (be patient, this may take some time...)')
   # mix = nmf_model.fit_transform(spectra)
    # mix /= np.sum(mix, axis=-1, keepdims=True)
    # print('renormalizing components and coefficients...')
    U, A1, P = np.linalg.svd(spectra, full_matrices=True)
    A = np.zeros((spectra.shape[0],spectra.shape[1]),dtype=float)
    #A[:spectra.shape[1], :spectra.shape[1]] = np.diag(A1)
    np.fill_diagonal(A,A1)
    R = np.dot(U, A) # to obtain score array A*U
    #pca_fit = pca.fit(spectra)
    score = R[:,:n_components]
    spectra_recons = np.dot(R[:,:n_components],P[:n_components,])
    # Note constraint order matters
    mcrar = McrAR(max_iter=120, st_regr='NNLS', c_regr='OLS',
                  c_constraints=[ConstraintNorm(), ConstraintNonneg()],
                  st_constraints=[ConstraintNonneg()])

    mcrar.fit(spectra, C=score)
    #%%
    #components_norma = np.sum(components, axis=-1, keepdims=True)
    components = mcrar.ST_opt_
    score = mcrar.C_opt_
    reconstructed_spectra = mcrar.D_opt_
    if visualize_components:
        visualize_components = vis.AllMaps(score.reshape(shape),
                                           components=components,
                                           components_sigma=components_sigma,col_lim=col_lim)
        if hasattr(da, 'attrs'):
            da.attrs["score_Components_visu"] = visualize_components

    if visualize_err:
        plt.figure()
        sqerr = np.sum((spectra - reconstructed_spectra)**2, axis=-1)
        if hasattr(da, 'attrs'):
            plt.imshow(sqerr.reshape(da.ScanShape))
        else : 
            plt.imshow(sqerr.reshape(scanshape))

    if visualize_compare:
        _s = np.stack((spectra, reconstructed_spectra), axis=-1)
        label = ["original spectra", "reconstructed spectra"]
        visualize_result = vis.ShowSpectra(_s, components_sigma,
                                           label=label)
        if hasattr(da, 'attrs'):
            da.attrs["NMF_deconvolution_visu"] = visualize_result

    if assign:
        da = da.expand_dims({"components_pca": n_components}, axis=1)
        da = da.assign_coords(pca_components=(("components_pca", "RamanShifts"),
                              components))
        da = da.assign_coords(pca_mixture_coeffs = (("points", "components_pca"),
                              reconstructed_spectra))
        return da
    else:
        return pp.giveback_same(da, reconstructed_spectra)

        

    
def hca(da, method='ward',
        metric ='euclidean',visualize=False):
    
    spectra_data = da.data
    #spectra = da.data
    #pca = decomposition.PCA(n_components, **kwargs)
    #pca_fit = pca.fit(spectra)
    #spectra_reduced = pca_fit.transform(spectra)
    #spectra_cleaned = pca_fit.inverse_transform(spectra_reduced)
    #spectra_data = scale(spectra, with_mean=True, with_std=True)
    spectra_hca = linkage(spectra_data,method=method,metric=metric)
    #spectra_data = pd.DataFrame(spectra_data)
    
    if visualize :
        plt.figure()
        dendrogram(spectra_hca,labels = pd.DataFrame(spectra_data).index)
    return pp.giveback_same(da, spectra_data),spectra_hca
