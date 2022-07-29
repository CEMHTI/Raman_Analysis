#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:46:49 2022

@author: dejan
"""
from warnings import warn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from skimage import morphology
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
import preprocessing as pp
import calculate as cc
import visualize as vis

# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn()
# except ModuleNotFoundError:
#     pass




#%%
def giveback_same(input_object, output_spectra):

    if isinstance(input_object, xr.DataArray):
        output_object = xr.DataArray(output_spectra,
                                     dims=input_object.dims,
                                     coords=input_object.coords,
                                     attrs=input_object.attrs)
    else:
        output_object = output_spectra
    return output_object


def find_zeros(inputspectra):
    """
    Find the indices of zero spectra.

    Parameters
    ----------
    spectra : xr.DataArray
        your raw spectra.

    Returns
    -------
    1D numpy array of ints
        indices of zero spectra.

    """
    spectra = inputspectra.data
    zero_idx = np.unique(np.where((np.max(spectra, axis=-1) == 0) &
                                  (np.min(spectra, axis=-1) == 0))[0])
    if len(zero_idx) > 0:
        print(f"{len(zero_idx)} zero-spectra found.")
    else:
        print("No zero-spectra found.")
    return zero_idx


def find_saturated(inputspectra, saturation_limit=90000):
    """
    Identify the saturated instances in the spectra.

    This function uses a default number of 90000 determined empirically
    as a value close to saturation limit. Depending on the spectrometer,
    this value should be adapted.

    IMPORTANT: This function will work only before any scaling is done!

    Parameters
    ----------
    inputspectra : xr.DataArray
        Your input spectra.
    saturation_limit: float
        limit close to, but not higher then, saturation limit of your detector.
    Returns
    -------
    np.array of full indices of saturated spectra.
    """
    spectra = inputspectra.data
    razlika = np.abs(np.diff(spectra, n=1, axis=-1,
                             append=spectra[:,-2][:, np.newaxis]))
    saturated_indices = np.nonzero(razlika > saturation_limit)
    spectra_with_zeros = np.unique(np.nonzero(spectra == 0)[0])

    if set(saturated_indices[0]) != set(spectra_with_zeros):
           warn("You have zeros in your spectra that are not saturations\n"
                f"{set(saturated_indices[0])} saturated, "
                f"and {set(spectra_with_zeros)} spectra with zeros.")
    if len(saturated_indices[0])==0:
        print("No saturated spectra is found.")
    else:
        print(f"{len(np.unique(saturated_indices[0]))} saturated spectra found.")
    return saturated_indices


def get_neighbourhood(map_shape, indices, diskradius=3, return_faulty=False):
    """
    Recover the neighbourhood around each of the indices

    Parameters
    ----------
        map_shape: tuple of ints
        indices: numpy 1D ndarray of ints
            indicies for wich you want to find the neighbours.
        diskradius: int
            what is the neghbourhood you want
    Returns
    -------
        enlarged ndarray of indices so it englobes the neighbours of initial inds
    """
    maska = np.zeros(np.prod(map_shape), dtype=bool)
    maska[indices] = True
    maska = maska.reshape(map_shape)
    selem = morphology.disk(3)
    if not return_faulty:
        selem[diskradius, diskradius] = False
    try:
        neighbourhood = np.nonzero(morphology.binary_dilation(maska,
                                                   footprint=selem).ravel())[0]
    except TypeError:
        neighbourhood = np.nonzero(morphology.binary_dilation(maska,
                                                   selem=selem).ravel())[0]
    return neighbourhood


def correct_zeros(inputspectra, copy=False):
    """Replace all zero-spectra with median values of the neighbours."""
    map_shape = inputspectra.attrs["ScanShape"]
    if copy:
        spectra = np.copy(inputspectra.data)
    else:
        spectra = inputspectra.data
    zero_idx = find_zeros(spectra)
    if len(zero_idx) > 0:
        for ind in zero_idx:
            neighbouring_indices = get_neighbourhood(map_shape, ind)
            spectra[ind] = np.median(spectra[neighbouring_indices], axis=0)
        print(f"{len(zero_idx)} zero spectra corrected.")
    return giveback_same(inputspectra, spectra)


def correct_faulty_spectra(inputspectra, faulty_spectra_indices, copy=False,
                     n_nearest_features=8, max_iter=44,
                     smoothen=True, lam=None, visualize_result=False):
    """
    Correct saturated spectra.

    Replaces saturated portions of spectra, "inventing" new values
    using the similarity with non-saturated neighbours.
    The values to be replaced are expected to be equal to zero.

    Parameters:
    -----------
    inputspectra: xr.DataArray
        Your raw (!) input spectra that you want to correct.
    faulty_spectra_indices: np.array of ints
        The indices in inputspectra.data array indicating faulty spectra.
        (As one would obtain with `find_saturated` function, for example)

    Returns:
    --------
    outputspectra: xr.DataArray
        the same as inputspectra, only with punctual saturations corrected
        using IterativeImputer from ScikitLearn"""

    spectra = inputspectra.data
    map_shape = inputspectra.attrs["ScanShape"]
    if lam == None:
        lam = inputspectra.shape[-1]//10
    faulty_spectra = np.unique(faulty_spectra_indices[0])
    if len(faulty_spectra) > 0:
        faulty_spectra = faulty_spectra.squeeze()
        neighbours_idx = get_neighbourhood(map_shape,
                                           faulty_spectra,
                                           return_faulty=True)
        neighbours = spectra[neighbours_idx]
        # create an array `faulty_in_da_hood` to isolate faulty from the hood
        # neighbours[faulty_in_da_hood] = spectra[faulty_spectra]
        faulty_in_da_hood = np.in1d(neighbours_idx, faulty_spectra)
# =============================================================================
#         test_img = np.zeros(map_shape, dtype='uint8')
#         test_img.ravel()[neighbours_idx] = 150
#         test_img.ravel()[faulty_spectra] = 255
#         plt.imshow(test_img)
# =============================================================================
        # The most important part:
        min_value = 0.5 * np.max(neighbours, axis=0)
        imp = impute.IterativeImputer(n_nearest_features=n_nearest_features,
                                      max_iter=max_iter, skip_complete=True,
                                      min_value=min_value,
                                      missing_values=0)
        repaired_spectra = imp.fit_transform(neighbours)[faulty_in_da_hood]

        if smoothen:
            repaired_spectra  = cc.baseline_als(repaired_spectra, lam=lam,
                                               p=0.5, visualize_result=False)
        repaired_spectra = np.array(repaired_spectra).squeeze()
        if visualize_result:
            s = np.stack((spectra[faulty_spectra], repaired_spectra), axis=-1)
            label = ["original spectra", "correction"]
            if repaired_spectra.ndim == 1:
                plt.plot(s[:,0], label=label[0])
                plt.plot(s[:,1], label=label[1])
                plt.title(f"Correcting saturations Spectra N° {faulty_spectra}")
                plt.legend()
            else:
                visualize_result = vis.ShowSpectra(s,
                                                   title="Correcting saturations",
                                                   label=label)
        spectra[faulty_spectra] = repaired_spectra

    return giveback_same(inputspectra, spectra)


def correct_saturated(inputspectra, **kwargs):
    da = correct_zeros(inputspectra)
    indices = find_saturated(da)
    return correct_faulty_spectra(da, indices, **kwargs)


def remove_CRs(inputspectra, nx=0, ny=0,
               sensitivity=0.01, width=0.02, **initialization):
    return pp.remove_CRs(inputspectra, nx=0, ny=0,
               sensitivity=0.01, width=0.02, **initialization)
