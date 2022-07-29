#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:50:13 2021

@author: dejan
@co-author : Raoul
"""
from warnings import warn
from tqdm import tqdm
import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.optimize import minimize_scalar
from joblib import delayed, Parallel
import visualize as vis
import pybaselines


def calculate_ss(func_name:str, input_spectra, xmin=None, xmax=None,
                 is_reduced=False):
    """What to calculate in vis.ShowSelected.

    Parameters:
    -----------
    func_name: str
        one of ['area', 'barycenter_x', 'max_value', 'peak_position', 'peak_ratio']
    spectra: xr.DataArray
        your spectra
    xmin: float
    xmax: float
    is_reduced: bool
        whether or not you want to use the reduced spectra in the calculation
        removes the straight line connecting y[xmin] and y[xmax]

    Returns:
    --------

    """

    def calc_max_value(spectra, x):
        return np.max(spectra, axis=-1).reshape(shape)

    def calc_area(spectra, x):
        if np.ptp(x) == 0:
            return np.ones(shape)
        else:
            return np.trapz(spectra, x=x).reshape(shape)

    def calc_peak_position(spectra, x):
        peak_pos = np.argmax(spectra, axis=-1).reshape(shape)
        return x[peak_pos]  # How cool is that? :)

    def calc_barycenter_x(spectra, x):
        return find_barycentre(x, spectra,
                                  method="weighted_mean"
                                 )[0].reshape(shape)

    def calc_peak_ratio(spectra, x):
        return (spectra[:, 0] / spectra[:, -1]).reshape(shape)

    function_map = {"area": calc_area,
                    "barycenter_x": calc_barycenter_x,
                    "max_value": calc_max_value,
                    "peak_position": calc_peak_position,
                    "peak_ratio": calc_peak_ratio}

    if isinstance(input_spectra, xr.DataArray):
        x = input_spectra.shifts.data
        xmin = x.min() if xmin == None else xmin
        xmax = x.max() if xmax == None else xmax
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        # indmax = min(len(x) - 1, indmax)
        if indmax == indmin:  # if only one line
            indmax = indmin + 1
        x = x[indmin:indmax]
        shape = input_spectra.attrs["ScanShape"]
        spectra = input_spectra.data[:, indmin:indmax].copy()
    if is_reduced:
        a = (spectra[:, -1] - spectra[:, 0]) / (xmax - xmin)
        b = spectra[:, 0] - a * xmin
        baseline = np.outer(a, x) + b[:, np.newaxis]
        spectra -= baseline
        spectra -= np.min(spectra, axis=-1, keepdims=True)

    return function_map.get(func_name, calc_max_value)(spectra, x)

def find_barycentre(x: np.array, y: np.array,
                    method: str = 'simple vectorized') -> tuple((np.array, np.array)):
    """Calculate the coordinates of the barycentre value.

    Parameters
    ----------
    x : np.array
        ndarray containing your raman shifts
    y : np.array
        ndarray containing your intensity (counts) values
    method : string
        one of ['trapz_minimize', 'list_minimize', 'weighted_mean', 'simple vectorized']

    Returns
    -------
    (x_value, y_value): the coordinates of the barycentre
    """
    assert (method in ['trapz_minimize', 'list_minimize', 'weighted_mean', 'simple vectorized'])
    if x[0] == x[-1]:
        return x * np.ones(len(y)), y / 2
    if method == 'trapz_minimize':
        half = np.abs(np.trapz(y, x=x) / 2)

        def find_y(y0, xx=x, yy=y):
            """Internal function to minimize
            depending on the method chosen"""
            # Calculate the area of the curve above the y0 value:
            part_up = np.abs(np.trapz(
                yy[yy >= y0] - y0,
                x=xx[yy >= y0]))
            # Calculate the area below y0:
            part_down = np.abs(np.trapz(
                yy[yy <= y0],
                x=xx[yy <= y0]))
            # for the two parts to be the same
            to_minimize_ud = np.abs(part_up - part_down)
            # fto make the other part be close to half
            to_minimize_uh = np.abs(part_up - half)
            # to make the other part be close to half
            to_minimize_dh = np.abs(part_down - half)
            return to_minimize_ud ** 2 + to_minimize_uh + to_minimize_dh

        def find_x(x0, xx=x, yy=y):
            part_left = np.abs(np.trapz(
                yy[xx <= x0],
                x=xx[xx <= x0]))
            part_right = np.abs(np.trapz(yy[xx >= x0],
                                         x=xx[xx >= x0]))
            to_minimize_lr = np.abs(part_left - part_right)
            to_minimize_lh = np.abs(part_left - half)
            to_minimize_rh = np.abs(part_right - half)
            return to_minimize_lr ** 2 + to_minimize_lh + to_minimize_rh

        minimized_y = minimize_scalar(find_y, method='Bounded',
                                      bounds=(np.quantile(y, 0.01),
                                              np.quantile(y, 0.99)))
        minimized_x = minimize_scalar(find_x, method='Bounded',
                                      bounds=(np.quantile(x, 0.01),
                                              np.quantile(x, 0.99)))
        y_value = minimized_y.x
        x_value = minimized_x.x

    elif method == "list_minimize":
        yy = y
        xx = x
        ys = np.sort(yy)
        z2 = np.asarray(
            [np.abs(np.trapz(yy[yy <= y_val], x=xx[yy <= y_val]) - \
                    np.trapz(yy[yy >= y_val] - y_val, x=xx[yy >= y_val])) \
             for y_val in ys])
        y_value = ys[np.argmin(z2)]
        x_ind = np.argmin(np.abs(np.cumsum(yy) - np.sum(yy) / 2)) + 1
        x_value = xx[x_ind]

    elif method == 'weighted_mean':
        weighted_sum = np.dot(y, x)
        x_value = weighted_sum / np.sum(y, axis=-1)
        y_value = weighted_sum / np.sum(x)

    elif method == 'simple vectorized':
        xgrad = np.gradient(x)
        proizvod = y * xgrad
        sumprod = np.cumsum(proizvod, axis=-1)
        medo = np.median(sumprod, axis=-1, keepdims=True)  # this should be half area
        ind2 = np.argmin(np.abs(sumprod - medo), axis=-1)
        x_value = x[ind2]
        y_value = sumprod[:, -1] / (x[-1] - x[0])
    return x_value, y_value


def rolling_median(arr, w_size, ax=0, mode='nearest', *args):
    """Calculates the rolling median of an array
    along the given axis on the given window size.
    Parameters:
    -------------
        arr:ndarray: input array
        w_size:int: the window size
                    (should be less then the dimension along the given axis)
        ax:int: the axis along which to calculate the rolling median
        mode:str: to choose from ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
        see the docstring of ndimage.median_filter for details
    Returns:
    ------------
        ndarray of same shape as the input array"""
    shape = np.ones(np.ndim(arr), dtype=int)
    shape[ax] = w_size
    return ndimage.median_filter(arr, size=shape, mode=mode, *args)


def baseline_ials(da, x=None, lam=1e5, p=5e-5,lam1=1e-3, niter=12, visualize_result=False,
                 inplace=False):
    """Adapted from:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    To get the feel on how the algorithm works, you can think of it as
    if the rolling ball which comes from beneath the spectrum and thus sets
    the baseline.

    Then, to follow the image, schematic explanation of the params would be:

    Params:
    ----------
        y:          1D or 2D ndarray: the spectra on which to find the baseline

        lam:number: Can be viewed as the radius of the ball.
                    As a rule of thumb, this value should be around the
                    twice the width of the broadest feature you want to keep
                    (width is to be measured in number of points, since
                    for the moment no x values are taken into account
                    in this algorithm)

        p:number:   Can be viewed as the measure of how much the ball
                    can penetrate into the spectra from below

        niter:int:  number of iterations
                   (the resulting baseline should stabilize after
                    some number of iterations)

    Returns:
    -----------
        if inplace == True:
            same type of object as the input, with the baseline substracted from it.
        if inplace == False:
            same type of objet as the input containing the baseline as data

    Note:
    ----------
        It calculates around 250 spectra/sec with 10 iterations
        on i7 4cores(8threads) @1,9GHz

    """
    if isinstance(da, xr.DataArray):
        x = da.shifts.data
        y = da.data
    elif isinstance(da, np.ndarray):
        y = da
    elif x == None:
        x = np.arange(y.shape[-1])

    def _one_bl(yi, lam=lam, p=p, lam_1=lam1, niter=niter, crit=1e-3, z=None):

        return pybaselines.whittaker.iasls(yi, lam=lam, p=p, lam_1=lam1, max_iter=niter,
                                         tol=1e-3, weights=None)[0]
# =============================================================================
#     def _one_bl(yi, lam=lam, p=p, niter=niter, crit=1e-3, z=None):
#         if z is None:
#             L = yi.shape[-1]
#             D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
#             D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
#             w = np.ones(L)
#             W = sparse.spdiags(w, 0, L, L)
#         for i in range(niter):
#             W.setdiag(w)  # Do not create a new matrix, just update diagonal values
#             Z = W + D
#             z = sparse.linalg.spsolve(Z, w * yi)
#             w_new = p * (yi > z) + (1 - p) * (yi < z)
#             delta = np.linalg.norm(w_new - w) / np.linalg.norm(w)
#             w = w_new
#             if delta < crit:
#                 # print(f"Convergence achieved after {i}/{niter} iterations.")
#                 break
#         return z
# =============================================================================

# =============================================================================
#     def _one_bl(y, ratio=p, lam=lam, niter=10):
#         L = len(y)
#
#         diag = np.ones(L - 2)
#         D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
#
#         H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
#
#         w = np.ones(L)
#         W = sparse.spdiags(w, 0, L, L)
#
#         crit = 1
#         count = 0
#
#         while crit > ratio:
#             z = sparse.linalg.spsolve(W + H, W * y)
#             d = y - z
#             dn = d[d < 0]
#
#             m = np.mean(dn)
#             s = np.std(dn)
#
#             w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
#
#             crit = np.linalg.norm(w_new - w) / np.linalg.norm(w)
#
#             w = w_new
#             W.setdiag(w)  # Do not create a new matrix, just update diagonal values
#
#             count += 1
#
#             if count > niter:
#                 # print('Maximum number of iterations exceeded')
#                 break
#         # print(z.shape)
#         return z - np.max(z-y)/2
# =============================================================================
    # -----------------------------------------------------------------
    if y.ndim == 1:
        b_line = _one_bl(y)
    elif y.ndim == 2:
        b_line = np.asarray(Parallel(n_jobs=-1)(
                    delayed(_one_bl)(y[i])
                    for i in range(y.shape[0])))
    else:
        warn("This only works for 1D or 2D arrays")
    if visualize_result:
        if y.ndim == 1:
            _s = np.stack((y, b_line, y-b_line), axis=1)[np.newaxis,:,:]
        else:
            _s = np.stack((y, b_line, y-b_line), axis=-1)
        visualize_result = vis.ShowSpectra(_s, sigma=x, label=["spectra", "baseline", "corrected"])
        if isinstance(da, xr.DataArray):
            da.attrs["BaselineVisu"] = visualize_result
    if inplace:
        vrackalica = y - b_line
        if isinstance(da, xr.DataArray):
            vrackalica -= np.min(vrackalica.data, axis=-1)
            da.data = vrackalica
            return da
        else:
            vrackalica -= np.min(vrackalica, axis=-1, keepdims=True)
            return vrackalica
    else:
        vrackalica = b_line
        if isinstance(da, xr.DataArray):
            da_copy = da.copy()
            da_copy.data = vrackalica
            return da_copy

    # print("returning the tuple containing the baseline and the visualization.")
    return vrackalica#, visualize_result



def baseline_arpls(da, x=None, lam=1e5, niter=50, visualize_result=False,
                 inplace=False):
    """Adapted from:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    To get the feel on how the algorithm works, you can think of it as
    if the rolling ball which comes from beneath the spectrum and thus sets
    the baseline.

    Then, to follow the image, schematic explanation of the params would be:

    Params:
    ----------
        y:          1D or 2D ndarray: the spectra on which to find the baseline

        lam:number: Can be viewed as the radius of the ball.
                    As a rule of thumb, this value should be around the
                    twice the width of the broadest feature you want to keep
                    (width is to be measured in number of points, since
                    for the moment no x values are taken into account
                    in this algorithm)

        p:number:   Can be viewed as the measure of how much the ball
                    can penetrate into the spectra from below

        niter:int:  number of iterations
                   (the resulting baseline should stabilize after
                    some number of iterations)

    Returns:
    -----------
        if inplace == True:
            same type of object as the input, with the baseline substracted from it.
        if inplace == False:
            same type of objet as the input containing the baseline as data

    Note:
    ----------
        It calculates around 250 spectra/sec with 10 iterations
        on i7 4cores(8threads) @1,9GHz

    """
    if isinstance(da, xr.DataArray):
        x = da.shifts.data
        y = da.data
    elif isinstance(da, np.ndarray):
        y = da
    elif x == None:
        x = np.arange(y.shape[-1])

    def _one_bl(yi, lam=lam, niter=niter, crit=1e-3, z=None):

        return pybaselines.whittaker.arpls(yi, lam=lam, max_iter=niter,
                                            tol=1e-3, weights=None)[0]

    if y.ndim == 1:
        b_line = _one_bl(y)
    elif y.ndim == 2:
        b_line = np.asarray(Parallel(n_jobs=-1)(
                    delayed(_one_bl)(y[i])
                    for i in range(y.shape[0])))
    else:
        warn("This only works for 1D or 2D arrays")
    if visualize_result:
        if y.ndim == 1:
            _s = np.stack((y, b_line, y-b_line), axis=1)[np.newaxis,:,:]
        else:
            _s = np.stack((y, b_line, y-b_line), axis=-1)
        visualize_result = vis.ShowSpectra(_s, sigma=x, label=["spectra", "baseline", "corrected"])
        if isinstance(da, xr.DataArray):
            da.attrs["BaselineVisu"] = visualize_result
    if inplace:
        vrackalica = y - b_line
        if isinstance(da, xr.DataArray):
            vrackalica -= np.min(vrackalica.data, axis=-1)
            da.data = vrackalica
            return da
        else:
            vrackalica -= np.min(vrackalica, axis=-1, keepdims=True)
            return vrackalica
    else:
        vrackalica = b_line
        if isinstance(da, xr.DataArray):
            da_copy = da.copy()
            da_copy.data = vrackalica
            return da_copy

    # print("returning the tuple containing the baseline and the visualization.")
    return vrackalica#, visualize_result
