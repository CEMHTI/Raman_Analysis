#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:13:46 2021

@author: dejan
@co-author : Raoul
"""
from os import path
from warnings import warn
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io
from sklearn import decomposition
from matplotlib.widgets import (Slider, Button, RadioButtons, SpanSelector,
                                   CheckButtons, MultiCursor, TextBox)
import calculate as cc
import preprocessing as pp
from sklearn import preprocessing
from read_WDF import get_exif
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pandas as pd
from dfply import *
from sklearn.cluster import KMeans
from dfply import *
from sklearn.cluster import KMeans
from matplotlib.widgets import (Slider, Button, RadioButtons, SpanSelector,
                                   CheckButtons, MultiCursor, TextBox)
from kneed import KneeLocator
from matplotlib.patches import Ellipse
from matplotlib.artist import ArtistInspector
from cycler import cycler
from copy import copy
from itertools import chain
from utilities import pV
from utilities import multi_pV as fitting_function
from scipy.optimize import curve_fit
from uncertainties import unumpy

def show_grid(da, img) :

    img_exif = get_exif(img)
    img_arr = np.array(img.getdata()).reshape(img.height, img.width, 3)

    xres = img.width / float(img_exif["FocalPlaneXResolution"])  # in px/µm
    yres = img.height / float(img_exif["FocalPlaneYResolution"])  # in px/µm

    xminpx = round((da.InitialCoordinates[0] - img_exif["FocalPlaneXYOrigins"][0]) * xres)
    yminpx = round((da.InitialCoordinates[1] - img_exif["FocalPlaneXYOrigins"][1]) * yres)

    xmaxpx = xminpx + round(da.StepSizes[0] * da.NbSteps[0] * xres)
    ymaxpx = yminpx + round(da.StepSizes[1] * da.NbSteps[1] * yres)

    xminpx, xmaxpx = np.sort([xminpx, xmaxpx])
    yminpx, ymaxpx = np.sort([yminpx, ymaxpx])

    xsizepx = xmaxpx - xminpx
    ysizepx = ymaxpx - yminpx

    grid_in_image: bool = (xsizepx <= img.width) & (ysizepx <= img.height)

    if grid_in_image:
        fig, ax = plt.subplots()
        ax.imshow(img_arr)
        x_pxvals = np.linspace(xminpx, xmaxpx, da.NbSteps[0])
        y_pxvals = np.linspace(yminpx, ymaxpx, da.NbSteps[1])
        for xxx in x_pxvals:
            ax.vlines(xxx, ymin=yminpx, ymax=ymaxpx, lw=1, alpha=0.2)
        for yyy in y_pxvals:
            ax.hlines(yyy, xmin=xminpx, xmax=xmaxpx, lw=1, alpha=0.2)
        ax.scatter(xminpx, yminpx, marker="X", s=30, c='r')
        ax. xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        fig.show()


def set_img_coordinates(da, ax, unit="µm",
                        rowcoord_arr=None, colcoord_arr=None):

    if rowcoord_arr == None:
        rowcoord_arr = np.unique(da[da.RowCoord].data)
    if colcoord_arr == None:
        colcoord_arr = np.unique(da[da.ColCoord].data)

    def row_coord(y, pos):
        yind = int(y)
        if yind < len(rowcoord_arr):
            yy = f"{rowcoord_arr[yind]}{unit}"
        else:
            yy = ""
        return yy

    def col_coord(x, pos):
        xind = int(x)
        if xind < len(colcoord_arr):
            xx = f"{colcoord_arr[xind]}{unit}"
        else:
            xx = ""
        return xx

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(col_coord))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(row_coord))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
                   width=.2, labelsize=8)


class ShowCollection(object):
    """Visualize a collection of images.

    Parameters
    ----------
    image_pattern : str
        Can take asterixes as wildcards. For ex.: "./my_images/*.jpg" to select
        all the .jpg images from the folder "my_images"
    load_func : function
        The function to apply when loading the images
    first_frame : int
        The frame from which you want to stard your slideshow
    load_func_kwargs : dict
        The named arguments of the load function

    Outputs
    -------
    Interactive graph displaying the images one by one, whilst you can
    scroll trough the collection using the slider or the keyboard arrows

    Example
    -------
    >>> import numpy as np
    >>> from skimage import io, transform

    >>> def binarization_load(f, shape=(132,132)):
    >>>     im = io.imread(f, as_gray=True)
    >>>     return transform.resize(im, shape, anti_aliasing=True)

    >>> ss = ShowCollection(images, load_func=binarization_load,
                            shape=(128,128))
    """

    def __init__(self, image_pattern, load_func=io.imread, first_frame=0,
                 **load_func_kwargs):

        self.coll_all = io.ImageCollection(image_pattern, load_func=load_func,
                                           **load_func_kwargs)
        self.first_frame = first_frame
        self.nb_pixels = self.coll_all[0].size
        self.titles = np.arange(len(self.coll_all))
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        self.last_frame = len(self.coll_all)-1
        self.line = plt.imshow(self.coll_all[self.first_frame])
        self.ax.set_title(f"{self.titles[self.first_frame]}")

        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)

        self.sframe = Slider(self.axframe, 'Frame', self.first_frame,
                             self.last_frame, valinit=self.first_frame,
                             valfmt='%d', valstep=1)
        # calls the update function when changing the slider position
        self.sframe.on_changed(self.update)

        # Calling the press function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        # self.fig.show()

    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        img = self.coll_all[frame]
        self.line.set_data(img)
        self.ax.set_title(f"{self.titles[frame]}")
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use left and right arrow keys to scroll through frames one by one"""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.coll_all)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.coll_all[new_frame]
        self.line.set_data(img)
        self.ax.set_title(f"{self.titles[new_frame]}")
        self.fig.canvas.draw_idle()


# %%

class AllMaps(object):
    """Rapidly visualize maps of Raman spectra.

    You can also choose to visualize the map and plot the
    corresponding component side by side if you set the
    "components" parameter.

    Parameters
    ----------
    input_spectra : xr.DataArray
    components: 2D ndarray
        The most evident use-case would be to help visualize the decomposition
        results from PCA or NMF. In this case, the function will plot the
        component with the corresponding map visualization of the given
        components' presence in each of the points in the map.
        So, in this case, your input_spectra would be for example
        the matrix of components' contributions in each spectrum,
        while the "components" array will be your actual components.
        In this case you can ommit your sigma values or set them to
        something like np.arange(n_components)
    components_sigma: 1D ndarray
        in the case explained above, this would be the actual wavenumbers
    **kwargs: dict
        can only take 'title' as a key for the moment

    Returns
    -------
    The interactive visualization.
    (you can scroll through sigma values with a slider,
     or using left/right keyboard arrows)
    """

    def __init__(self, input_spectra, sigma=None, components=None,
                 components_sigma=None, var=None,col_lim=None,
                 line=None,ax2=None,ax=None,im=None,fig=None,**kwargs):

        if isinstance(input_spectra, xr.DataArray):
            shape = input_spectra.attrs["ScanShape"]
            self.map_spectra = input_spectra.data.reshape(shape + (-1,))
            self.sigma = input_spectra.shifts.data
        else:
            self.map_spectra = input_spectra
            self.sigma = sigma
            if sigma is None:
                self.sigma = np.arange(self.map_spectra.shape[-1])
        assert self.map_spectra.shape[-1] == len(
                self.sigma), "Check your Ramans shifts array"

        self.first_frame = 0
        self.last_frame = len(self.sigma)-1
        self.line = line
        self.ax2 = ax2
        self.ax = ax
        self.im = im
        self.fig = fig

        self.components = components
        self.var = var
        self.col_lim = col_lim
        if self.components is not None:
            if components_sigma is None:
                self.components_sigma = np.arange(components.shape[-1])
            else:
                self.components_sigma = components_sigma
            self.fig, (self.ax2, self.ax, self.cbax) = plt.subplots(
                ncols=3, gridspec_kw={'width_ratios': [40, 40, 1]})
            self.cbax.set_box_aspect(
                40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
        else:
            self.fig, (self.ax, self.cbax) = plt.subplots(
                ncols=2, gridspec_kw={'width_ratios': [40, 1]})
            self.cbax.set_box_aspect(
                40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
            # self.cbax = self.fig.add_axes([0.92, 0.3, 0.03, 0.48])
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)

        self.im = self.ax.imshow(self.map_spectra[:, :, 0], interpolation='gaussian',cmap="rainbow")
        if self.col_lim is None :
            self.im.set_clim(np.percentile(self.map_spectra[:, :, 0], [1, 99]))
        else :
            self.im.set_clim(self.col_lim[0], self.col_lim[1])
        if self.components is not None:
            self.line, = self.ax2.plot(
                self.components_sigma, self.components[0])
            self.ax2.set_box_aspect(
                self.map_spectra.shape[0]/self.map_spectra.shape[1])
            if self.var is not None :
                self.ax2.set_title("Component " +str(0)+ " : " + str(self.var[0]) +"%")
            else :
                self.ax2.set_title(f"Component {0}")
        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes(
            [0.15, 0.05, 0.7, 0.03], facecolor=self.axcolor)
        self.axnbre = self.fig.add_axes([0.15, 0.1, 0.1, 0.03])

        self.sframe = Slider(self.axframe, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)
        self.nbre_box = TextBox(self.axnbre, 'S.No :')

        self.my_cbar = mpl.colorbar.Colorbar(self.cbax, self.im)

        # calls the "update" function when changing the slider position
        self.sframe.on_changed(self.update)
        self.nbre_box.on_submit(self.submit)
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        # self.fig.show()

    def titled(self, frame):
        if self.components is None:
            if self.title is None:
                self.ax.set_title(f"Raman shift = {self.sigma[frame]:.1f}cm⁻¹")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")
        else:
            if self.var is not None :
                self.ax2.set_title("Component " + str(frame) + " : " + str(np.round(self.var[frame],3)) + "%")
            else :
                self.ax2.set_title(f"Component {frame}")
            if self.title is None:
                self.ax.set_title(f"Component n°{frame} contribution")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")
                
    def submit(self,nbre):
        frame = int(nbre)
        self.sframe.set_val(frame)
        img = self.map_spectra[:, :, frame]
        self.im.set_data(img)
        if self.col_lim is None :
            self.im.set_clim(np.percentile(img, [1, 99]))
        else :
            self.im.set_clim(self.col_lim[0], self.col_lim[1])
        #self.im.set_clim(np.percentile(img, [1, 99]))
        self.titled(frame)
        if self.components is not None:
            self.line.set_ydata(self.components[frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()

    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        img = self.map_spectra[:, :, frame]
        self.im.set_data(img)
        if self.col_lim is None :
            self.im.set_clim(np.percentile(img, [1, 99]))
        else :
            self.im.set_clim(self.col_lim[0], self.col_lim[1])
        #self.im.set_clim(np.percentile(img, [1, 99]))
        #self.im.set_clim(0,1)
        if self.components is not None:
            self.line.set_ydata(self.components[frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use left and right arrow keys to scroll through frames one by one."""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.sigma)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.map_spectra[:, :, new_frame]
        self.im.set_data(img)
        if self.col_lim is None :
            self.im.set_clim(np.percentile(img, [1, 99]))
        else :
            self.im.set_clim(self.col_lim[0], self.col_lim[1])
        #self.im.set_clim(np.percentile(img, [1, 99]))
        #self.im.set_clim(0,1)
        self.titled(new_frame)
        if self.components is not None:
            self.line.set_ydata(self.components[new_frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()

# %%


class ShowSpectra(object):
    """Rapidly visualize Raman spectra.

    Imortant: Your spectra can either be a 2D ndarray
    (1st dimension is for counting the spectra,
    the 2nd dimension is for the intensities)
    And that would be the standard use-case, But:
    Your spectra can also be a 3D ndarray,
    In which case the last dimension is used to store additional spectra
    (for the same pixel)
    Fo example, you can store spectra, the baseline
    and the corrected spectra all together.
    Parameters:
    -----------
    input_spectra = xr.DataArray or numpy ndarray
        in the latter case, you can provide multiple spectra stacked along the
        last axis
    sigma:
        what's on the x-axis, optional
    title: str or iterable of the same length as the spectra, optional
    labels: list of labels

    Returns
    -------
    The interactive visualization.\n
    (you can scroll through the spectra with a slider,
     or using left/right keyboard arrows)

    Note:
        When there's only one spectrum to visualize, it bugs.
    """

    def __init__(self, input_spectra, sigma=None, **kwargs):

        if isinstance(input_spectra, xr.DataArray):
            self.my_spectra = input_spectra.data
            self.sigma = input_spectra.shifts.data
        else:
            self.my_spectra = input_spectra
            if sigma is None:
                if self.my_spectra.ndim == 1:
                    self.sigma = np.arange(len(self.my_spectra))
                else:
                    self.sigma = np.arange(self.my_spectra.shape[1])
            else:
                self.sigma = sigma
        if self.my_spectra.ndim == 1:
            self.my_spectra = self.my_spectra[np.newaxis, :, np.newaxis]
        if self.my_spectra.ndim == 2:
            self.my_spectra = self.my_spectra[:, :, np.newaxis]

        assert self.my_spectra.shape[1] == len(self.sigma),\
               "Check your Raman shifts array. The dimensions " + \
               f"of your spectra ({self.my_spectra.shape[1]}) and that of " + \
               f"your Ramans shifts ({len(self.sigma)}) are not the same."

        self.first_frame = 0
        self.last_frame = len(self.my_spectra)-1
        self.fig, self.ax = plt.subplots()
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)
        self.label = kwargs.get('labels', [None])
        if (not hasattr(self.label[0], '__iter__')\
           or len(self.label[0]) != self.my_spectra.shape[0]\
           or isinstance(self.label[0], str))\
           and self.label[0] is not None:
            self.label = [self.label]*self.my_spectra.shape[0]
            self.spectrumplot = self.ax.plot(self.sigma, self.my_spectra[0],
                                             label=self.label[0])
        else:
            self.spectrumplot = self.ax.plot(self.sigma, self.my_spectra[0])

        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes([0.15, 0.05, 0.7, 0.03])
        self.axnbre = self.fig.add_axes([0.15, 0.1, 0.1, 0.03])
        # self.axframe.plot(self.sigma, np.median(self.my_spectra, axis=0))
        if len(self.my_spectra) > 1:
            self.sframe = Slider(self.axframe, 'N°',
                                 self.first_frame, self.last_frame, valfmt='%d',
                                 valinit=self.first_frame, valstep=1)
            self.nbre_box = TextBox(self.axnbre, 'S.No : ')
            # calls the "update" function when changing the slider position
            self.sframe.on_changed(self.update)
            self.nbre_box.on_submit(self.submit)
            # Calling the "press" function on keypress event
            # (only arrow keys left and right work)
            self.fig.canvas.mpl_connect('key_press_event', self.press)
        else:
            self.axframe.axis('off')
        # self.fig.show()

    def titled(self, frame):
        if self.title is None:
            self.ax.set_title(f"Spectrum N° {frame} /{self.last_frame + 1}")
        elif isinstance(self.title, str):
            self.ax.set_title(f"{self.title} n°{frame}")
        elif hasattr(self.title, '__iter__'):
            self.ax.set_title(f"{self.title[frame]}")
        if self.label[0] is not None:
            handles, _ = self.ax.get_legend_handles_labels()
            self.ax.legend(handles, self.label[frame])
            
            
    def submit(self,nbre) :
        frame = int(nbre)
        self.sframe.set_val(frame)
        current_spectrum = self.my_spectra[frame]
        for i, line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.titled(frame)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()


    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        current_spectrum = self.my_spectra[frame]
        for i, line in enumerate(self.spectrumplot):
            # self.ax.cla()
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use left and right arrow keys to scroll through frames one by one."""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < self.last_frame:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        current_spectrum = self.my_spectra[new_frame]
        for i, line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.titled(new_frame)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

# %%


class NavigationButtons(object):
    """Interactivly visualize multispectral data.

    Navigate trough your spectra by simply clicking on the navigation buttons.

    Parameters
    ----------
        sigma: 1D ndarray
            1D numpy array of your x-values (raman shifts, par ex.)
        spectra: 2D or 3D ndarray
            3D or 2D ndarray of shape (n_spectra, len(sigma), n_curves).
            The last dimension may be ommited it there is only one curve
            to be plotted for each spectra).
        autoscale: bool
            determining if you want to adjust the scale to each spectrum
        title: str
            The initial title describing where the spectra comes from
        label: list
            A list explaining each of the curves. len(label) = n_curves

    Output
    ------
        matplotlib graph with navigation buttons to cycle through spectra

    Example
    -------
        Let's say you have a ndarray containing 10 spectra,
        and let's suppose each of those spectra contains 500 points.

        >>> my_spectra.shape
        (10, 500)
        >>> sigma.shape
        (500, )

        Then let's say you show the results of baseline substraction.

        >>> my_baseline[i] = baseline_function(my_spectra[i])
        your baseline should have the same shape as your initial spectra.
        >>> multiple_curves_to_plot = np.stack(
                (my_spectra, my_baseline, my_spectra - my_baseline), axis=-1)
        >>> NavigationButtons(sigma, multiple_curves_to_plot)
    """
    ind = 0

    def __init__(self, spectra, sigma=None, autoscale_y=False, title='Spectrum',
                 label=False, as_series=False, axis="shorter", **kwargs):

        if as_series:
            spectra = pp.as_series(spectra)
        if isinstance(spectra, xr.DataArray):
            sigma = spectra.shifts.data
            spectra = spectra.data
        elif sigma==None:
            sigma = np.arange(spectra.shape[-1])

        if len(spectra.shape) == 2:
            self.s = spectra[:, :, np.newaxis]
        elif len(spectra.shape) == 3:
            self.s = spectra
        else:
            raise ValueError("Check the shape of your spectra.\n"
                             "It should be (n_spectra, n_points, n_curves)\n"
                             "(this last dimension might be ommited"
                             "if it's equal to one)")
        self.y_autoscale = autoscale_y
        self.n_spectra = self.s.shape[0]
        if isinstance(title, list) or isinstance(title, np.ndarray):
            if len(title) == spectra.shape[0]:
                self.title = title
            else:
                raise ValueError(f"you have {len(title)} titles,\n"
                                 f"but you have {len(spectra)} spectra")
        else:
            self.title = [title]*self.n_spectra

        self.sigma = sigma
        if label:
            if len(label) == self.s.shape[2]:
                self.label = label
            else:
                warn(
                    "You should check the length of your label list.\n"
                    "Falling on to default labels...")
                self.label = ["Curve n°"+str(numb)
                              for numb in range(self.s.shape[2])]
        else:
            self.label = ["Curve n°"+str(numb)
                          for numb in range(self.s.shape[2])]

        self.figr, self.axr = plt.subplots(**kwargs)
        self.axr.set_title(f'{title[0]}')
        self.figr.subplots_adjust(bottom=0.2)
        # l potentially contains multiple lines
        self.line = self.axr.plot(self.sigma, self.s[0], lw=2, alpha=0.7)
        self.axr.legend(self.line, self.label)
        self.axprev1000 = plt.axes([0.097, 0.05, 0.1, 0.04])
        self.axprev100 = plt.axes([0.198, 0.05, 0.1, 0.04])
        self.axprev10 = plt.axes([0.299, 0.05, 0.1, 0.04])
        self.axprev1 = plt.axes([0.4, 0.05, 0.1, 0.04])
        self.axnext1 = plt.axes([0.501, 0.05, 0.1, 0.04])
        self.axnext10 = plt.axes([0.602, 0.05, 0.1, 0.04])
        self.axnext100 = plt.axes([0.703, 0.05, 0.1, 0.04])
        self.axnext1000 = plt.axes([0.804, 0.05, 0.1, 0.04])

        self.bprev1000 = Button(self.axprev1000, 'Prev.1000')
        self.bprev1000.on_clicked(self.prev1000)
        self.bprev100 = Button(self.axprev100, 'Prev.100')
        self.bprev100.on_clicked(self.prev100)
        self.bprev10 = Button(self.axprev10, 'Prev.10')
        self.bprev10.on_clicked(self.prev10)
        self.bprev = Button(self.axprev1, 'Prev.1')
        self.bprev.on_clicked(self.prev1)
        self.bnext = Button(self.axnext1, 'Next1')
        self.bnext.on_clicked(self.next1)
        self.bnext10 = Button(self.axnext10, 'Next10')
        self.bnext10.on_clicked(self.next10)
        self.bnext100 = Button(self.axnext100, 'Next100')
        self.bnext100.on_clicked(self.next100)
        self.bnext1000 = Button(self.axnext1000, 'Next1000')
        self.bnext1000.on_clicked(self.next1000)

    def update_data(self):
        _i = self.ind % self.n_spectra
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next1(self, event):
        self.ind += 1
        self.update_data()

    def next10(self, event):
        self.ind += 10
        self.update_data()

    def next100(self, event):
        self.ind += 100
        self.update_data()

    def next1000(self, event):
        self.ind += 1000
        self.update_data()

    def prev1(self, event):
        self.ind -= 1
        self.update_data()

    def prev10(self, event):
        self.ind -= 10
        self.update_data()

    def prev100(self, event):
        self.ind -= 100
        self.update_data()

    def prev1000(self, event):
        self.ind -= 1000
        self.update_data()

# %%


class ShowSelected(object):
    """Select a span and plot a map of a chosen function in that span.
    Right-Click (or middle-click) on the image to see the spectrum
    corresponding to that pixel.

    To be used for visual exploration of the maps.
    The lower part of the figure contains the spectra you can scroll through
    using the slider just beneath the spectra.
    You can use your mouse to select a zone in the spectra and a map plot
    should appear in the upper part of the figure.
    On the left part of the figure you can select what kind of function
    you want to apply on the selected span.
    ['area', 'barycenter_x', 'max_value', 'peak_position', 'peak_ratio']
    """

    def __init__(self, input_spectra, x=None, interpolation='gaussian', **kwargs):

        self.da = input_spectra.copy()
        self.interp = interpolation
        self.f_names = ['area',
                        'barycenter_x',
                        'max_value',
                        'peak_position',
                        'peak_ratio']
        self.xmin = self.da.shifts.data.min()
        self.xmax = self.da.shifts.data.max()
        # Get some basic info on the spectra:
        self.nshifts = self.da.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da.attrs["ScanShape"]
        self.scan_type = self.da.attrs["MeasurementType"]
        self.cmap = kwargs.pop("cmap", "viridis")
        self.file_location = kwargs.pop("file_location", "./")
        self.extent = kwargs.pop("extent", "full")
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

        # Preparing the plots:
        figsize = kwargs.pop("figsize", (14, 8))
        self.fig = plt.figure(figsize=figsize, **kwargs)
        # Add all the axes:
        self.aximg = self.fig.add_axes([.21, .3, .74, .6]) # main frame
        self.axspectrum = self.fig.add_axes([.05, .075, .9, .15])
        self.axradio = self.fig.add_axes([.05, .3, .1, .6])
        self.axreduce = self.fig.add_axes([.05, .275, .1, .05])
        self.axabsscale = self.fig.add_axes([.05, .22, .1, .05])
        self.axsave = self.fig.add_axes([.05, .905, .1, .08])
        self.axscroll = self.fig.add_axes([.05, .02, .9, .02])
        self.axradio.axis('off')
        self.axreduce.axis('off')
        self.axabsscale.axis('off')
        # self.axsave.axis('off')
        self.axscroll.axis('off')
        self.first_frame = 0
        if self.scan_type != "Single":
            # Slider to scroll through spectra:
            self.last_frame = len(self.da.data)-1
            self.sframe = Slider(self.axscroll, 'S.N°',
                                 self.first_frame, self.last_frame,
                                 valinit=self.first_frame, valfmt='%d', valstep=1)
            self.sframe.on_changed(self.scroll_spectra)
        # Show the spectrum:
        self.spectrumplot, = self.axspectrum.plot(self.da.shifts.data,
                                                  self.da.data[self.first_frame])
        self.titled(self.axspectrum, self.first_frame)
        self.vline = None
        self.func = "max"  # Default function
        self.xmin = self.da.shifts.data.min()
        self.xmax = self.da.shifts.data.max()
        self.span = SpanSelector(self.axspectrum, self.onselect, 'horizontal',
                                 useblit=True, span_stays=True,
                                 rectprops=dict(alpha=0.5,
                                                facecolor='tab:blue'))
        self.func_choice = RadioButtons(self.axradio, self.f_names)
        self.func_choice.on_clicked(self.determine_func)
        self.reduced_button = CheckButtons(self.axreduce, ["reduced"])
        self.reduced_button.on_clicked(self.is_reduced)
        self.reduced = self.reduced_button.get_status()[0]
        self.abs_scale_button = CheckButtons(self.axabsscale, ["abs. scale"])
        self.abs_scale_button.on_clicked(self.is_absolute_scale)
        self.absolute_scale = self.abs_scale_button.get_status()[0]
        self.save_button = Button(self.axsave, "Save Image")
        self.save_button.on_clicked(self.save_image)
        self.func = self.func_choice.value_selected
        # Plot the empty "image":
        if self.scan_type == "Map":
            self.imup = self.aximg.imshow(cc.calculate_ss(
                                        self.func,
                                        self.da),
                                        interpolation=self.interp,
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

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # self.fig.show()
    def is_reduced(self, label):
        self.reduced = self.reduced_button.get_status()[0]
        self.draw_img()

    def is_absolute_scale(self, label):
        self.absolute_scale = self.abs_scale_button.get_status()[0]
        self.draw_img()

    def increment_filename(self, filename):

        if not path.exists(filename):
            return filename
        file, extension = path.splitext(filename)
        counter = file[-3:]
        if counter.isnumeric():
            file = file[:-3]
            counter = f"{int(counter) + 1:03d}"
        else:
            counter = "001"

        filename = file + counter + extension
        return self.increment_filename(filename)

    def full_extent(self, padx=0.2, pady=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""

        # ax.figure.canvas.draw()
        items = self.aximg.get_xticklabels() + self.aximg.get_yticklabels()
        items += [self.aximg, self.aximg.title, self.aximg.xaxis.label, self.aximg.yaxis.label]
        items += [self.aximg.title]
        bbox = mpl.transforms.Bbox.union([item.get_window_extent() for item in items])
        dx, dy = self.aximg.transAxes.transform((0.15, 0.025)) -\
                 self.aximg.transAxes.transform((0, 0))
        return bbox.expanded(1.0 + padx, 1.0 + pady).translated(dx, dy)

    def save_image(self, event):
        filename = path.join(self.file_location, self.func+"_000.png")
        figsize = self.fig.get_size_inches()
        self.fig.set_size_inches(16, 10, forward=False)
        if self.extent == "full":
            myextent = self.full_extent().transformed(self.fig.dpi_scale_trans.inverted())
        elif self.extent == "tight":
            myextent = self.aximg.get_tightbbox(self.fig.canvas.renderer).transformed(self.fig.dpi_scale_trans.inverted())
        else:
            warn("The 'extent' keyword must be one of [\"full\", \"tight\"]")
        savename = self.increment_filename(filename)
        # print(savename)
        self.fig.savefig(savename, bbox_inches=myextent, transparent=True, dpi=120)
        self.fig.set_size_inches(figsize)


    def onclick(self, event):
        """Right-Clicking on a pixel will show the spectrum
        corresponding to that pixel on the bottom plot"""
        if event.inaxes == self.aximg:
            if event.button != 1:
                x_pos = round(event.xdata)
                y_pos = round(event.ydata)
                if isinstance(self.imup, mpl.image.AxesImage): # if image
                    if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >= 0:
                        broj = round(y_pos * self.nx + x_pos)
                        self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                elif isinstance(self.imup, mpl.lines.Line2D):
                    broj = x_pos
                    self.sframe.set_val(broj)
                    self.scroll_spectra(broj)
            else:
                pass

    def determine_func(self, label):
        "Recover the function name from radio button clicked"""
        self.func = label
        self.draw_img()

    def onselect(self, xmin, xmax):
        """When you select a region of the spectra."""
        self.xmin = xmin
        self.xmax = xmax
        if self.vline:
            self.axspectrum.lines.remove(self.vline)
            self.vline = None
        self.draw_img()

    def draw_img(self):
        """Draw/update the image."""
        # calculate the function:
        whole_data = cc.calculate_ss(self.func, self.da, self.xmin, self.xmax,
                                     is_reduced=self.reduced)
        naj = np.max(whole_data)
        img = whole_data / naj 
        if self.scan_type == "Map":
            self.imup.set_data(img)
            limits = np.percentile(img, [0, 100])
            if self.absolute_scale:
                limits = [0, limits[-1]]
            self.imup.set_clim(limits)
            self.cbar.mappable.set_clim(*limits)
        elif self.scan_type == 'Single':
            self.imup.set_text(img[0][0])
        else:
            self.imup.set_ydata(img.squeeze())
            self.aximg.relim()
            self.aximg.autoscale_view(None, False, True)

        self.aximg.set_title(f"Calculated {'reduced'*self.reduced} {self.func} "
                             f"between {self.xmin:.1f} and {self.xmax:.1f} cm-1"
                             f" / {naj:.2f}\n")
        self.fig.canvas.draw_idle()

    def scroll_spectra(self, val):
        """Use the slider to scroll through individual spectra"""
        frame = int(self.sframe.val)
        current_spectrum = self.da.data[frame]
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()

    def titled(self, ax, frame):
        """Set the title for the spectrum plot"""
        if self.scan_type == "Single":
            new_title = self.da.attrs["Title"]
        elif self.scan_type == "Series":
            new_title = f"Spectrum @ {np.datetime64(self.da.Time.data[frame], 's')}"
        else:
            new_title = "Spectrum @ "+\
                f"{self.da.RowCoord}: {self.da[self.da.RowCoord].data[frame]}"\
              + f"; {self.da.ColCoord}: {self.da[self.da.ColCoord].data[frame]}"
        ax.set_title(new_title, x=0.28)


class FindBaseline(object):
    """Visualy adjust parameters for the baseline.

    Parameters
    ----------
    my_spectra: 2D ndarray

    Returns
    -------
    The interactive graph facilitating the parameter search.
    You can later recover the parameters with:
        MyFindBaselineInstance.p_val
        MyFindBaselineInstance.lam_val

    Note that you can use the same function for smoothing
    (by setting the `p_val` to 0.5 and `lam_val` to some "small" value (like 13))
    """

    def __init__(self, my_spectra, sigma=None, **kwargs):
        if my_spectra.ndim == 1:
            self.my_spectra = my_spectra[np.newaxis, :]
        else:
            self.my_spectra = my_spectra
        if sigma is None:
            self.sigma = np.arange(my_spectra.shape[1])
        else:
            assert my_spectra.shape[-1] == len(
                sigma), "Check your Raman shifts array"
            self.sigma = sigma

        self.nb_spectra = len(self.my_spectra)
        self.current_spectrum = self.my_spectra[0]
        self.title = kwargs.get('title', None)
        self.p_val = 5e-5
        self.lam_val = 1e5
        self.lam1_val = 1e-2

        self.fig = plt.figure(figsize=(14, 10))
        # Add all the axes:
        self.ax = self.fig.add_axes([.2, .15, .75, .8])  # [left, bottom, width, height]
        self.axpslider = self.fig.add_axes([.05, .15, .02, .8], yscale='log')
        self.axlamslider = self.fig.add_axes([.1, .15, .02, .8], yscale='log')
        self.axlam1slider = self.fig.add_axes([.15, .15, .02, .8], yscale='log')
        if self.nb_spectra > 1:  # scroll through spectra if there are many
            self.axspectrumslider = self.fig.add_axes([.2, .05, .75, .02])
            self.spectrumslider = Slider(self.axspectrumslider, 'Frame',
                                         0, self.nb_spectra-1,
                                         valinit=0, valfmt='%d', valstep=1)
            self.spectrumslider.on_changed(self.spectrumupdate)

        self.pslider = Slider(self.axpslider, 'p-val',
                              1e-10, 1, valfmt='%.2g',
                              valinit=self.p_val,
                              orientation='vertical')
        self.lamslider = Slider(self.axlamslider, 'lam-val',
                                .1, 1e10, valfmt='%.2g',
                                valinit=self.lam_val,
                                orientation='vertical')
        self.lam1slider = Slider(self.axlam1slider, 'lam1-val',
                                1e-4, 10, valfmt='%.2g',
                                valinit=self.lam1_val,
                                orientation='vertical')
        self.pslider.on_changed(self.blupdate)
        self.lamslider.on_changed(self.blupdate)
        self.lam1slider.on_changed(self.blupdate)

        self.spectrumplot, = self.ax.plot(self.sigma, self.current_spectrum,
                                          label="original spectrum")
        self.bl = cc.baseline_ials(self.current_spectrum, p=self.p_val,
                                  lam=self.lam_val,lam1=self.lam1_val,niter=50)
        self.blplot, = self.ax.plot(self.sigma, self.bl, label="baseline")
        self.corrplot, = self.ax.plot(self.sigma,
                                      self.current_spectrum - self.bl,
                                      label="corrected_plot")
        self.ax.legend()
        self.titled(0)

        # self.fig.show()

    def titled(self, frame):
        if self.title is None:
            self.ax.set_title(f"Spectrum N° {frame} /{self.nb_spectra}")
        else:
            self.ax.set_title(f"{self.title} n°{frame}")

    def spectrumupdate(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.spectrumslider.val)
        self.current_spectrum = self.my_spectra[frame]
        self.spectrumplot.set_ydata(self.current_spectrum)
        self.blupdate(val)
        self.ax.relim()
        self.ax.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def blupdate(self, val):
        self.p_val = self.pslider.val
        self.lam_val = self.lamslider.val
        self.lam1_val = self.lam1slider.val
        self.bl = cc.baseline_ials(self.current_spectrum, p=self.p_val,
                                  lam=self.lam_val,lam1=self.lam1_val)
        self.blplot.set_ydata(self.bl)
        self.corrplot.set_ydata(self.current_spectrum - self.bl)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

        
        
        
class FindBaseline1(object):
    """Visualy adjust parameters for the baseline.

    Parameters
    ----------
    my_spectra: 2D ndarray

    Returns
    -------
    The interactive graph facilitating the parameter search.
    You can later recover the parameters with:
        MyFindBaselineInstance.p_val
        MyFindBaselineInstance.lam_val

    Note that you can use the same function for smoothing
    (by setting the `p_val` to 0.5 and `lam_val` to some "small" value (like 13))
    """

    def __init__(self, my_spectra, sigma=None, **kwargs):
        if my_spectra.ndim == 1:
            self.my_spectra = my_spectra[np.newaxis, :]
        else:
            self.my_spectra = my_spectra
        if sigma is None:
            self.sigma = np.arange(my_spectra.shape[1])
        else:
            assert my_spectra.shape[-1] == len(
                sigma), "Check your Raman shifts array"
            self.sigma = sigma

        self.nb_spectra = len(self.my_spectra)
        self.current_spectrum = self.my_spectra[0]
        self.title = kwargs.get('title', None)
        self.lam_val = 1e5

        self.fig = plt.figure(figsize=(14, 10))
        # Add all the axes:
        self.ax = self.fig.add_axes([.2, .15, .75, .8])  # [left, bottom, width, height]
        self.axlamslider = self.fig.add_axes([.1, .15, .02, .8], yscale='log')
        if self.nb_spectra > 1:  # scroll through spectra if there are many
            self.axspectrumslider = self.fig.add_axes([.2, .05, .75, .02])
            self.spectrumslider = Slider(self.axspectrumslider, 'Frame',
                                         0, self.nb_spectra-1,
                                         valinit=0, valfmt='%d', valstep=1)
            self.spectrumslider.on_changed(self.spectrumupdate)

        self.lamslider = Slider(self.axlamslider, 'lam-val',
                                .1, 1e10, valfmt='%.2g',
                                valinit=self.lam_val,
                                orientation='vertical')
        self.lamslider.on_changed(self.blupdate)

        self.spectrumplot, = self.ax.plot(self.sigma, self.current_spectrum,
                                          label="original spectrum")
        self.bl = cc.baseline_arpls(self.current_spectrum,
                                  lam=self.lam_val,niter=50)
        self.blplot, = self.ax.plot(self.sigma, self.bl, label="baseline")
        self.corrplot, = self.ax.plot(self.sigma,
                                      self.current_spectrum - self.bl,
                                      label="corrected_plot")
        self.ax.legend()
        self.titled(0)

        # self.fig.show()

    def titled(self, frame):
        if self.title is None:
            self.ax.set_title(f"Spectrum N° {frame} /{self.nb_spectra}")
        else:
            self.ax.set_title(f"{self.title} n°{frame}")

    def spectrumupdate(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.spectrumslider.val)
        self.current_spectrum = self.my_spectra[frame]
        self.spectrumplot.set_ydata(self.current_spectrum)
        self.blupdate(val)
        self.ax.relim()
        self.ax.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()
    
        
        
    def press(self, event):
        
        """Use left and right arrow keys to scroll through frames one by one."""
        frame = int(self.spectrumslider.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < self.last_frame:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.spectrumslider.set_val(new_frame)
        self.current_spectrum = self.my_spectra[new_frame]
        self.spectrumplot.set_ydata(self.current_spectrum)
        self.blupdate(val)
        self.ax.relim()
        self.ax.autoscale_view()
        self.titled(new_frame)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def blupdate(self, val):
        self.lam_val = self.lamslider.val
        self.bl = cc.baseline_arpls(self.current_spectrum,
                                  lam=self.lam_val)
        self.blplot.set_ydata(self.bl)
        self.corrplot.set_ydata(self.current_spectrum - self.bl)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        
        
        


class PCA_clean(object):
    """Choose which pca components to use to reconstruct the spectra with."""

    def __init__(self, da, n_components):
        self.spectra = da.data
        # self.original_spectra = np.copy(da.data)
        self.mean_spectra = np.mean(self.spectra, axis=-1, keepdims=True)
        self.mean_spectra2 = np.mean(self.spectra, axis=0, keepdims=False)
        self.x_values = da.shifts.data
        self.shape = da.attrs["ScanShape"]# + (-1, )
        self.n = n = n_components
        self.x_ind = 0
        self.point_ind = 0
        pca = decomposition.PCA(self.n)
        # Seems to produce better interprable results:
        pca_fit = pca.fit(self.spectra.T)
        self.components = pca_fit.transform(self.spectra.T).T
        self.coeffs = pca_fit.components_.T

        # Construct the figure:
        self.fig = plt.figure()
        # Add all the axes and the buttons:
        self.aximg = self.fig.add_axes([.31, .3, .64, .6])  # image
        self.img_data = self.aximg.imshow(self.mean_spectra.reshape(self.shape))
        limits = (1, 1 + np.max(np.ptp(self.spectra, axis=0)))
        self.img_data.set_clim(limits)
        self.axspectrum = self.fig.add_axes([.05, .075, .9, .15])  # spectrum
        self.spec_data, = self.axspectrum.plot(self.x_values, self.mean_spectra2)
        self.axspectrum.vline_present = False
        # self.axscroll = self.add_naked(self.fig, [.05, .02, .9, .02])  # for the slider?
        # self.slider = Slider(self.axscroll, r'$\lambda$',
        #                      self.x_values.min(), self.x_values.max(),
        #                      valinit = self.x_values.min(), valstep=1)
        # self.slider.on_changed(self.slider_change)
        # for the "all" button:
        self.axall = self.add_naked(self.fig, [.05, .91, .08, .03])
        self.button_all = Button(self.axall, "All")
        # for the "none" button
        self.axnone = self.add_naked(self.fig, [.15, .91, .08, .03])
        self.button_none = Button(self.axnone, "None")
        # for the checkboxes (to select components):
        self.axchck = self.add_naked(self.fig, [0.01, 0.3, 0.04, 0.6])
        self.color_spines(self.axchck, "w")
        self.comp_labels = ["c"+str(i) for i in range(self.n)]
        self.check = CheckButtons(self.axchck, self.comp_labels, [True]*self.n)
        # Add the axes and plot the components, also adjust the checkbuttons:
        self.axcomp = []
        # print(self.components.shape)
        for i, comp in enumerate(self.components):
            self.axcomp.append(self.add_naked(self.fig,
                                              [.05, .9-(i+1)*.6/n, .18, .6/n]))
            self.axcomp[i].plot(self.x_values, comp)
            self.axcomp[i].vline_present = False
            # Now we need to adjust the positions of the checkboxes (rectangles,
            # lines and labels) for them to be aligned with axcomp axes:
            y = 1-(i+.7)/n
            height = .4/n
            self.check.rectangles[i].set(x=.65, y=y, width=.3, height=height)
            self.check.lines[i][0].set_data([.65, .95], [y+height, y])
            self.check.lines[i][1].set_data([.65, .95], [y, y+height])
            self.check.labels[i].set_position((.02, y+height/2))
        # Add the interactivity
        self.button_all.on_clicked(self.select_all)
        self.button_none.on_clicked(self.select_none)
        self.check.on_clicked(self.just_checking)
        self.spec_cursor = MultiCursor(self.fig.canvas, [self.axspectrum],
                                     color='r', lw=1)
        self.pos_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                   self.draw_vline)
        self.img_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                   self.onclick)

    def add_naked(self, fig, pos):
        """Add new "naked" axes to the figure at the given position `pos`.

        Parameters:
        -----------
        pos: list of floats
            [left, bottom, width, height]
        """
        ax = fig.add_axes(pos)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(bottom=False, left=False)
        return ax

    def color_spines(self, ax, color):
        """Change the color of all spines for an axes"""
        for i in ax.spines:
            ax.spines[i].set(edgecolor=color)#, linewidth=3)

    def switch_spine_color(self, ax):
        """Switch spines color from red to green (and vice versa)."""
        for i in ax.spines:
            boja = np.array(ax.spines[i]._edgecolor, dtype=bool)
            nova_boja = boja + np.array([1, 0, 1, 0], dtype=bool)
            ax.spines[i].set(edgecolor=nova_boja)

    def select_all(self, event):
        """Set all checkboxes to active"""
        for i, s in enumerate(self.check.get_status()):
            if not s:
                self.check.set_active(i)
        self.just_checking("all")

    def select_none(self, event):
        """Set all checkboxes to inactive"""
        for i, s in enumerate(self.check.get_status()):
            if s:
                self.check.set_active(i)
        self.just_checking("none")

    def just_checking(self, label):
        facecolors = ["lightgray", "white"]
        checked = self.check.get_status()
        if label in self.comp_labels:
            clicked_idx = self.comp_labels.index(label)
            color_idx = checked[clicked_idx] ^ False
            self.axcomp[clicked_idx].set_facecolor(facecolors[color_idx])
        else:
            for i, c in enumerate(checked):
                self.axcomp[i].set_facecolor(facecolors[c])
        # used_comps = components[checked]
        self.spectra = np.dot(self.coeffs[:, checked],
                              self.components[checked, :]) + self.mean_spectra
        self.draw_image()
        self.draw_spectrum()

    def draw_image(self):
            new_img_data = self.spectra[:, self.x_ind].reshape(self.shape)
            self.img_data.set_data(new_img_data)
            limits = np.percentile(new_img_data, (1, 99))
            self.img_data.set_clim(limits)
            self.fig.canvas.draw_idle()

    def draw_vline(self, event):
        if event.inaxes in [self.axspectrum]:
            x = event.xdata
            self.x_ind = np.nanargmin(np.abs(self.x_values - x))
        else:
            x = self.x_values[self.x_ind]
        if self.axspectrum.vline_present:
            self.axspectrum.lines.remove(self.axspectrum.vlineid)
        self.axspectrum.vlineid = self.axspectrum.axvline(x, c='g', lw=.5)
        self.axspectrum.vline_present = True
        for i in range(self.n):
            if self.axcomp[i].vline_present:
                self.axcomp[i].lines.remove(self.axcomp[i].vlineid)
            self.axcomp[i].vlineid = self.axcomp[i].axvline(x, c='g', lw=.5)
            self.axcomp[i].vline_present = True
        self.draw_image()

    def onclick(self, event):
        """Right-Clicking on a pixel will show the spectrum
        corresponding to that pixel on the bottom plot"""
        if event.inaxes == self.aximg:
            x_pos = round(event.xdata)
            y_pos = round(event.ydata)
            nx, ny = self.shape[1], self.shape[0]
            if event.button != 1:
                if x_pos <= nx and y_pos <= ny and x_pos * y_pos >= 0:
                    # print(x_pos, y_pos)
                    self.point_ind = round(y_pos * nx + x_pos)
                    self.draw_spectrum()

    # def slider_change(self, val):
    #     self.x_ind = np.nanargmin(np.abs(self.x_values - val))
    #     self.draw_vline(0)

    def draw_spectrum(self):
        self.spec_data.set_ydata(self.spectra[self.point_ind])
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.fig.canvas.draw_idle()


def draw_aggregated(da, style="dark_background", facecolor="black", note=False,
                    binning=np.geomspace, shading="auto", units="cm$^{{-1}}$",
                    cmap="inferno", add=False, **kwargs):
    """Draw aggregated spectra.
    It puts spectra into bins, color intensity reflects the number of spectra
    present in the given bin.

    Parameters:
        da: xr.DataArray
            Your spectra.
        n_bins: int
            The number of bins you want to use
        style:
            on ocoloredf matplotlib.pyplot.styles.available
        facecolor:
            matplotlib color
        binning:
            The function to use for binning (np.geomspace, np.linspace,...)
        add: bool
            Weather to add the spectra to an existing figure/axes
            or to draw new figure

    """

    def tocm_1(x, pos):
        """Add units to x_values"""
        nonlocal n
        if x < n:
            xx = f"{da.shifts.data[int(x)]:.0f}{units}"
        else:
            xx = ""
        return xx

    def restablish_zero(y, pos):
        """Restablishes zero values (removed because of log)."""
        nonlocal bins
        yind = int(y)
        if yind < len(bins):
            yy = f"{bins[int(y)] - 1:.2g}"
        else:
            yy = ""
        return yy

    def forward(x):
        return np.exp(x)

    def backward(x):
        return np.log(x)

    @mpl.ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x:.2f}"

    n = len(da.shifts)
    nm = kwargs.pop("n_bins", min(n, len(da.values)))
    da.values -= (np.min(da.values, axis=-1, keepdims=True) - 1) # We can't have zeros
    bins = binning(np.min(da.values), np.max(da.values), nm)
    binda = np.empty((n, nm), dtype=int)
    prd = []
    for i in range(n):
        bin_data = np.digitize(da.values[:, i], bins=bins, right=False)
        prebroj = np.bincount(bin_data, minlength=nm)
        prd.append(len(prebroj))
        binda[i] = prebroj[:nm]


    norm = kwargs.pop("norm", mpl.colors.LogNorm())
    my_title = kwargs.pop("title", "")
    alpha = kwargs.pop("alpha", .6)
    figsize = kwargs.pop("figsize", (18, 10))
    if add:
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    if style:
        plt.style.use(style)
    if facecolor:
        ax.set_facecolor(facecolor)
    plt.pcolormesh(binda.T, norm=norm, shading=shading, cmap=cmap, alpha=alpha)
    plt.title(my_title)
    # poz_y = plt.yticks()[0][1:-1].astype(int)
    # plt.yticks(poz_y, bins[poz_y].astype(int))
    # poz_x = plt.xticks()[0][1:-1].astype(int)
    # plt.xticks(poz_x, da.shifts.data[poz_x].astype(int))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(tocm_1))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(restablish_zero))
    # ax.yaxis.set_major_formatter(major_formatter)
    plt.ylabel("Intensity")
    # scaler = mpl.scale.FuncScale(plt.gca(), (forward, backward))
    # plt.yscale(scaler)
    if note:
        explanation = (f"NOTE: This plot shows all of the {len(da)}"
                       f" spectra binned in {nm} bins.\n"
                       f"The binning is done with {binning.__name__}")
        fig.supxlabel(f"{explanation:<50s}", fontsize=10)#, transform=plt.gca().transAxes)
    try:
        plt.colorbar(shrink=.5, label="Number of spectra in the bin.")
    except:
        pass
    # plt.show();

    
    
    
class ShowHca(object):
    
    def __init__(self, input_spectra, input_spectra1,spectra_hca, t, criterion, x=None, **kwargs):
        
        self.da1 = input_spectra1.copy()
        self.da = input_spectra.copy()
        self.hcarep = spectra_hca.copy()
        self.xmin = self.da.shifts.data.min()
        self.xmax = self.da.shifts.data.max()
        self.nshifts = self.da.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da.attrs["ScanShape"]
        self.scan_type = self.da.attrs["MeasurementType"]
        
        self.groupes = fcluster(self.hcarep, t=t, criterion = "distance")
        self.spectra_data = pd.DataFrame(self.da.data)
        self.clas_spectra = pd.DataFrame(self.groupes, index = self.spectra_data.index, columns = ["classe"])
        
        self.dico = dict()
        for i in range(1,max(self.groupes),1) : 
            for j in self.clas_spectra.classe :
                if i==j :
                    self.dico[f'classe {i}'] = list(self.clas_spectra[self.clas_spectra.classe==i].index)
        
        self.data_spectra_hat = pd.DataFrame(self.da1.data)
        self.data_clas_spectra = pd.concat([self.data_spectra_hat,self.clas_spectra],axis=1)
        
        try : 
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
        
        figsize = kwargs.pop("figsize", (14, 8))
        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.aximg = self.fig.add_axes([.05, .25, .4, .6])
        self.axspectrum = self.fig.add_axes([.5, .25, .4, .6])
        self.axscroll = self.fig.add_axes([.05, .15, .7, .02])
        self.first_frame = 0
        self.last_frame = len(self.da.data)-1
        
        self.sframe = Slider(self.axscroll, 'S.N°',
                                 self.first_frame, self.last_frame,
                                 valinit=self.first_frame, valfmt='%d', valstep=1)
        self.sframe.on_changed(self.scroll_spectra)
        self.spectrumplot, = self.axspectrum.plot(self.da.shifts.data,
                                                  self.da1.data[self.first_frame])
        self.titled(self.axspectrum, self.first_frame)
        
        self.imup = self.aximg.imshow(np.array(self.data_clas_spectra['classe']).reshape(self.ny,self.nx))
        self.aximg.set_xlabel(f"{self.xlabel}")
        self.aximg.xaxis.set_label_position('top')
        self.aximg.set_ylabel(f"{self.ylabel}")
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        
    def ret_data(self):
        #return pp.giveback_same(self.da, self.data_clas_spectra),max(self.groupes)
        return self.data_clas_spectra,max(self.groupes)
    
    def onclick(self, event):
        if event.inaxes == self.aximg:
            if event.button != 1:
                x_pos = round(event.xdata)
                y_pos = round(event.ydata)
                if isinstance(self.imup, mpl.image.AxesImage): # if image
                    if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >= 0:
                        broj = round(y_pos * self.nx + x_pos)
                        self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                elif isinstance(self.imup, mpl.lines.Line2D):
                    broj = x_pos
                    self.sframe.set_val(broj)
                    self.scroll_spectra(broj)
            else:
                pass
        
        
    def scroll_spectra(self, val):
        frame = int(self.sframe.val)
        current_spectrum = self.da1.data[frame]
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()
            
    def titled(self, ax, frame):
        if self.scan_type == "Single":
            new_title = self.da.attrs["Title"]
        elif self.scan_type == "Series":
            new_title = f"Spectrum @ {np.datetime64(self.da.Time.data[frame], 's')}"
        else:
            new_title = "Spectrum @ "+\
                f"{self.da.RowCoord}: {self.da[self.da.RowCoord].data[frame]}"\
                + f"; {self.da.ColCoord}: {self.da[self.da.ColCoord].data[frame]}"
        ax.set_title(new_title, x=0.28)
   
class ShowIHCA(object):
    
    
    
    
    
    def __init__(self, input_spectra, input_spectra1, nb_clas, x=None, **kwargs):
        
        self.da = input_spectra.copy()
        self.da1 = input_spectra1.copy()
        self.xmin = self.da1.shifts.data.min()
        self.xmax = self.da1.shifts.data.max()
        self.nshifts = self.da1.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da1.attrs["ScanShape"]
        self.scan_type = self.da1.attrs["MeasurementType"]
        
        self.ref = []
        for k in range(1,nb_clas+1):
            self.gp = self.da >> row_slice(X.classe==k) >> select(~X.classe)
            self.mean_clas = np.array(self.gp.mean())
            self.dist_clas = self.gp.values
            
            for i in range(self.dist_clas.shape[0]) :
                for j in range(self.dist_clas.shape[1]) :
                    self.dist_clas[i,j] = (self.dist_clas[i,j] - self.mean_clas[j])**2
            
            self.dist_clas = np.sum(self.dist_clas, axis=1)
            self.dist_clas = pd.DataFrame({'Distance_Centre' : self.dist_clas})
            self.dist_clas.index = self.gp.index
            
            self.dist_clas = self.dist_clas >> arrange(X.Distance_Centre, ascending=True)
            self.ref.append(self.dist_clas.index[0])
        
        self.spectra_kmeans = KMeans(n_clusters=nb_clas, init=self.da.loc[self.ref,:],n_init=1).fit(self.da)
        self.lab = self.spectra_kmeans.labels_.reshape(self.ny, self.nx)
        
        try :
            xlabel = self.da1.attrs["ColCoord"]
            ylabel = self.da1.attrs["RowCoord"]
            if (self.scan_type == "Map") and (self.da1.MapAreaType != "Slice") :
                self.xlabel = f"{xlabel} [{input_spectra1[xlabel].units}]"
                self.ylabel = f"{ylabel} [{input_spectra1[ylabel].units}]"
            else : 
                self.xlabel = xlabel
                self.ylabel = ylabel
        except : 
            self.xlabel = "X"
            self.ylabel = "Y"
            
        figsize = kwargs.pop("figsize", (14,8))
        self.fig = plt.figure(figsize=figsize,**kwargs)
        self.aximg = self.fig.add_axes([0.05, 0.25, 0.4, 0.6])
        self.axspectrum = self.fig.add_axes([0.5, 0.25, 0.4, 0.6])
        self.axscroll = self.fig.add_axes([0.05, 0.15, 0.7, 0.02])
        self.first_frame = 0
        self.last_frame = len(self.da1.data) - 1
        
        self.sframe = Slider(self.axscroll, 'S.No', self.first_frame, self.last_frame,
                            valinit = self.first_frame, valfmt= '%d', valstep=1)
        self.sframe.on_changed(self.scroll_spectra)
        self.spectrumplot, = self.axspectrum.plot(self.da1.shifts.data,self.da1.data[self.first_frame])
        
        self.titled(self.axspectrum, self.first_frame)
        self.imup = self.aximg.imshow(self.lab)
        self.aximg.set_xlabel(f"{self.xlabel}")
        self.aximg.xaxis.set_label_position('top')
        self.aximg.set_ylabel(f"{self.ylabel}")
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        
    def onclick(self,event):
        if event.inaxes == self.aximg :
            if event.button != 1 :
                x_pos = round(event.xdata)
                y_pos = round(event.ydata)
                if isinstance(self.imup, mpl.image.AxesImage) : 
                    if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >=0:
                        broj = round(y_pos * self.nx + x_pos)
                        self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                    elif isinstance(self.imup, mpl.lines.Line2D):
                        broj = x_pos
                        self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                else:
                    pass
            
    def scroll_spectra(self, val):
        frame  = int(self.sframe.val)
        current_spectrum = self.da1.data[frame]
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()
        
    def titled(self, ax, frame) :
        if self.scan_type == "Single" :
            new_title = self.da1.attrs['Title']
        elif self.scan_type == "Series" :
            new_title = f"Spectrum @ {np.datetime64(self.da1.Time.data[frame],'s')}"
        else :
            new_title = "Spectrum @"+\
                f"{self.da1.RowCoord}: {self.da1[self.da1.RowCoord].data[frame]}"\
                + f";{self.da1.ColCoord}: {self.da1[self.da1.ColCoord].data[frame]}"
        ax.set_title(new_title, x=0.28)
        
        

        
        
def ShowElbpt(nmax,input_spectra) :
    """Elbow Method :
    
    
    This function will show the plot of inertia_values
    nmax : maximal number of cluster
    input_spectra : reconstructed spectra after PCA.
    
    
    """
    
    da = input_spectra.copy()
    inertia = []
    for i in range(1,nmax):
        model = KMeans(n_clusters = i).fit(da.data)
        inertia.append(model.inertia_)
    fig,ax = plt.subplots()
    ax.plot(range(1,nmax),inertia)
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Cluster")
    ax.set_ylabel("Inertia")
    
    return inertia

def print_elbowpt(nmax,inertia,curve="convex",direction="decreasing"):
    """
    Print number of cluster :
    nmax : maximal number of cluster using to plot the Elbow curve
    inertia : sum of squared distance between each point and the centroid in a cluster
    curve : convex or concave
    direction: decreasing or increasing
    
    """
    
    knee = KneeLocator(range(1,nmax),inertia,curve = curve, direction = direction)
    
    return knee.elbow
    
    
        
        
        
   
class ShowKMeans(object):
    
    
    
    
    
    def __init__(self, input_spectra, nb_clas, x=None, **kwargs):
        
        self.da = input_spectra.copy()
        #self.da1 = input_spectra1.copy()
        self.xmin = self.da.shifts.data.min()
        self.xmax = self.da.shifts.data.max()
        self.nshifts = self.da.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da.attrs["ScanShape"]
        self.scan_type = self.da.attrs["MeasurementType"]
        
      
        self.spectra_kmeans = KMeans(n_clusters=nb_clas).fit(self.da)
        self.lab = self.spectra_kmeans.labels_.reshape(self.ny, self.nx)
        
        try :
            xlabel = self.da.attrs["ColCoord"]
            ylabel = self.da.attrs["RowCoord"]
            if (self.scan_type == "Map") and (self.da.MapAreaType != "Slice") :
                self.xlabel = f"{xlabel} [{input_spectra[xlabel].units}]"
                self.ylabel = f"{ylabel} [{input_spectra[ylabel].units}]"
            else : 
                self.xlabel = xlabel
                self.ylabel = ylabel
        except : 
            self.xlabel = "X"
            self.ylabel = "Y"
            
        figsize = kwargs.pop("figsize", (14,8))
        self.fig = plt.figure(figsize=figsize,**kwargs)
        self.aximg = self.fig.add_axes([0.05, 0.25, 0.4, 0.6])
        self.axspectrum = self.fig.add_axes([0.5, 0.25, 0.4, 0.6])
        self.axscroll = self.fig.add_axes([0.05, 0.15, 0.7, 0.02])
        self.first_frame = 0
        self.last_frame = len(self.da.data) - 1
        
        self.sframe = Slider(self.axscroll, 'S.No', self.first_frame, self.last_frame,
                            valinit = self.first_frame, valfmt= '%d', valstep=1)
        self.sframe.on_changed(self.scroll_spectra)
        self.spectrumplot, = self.axspectrum.plot(self.da.shifts.data,self.da.data[self.first_frame])
        
        self.titled(self.axspectrum, self.first_frame)
        self.imup = self.aximg.imshow(self.lab)
        self.aximg.set_xlabel(f"{self.xlabel}")
        self.aximg.xaxis.set_label_position('top')
        self.aximg.set_ylabel(f"{self.ylabel}")
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        
    def onclick(self,event):
        if event.inaxes == self.aximg :
            if event.button != 1 :
                x_pos = round(event.xdata)
                y_pos = round(event.ydata)
                if isinstance(self.imup, mpl.image.AxesImage) : 
                    if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >=0:
                        broj = round(y_pos * self.nx + x_pos)
                        self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                    elif isinstance(self.imup, mpl.lines.Line2D):
                        broj = x_pos
                        self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                else:
                    pass
            
    def scroll_spectra(self, val):
        frame  = int(self.sframe.val)
        current_spectrum = self.da.data[frame]
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()
        
    def titled(self, ax, frame) :
        if self.scan_type == "Single" :
            new_title = self.da.attrs['Title']
        elif self.scan_type == "Series" :
            new_title = f"Spectrum @ {np.datetime64(self.da.Time.data[frame],'s')}"
        else :
            new_title = "Spectrum @"+\
                f"{self.da.RowCoord}: {self.da[self.da.RowCoord].data[frame]}"\
                + f";{self.da.ColCoord}: {self.da[self.da.ColCoord].data[frame]}"
        ax.set_title(new_title, x=0.28)
        
        

        
class FitAllSpectra(object) :
    
    '''This class is used to interactively draw pseudo-voigt (or other type)
    peaks, on top of your spectra.
    It was originaly created to help defining initial fit parameters to pass
    on to SciPy CurveFit.
    Parameters :
        y : xr.DataArray (your spectra with baseline correction)
        initial_GaussToLorentnz_ratio : float between 0 and 1, default=0.5
            Pseudo-Voigt peak is composed of Gaussian and of a Laurentzian
            part. This ratio defines the proportion of those parts.
        scrolling_speed = float>0, default=1.
            defines how quickly your scrolling widens peaks
        initial_width : float>0, default = 5
            defines initial width of peaks
        **kwargs : dictionary, for exemple {'figsize' :(9,9)}
            whatever you want to pass to plt.sbplots(**kwargs)
    Returns :
        Nothing, but you can access the attributes using class instance, like
        fitallspectra.pic : dictionnary containing the parameters of each peak added
        fiallspectra.sum_peak : list containing cumulated graph line
            to get the y_values, use sum_peak[-1][0].get_ydata()
        fitallspectra.peak_counter: int giving the number of peaks present
        ...
    '''
    
    #pic = {}
    #pic['line'] = []
    #pic['h'] = []
    #pic['x0'] = []
    #pic['w'] = []
    #pic['GL'] = []
    
    rc = 0 # counter to store number of right click
    #sum_peak = []
    peaks_counter = 0 # number of peaks on the graph
    scroll_count = 0 # store the cumulative values of scrolling
    #artists = []
    
    #initial_width = 5
    #scrolling_speed = 1
    
    def __init__(self, y, x=None, initial_GaussToLoretnz_ratio=0.5,
                 scrolling_speed=1,
                 initial_width=5,pic={},clicked_indice = -1,artists=[],
                 sum_peak = [],**kwargs) :
        
        if isinstance(y, xr.DataArray) :
            self.y = y.data
            self.x = y.shifts.data
        else :
            self.y = y
            if x is None :
                if self.y.ndim == 1 :
                    self.x = np.arange(len(self.y))
                else :
                    self.x = np.arange(self.y.shape[1])
            else :
                self.x = x
        if self.y.ndim == 1 :
            self.y = self.y[np.newaxis, :, np.newaxis]
        if self.y.ndim ==2 :
            self.y = self.y[:, :, np.newaxis]
        assert self.y.shape[1] == len(self.x),\
        "Check your Raman shifts array. The dimensions " + \
        f"of your spectra ({self.y[1]}) and that of " + \
        f"your Ramans shifts ({len(self.x)}) are not the same."
            
        self.first_frame = 0
        self.last_frame = len(self.y) - 1
        self.GL = initial_GaussToLoretnz_ratio
        self.scrolling_speed = scrolling_speed
        self.initial_width = initial_width
        self.fig, self.ax = plt.subplots(**kwargs)
        self.spectrumplot = self.ax.plot(self.x, self.y[0], c='k', alpha=0.5)
        self.ax.set_title('Left-click to add/remove peaks,'
                          'Scroll to adjust width, Right-click to draw the sum')
        self.x_size = self.set_size(self.x)
        self.y_size = 2*self.set_size(self.y)
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.axframe = self.fig.add_axes([0.15, 0.05, 0.7, 0.03])
        self.axnbre = self.fig.add_axes([-0.15, 0.1, 0.1, 0.03]) # create box to enter numbr of spectra
        if len(self.y) > 1:
            self.sframe = Slider(self.axframe, 'Frame N°',
                                self.first_frame, self.last_frame, valfmt = '%d',
                                valinit = self.first_frame, valstep =1)
            self.nbre_box = TextBox(self.axnbre, 'S.No : ')
            self.sframe.on_changed(self.update)
            self.nbre_box.on_submit(self.submit)
            self.fig.canvas.mpl_connect('key_press_event', self.press)
        else :
            self.axframe.axis('off')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('scroll_event',self.onclick)
        self.pic = pic
        self.pic['line'] = [] # list containing matplotlib.Line2D object for each peak
        self.pic['h'] = [] # list that will contain heights of each peak 
        self.pic['x0'] = [] # list that will contain central positions of each peak
        self.pic['w'] = [] # list containing widths
        self.pic['GL'] = [] # list containing G/L ratio
        self.clicked_indice= clicked_indice
        self.artists=[] # will be used to store the elipses on tops of the peaks
        self.sum_peak = [] # list of cumulated graphs
        
        # point size 
    def set_size(self, variable, rapport = 70):
        return (variable.max() - variable.min())/rapport
    
        
    def submit(self,nbre) :
        # enter number in the text box to show a spectra
        frame = int(nbre)
        self.sframe.set_val(frame)
        current_spectrum = self.y[frame]
        for i, line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        #self.titled(frame)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()    
    
    def onclick(self,event):
        
        if event.inaxes == self.ax : # if you click inside the plot
            if event.button == 1 : # left click
                # create list of all elipses and check if the click was inside
                click_in_artist = [artist.contains(event)[0] for artist in self.artists]
                if not any(click_in_artist): # if the click was not on any elipis
                    self.peaks_counter +=1
                    self.one_elipsis = self.ax.add_artist(Ellipse((
                        event.xdata,event.ydata),self.x_size, self.y_size, alpha=0.5,
                        gid=str(self.peaks_counter)))
                    self.artists.append(self.one_elipsis)
                    h = event.ydata
                    x0 = event.xdata
                    yy = pV(x=self.x, h=h,x0=x0, w=self.x_size*self.initial_width, 
                            factor=self.GL)
                    
                    self.pic['line'].append(self.ax.plot(self.x, yy, alpha=0.75, lw=2.5, picker=5))
                    self.pic['h'].append(h)
                    self.pic['x0'].append(x0)
                    self.pic['w'].append(self.x_size*self.initial_width)
                    self.fig.canvas.draw_idle()
                    
                elif any(click_in_artist) : # if the click was on one of the elipses
                    self.clicked_indice = click_in_artist.index(True)
                    self.artists[self.clicked_indice].remove()
                    self.artists.pop(self.clicked_indice)
                    if self.pic['line'][self.clicked_indice][0] in self.ax.lines :
                        self.ax.lines.remove(self.pic['line'][self.clicked_indice][0])
                    self.pic['line'].pop(self.clicked_indice)
                    self.pic['x0'].pop(self.clicked_indice)
                    self.pic['h'].pop(self.clicked_indice)
                    self.pic['w'].pop(self.clicked_indice)
                    self.peaks_counter -=1
                    self.fig.canvas.draw_idle()
                    
            elif event.button ==3 and not event.step : # suppose that middle click and right have the same values
                if self.rc > 0 : # checks if there is already a cumulated graph plotted
                    #remove the last cumulated graph from the fig
                    self.ax.lines.remove(self.sum_peak[-1][0])
                    self.sum_peak.pop()
                # sum all the y values from all the peaks        
                self.sumy = np.sum(np.asarray([self.pic['line'][i][0].get_ydata() 
                                             for i in range(self.peaks_counter)]), axis=0)
                # added this condition for the case where you removed all peaks,
                # but the cumulated graph is left
                # the right-clicking need to remoe that one as well
                if self.sumy.shape == self.x.shape :
                    # plot the cumulated graph :
                    self.sum_peak.append(self.ax.plot(self.x,self.sumy, '--', color='lightgreen'
                                                         ,lw=3, alpha=0.6))
                    self.rc +=1 # one cumulated graph added
                else :
                    # if you right clicked on the grph with no peaks,
                    # you removed the cumulated graph as welll
                    self.rc = -1
                self.fig.canvas.draw_idle()
                    
            if event.step !=0 :
                if self.peaks_counter :
                    peak_identifier = -1 # -1 means that scroling will only affect the last plotted peak
                    # this adjust the "speed" of width change with scrolling :
                    self.scroll_count += self.x_size*np.sign(event.step)*self.scrolling_speed/10
                    if self.scroll_count > -self.x_size*self.initial_width*0.999:
                        w2 = self.x_size*self.initial_width + self.scroll_count
                    else :
                        w2 = self.x_size * self.initial_width / 1000
                        #this doesn't allow you to scroll to negative values
                        # (basic width is x_size) aliased_name
                        self.self.scroll_count = -self.x_size*self.initial_width*0.999
                        
                    center2 = self.pic['x0'][peak_identifier]
                    h2 = self.pic['h'][peak_identifier]
                    self.pic['w'][peak_identifier] = w2
                    yy = pV(x=self.x,x0=center2, h=h2, w=w2, factor=self.GL)
                    active_line = self.pic['line'][peak_identifier][0]
                    active_line.set_ydata(yy)
                    self.ax.draw_artist(active_line)
                    self.fig.canvas.draw_idle()
                        
            if event.button !=1 and event.dblclick :
                block = True
                self.pic['GL'] = [self.GL] * self.peaks_counter
                self.fig.canvas.mpl_disconnect(self.cid)
                self.fig.canvas.mpl_disconnect(self.cid2)
                
    def press(self, event) :
        # use left and right array keys to scroll through frames one by one.
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        
        elif event.key == 'right' and frame < self.last_frame:
            new_frame = frame + 1
        else :
            new_frame = frame
        self.sframe.set_val(new_frame)
        current_spectrum = self.y[new_frame]    
        for i, line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        
    def update(self, val) :
        # use the slider to scroll through frames
        frame = int(self.sframe.val)
        current_spectrum = self.y[frame]
        for i, line in enumerate(self.spectrumplot) :
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        
        
        
        
class FitParams(object) :
    '''use this class to show your fitting result. it will also shows a table 
    containing your peaks parameters (center, height...)
    Parameters :
        y : xr.DataArray (your spectra with baseline correction)
        x_size : x point size that you used during your fit
        peaks_present : number of your peaks
        pic: dictionnary wwith the parameters of each peak
        sum_peak = list with cumulated graph line
    '''
    
    
    def __init__(self, y, x_size, peaks_present, pic, 
                 sum_peak, x =None, **kwargs) :
        
        if isinstance(y, xr.DataArray) :
            self.y = y.data
            self.x = y.shifts.data
        else :
            self.y = y
            if x is None :
                if self.y.ndim == 1 :
                    self.x = np.arange(len(self.y))
                else :
                    self.x = np.arange(self.y.shape[1])
            else :
                self.x = x
        if self.y.ndim == 1 :
            self.y = self.y[np.newaxis, :, np.newaxis]
        if self.y.ndim ==2 :
            self.y = self.y[:, :, np.newaxis]
        
        self.x_size = x_size
        self.peaks_present = peaks_present
        self.pic = pic
        self.sum_peak = sum_peak
        # create the list of initial parameters from your manual input
        self.manualfit_components_params = copy(list(map(list,zip(
        self.pic['h'], self.pic['x0'], self.pic['w'], self.pic['GL']))))
        
        # to transform the list of lists into one single list:
        self.manualfit_components_params = list(chain(*self.manualfit_components_params))
        # the sum of manually created peaks:
        assert len(self.sum_peak) >0, 'No peaks initiated'
        self.manualfit = self.sum_peak[0][0].get_data()[1]
        # Setting the bounds based on your input
        # (you can play with this part if you feel like it
        # but leaving it as it is should be ok for basic usage)
        # set the initial bounds as infinities:
        #upper_bounds = np.ones_like(manualfit_components_params)*np.inf
        self.upper_bounds = np.ones(len(self.manualfit_components_params))*np.inf
        self.lower_bounds = np.ones(len(self.manualfit_components_params))*(-np.inf)
        
        # setting reasonable bounds for the peak amplitude
        # as a portion to your initial manual estimate
        self.upper_bounds[0::4] = [A*1.4 for A in self.manualfit_components_params[0::4]]
        self.lower_bounds[0::4] = [A*0.7 for A in self.manualfit_components_params[0::4]]
        # setting reasonable bounds for the peak position
        #as a shift in regard to your initial manual position
        self.upper_bounds[1::4] = [A + 2*self.x_size for A in self.manualfit_components_params[1::4]]
        self.lower_bounds[1::4] = [A - 2*self.x_size for A in self.manualfit_components_params[1::4]]
        # setting the bounds for the widths
        self.upper_bounds[2::4] = [A*10 for A in self.manualfit_components_params[2::4]]
        self.lower_bounds[2::4] = [A*0.5 for A in self.manualfit_components_params[2::4]]
        # setting the bounds for the Gauss/Lorentz ratio
        self.upper_bounds[3::4] = 1
        self.lower_bounds[3::4] = 0
        self.bounds = (self.lower_bounds, self.upper_bounds)
        # the curve-fitting part : (I choose the means of all spectra)
        #self.fitted_params, self.b = curve_fit(fitting_function, self.x, np.average(self.y.reshape(y.data.shape),axis=0), method='trf'
        #                                       , p0=self.manualfit_components_params
        #                                       ,absolute_sigma=False, bounds=self.bounds)
        self.fitted_params, self.b = curve_fit(fitting_function, self.x, self.y.reshape(y.data.shape)[1], method='trf'
                                               , p0=self.manualfit_components_params
                                               ,absolute_sigma=False, bounds=self.bounds)
        
        self.fitting_err = np.sqrt(np.diag(self.b))
        self.y_fitted = fitting_function(self.x, *self.fitted_params)
        assert self.y.shape[1] == len(self.x),\
        "Check your Raman shifts array. The dimensions " + \
        f"of your spectra ({self.y[1]}) and that of " + \
        f"your Ramans shifts ({len(self.x)}) are not the same."
            
        self.first_frame = 0
        self.last_frame = len(self.y) - 1
        # Plotting the results of the optimization:
        self.fig, self.ax = plt.subplots(**kwargs)
        self.spectrumplot = self.ax.plot(self.x, self.y[0], c='k', alpha=0.3,
                                        label ='original spectra')
        self.fig.subplots_adjust(bottom=0.15, right=0.89)
        self.axframe = self.fig.add_axes([0.05, 0.15, 0.02, 0.8])
        self.axnbre = self.fig.add_axes([0.08, 0.02, 0.1, 0.03])
        #self.errax = fig.add_axes([0.125, 0.1, 0.775, 0.1])
        #self.spectrumplot2 = self.ax.plot(self.x, self.manualfit, '--g', alpha=0.5, label='initial manual fit')
        #self.spectrum3 = self.ax.plot(self.x, self.y_fitted, '--r', lw=4, alpha=0.6, label ='after optimization')
        #self.errax.set_facecolor('w')
        #self.errax.plot(self.x, self.y-self.y_fitted)
        #self.errax.set_ylabel('error\n(data-fit)')
        self.ax.plot(self.x, self.y_fitted, '--k',lw=2,alpha=0.6, label = 'fit')
        self.par_nam = ['h', 'x0', 'w', 'G/L']
        for i in range(self.peaks_present):
            self.fit_res = list(zip(self.par_nam, self.fitted_params[i*4:i*4+4],
                               self.fitting_err[i*4:i*4+4]))
            self.label = [f"{P}={v:.2f}\U000000B1{e:.1f}" for P, v, e in self.fit_res]
            self.yy_i = pV(self.x, *self.fitted_params[i*4:i*4+4])
            self.peak_i, = self.ax.plot(self.x, self.yy_i, alpha=0.5, label=self.label)
            self.ax.fill_between(self.x, self.yy_i, facecolor=self.peak_i.get_color(), alpha=0.3)
            
        self.ax.set_title('Showing the individual peaks as found by fitting procedure')
        plt.subplots_adjust(left=0.2, bottom=0.3)
        self.rows = ["Peak %d" % i for i in range(self.peaks_present)]
        self.cell_text1 = np.round(self.fitted_params,3).reshape(len(self.fitted_params)//4,4).tolist()
        self.cell_text2 = np.round(self.fitting_err,3).reshape(len(self.fitting_err)//4,4).tolist()
        self.after = unumpy.uarray(self.cell_text1,self.cell_text2)
        self.colors = plt.cm.Greys(np.linspace(0, 0.5, len(self.rows)))
        self.columns = ['Height','Center','FWMH','Ratio G/L']
        
        self.the_table = self.ax.table(cellText=self.after,
                          rowLabels=self.rows,
                          rowColours=self.colors,
                          colLabels=self.columns,
                          loc='bottom',bbox=[0.1, -0.5, 0.9, 0.4])
        
        if len(self.y) > 1:
            self.sframe = Slider(self.axframe, 'Frame N°',
                                self.first_frame, self.last_frame, valfmt = '%d',
                                valinit = self.first_frame, valstep =1,
                                orientation='vertical')
            self.nbre_box = TextBox(self.axnbre, 'S.No : ')
            self.sframe.on_changed(self.update)
            self.nbre_box.on_submit(self.submit)
            self.fig.canvas.mpl_connect('key_press_event', self.press)
        else :
            self.axframe.axis('off')
            
    def submit(self,nbre) :
        # enter number in the text box to show a spectra
        frame = int(nbre)
        self.sframe.set_val(frame)
        current_spectrum = self.y[frame]
        for i, line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        #self.titled(frame)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        
        
        
    def press(self, event) :
        # use up and down array keys to scroll through frames one by one.
        frame = int(self.sframe.val)
        if event.key == 'down' and frame > 0:
            new_frame = frame - 1
        
        elif event.key == 'up' and frame < self.last_frame:
            new_frame = frame + 1
        else :
            new_frame = frame
        self.sframe.set_val(new_frame)
        current_spectrum = self.y[new_frame]    
        for i, line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        
    def update(self, val) :
        # use the slider to scroll through frames
        frame = int(self.sframe.val)
        current_spectrum = self.y[frame]
        for i, line in enumerate(self.spectrumplot) :
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
    

    
def f_between(peaks_present,para_names,allfitted_params,allstd_err,x) :
    
    """This fonction is used to calculate the fitted data, and area under each peak"""
    
    allyy = np.zeros((peaks_present,len(x)))
    all_area = np.zeros((peaks_present))
    for i in range(peaks_present):
        
        #fit_res = list(zip(para_names,allfitted_params[i*4:i*4+4],
                          # allstd_err[i*4:i*4+4]))
        #label = [f"{P}={v:.2f}\U000000B1{e:.1f}" for P, v, e in fit_res]
        allyy[i,] = pV(x, *allfitted_params[i*4:i*4+4])
        all_area[i] = np.trapz(allyy[i,],x)
        #peak_i, = axes.plot(x, yy_i, alpha=0.5, label=label)
        #axes.fill_between(x,yy_i, facecolor=peak_i.get_color(), alpha=0.3)
    return allyy, all_area    
    
    
def param_values(all_area, peaks_present,allfitted_params,allstd_err,axes) :
    
    """This fonction is used to save peak parameters ('Height','Center','FWMH','Ratio G/L','Area')
    in table and plot it. it return also all peaks height and all peaks area which are
    used to plot the peak ratio
    """
    
    rows = ["Peak %d" % i for i in range(peaks_present)]
    cell_text1 = np.round(allfitted_params,3).reshape(len(allfitted_params)//4,4).tolist()
    cell_text2 = np.round(allstd_err,3).reshape(len(allstd_err)//4,4).tolist()
    after = unumpy.uarray(cell_text1,cell_text2)
    after1 = []
    cell_text11 = []
    for i in range(len(after)) :
        after1.append(np.concatenate((after[i,], np.round(np.array([all_area[i]]),3)),axis=0))
        cell_text11.append(np.concatenate((np.array(cell_text1)[i,], np.array([all_area[i]])),axis=0))
    after1 = np.array(after1)  
    cell_text11 = np.array(cell_text11)
    colors = plt.cm.Greys(np.linspace(0, 0.5, len(rows)))
    columns = ['Height','Center','FWMH','Ratio G/L','Area']
    the_table = axes.table(cellText=after1,
                          rowLabels=rows,
                          rowColours = colors,
                          colLabels=columns,
                          loc='bottom',bbox=[0.1, -0.5, 0.9, 0.4]) 
    return cell_text11[:,0] , cell_text11[:,4]

    
    
    
class FitParams1(object) :
    '''use this class to show your fitting result. it will also shows a table 
    containing your peaks parameters (center, height...)
    
    Parameters :
        y : xr.DataArray (your spectra with baseline correction)
        x_size : x point size that you used during your fit
        peaks_present : number of your peaks
        pic: dictionnary wwith the parameters of each peak
        sum_peak = list with cumulated graph line
    '''
    
    
    def __init__(self,input_spectra, x_size, peaks_present, pic, 
                 sum_peak, pic_a=None, pic_h=None, x =None, **kwargs) :
        #self.input_spectra = input_spectra
        self.shape = input_spectra.data.shape 
        self.pic_h = pic_h
        self.pic_a = pic_a
        if isinstance(input_spectra, xr.DataArray) :
            self.y = input_spectra.data
            self.x = input_spectra.shifts.data
        else :
            self.y = input_spectra
            if x is None :
                if self.y.ndim == 1 :
                    self.x = np.arange(len(self.y))
                else :
                    self.x = np.arange(self.y.shape[1])
            else :
                self.x = x
        if self.y.ndim == 1 :
            self.y = self.y[np.newaxis, :, np.newaxis]
        if self.y.ndim ==2 :
            self.y = self.y[:, :, np.newaxis]
        
        self.x_size = x_size
        self.peaks_present = peaks_present
        self.pic = pic
        self.sum_peak = sum_peak
        self.pic_h = np.zeros((self.y.shape[0],self.peaks_present))
        self.pic_a = np.zeros((self.y.shape[0],self.peaks_present))
        # create the list of initial parameters from your manual input
        self.manualfit_components_params = copy(list(map(list,zip(
        self.pic['h'], self.pic['x0'], self.pic['w'], self.pic['GL']))))
        
        # to transform the list of lists into one single list:
        self.manualfit_components_params = list(chain(*self.manualfit_components_params))
        # the sum of manually created peaks:
        assert len(self.sum_peak) >0, 'No peaks initiated'
        self.manualfit = self.sum_peak[0][0].get_data()[1]
        # Setting the bounds based on your input
        # (you can play with this part if you feel like it
        # but leaving it as it is should be ok for basic usage)
        # set the initial bounds as infinities:
        #upper_bounds = np.ones_like(manualfit_components_params)*np.inf
        self.upper_bounds = np.ones(len(self.manualfit_components_params))*np.inf
        self.lower_bounds = np.ones(len(self.manualfit_components_params))*(-np.inf)
        
        # setting reasonable bounds for the peak amplitude
        # as a portion to your initial manual estimate
        self.upper_bounds[0::4] = [A*1.4 for A in self.manualfit_components_params[0::4]]
        self.lower_bounds[0::4] = [A*0.7 for A in self.manualfit_components_params[0::4]]
        # setting reasonable bounds for the peak position
        #as a shift in regard to your initial manual position
        self.upper_bounds[1::4] = [A + 2*self.x_size for A in self.manualfit_components_params[1::4]]
        self.lower_bounds[1::4] = [A - 2*self.x_size for A in self.manualfit_components_params[1::4]]
        # setting the bounds for the widths
        self.upper_bounds[2::4] = [A*10 for A in self.manualfit_components_params[2::4]]
        self.lower_bounds[2::4] = [A*0.5 for A in self.manualfit_components_params[2::4]]
        # setting the bounds for the Gauss/Lorentz ratio
        self.upper_bounds[3::4] = 1
        self.lower_bounds[3::4] = 0
        self.bounds = (self.lower_bounds, self.upper_bounds)
        #self.allfitted_params = np.zeros((self.y.data.shape[0],len(self.manualfit_components_params)))
        #self.allstd_err = np.zeros_like(self.allfitted_params)
        #self.allcov = np.zeros((self.y.data.shape[0],len(self.manualfit_components_params),
        #                       len(self.manualfit_components_params)))
        self.allfitted_params, self.allcov = curve_fit(fitting_function, self.x, self.y.reshape(self.shape)[0], method='trf'
                                               , p0=self.manualfit_components_params
                                               ,absolute_sigma=False, bounds=self.bounds)
        self.allstd_err = np.sqrt(np.diag(self.allcov))
        
        # calculate peaks parameters ( height and area for all spectra)

        #self.fitting_err = np.sqrt(np.diag(self.b))
        #self.ally_fitted = np.zeros_like(self.y.reshape(y.data.shape))
        #for i in range(50) :
        self.ally_fitted = fitting_function(self.x, *self.allfitted_params)
        self.error = self.y.reshape(self.shape)[0] - self.ally_fitted
        # calaculate the r_squared of the model
        self.ss_res = np.matmul(self.error,self.error)
        self.ss_tot = np.matmul(self.y.reshape(self.shape)[0] - np.mean(self.y.reshape(self.shape)[0]),
                               self.y.reshape(self.shape)[0] - np.mean(self.y.reshape(self.shape)[0]))
        
        self.r_squared = 1 - self.ss_res/self.ss_tot
        
        assert self.y.shape[1] == len(self.x),\
        "Check your Raman shifts array. The dimensions " + \
        f"of your spectra ({self.y[1]}) and that of " + \
        f"your Ramans shifts ({len(self.x)}) are not the same."
            
        self.first_frame = 0
        self.last_frame = len(self.y) - 1
        # Plotting the results of the optimization:
        self.fig, self.ax = plt.subplots(**kwargs)
        self.spectrumplot, = self.ax.plot(self.x, self.y[0], c='k', alpha=0.3)
        self.fig.subplots_adjust(bottom=0.15, right=0.89)
        self.axframe = self.fig.add_axes([0.05, 0.15, 0.02, 0.8])
        self.axnbre = self.fig.add_axes([0.08, 0.02, 0.1, 0.03])
        #self.errax = fig.add_axes([0.125, 0.1, 0.775, 0.1])
        #self.spectrumplot2 = self.ax.plot(self.x, self.manualfit, '--g', alpha=0.5, label='initial manual fit')
        #self.spectrum3 = self.ax.plot(self.x, self.y_fitted, '--r', lw=4, alpha=0.6, label ='after optimization')
        #self.errax.set_facecolor('w')
        #self.errax.plot(self.x, self.y-self.y_fitted)
        #self.errax.set_ylabel('error\n(data-fit)')
        self.label = [f"R² = {np.round(self.r_squared,3)}"]
        self.y_fittedplot, = self.ax.plot(self.x, self.ally_fitted,'--k',lw=2,alpha=0.6, label = self.label)
        self.ax.legend()
        self.par_nam = ['h', 'x0', 'w', 'G/L']
        self.allyy,self.all_area  = f_between(peaks_present=self.peaks_present,para_names=self.par_nam
                  ,allfitted_params=self.allfitted_params,allstd_err=self.allstd_err
                  ,x=self.x)
        self.lineD = []
        self.polycol = []
        for i in range(self.peaks_present) :
            self.allyy_i, = self.ax.plot(self.x, self.allyy[i,], alpha=0.5)
            self.lineD.append(self.allyy_i)
            self.fill_b = self.ax.fill_between(self.x,self.allyy[i,], facecolor=self.allyy_i.get_color(), alpha=0.3)
            self.polycol.append(self.fill_b)
        self.ax.set_title('Showing the individual peaks as found by fitting procedure')
        plt.subplots_adjust(left=0.2, bottom=0.3)
        self.pic_h[0,], self.pic_a[0,] = param_values(self.all_area, peaks_present=self.peaks_present
                     ,allfitted_params=self.allfitted_params
                    ,allstd_err=self.allstd_err
                    ,axes=self.ax) 
        
        
                
        for i in range(self.shape[0]) :
                
            
                         
            self.a, self.b = curve_fit(fitting_function, self.x, self.y.reshape(self.shape)[i], method='trf'
                                               , p0=self.manualfit_components_params
                                               ,absolute_sigma=False, bounds=self.bounds)
            
                        
            self.cell_text1 = np.round(self.a,3).reshape(len(self.a)//4,4)
            self.aly = np.zeros((self.peaks_present,len(self.x)))
            self.al_area = np.zeros(self.peaks_present)
            self.cell_text11 = []            
            for j in range(self.peaks_present):
                self.aly[j,] = pV(self.x, *self.a[j*4:j*4+4])
                self.al_area[j] = np.trapz(self.aly[j,],x)
                self.cell_text11.append(np.concatenate((self.cell_text1[j,], np.array([self.al_area[j]])),axis=0))
   
            self.cell_text11 = np.array(self.cell_text11)
            
            #self.al_fit = fitting_function(self.x, *self.a)
            #self.errori = self.y.reshape(self.shape)[i] - self.al_fit
            #self.ss_resi = np.matmul(self.errori,self.errori)
            #self.ss_toti = np.matmul(self.y.reshape(self.shape)[i] - np.mean(self.y.reshape(self.shape)[i]),
            #                   self.y.reshape(self.shape)[i] - np.mean(self.y.reshape(self.shape)[i]))
        
            #self.r_squaredi = 1 - self.ss_resi/self.ss_toti
            #if self.r_squaredi < 0.5 :
            #    self.pic_h[i,] = np.array([1,1,1])
            #else :
            self.pic_h[i,] = self.cell_text1[:,0]
            self.pic_a[i,] = self.cell_text11[:,4]
        if len(self.y) > 1:
            self.sframe = Slider(self.axframe, 'Frame N°',
                                self.first_frame, self.last_frame, valfmt = '%d',
                                valinit = self.first_frame, valstep =1,
                                orientation='vertical')
            self.nbre_box = TextBox(self.axnbre, 'S.No : ')
            self.sframe.on_changed(self.update)
            self.nbre_box.on_submit(self.submit)
            self.fig.canvas.mpl_connect('key_press_event', self.press)
        else :
            self.axframe.axis('off')
            
    def submit(self,nbre) :
        # enter number in the text box to show a spectra
        frame = int(nbre)
        self.sframe.set_val(frame)
        self.current_spectrum = self.y[frame]
        #for i, line in enumerate(self.spectrumplot):
        #    line.set_ydata(current_spectrum[:, i])
        #    self.ax.relim()
        #    self.ax.autoscale_view()
        #self.titled(frame)
        self.spectrumplot.set_ydata(self.current_spectrum)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        
        
        
    def press(self, event) :
        # use top and bottom array keys to scroll through frames one by one.
        frame = int(self.sframe.val)
        if event.key == 'down' and frame > 0:
            new_frame = frame - 1
        
        elif event.key == 'up' and frame < self.last_frame:
            new_frame = frame + 1
        else :
            new_frame = frame
        self.sframe.set_val(new_frame)
        self.current_spectrum = self.y[new_frame] 
        self.spectrumplot.set_ydata(self.current_spectrum)
        #for i, line in enumerate(self.spectrumplot):
        #    line.set_ydata(current_spectrum[:, i])
        #    self.ax.relim()
        #    self.ax.autoscale_view()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        
    def update(self, val) :
        # use the slider to scroll through frames
        frame = int(self.sframe.val)
        self.current_spectrum = self.y[frame]
        
        self.spectrumplot.set_ydata(self.current_spectrum)
        self.allfitted_params, self.allcov = curve_fit(fitting_function, self.x, self.y.reshape(self.shape)[frame], method='trf'
                                               , p0=self.manualfit_components_params
                                               ,absolute_sigma=False, bounds=self.bounds)
        #self.cell_text1 = np.round(self.allfitted_params,3).reshape(len(self.allfitted_params)//4,4)
        #self.pic_h[frame,] = self.cell_text1[:,0]
        self.allstd_err = np.sqrt(np.diag(self.allcov))
        self.current_std_err = self.allstd_err
        self.current_fitparams = self.allfitted_params
        #self.fitting_err = np.sqrt(np.diag(self.b))
        #self.ally_fitted = np.zeros_like(self.y.reshape(y.data.shape))
        #for i in range(50) :
        self.ally_fitted = fitting_function(self.x, *self.allfitted_params)
        
        self.current_fit = self.ally_fitted
        self.error = self.y.reshape(self.shape)[frame] - self.current_fit
        self.ss_res = np.matmul(self.error,self.error)
        self.ss_tot = np.matmul(self.y.reshape(self.shape)[frame] - np.mean(self.y.reshape(self.shape)[frame]),
                               self.y.reshape(self.shape)[frame] - np.mean(self.y.reshape(self.shape)[frame]))
        
        self.r_squared = 1 - self.ss_res/self.ss_tot
        handles,_= self.ax.get_legend_handles_labels()
        self.ax.legend(handles, [f"R² = {np.round(self.r_squared,3)}"])
        self.y_fittedplot.set_ydata(self.current_fit)
        
        self.current_allyy,self.current_all_area = f_between(peaks_present=self.peaks_present,para_names=self.par_nam
                  ,allfitted_params=self.allfitted_params,allstd_err=self.allstd_err
                  ,x=self.x)
        
        for i in range(self.peaks_present):
            self.lineD[i].set_ydata(self.current_allyy[i,])
            dummy = self.ax.fill_between(self.x,self.current_allyy[i,], facecolor=self.lineD[i].get_color(), alpha=0.3)
            dp = dummy.get_paths()[0]
            dummy.remove()
        #update the vertices of the PolyCollection
            self.polycol[i].set_paths([dp.vertices])
            #self.fill_b.set_ydata(self.current_allyy[i,])
        
        param_values(all_area=self.current_all_area,peaks_present=self.peaks_present
                     ,allfitted_params=self.allfitted_params
                     ,allstd_err=self.allstd_err
                     ,axes=self.ax) 
        #self.pic_h[frame,] = param_values(peaks_present=self.peaks_present
        #             ,allfitted_params=self.allfitted_params
        #             ,allstd_err=self.allstd_err
        #             ,axes=self.ax) 
        #del param_values
        self.ax.relim()
        self.ax.autoscale_view()
        #for i, line in enumerate(self.spectrumplot) :
        #    line.set_ydata(current_spectrum[:, i])
        #    self.ax.relim()
         #   self.ax.autoscale_view()
        #for i, line in enumerate(self.y_fittedplot) :
         #   line.set_ydata(current_fit[i])
         #   self.ax.relim()
          #  self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

        
        
def pic_ratio(da, pic_h, ratio=None, col_lim = None, scanshape=None,components_sigma=None,
              **kwargs) :
    
    """ Use this fonction to plot peaks_ratio using height and area
    
    pic_h : all peaks height for all spectra
    
    """
    if isinstance(da, xr.DataArray) :
        
        spectra = da.data# - da.data.mean())#/da.data.std()
    
        shape = da.attrs["ScanShape"] + (-1, )
        components_sigma = da.shifts.data
        
    else :
        spectra = da
        shape = scanshape + (-1,)
        components_sigma = components_sigma
    
    if len(ratio) == 2 :
        
        pic_ratio = pic_h[:,ratio[0]]/pic_h[:,ratio[1]]
        visualize_components = AllMaps(pic_ratio.reshape(shape),
                                       components_sigma=components_sigma,col_lim = col_lim)
    elif len(ratio) > 2 :
        pic_sum = np.zeros(len(pic_h[:,ratio[0]]))
        for i in range(len(ratio)-1) :
            pic_sum += pic_h[:,ratio[i+1]]
        pic_ratio = pic_h[:,ratio[0]]/pic_sum
        visualize_components = AllMaps(pic_ratio.reshape(shape),
                                       components_sigma=components_sigma,col_lim = col_lim)
                                           #components=P[:n_components,],
                                          # var = each_cp_var[0:n_components]*100,
                                       

                    
                    
class SVD(object) :
    
    """ This class is used to show pca result. 
    Right click on the score image to show the corresponding spectra
    
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
    visualize_var: bool
        Wheather to visualize the scree plot
    assign: bool
        Wheather to assign the results of pca decomposition to the returned
        xr.DataArray.
        If True, the resulting xr.DataArray will be attributed new coordinates:
            `pca_comp[N]` along the `"RamanShifts"` dimension, and
            `pca_comp[N]_coeffs` along the `"points"` dimension
    Returns:
    
        SVD.svd_element  <---- data after SVD, explianed variance, scores array , loadings array
    --------
  
    
    """
   
    def __init__(self, da, n_components=10,    
                visualize_clean=False, visualize_components=False, visualize_var =False,
                col_lim = None,scanshape=None,components_sigma=None,
                feature_range = (-0.05,0.05),**kwargs) :
        self.da = da
        self.feature_range = feature_range
        self.n_components = n_components
        
        if isinstance(self.da, xr.DataArray) :
            self.spectra = self.da.data.copy()
            self.shape = self.da.attrs["ScanShape"] + (-1, )
            self.ny, self.nx = self.da.attrs["ScanShape"]
            self.components_sigma = self.da.shifts.data
            
        else :
            self.spectra = self.da.copy()
            self.shape = scanshape + (-1,)
            self.ny, self.nx = scanshape
            self.components_sigma = components_sigma
        
        self.U, self.A1, self.P = np.linalg.svd(self.spectra, full_matrices=True)
        self.A = np.zeros((self.spectra.shape[0],self.spectra.shape[1]),dtype=float)
        np.fill_diagonal(self.A,self.A1)
        self.R = np.dot(self.U, self.A)
        self.score = self.R[:,:n_components]
        self.spectra_recons = np.dot(self.R[:,:n_components],self.P[:n_components,])
    # remarquer le nombre de variance expliqué par chaque CP
        self.var_expl = self.A1**2/(self.spectra.shape[0] - 1)
        self.tot_var_expl = np.sum(self.var_expl)
        self.each_cp_var = np.round((self.var_expl / self.tot_var_expl),3)
                #spectra = scaler.transform(da.data)
        self.L = preprocessing.minmax_scale(self.da, feature_range= feature_range ,axis=-1, copy=False)
    #n_components = int(pca_fit.n_components_)
        if visualize_components:
            self.visualize_components = AllMaps(self.score.reshape(self.shape),
                                                    components=self.P[:n_components,],
                                                    var = self.each_cp_var[0:n_components]*100,
                                                    components_sigma=self.components_sigma,col_lim = col_lim)
            
            self.line2, = self.visualize_components.ax2.plot(self.components_sigma,self.L[0])
        #visualize_components.line, = self.visualize_components.ax2.plot(self.da.shifts.data,self.da[0])
        if hasattr(self.da, 'attrs'):
            self.da.attrs["score_Components_visu"] = self.visualize_components

   # if visualize_err:
        #plt.figure()
        #sqerr = np.sum((spectra - spectra_cleaned)**2, axis=-1)
        #plt.imshow(sqerr.reshape(da.ScanShape))

        if visualize_clean:
            
            self._s = np.stack((self.spectra, self.spectra_recons), axis=-1)
            self.label = ["original spectra", "svd reconstruction"]
            self.visualize_result = ShowSpectra(self._s, self.components_sigma,
                                           label=self.label)
        if hasattr(self.da, 'attrs'):
            self.da.attrs["svd_reconstruction_visu"] = self.visualize_result
           
        if visualize_var :
            plt.figure()
            self.ncomp = np.arange(n_components) + 1
            plt.plot(self.ncomp, self.each_cp_var[0:n_components], 'o-',linewidth=3,color='green')
            plt.title('Scree Plot')
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained')
            
            
        self.visualize_components.fig.canvas.mpl_connect('button_press_event', self.onclick)
        #self.visualize_components.Line = self.visualize_components.line
        
    
        
    def onclick(self,event):
            
        if event.inaxes == self.visualize_components.ax :
            if event.button != 1 :
                x_pos = round(event.xdata)
                y_pos = round(event.ydata)
                if isinstance(self.visualize_components.im, mpl.image.AxesImage) : 
                    if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >=0:
                        broj = round(y_pos * self.nx + x_pos)
                        #self.visualize_components.line.set_ydata(self.da.data[broj])
                        self.line2.set_ydata(self.L[broj])
                        self.visualize_components.ax2.relim()
                        self.visualize_components.ax2.autoscale_view()
                        self.visualize_components.fig.canvas.draw_idle()
                        #self.sframe.set_val(broj)
                        #self.scroll_spectra(broj)
                    elif isinstance(self.visualize_components.im, mpl.lines.Line2D):
                        broj = x_pos
                        #self.visualize_components.line.set_ydata(self.da.data[broj])
                        self.line2.set_ydata(self.L[broj])
                        self.visualize_components.ax2.relim()
                        self.visualize_components.ax2.autoscale_view()
                        self.visualize_components.fig.canvas.draw_idle()
                        #self.sframe.set_val(broj)
                        #self.scroll_spectra(broj)
                else:
                    pass

    #else:
    def svd_element(self,da) :
        n_components = self.n_components
        return pp.giveback_same(da, self.spectra_recons) ,self.each_cp_var[0:n_components]*100,self.score,self.P[:n_components,]
        
 #   return giveback_same(da, spectra_recons) ,each_cp_var[0:n_components]*100,score,P[:n_components,],visualize_components
