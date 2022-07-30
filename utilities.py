# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:55:29 2019

@author: jacquemin
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import integrate
from sklearn import decomposition
from matplotlib.patches import Ellipse
from matplotlib.artist import ArtistInspector
from cycler import cycler
from copy import copy
from itertools import chain
from uncertainties import unumpy
from scipy.optimize import curve_fit

# =============================================================================
# Graph functions
# =============================================================================

def force_aspect(ax, aspect):
    """ Forces the aspect ratio of an axis. """
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    ax.set_aspect(abs((xmax-xmin)/(ymax-ymin))/aspect)

    
def save_fig(Do_Save, Save_Name, PNG=False, PDF=False):
    """ Saves a figure to PNG and/or PDF. """

    if Do_Save:
        if PDF:
            plt.savefig(Save_Name + '.pdf')
        if PNG:
            plt.savefig(Save_Name + '.png')
            
def colorset(n_col=6, cmap='viridis'):
    """ Returns colors from a color map """
    
    col_norm = mpl.colors.Normalize(vmin = 0, vmax = n_col)
    temp = cm.ScalarMappable(norm = col_norm, cmap = cmap)
    cset = []
    for i in range(n_col):
        cset.append(temp.to_rgba(i))
    
    return cset
            
# =============================================================================
# Calc functions
# =============================================================================
    
def long_correction(sigma, lambda_laser, T=21):
     """
     Function computing the Long correction factor according to Long
     1977. This function can operate on numpy.ndarrays as well as on
     simple numbers.
    
     Parameters
     ----------
     sigma : numpy.ndarray
         Wavenumber in cm-1
     lambda_inc : float
         Laser wavelength in nm.
     T : float
         Actual temperature in °C
     T0 : float
         The temperature to which to make the correction in °C
     Returns:
     ----------
     lcorr: numpy.ndarray of the same shape as sigma
    
     Examples
     --------
     >>> sigma, spectra_i = deconvolution.acquire_data('my_raman_file.CSV')
     >>> corrected_spectra = spectra_i * long_correction(sigma)
     """
     c = 2.998e10                          # cm/s
     lambda_inc = lambda_laser * 1e-7      # cm
     sigma_inc = 1. / lambda_inc           # cm-1
     h = 6.63e-34                          # J.s
     TK = 273.0 + T                        # K
     kB = 1.38e-23                         # J/K
     ss = sigma_inc / sigma
     cc = h*c/kB
     return (ss**3 / (ss - 1)**4
             * (1 - np.exp(-cc*sigma*(1/TK))))
 
def bc(spectra):
    """ Removes the baseline offset """
    
    spectra_bc = np.copy(spectra)

    if np.shape(spectra_bc)[0] == 0 :
        spectra_bc -= spectra_bc.min()
    else :
        spectra_bc -= spectra_bc.min(axis=1)[:, np.newaxis]
        
    return spectra_bc

def norm(spectra, mode='area'):
    """ Normalizes the data """
    
    spectra_norm = np.copy(spectra)

    if mode == 'max':
        spectra_norm /= spectra_norm.max(axis=1)[:, np.newaxis]
    elif mode == 'area':
        spectra_norm /= integrate.trapz(spectra_norm)[:, np.newaxis]
    
    return spectra_norm


def clean(sigma, raw_spectra, mode='area', delete=None, long_cor=514, T=21):
    """
    Cleans the spectra to remove abnormal ones, remove the baseline offset,
    correct temperature & frequency effects, and make them comparable
    by normalizing them according to their area or their maximum.

    Parameters
    ----------
    sigma : numpy.ndarray
        Wavenumber in cm-1
    raw_spectra : numpy.ndarray, n_spectra * n_features
        Input spectra
    mode : {'area', 'max'}
        Controls how spectra are normalized
    delete : list of int, default None
        Spectra that should be removed, eg outliers
    long_cor : float, optional
        Laser wavelength in nm. If given, then temperature-frequence correction
        will be applied. If None or False, no correction is applied.
    """
    # Remove abnormal spectra
    if delete is not None:
        clean_spectra = np.delete(raw_spectra, delete, axis=0)
    else:
        clean_spectra = np.copy(raw_spectra)
    # Remove the offset
    if np.shape(clean_spectra)[0] == 0 :
        clean_spectra -= clean_spectra.min()
    else :
        clean_spectra -= clean_spectra.min(axis=1)[:, np.newaxis]
    # Apply Long correction
    if long_cor:
        clean_spectra *= long_correction(sigma, long_cor, T)
    # Normalize the spectra
    if mode == 'max':
        clean_spectra /= clean_spectra.max(axis=1)[:, np.newaxis]
    elif mode == 'area':
        clean_spectra /= integrate.trapz(clean_spectra)[:, np.newaxis]
    elif mode == None:
        clean_spectra = np.copy(clean_spectra)

    return clean_spectra

def barycentre(x, y, area, baseline=True):
    """
    
    Parameters
    ----------
    x : numpy.ndarray
        Wavenumber in cm-1
    y : numpy.ndarray, n_spectra * n_features
        Input spectra
    area : list
        x range for barycentre calculation
    baseline : boolean, optional
        Substracts a linear baseline to flatten the data. The default is True.

    Returns
    -------
    bar : the position of the barycentre on the x axis

    """
    
    # Truncating the data
    condition = (x >= area[0]) & (x <= area[1])
    x_bar = np.copy(x[condition])
    
    if len(y.shape)==1:
        y = y[None,:]
    y_bar = np.copy(y[:,condition])
    
    n = len(y)
    
    # Baseline correction
    if baseline == True:
        
        x1, y1 = x_bar[0], y_bar[:,0]
        x2, y2 = x_bar[-1], y_bar[:,-1]
        
        a = (y2-y1)/(x2-x1)
        b = (x2*y1-x1*y2)/(x2-x1)

        line = np.ones((n,len(x_bar)))
        for _i in range(n):
            line[_i,:] = a[_i]*x_bar.T + b[_i]
        y_bar = y_bar-line
    
    # Calculation
    if x[0]>x[-1]:
        y_calc = np.fliplr(y_bar)
        x_calc = np.flipud(x_bar)
    else:
        x_calc, y_calc = x_bar, y_bar

    bar = np.zeros(n)
    areatot = integrate.trapz(y_calc, x_calc)

    for _k in range(n):
          seuil = areatot[_k]/2
          c = 0
          int_partielle = integrate.trapz(y_calc[_k,:c+1],x_calc[:c+1])
          while int_partielle < seuil:
              c += 1 
              int_partielle = integrate.trapz(y_calc[_k,:c+1],x_calc[:c+1])
          bar[_k] = x_calc[c]
    
    return bar

def trunc(x, spectra, xmin, xmax):
    """ Truncates spectra over the [xmin,xmax] interval """
    
    interval = [xmin, xmax]

    if not interval[0]:
        interval[0] = np.min(x)
    if not interval[1]:
        interval[1] = np.max(x)
    
    _condition = (x >= interval[0]) & (x <= interval[1])
    x_kept = np.copy(x[_condition])
    spectra_kept = np.copy(spectra[:, _condition])
    
    return x_kept, spectra_kept

def PCA_denoising(x, spectra, n_comp, plot=False):
    """ PCA denoising of the dataset """
    
    pca = decomposition.PCA(n_comp)
    pca_fit = pca.fit(spectra)
    pca.n_components = min(n_comp, len(spectra), len(x))
    spectra_reduced = pca.fit_transform(spectra)
    spectra_denoised = pca.inverse_transform(spectra_reduced)
    
    if plot==True:
        fig, ax = plt.subplots()
        ax.plot(np.arange(1,n_comp+1,1),pca.explained_variance_ratio_, '-o') 
        ax.set_title('Scree plot')           
        ax.set_xlabel('Number of PCA components')
        ax.set_xlim(0,n_comp+1)
        ax.set_ylabel('explained variance')
        ax.grid()
        plt.tight_layout()
    
    return spectra_denoised

def deconvolute_nmf(inputspectra, n_components, **kwargs):
    """ NMF Deconvolution of a dataset into n components """
    
    nmf_model = decomposition.NMF(n_components=n_components,
                                  init='nndsvda', solver='mu', max_iter=1000)
    mix = nmf_model.fit_transform(inputspectra)
    components = nmf_model.components_
    reconstructed_spectra = nmf_model.inverse_transform(mix)
    print('Reconstruction error: {:.3e}'.format(nmf_model.reconstruction_err_))
    
    return mix, components, reconstructed_spectra

def find_idx(x, val):
    
    return np.argmin(abs(x - val))

def pV(x:np.ndarray, h:float, x0:float=None, w:float=None, factor:float=0.5):
    '''Creates an pseudo-Voigt profile.
    Parameters:
    ------------
    x : 1D ndarray
        Independent variable (Raman shift for ex.)
    h : float
        height of the peak
    x0 : float
        The position of the peak on the x-axis.
        Default value is at the middle of the x
    w : float
        FWHM - The width
        Default value is 1/3 of the x
    factor : float
        The ratio of Gauss vs Lorentz in the peak
        Default value is 0.5
    Returns:
    --------------
    y : np.ndarray :
        1D-array of the same length as the input x-array
    ***************************
    Example :
    --------------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(150, 1300, 1015)
    >>> plt.plot(x, pV(x, 200))
    '''

    def Gauss(x, w):
        return((2/w) * np.sqrt(np.log(2)/np.pi) * np.exp(
                -(4*np.log(2)/w**2) * (x - x0)**2))

    def Lorentz(x, w):
        return((1/np.pi)*(w/2) / (
                (x - x0)**2 + (w/2)**2))

    if x0 == None:
        x0 = x[int(len(x)/2)]
    if w == None:
        w = (x.max() - x.min()) / 3

    intensity = h * np.pi * (w/2) /\
                    (1 + factor * (np.sqrt(np.pi*np.log(2)) - 1))

    return(intensity * (factor * Gauss(x, w)
                        + (1-factor) * Lorentz(x, w)))


def fitting_function(x, *params):
    '''
    The function giving the sum of the pseudo-Voigt peaks.
    Parameters:
    *params: is a list of parameters. Its length is = 4 * "number of peaks",
    where 4 is the number of parameters in the "pV" function.
    Look in the docstring of pV function for more info on theese.
    '''
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 4):
        result += pV(x, *params[i:i+4])  # h, x0, w, r)
    return result


def multi_pV(x, *params, peak_function=pV):
    '''
    This function returns the spectra as the sum of the pseudo-Voigt peaks,
    given the independent variable `x` and a set of parameters for each peak.
    (one sublist for each Pseudo-Voigt peak).
    Parameters :
    -----------------
    x : np.ndarray
        1D ndarray - independent variable.
    *params : list[list[float]]
        The list of lists containing the peak parameters. For each infividual
        peak to be created there should be a sublist of parameters to be
        passed to the pV function. So that `params` list finally contains
        one of these sublists for each Pseudo-Voigt peak to be created.
        Look in the docstring of pV function for more info on theese params.
    Returns :
    -----------------
    y : np.ndarray
        1D ndarray of the same length as the input x-array
    Example :
    -----------------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(150, 1300, 1015) # Create 1015 equally spaced points
    >>> mpar = [[40, 220, 100], [122, 440, 80], [164, 550, 160], [40, 480, 340]]
    >>> plt.plot(x, multi_pV(x, *mpar))
    '''
    result = np.zeros_like(x, dtype=float)
    n_peaks = int((len(params)+0.1)/4) # Number of peaks
    ipp = np.asarray(params).reshape(n_peaks, 4)
    for pp in ipp:
        result += peak_function(x, *pp)  # h, x0, w, r)
    return result





class fitonclick(object):
    '''This class is used to interactively draw pseudo-voigt (or other type)
    peaks, on top of your data.
    It was originaly created to help defining initial fit parameters to
    pass on to SciPy CurveFit.
    IMPORTANT! See the Example below, to see how to use the class
    Parameters:
        x: independent variable
        y: your data
        initial_GaussToLorentz_ratio:float between 0 and 1, default=0.5
            Pseudo-Voigt peak is composed of a Gaussian and of a Laurentzian
            part. This ratio defines the proportion of those parts.
        scrolling_speed: float>0, default=1
            defines how quickly your scroling widens peaks
        initial_width: float>0, default=5
            defines initial width of peaks
        **kwargs: dictionary, for exemple {'figsize':(9,9)}
            whatever you want to pass to plt.subplots(**kwargs)
    Returns:
        Nothing, but you can access the atributes using class instance, like
        fitonclick.pic: dictionnary containing the parameters of each peak added
        fitonclick.sum_peak: list containing cumulated graph line
            to get the y-values, use sum_peak[-1][0].get_ydata()
        fitonclick.peak_counter: int giving the number of peaks present
        etc.
    Example:
        >>>my_class_instance = fitonclick(x, y)
        >>>while my_class_instance.block:
        >>>    plt.waitforbuttonpress(timeout=-1)
    '''
    # Initiating variables to which we will atribute peak caractéristics:
    pic = {}
    pic['line'] = []  # List containing matplotlib.Line2D object for each peak
    pic['h'] = []  # List that will contain heights of each peak
    pic['x0'] = []  # List that will contain central positions of each peak
    pic['w'] = []  # List containing widths
    pic['GL'] = []
    # List of cumulated graphs
    # (used later for updating while removing previous one)
    sum_peak = []
    peak_counter: int = 0  # number of peaks on the graph
    cum_graph_present: int = 0  # only 0 or 1
    scroll_count = 0  # counter to store the cumulative values of scrolling
    artists = []  # will be used to store the elipses on tops of the peaks

    block = True

    def __init__(self, x, y,
                 initial_GaussToLoretnz_ratio=0.5,
                 scrolling_speed=1,
                 initial_width=5,
                 **kwargs):
        plt.ioff()
        self.x = x
        self.y = y
        self.GL = initial_GaussToLoretnz_ratio
        self.scrolling_speed = scrolling_speed
        self.initial_width = initial_width
        # Setting up the plot:
        self.fig, self.ax = plt.subplots(**kwargs)
        self.ax.plot(self.x, self.y,
                     linestyle='none', marker='o', c='k', ms=4, alpha=0.5)
        self.ax.set_title('Left-click to add/remove peaks,'
                          'Scroll to adjust width, \nRight-click to draw sum,'
                          ' Double-Right-Click when done')
        self.x_size = self.set_size(self.x)
        self.y_size = 2*self.set_size(self.y)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('scroll_event', self.onclick)

    def set_size(self, variable, rapport=70):
        return (variable.max() - variable.min())/rapport

    def _add_peak(self, event):
        self.peak_counter += 1
        one_elipsis = self.ax.add_artist(
                        Ellipse((event.xdata, event.ydata),
                                self.x_size, self.y_size, alpha=0.5,
                                gid=str(self.peak_counter)))
        h = event.ydata
        x0 = event.xdata
        yy = pV(x=self.x, h=h,
                x0=x0, w=self.x_size*self.initial_width, factor=self.GL)
        one_elipsis = self.ax.add_artist(
                        Ellipse((x0, h),
                                self.x_size, self.y_size, alpha=0.5,
                                gid=str(self.peak_counter)))
        self.artists.append(one_elipsis)
        self.pic['line'].append(self.ax.plot(self.x, yy,
                                alpha=0.75, lw=2.5,picker=5))
        #self.pic['line'][-1][0].set_pickradius(5)
        # ax.set_ylim(auto=True)
        self.pic['h'].append(h)
        self.pic['x0'].append(x0)
        self.pic['w'].append(self.x_size*self.initial_width)
        self.fig.canvas.draw_idle()
#        return(self.artists, self.pic)

    def _adjust_peak_width(self, event, peak_identifier=-1):
        self.scroll_count += self.x_size * np.sign(event.step) *\
                             self.scrolling_speed/10

        if self.scroll_count > -self.x_size*self.initial_width*0.999:
            w2 = self.x_size*self.initial_width + self.scroll_count
        else:
            w2 = self.x_size * self.initial_width / 1000
            # This doesn't allow you to sroll to negative values
            # (basic width is x_size)
            self.scroll_count = -self.x_size * self.initial_width * 0.999

        center2 = self.pic['x0'][peak_identifier]
        h2 = self.pic['h'][peak_identifier]
        self.pic['w'][peak_identifier] = w2
        yy = pV(x=self.x, x0=center2, h=h2, w=w2, factor=self.GL)
        active_line = self.pic['line'][peak_identifier][0]
        # This updates the values on the peak identified
        active_line.set_ydata(yy)
        self.ax.draw_artist(active_line)
        self.fig.canvas.draw_idle()
#        return(scroll_count, pic)

    def _remove_peak(self, clicked_indice):
        self.artists[clicked_indice].remove()
        self.artists.pop(clicked_indice)
        self.ax.lines.remove(self.pic['line'][clicked_indice][0])
        self.pic['line'].pop(clicked_indice)
        self.pic['x0'].pop(clicked_indice)
        self.pic['h'].pop(clicked_indice)
        self.pic['w'].pop(clicked_indice)
        self.fig.canvas.draw_idle()
        self.peak_counter -= 1
#        return(artists, pic)

    def _draw_peak_sum(self):
        if self.peak_counter < 1:
            return

        def _remove_sum(self):
            assert self.cum_graph_present == 1, "no sum drawn, nothing to remove"
            self.ax.lines.remove(self.sum_peak[-1][0])
            self.sum_peak.pop()
            self.cum_graph_present -= 1
#            return sum_peak

        def _add_sum(self, sumy):
            assert sumy.shape == self.x.shape, "something's wrong with your data"
            self.sum_peak.append(self.ax.plot(self.x, sumy, '--',
                                              color='lightgreen',
                                              lw=3, alpha=0.6))
            self.cum_graph_present += 1
#            return sum_peak

        # Sum all the y values from all the peaks:
        sumy = np.sum(np.asarray(
                [self.pic['line'][i][0].get_ydata() for i in range(self.peak_counter)]),
                axis=0)
        # Check if there is already a cumulated graph plotted:
        if self.cum_graph_present == 1:
            # Check if the sum of present peaks correponds to the cumulated graph
            if np.array_equal(self.sum_peak[-1][0].get_ydata(), sumy):
                pass
            else:  # if not, remove the last cumulated graph from the figure:
                _remove_sum(self)
                # and then plot the new cumulated graph:
                _add_sum(self, sumy=sumy)
        # No cumulated graph present:
        elif self.cum_graph_present == 0:
            # plot the new cumulated graph
            _add_sum(self, sumy=sumy)

        else:
            raise("WTF?")
        self.fig.canvas.draw_idle()
#        return(cum_graph_present, sum_peak)

    def onclick(self, event):
        if event.inaxes == self.ax:  # if you click inside the plot
            if event.button == 1:  # left click
                # Create list of all elipes and check if the click was inside:
                click_in_artist = [art.contains(event)[0] for art in self.artists]
                if any(click_in_artist):  # if click was on any of the elipsis
                    clicked_indice = click_in_artist.index(True) # identify the one
                    self._remove_peak(clicked_indice=clicked_indice)

                else:  # if click was not on any of the already drawn elipsis
                    self._add_peak(event)

            elif event.step:
                if self.peak_counter:
                    self._adjust_peak_width(event, peak_identifier=-1)
                    # -1 means that scrolling will only affect the last plotted peak

            elif event.button !=1 and not event.step:
                # On some computers middle and right click have both the value 3
                self._draw_peak_sum()

                if event.dblclick:
                    print('kraj')
                    # Double Middle (or Right?) click ends the show
                    assert len(self.pic['line']) == self.peak_counter
                    assert self.cum_graph_present == len(self.sum_peak)
                    self.fig.canvas.mpl_disconnect(self.cid)
                    self.fig.canvas.mpl_disconnect(self.cid2)
                    self.pic['GL'] = [self.GL] * self.peak_counter
                    self.block = False
                    
                    
                    
                    
                    
                    
class fitspectra(object) :
    '''This class is used to interactively draw pseudo-voigt (or other type)
    peaks, on top of your data.
    It was originaly created to help defining initial fit parameters to
    pass on to SciPy CurveFit.
    IMPORTANT! See the Example below, to see how to use the class
    Parameters:
        x: independent variable
        y: your data
        initial_GaussToLorentz_ratio:float between 0 and 1, default=0.5
            Pseudo-Voigt peak is composed of a Gaussian and of a Laurentzian
            part. This ratio defines the proportion of those parts.
        scrolling_speed: float>0, default=1
            defines how quickly your scroling widens peaks
        initial_width: float>0, default=5
            defines initial width of peaks
        **kwargs: dictionary, for exemple {'figsize':(9,9)}
            whatever you want to pass to plt.subplots(**kwargs)
    Returns:
        Nothing, but you can access the atributes using class instance, like
        fitonclick.pic: dictionnary containing the parameters of each peak added
        fitonclick.sum_peak: list containing cumulated graph line
            to get the y-values, use sum_peak[-1][0].get_ydata()
        fitonclick.peak_counter: int giving the number of peaks present
        etc.
    Example:
        >>>my_class_instance = fitonclick(x, y)
        >>>while my_class_instance.block:
        >>>    plt.waitforbuttonpress(timeout=-1)
    '''
    #pic = {}
    #pic['line'] = []
    #pic['h'] = []
    #pic['x0'] = []
    #pic['w'] = []
    #pic['GL'] = []
    #sum_peak = []
    rc = 0
    peaks_counter = 0
    scroll_count = 0
    #artists = []
    #initial_width = 5
    #scrolling_speed = 1
    
    def __init__(self,x, y, initial_GaussToLoretnz_ratio=0.5,
                 scrolling_speed=1,initial_width=5,pic={},
                 clicked_indice = -1,artists=[],sum_peak = [],**kwargs) :
        self.x = x
        self.y = y
        self.GL = initial_GaussToLoretnz_ratio
        self.scrolling_speed = scrolling_speed
        self.initial_width = initial_width
        self.fig, self.ax = plt.subplots(**kwargs)
        self.ax.plot(self.x, self.y, c='k', alpha=0.5)
        self.ax.set_title('Left-click to add/remove peaks,'
                          'Scroll to adjust width, Right-click to draw the sum')
        self.x_size = self.set_size(self.x)
        self.y_size = 2*self.set_size(self.y)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('scroll_event',self.onclick)
        self.pic = pic
        self.pic['line'] = []
        self.pic['h'] = []
        self.pic['x0'] = []
        self.pic['w'] = []
        self.pic['GL'] = []
        self.clicked_indice= clicked_indice
        self.artists=[]
        self.sum_peak = []
    def set_size(self, variable, rapport = 70):
        return (variable.max() - variable.min())/rapport
    
    def reset(self):
        self.__init__()
    
    def onclick(self,event):
        
        if event.inaxes == self.ax :
            if event.button == 1 :
                click_in_artist = [artist.contains(event)[0] for artist in self.artists]
                if not any(click_in_artist):
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
                    
                elif any(click_in_artist) :
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
                    
            elif event.button ==3 and not event.step :
                if self.rc > 0 :
                    self.ax.lines.remove(self.sum_peak[-1][0])
                    self.sum_peak.pop()
                        
                self.sumy = np.sum(np.asarray([self.pic['line'][i][0].get_ydata() 
                                             for i in range(self.peaks_counter)]), axis=0)
                if self.sumy.shape == self.x.shape :
                    self.sum_peak.append(self.ax.plot(self.x,self.sumy, '--', color='lightgreen'
                                                         ,lw=3, alpha=0.6))
                    self.rc +=1
                else :
                    self.rc = -1
                self.fig.canvas.draw_idle()
                    
            if event.step !=0 :
                if self.peaks_counter :
                    peak_identifier = -1
                    self.scroll_count += self.x_size*np.sign(event.step)*self.scrolling_speed/10
                    if self.scroll_count > -self.x_size*self.initial_width*0.999:
                        w2 = self.x_size*self.initial_width + self.scroll_count
                    else :
                        w2 = self.x_size * self.initial_width / 1000
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
                                
                    
def fitparams(x , y, x_size, peaks_present, pic, sum_peak,**kwargs) :
    # this function will plot our fit result and show fit parameters in a table
    # creating the list of initial parameters from your manual input:
    # (as a list of lists)
    manualfit_components_params = copy(list(map(list, zip(
        pic['h'], pic['x0'], pic['w'], pic['GL']))))
    
    
    # to transform the list of lists into one single list:
    manualfit_components_params = list(chain(*manualfit_components_params))

    # the sum of manually created peaks:
    assert len(sum_peak) > 0, 'No peaks initiated'
    manualfit = sum_peak[0][0].get_data()[1]
    
    # Setting the bounds based on your input
    # (you can play with this part if you feel like it,
    # but leaving it as it is should be ok for basic usage)

    # set the initial bounds as infinities:
    #upper_bounds = np.ones_like(manualfit_components_params)*np.inf
    upper_bounds = np.ones(len(manualfit_components_params))*np.inf
    lower_bounds = np.ones(len(manualfit_components_params))*(-np.inf)
    #lower_bounds = np.ones_like(manualfit_components_params)*(-np.inf)
    
    # setting reasonable bounds for the peak amplitude
    # as a portion to your initial manual estimate
    upper_bounds[0::4] = [A*1.4 for A in manualfit_components_params[0::4]]
    #upper_bounds[:, 0] = np.asarray([A[0]*1.4 for A in manualfit_components_params])
    #lower_bounds[:, 0] = np.asarray([A[0]*0.7 for A in manualfit_components_params])
    lower_bounds[0::4] = [A*0.7 for A in manualfit_components_params[0::4]]
    
    
    # setting reasonable bounds for the peak position
    #as a shift in regard to your initial manual position
    #upper_bounds[:, 1] = np.asarray([A[1] + 2*x_size for A in manualfit_components_params])
    upper_bounds[1::4] = [A + 2*x_size for A in manualfit_components_params[1::4]]
    #lower_bounds[:, 1] = np.asarray([A[1] - 2*x_size for A in manualfit_components_params])
    lower_bounds[1::4] = [A - 2*x_size for A in manualfit_components_params[1::4]]
    # setting the bounds for the widths
    
    #upper_bounds[:, 2] = np.asarray([A[2]*16 for A in manualfit_components_params])
    upper_bounds[2::4] = [A*10 for A in manualfit_components_params[2::4]]
    #lower_bounds[:, 2] = np.asarray([A[2]*0.5 for A in manualfit_components_params])
    lower_bounds[2::4] = [A*0.5 for A in manualfit_components_params[2::4]]
    
    # setting the bounds for the Gauss/Lorentz ratio
    #upper_bounds[:, 3] = 1
    #lower_bounds[:, 3] = 0
    
    upper_bounds[3::4] = 1
    lower_bounds[3::4] = 0
    
    
    #bounds = (lower_bounds.ravel(), upper_bounds.ravel())
    bounds = (lower_bounds, upper_bounds)
    
    
    mfcp = np.asarray(manualfit_components_params).ravel()
    # The curve-fitting part:
    
    fitted_params, b = curve_fit(fitting_function, x, y, method='trf'
                                 , p0=manualfit_components_params
                                ,absolute_sigma=False, bounds=bounds)
    
    fitting_err = np.sqrt(np.diag(b))
    y_fitted = fitting_function(x, *fitted_params)
    
    # Plotting the results of the optimization:
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    errax = fig.add_axes([0.125, 0.1, 0.775, 0.1])
    errax.set_facecolor('w')
    ax.plot(x, manualfit, '--g', alpha=0.5, label='initial manual fit')
    ax.plot(x, y, alpha=0.3, label ='original data')
    ax.plot(x, y_fitted, '--r', lw=4, alpha=0.6, label ='after optimization')
    
    ax.legend()
    
    errax.plot(x, y-y_fitted)
    errax.set_ylabel('error\n(data-fit)')
    errax.set_xlabel(f'fitting error = '
                    f'{np.sum(fitting_err/np.ceil(fitted_params))/peaks_present:.3f}'
                    f'\n\u03A3(\u0394param/param) /n_peaks')
    
    errax2 = errax.twinx()
    errax2.set_yticks([])
    errax2.set_ylabel(f'\u03A3(\u0394y) = {np.sum(y-y_fitted):.2f}',
                     fontsize='small')
    ax.set_title('after fitting')
    
    # Plotting the individual peaks after fitting
    
    pfig, pax = plt.subplots()
    
    pax.plot(x, y, alpha=0.3)
    pax.plot(x, y_fitted, '--k',lw=2,alpha=0.6, label = 'fit')
    
    par_nam = ['h', 'x0', 'w', 'G/L']
    for i in range(peaks_present):
        fit_res = list(zip(par_nam, fitted_params[i*4:i*4+4],
                          fitting_err[i*4:i*4+4]))
        label = [f"{P}={v:.2f}\U000000B1{e:.1f}" for P, v, e in fit_res]
        yy_i = pV(x, *fitted_params[i*4:i*4+4])
        peak_i, = pax.plot(x, yy_i, alpha=0.5, label=label)
        pax.fill_between(x, yy_i, facecolor=peak_i.get_color(), alpha=0.3)
    pax.set_title('Showing the individual peaks as found by fitting procedure')
    plt.subplots_adjust(left=0.2, bottom=0.3)
    rows = ["Peak %d" % i for i in range(peaks_present)]
    cell_text1 = np.round(fitted_params,3).reshape(len(fitted_params)//4,4).tolist()
    cell_text2 = np.round(fitting_err,3).reshape(len(fitting_err)//4,4).tolist()
    after = unumpy.uarray(cell_text1,cell_text2)
    colors = plt.cm.Greys(np.linspace(0, 0.5, len(rows)))
    columns = ['Height','Center','FWMH','Ratio G/L']
    
    #fig.patch.set_visible(False)
    the_table = pax.table(cellText=after,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom',bbox=[0.1, -0.5, 0.9, 0.4])
    
         
        
                      
                      
                        
    