import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from scipy import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from tifffile import tifffile
import seaborn as sns
from typing import Dict, List

class NanoImgPro():
  def __init__(self, tiff_stack_path:str, roi_size:int, stim_side:str='left',
               stim_frame:int = 200, ma_tail_start:int = 485):
    """
    Initiate the image processing tool

    ## Parameters
    tiff_stack_path : str
          Path to file as a string
    roi_size : int
          ROI box size (e.g. a value of 15 will result in roi boxes that are 15 x 15 pixels)
    stim_side : str='left', options are "right" or "left", optional
          Places incomplete boxes farthest away from stimulator location
    stim_frame : int = 200, optional
          ID 
    ma_tail_start:int = 485



    """
    self.tiff_stack_path:str = tiff_stack_path
    self.stack:np.ndarray = tifffile.imread(tiff_stack_path) / 255 # image file
    self.roi_size:int = roi_size # size of the roi in pixels
    self.stim_side:str = stim_side # location of the stimulator
    self.stim_frame:int = stim_frame
    self.ma_tail_start:int = ma_tail_start
    self.sig_roi_traces:Dict[str:np.ndarray] = {}
    self.sig_roi_metrics:Dict[str:np.ndarray] = {}
    self.sig_roi_fit_trace:Dict[str:np.ndarray] = {}
    self.traces_df:pd.DataFrame = pd.DataFrame()
    self.metrics_df = pd.DataFrame()
    self.fit_df = pd.DataFrame()
    self.save_path:str = ''
    self.version:str = '_v0-2'

    # private variables 
    self._xsplits:np.ndarray = np.array([]) # where to break on x axis
    self._ysplits:np.ndarray = np.array([]) # where to break on y axis
    
    # get ready to save
    name_suffix = '_'+str(self.roi_size) + '_pxls' + self.version

    if "MMStack_Pos0" in self.tiff_stack_path:
      self.save_path:str = self.tiff_stack_path.replace('MMStack_Pos0.ome.tif', name_suffix)

    else:
      self.save_path:str = self.tiff_stack_path.replace('.ome.tif', name_suffix)


  def process_file(self, loading_bar:bool=True) -> None:
    """
    Process the file
     
    ## Parameters

    loading_bar : bool, optional
         Turns tqdm loading bar on/off

    ##  Returns 

    None
         Updates are applied to class variables


    """
    self.__pixel_splits()
    roi_num_counter = 0
    if loading_bar:
      with tqdm(total=100) as pbar:
        for x_iter in range(len(self._xsplits)-1):
          for y_iter in range(len(self._ysplits)-1):
            roi_array = self.stack[:,
                              self._xsplits[x_iter]:self._xsplits[x_iter+1],
                              self._ysplits[y_iter]:self._ysplits[y_iter+1]]
            roi_data_pooled = np.mean(roi_array, axis=(1,2))
            self.__roi_proccessing(roi_data_pooled, x_iter, y_iter, roi_num_counter)
            roi_num_counter+=1
          

          pbar.update(np.around(100/len(self._xsplits)))
    
    else:
      for x_iter in range(len(self._xsplits)-1):
          for y_iter in range(len(self._ysplits)-1):
            roi_array = self.stack[:,
                              self._xsplits[x_iter]:self._xsplits[x_iter+1],
                              self._ysplits[y_iter]:self._ysplits[y_iter+1]]
            roi_data_pooled = np.mean(roi_array, axis=(1,2))
            self.__roi_proccessing(roi_data_pooled, x_iter, y_iter, roi_num_counter)
            roi_num_counter+=1

    self.traces_df = pd.DataFrame.from_dict(self.sig_roi_traces, orient='index')
    self.metrics_df = pd.DataFrame.from_dict(self.sig_roi_metrics, orient='index')
    self.fit_df = pd.DataFrame.from_dict(self.sig_roi_fit_trace, orient='index')
    

  def plot_metric_heatmaps(self, save:bool=False, display:bool=False, dpi:int=250) -> None:
    """
    Create Heatmaps for dFoF and Tau Off and save them if desired
     
    ## Parameters

    save : bool, optional, default = False
         Save the generated heatmap as a png
    
    display : bool, optional, default = False
         Should the figure be displayed

    dpi : int, optional, default = 250
         Resolution of generated figure

    ##  Returns 

    None
    
    """
    plt.close()
    plt.clf()

    significant_rois = np.zeros((512, 640))
    tau_off_img = np.zeros((512, 640))

    for roi in self.metrics_df.index:
      roi_info = self.metrics_df.loc[roi]
      x_start = roi_info['x_begin']
      x_end = roi_info['x_end']
      y_start = roi_info['y_begin']
      y_end = roi_info['y_end']
      
      significant_rois[int(x_start):int(x_end), int(y_start):int(y_end)] = roi_info['max_dFoF']
      tau_off_img[int(x_start):int(x_end), int(y_start):int(y_end)] = roi_info['tau_off'] 

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5), dpi=dpi)
    plt.subplot(1, 2, 1)
    plt.title('Max dF over F0')
    axes[0] = plt.imshow(significant_rois, vmin=0, vmax=1, cmap='Reds', aspect='auto')
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Tau Off')
    axes[1] = plt.imshow(tau_off_img, vmin=0, vmax=25, cmap='Blues', aspect='auto')
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    plt.colorbar()
    
    if save:
      plt.savefig(self.save_path+'_heatmaps.png', dpi=dpi)

    if display:
      plt.show()

  def plot_average_trace(self, save:bool=False, display:bool=False, dpi:int=250):
    """
    Call this fuction to plot the average dF over F0 trace 
    """
  
    plt.close()

    traces_df_melt = self.traces_df.melt(var_name='Time (s)', value_name='dFoF (a.u.)')
    traces_df_melt['Time (s)'] = traces_df_melt['Time (s)']/8.33
    traces_df_melt['Time (s)']
    fig = plt.figure(figsize=(8,5), dpi=dpi)
    sns.lineplot(data = traces_df_melt, x='Time (s)', y='dFoF (a.u.)', ci='sd')

    if save:
      plt.savefig(self.save_path+'_avg_trace.png', dpi=dpi)

      
    if display:
      plt.show()

  def save_data(self, path=None) -> None:
    """
    Save the pandas class variables that store the processed data
    """
    self.traces_df.to_excel(self.save_path+'_traces.xlsx')
    self.metrics_df.to_excel(self.save_path+'_metrics.xlsx')
    self.fit_df.to_excel(self.save_path+'_fits.xlsx')


    
  # staticmethod
  def __baseline_als(self, y_als:np.ndarray, lam:int=1E7, p:float=0.25, niter:int=20) -> np.ndarray:
    """
    Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens in 2005. 
    Adapted from Stack Overflow Answer https://stackoverflow.com/a/57431514
    """
    
    L = y_als.shape[0]
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y_als)
        w = p * (y_als > z) + (1-p) * (y_als < z)
    return z # return array to subtract from trace  

  # staticmethod
  def __baseline_ma(self, trace:np.ndarray, window:int=40, early_frame:int=190, late_frame:int=485):
    """
    Calculate a moving average baseline of the trace
    This method exempts the center of the trace
    In other words it only uses the time before the stimulation and the last 15s to create a trace
    the middle component is just a line
    """
    ma_bl = np.zeros(trace.shape) # moving average baseline initiation
    ma_bl[window-1:] = self.__moving_average(trace, window) #forward pass
    ma_bl[:window] = np.flip(self.__moving_average(np.flip(trace), window))[:window] #backward pass
    #fill in center with a line to avoid stimulation
    # if this is a tad short add one more value to the end

    # deal with exception that cant calculate flat traces
    try : linear_fill_in = np.arange(ma_bl[early_frame-1], ma_bl[late_frame-1],
                          (ma_bl[late_frame-1] - ma_bl[early_frame-1])/(late_frame-early_frame))
    except ValueError:
      linear_fill_in = np.ones(int(late_frame-early_frame)) * np.mean([ma_bl[early_frame-1], ma_bl[late_frame-1]])

    if len(linear_fill_in) == len(ma_bl[early_frame:late_frame]):
      ma_bl[early_frame:late_frame] = linear_fill_in

    elif len(linear_fill_in) < len(ma_bl[early_frame:late_frame]):
      for i in range(len(ma_bl[early_frame:late_frame]) - len(linear_fill_in)):
        linear_fill_in = linear_fill_in.append(ma_bl[late_frame])
      ma_bl[early_frame:late_frame] = linear_fill_in

    elif len(linear_fill_in) > len(ma_bl[early_frame:late_frame]):
      to_cut = int(len(linear_fill_in)-len(ma_bl[early_frame:late_frame]))
      ma_bl[early_frame:late_frame] = linear_fill_in[:-to_cut]
      
    return ma_bl # return the MA baseline 


  # staticmethod
  def __moving_average(self, a:np.ndarray, window:int) -> np.ndarray:
    """ 
    Calculate a moving average 
    a: array to average
    window: size of the rolling average window

    from https://stackoverflow.com/a/14314054 
    """
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

    
  def __baseline_trace(self, trace):
    """ Workflow for baselining a trace. 
    Input:
    pooled trace must be a 1d array  
     """
    baseline_flouresence_divisor = trace[self.stim_frame-30:self.stim_frame-5].mean()
    trace_magnitude_adjusted = trace.copy() / baseline_flouresence_divisor
    baselined_trace = trace_magnitude_adjusted - self.__baseline_als(trace_magnitude_adjusted)
    dFoF_trace = baselined_trace - self.__baseline_ma(baselined_trace,
                                                early_frame = self.stim_frame-10,
                                                late_frame = self.ma_tail_start)
    return dFoF_trace


  def __pixel_splits(self)-> np.ndarray:
    """
    The purpose of this function is to determine at which pixels we should split 
    the image up
    """
    x_pixels, y_pixels = self.stack[0, :, :].shape # get size information 

    if self.stim_side == 'left': # put fringe ROIs far from stimulator
    ### switch to append for compatability with numba
      self._xsplits = np.insert(np.flip(np.arange(x_pixels, 0, -self.roi_size)), 0, 0)
      self._ysplits = np.insert(np.flip(np.arange(y_pixels, 0, -self.roi_size)), 0, 0)
    else: # put fringe ROIs far from stimulator (assumes right side)
      self._xsplits = np.insert((np.arange(0, x_pixels, self.roi_size)), -1, x_pixels)
      self._ysplits = np.insert((np.arange(0, y_pixels, self.roi_size)), -1, y_pixels)
    
    #future versions could add up and down

  # staticmethod
  def __roi_function_fit(self, baselined_trace:np.ndarray, starting_frame=200):
    """
    Fit "fitting_function" to the baselined trace 
    Input: 
    Baselined Trace : np.ndarray
    Starting Frame : stimulation frame 

    return 
    Fit array : np.ndarray 
    Function Scale factor : int
    Tau On : int
    Tau Off : int
    Function Shift : int
    """

    window_start = starting_frame - 5 
    y_window = baselined_trace[window_start:]
    x_window = np.arange(0, 600)[window_start:]/8.33 - window_start / 8.33
    fitting_bounds = ([-1,0.001,0.001,-1], [5,200,100,1])

    try:  optimal_values, _ = curve_fit(self.__fitting_function, x_window, y_window, bounds=fitting_bounds)
    except RuntimeError: optimal_values = (5, 100, 100, 1)
    shifted_trace = np.zeros(len(baselined_trace))
    shifted_trace[window_start:] = self.__fitting_function(x_window, *optimal_values)
    fit_trace = np.where(shifted_trace<0, 0, shifted_trace)
    
    return fit_trace, optimal_values[0], optimal_values[1], optimal_values[2], optimal_values[3]

  # staticmethod
  def __fitting_function(self, x, scale_factor, tau_on, tau_off, shift):
    """
    Used for fitting in roi_function_fit
    """
    return scale_factor*(1-np.exp(-x/tau_on))*np.exp(-x/tau_off) + shift 
    


  def __roi_proccessing(self, pooled_trace:np.ndarray, x_iter:int, y_iter:int, roi_num:int):
    """ Process an individual ROI """

    baselined_roi_trace = self.__baseline_trace(pooled_trace)

    ## check for significance --- 3 S.D. + mean of 10 frames before stim
    sig_threshold = ((3 * baselined_roi_trace[:self.stim_frame-5].std())
     + baselined_roi_trace[self.stim_frame-12:self.stim_frame-2].mean())
    
    trace_argmax = np.argmax(baselined_roi_trace[self.stim_frame:self.stim_frame+50])
    max_dfof =  baselined_roi_trace[self.stim_frame+trace_argmax:self.stim_frame+trace_argmax+5].mean()

    neg_sig_threshold= (baselined_roi_trace[self.stim_frame-12:self.stim_frame-2].mean() -
                      (3 * baselined_roi_trace[:self.stim_frame-5].std()))
    min_dfof =  baselined_roi_trace[self.stim_frame:self.stim_frame+200].min()

    # if min_dfof < 0:
    #   print(min_dfof, trace_argmin)

    if max_dfof >= sig_threshold:
      if min_dfof >= neg_sig_threshold:
        # if we saw a max value above significance in the next 50 frames
        fit_trace, scale, tau_on, tau_off, shift = self.__roi_function_fit(
            baselined_roi_trace, starting_frame=self.stim_frame)
        
        sig_shadow = (baselined_roi_trace[self.stim_frame-12:self.stim_frame-2].mean() -
                      (3 * baselined_roi_trace[:self.stim_frame-5].std()))
      
        self.sig_roi_traces['roi_'+str(roi_num)] = baselined_roi_trace
        self.sig_roi_fit_trace['roi_'+str(roi_num)] = fit_trace
        self.sig_roi_metrics['roi_'+str(roi_num)] = {
            'x_begin': int(self._xsplits[x_iter]),
            'x_end': int(self._xsplits[x_iter+1]),
            'y_begin': int(self._ysplits[y_iter]),
            'y_end': int(self._ysplits[y_iter+1]),
            'x_center': int(np.mean((self._xsplits[x_iter+1],
                                    self._xsplits[x_iter]))),
            'y_center': int(np.mean((self._ysplits[y_iter+1],
                                    self._ysplits[y_iter]))),
            'max_dFoF':max_dfof,
            'min_dFoF':min_dfof,
            'tau_on': tau_on, 
            'tau_off': tau_off, 
            'shadow_suspected':baselined_roi_trace[self.stim_frame:self.stim_frame+10].min() < sig_shadow}
 

  def __draw_roi_grid(self):
    pass

  def __reconstruct_trace(self):
    """
    Deep learning based bubble corrections: currently work in progress
    """
    pass