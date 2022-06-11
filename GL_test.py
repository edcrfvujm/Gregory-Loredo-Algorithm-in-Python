"""
@author: Xiars

Module for the Gregory-Loredo algorithm.

It is a python implement of 

Gregory, P.C. and Loredo, T.J. (1992) 
‘A new method for the detection of a periodic signal of unknown shape and period’,
The Astrophysical Journal, 398, p. 146. doi:10.1086/171844.

"""
import sys
import copy
import scipy
from scipy.special import comb,factorial
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

def lfac(n): 
    """
    Comput lg(n!) in a fast way
    """
    n=np.array(n,'float')
    n[n>170]=0.5*np.log10(2*np.pi*n[n>170])+n[n>170]*np.log10(n[n>170]/np.e)
    n[n<=170]=np.log10(factorial(n[n<=170]))
    return n



def lsum(lgx):
    """
    Compute the lg sum of x
    input : lg(x)
    output : lg(sum(x))
    """
    return np.max(lgx) + np.log10(np.sum(10.**(lgx - np.max(lgx))))

def lint_newton(lgx,delta): 
    """
    Compute the lg integral of x using Newton method
    input : lg(x)
    output : lg(integral(x,delta))
    """
    return lsum(np.array([lsum(lgx[1:]),lsum(lgx[:-1])]))-np.log10(2)+np.log10(delta)

def lint_delta(lgx,delta): 
    """
    Compute the lg integral of x using discrete method
    input : lg(x)
    output : lg(integral(x,delta))
    """
    return lsum(lgx) + np.log10(delta)

class GL(object):
    """
    Class to handle a time series with potential observation gaps.
    
    One should generate one of these objects for each appropriate epoch of a time series, in which there
    should not be too much gaps. 
    
    """
    def __init__(self, obs_t, T=None, obs_cover=None):
        """
        Args:
            obs_t ('array' or 'list')
                An array of arrival times of photons.
            T ('float')
                The baseline of the time series. Also the range (start at 0 and end at T)
                that one wants to estimate the light curve shape.
            obs_cover (['array','array'])
                Contains two arrays standing for observation covered range.
                One array is start times of each observation.
                The other is stop times of them.
        """
        self.N = np.float64(len(obs_t))
        self.obs_t = np.array(obs_t, 'float')
            
        if obs_cover is not None:
            
            self.t_start = np.array(obs_cover[0], 'float')
            self.t_stop = np.array(obs_cover[1], 'float')
            if T is not None: self.T = np.float64(T)
            else: 
                self.T = self.t_stop [-1] - self.t_start[0]
                self.t_stop = self.t_stop - self.t_start[0]
                self.obs_t = self.obs_t - self.t_start[0]
                self.t_start = np.array([0.], 'float')
               
        else:
            if T is not None:
                self.T = np.float64(T)
                self.t_start = np.array([0.], 'float')
                self.t_stop = np.array([self.T], 'float')
                
            else:
                self.t_start = np.array([0.], 'float')
                self.t_stop = np.array(np.ptp(obs_t), 'float')             
                self.T = self.t_stop
                self.obs_t = obs_t - np.min(obs_t)
        
        self.t0 = None
        self.f_est = None
        self.sigma_est = None
        self.lgO_period = None
        self.P_period = None
        self.m_opt = None
        self.P_w = None
        self.w_array = None
        self.w_peak = None
        self.w_mean = None
        self.w_comf = None       
        self.lgOm1 = None
        self.lgOm1_w = None       
        self.w_lo = None
        self.delta_w = None        
        self.m_max = None
        self.fig = None
        
    def ss(self, t_0, t_1): 
        """
        Compute Start time to Stop time in specific range

        Args:
            t_0 : the start of the range
            t_1 : the end of the range
        """
        a = self.t_start
        b = self.t_stop
        s0 = a[(a > t_0) &(a < t_1 )]
        s1 = b[(b > t_0) &(b < t_1 )]
        if np.size(s0)==0 and np.size(s1)==0 and np.sum(a<=t_0) == np.sum(b<t_1):
            return np.array([-1]),np.array([-1])
        else: 
            if np.size(s0)==0 or (np.size(s1)!=0 and s0[0] > s1[0]) : s0=np.insert(s0,0,t_0)
            if np.size(s1)==0 or s0[-1] > s1[-1]: s1=np.append(s1, t_1)
        return s0, s1

    def compute_ns(self, m, w, phi):
        """
        n_j : the histogram after folding the period
        s_j : the sum_fraction of (Start time to Stop time) in specific range
        time_grid : temporal histogram scheme for compute n_j and s_j
        """
        period = 2*np.pi/w
        time_grid = np.linspace(-period*phi/(2*np.pi), period*(self.T//period+1), int(m*(self.T//period+1)+1))

        n_grid, _= np.histogram(self.obs_t, bins= time_grid)
        n_grid= np.array(n_grid, 'float')
        n_grid = np.append(n_grid, np.repeat(0,m-len(n_grid)%m))
        n_j = np.sum(n_grid.reshape(-1,m), axis=0)
        
        if np.sum(n_j)!=len(self.obs_t): print("Error: not all photons are included in n_j\n"+str(n_j))
        
        tau = np.zeros(len(time_grid)-1)
        for i in range(len(tau)):
            s0, s1 = self.ss(time_grid[i], time_grid[i+1])
            tau[i] = np.sum(s1 - s0)
        tau = np.append(tau, np.repeat(0,m-len(tau)%m))
        s_j = np.float64(m)*np.sum(tau.reshape(-1,m), axis=0)/np.sum(self.t_stop-self.t_start)
        return n_j, s_j, time_grid

    def compute(self, m_max=12, w_range=None, delta_phi=None, delta_w=None, est_step=1000):
        """
        m_max : the maximum bins for each period, experically 12-15
        w_range : (['float', 'float']) standing for the range of angular frequency considered to traverse.
        delta_phi : the step to traverse the phase
        delta_w : the step to traverse the angular frequency
        est_step : the resolution of the estimated light curve
        """

        if w_range is None: w_lo, w_hi = (2*np.pi/self.T, np.minimum(np.pi*200/self.T,np.pi*self.N/self.T))
        else: w_lo, w_hi = w_range
        if delta_phi is None: delta_phi = 0.1*np.pi
        if delta_w is None: delta_w=np.pi/self.T

        t0 = np.linspace(0, self.T, est_step)
        v = m_max - 1
        w_array = np.arange(w_lo, w_hi, delta_w)
        lgOm1_w = np.zeros((m_max, len(w_array)), 'float')

        f_m = np.zeros((m_max,len(t0)))
        sigma_m = np.zeros((m_max,len(t0)))
        for m in np.arange(1, m_max+1):
            sys.stdout.write('\r'+str(m)+"/"+str(m_max))
            lgP_wphi = []
            f_wphi = []
            sigma_wphi = []
            for i,w in zip(range(len(w_array)),w_array):
                phi_array = np.arange(0, 2*np.pi/m, delta_phi)
                y = np.zeros(len(phi_array), 'float')
                for k in range(0, len(y)):
                    n_j, s_j, time_grid= self.compute_ns(m, w, phi_array[k])
                    lgS =  -np.sum(n_j[s_j!=0]*np.log10(s_j[s_j!=0]))
                    y[k] = np.sum(lfac(n_j))+lgS

                    h_j = (n_j+1)/((self.N+m))
                    h_j[s_j == 0 ] = 1/m
                    h_j[s_j !=0 ] = h_j[s_j !=0 ]/s_j[s_j !=0 ]
                    if np.any(h_j>10):print(s_j)
                    sigma = np.sqrt(abs(n_j/self.N*(1-n_j/self.N)/(self.N+m+1)))
                    sigma[s_j == 0 ] = 1
                    sigma[s_j != 0 ] = sigma[s_j !=0 ]/s_j[s_j !=0 ]
                    filllist = np.tile(h_j, int(np.ceil((len(time_grid))/len(n_j))))
                    fulllist = np.tile(sigma, int(np.ceil((len(time_grid))/len(n_j))))
                    repeat_n, _ = np.histogram(t0, bins= time_grid)
                    if np.sum(repeat_n) != len(t0): print("error")
                    roll = int(np.sum(t0 > time_grid[-1]))
                    f_wphi.append(np.roll(np.repeat(filllist[:len(time_grid)-1], repeat_n), -roll))
                    sigma_wphi.append(np.roll(np.repeat(fulllist[:len(time_grid)-1], repeat_n), -roll))
                    lgP_wphi.append(-np.log10(w)-lfac(self.N)+np.sum(lfac(n_j))+lgS)
                lgX_wphi = lint_delta(y, delta_phi)

                lgOm1_w[m-1, i] = self.N*np.log10(m)-np.log10(2*np.pi*v)+lfac(m-1)-lfac(self.N+m-1) + lgX_wphi +np.log10(m) 

            lgP = np.array(lgP_wphi) - lsum(np.array(lgP_wphi))
            f_m[m-1,:] = np.sum(np.array(f_wphi) * (10**lgP).reshape(-1,1),axis=0)
            sigma_m[m-1,:] =  (np.sum(np.array(sigma_wphi) * 10.**lgP.reshape(-1,1),axis=0))    

        lgOm1 = np.zeros(m_max)
        for i in range(m_max):
            lgOm1[i] = lint_delta(lgOm1_w[i]-np.log10(w_array), delta_w) - lint_delta(-np.log10(w_array), delta_w)
        lgOm1[0] += np.log10(v)

        self.f_est = np.sum((np.arange(1, m_max+1).reshape(-1,1)*(10**(lgOm1-lsum(lgOm1)).reshape(-1,1)*np.array(f_m)))[:],axis=0)
        self.sigma_est = np.sum((np.arange(1, m_max+1).reshape(-1,1)*(10**(lgOm1-lsum(lgOm1)).reshape(-1,1)*np.array(sigma_m)))[:],axis=0)
        self.lgOm1 = lgOm1
        self.lgOm1_w = lgOm1_w
        m_opt = np.argmax(lgOm1)+1

        P_w = 10** ((lgOm1_w[m_opt-1]-np.log10(w_array)) - lsum(lgOm1_w[m_opt-1]-np.log10(w_array)))
        lgO_period = lsum(lgOm1)
        P_period = 1/(1+(10**(-lgO_period)))
        cdf = np.zeros_like(P_w)
        for i in range(0,np.size(P_w)):
            cdf[i]=np.trapz(P_w[0:i],w_array[0:i])
        wr=np.extract(np.logical_and(cdf>.016, cdf<.084),w_array)
        w_peak=w_array[np.argmax(P_w)]
        w_mean=np.trapz(P_w*w_array, w_array)
        if np.size(wr)>0:
            w_conf=[np.min(wr),np.max(wr)]
        else:
            w_conf=[w_peak,w_peak]

        self.t0 = t0
        self.lgO_period = lgO_period
        self.P_period = P_period
        self.m_opt = m_opt
        self.P_w = P_w
        self.w_array = w_array
        self.w_peak = w_peak
        self.w_mean = w_mean
        self.w_comf = w_conf
        self.m_max = m_max
        self.w_lo = w_lo
        self.delta_w = delta_w
        sys.stdout.write('\r'+"Completed!\n")
        
        
    def save_diag(self, save_path):
        self.fig.savefig(save_path)
        plt.close('all')
        
    def diagram(self, bins_for_show=50, save_path=None):
        """
        bins_for_show : the number of bins of the histogram along with estimated light curve
        """
        
        fig, axs = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)']], figsize=(10,7), dpi=300)
        ax1 = axs['a)']
        ax2 = axs['b)']
        ax3 = axs['c)']
        
        ax1.fill_between(self.t0, self.f_est-self.sigma_est,self.f_est+self.sigma_est, color='orange', alpha=0.35, label="$1-\sigma$ uncertainties")
        ax1.plot(self.t0, self.f_est, color='orange', label="estimated lc. shape")
        ax1.hist(self.obs_t, bins=bins_for_show, weights= int(self.N)*[bins_for_show/self.N], alpha=0.5, label="reduced hist. of photons")
        ylim = ax1.get_ylim()
        s0, s1 = self.ss(0., self.T)
        if len(s0) == len(s1):
            for i in range(len(s0)):
                ax1.fill_betweenx(ylim , s0[i], s1[i], color='g' ,alpha=0.2)
        ax1.set_ylim(0,ylim[1])
        ax1.set_xlim(0,self.T)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("norm. counts (rate)")
        ax1.legend()

        ax2.plot(np.arange(1, self.m_max+1),self.lgOm1)
        ax2.set_xticks(ticks=np.arange(1, self.m_max+1))
        ax2.grid()
        ax2.axhline(y=0,ls=":",c="orange")
        ax2.set_ylabel("$lgO_{m1}$", fontsize=12)
        ax2.set_xlabel("m", fontsize=12)

        ax3.plot(self.w_array, (self.lgOm1_w[np.argmax(self.lgOm1[1:])]-np.log10(self.w_array))-lsum(self.lgOm1_w[np.argmax(self.lgOm1[1:])]-np.log10(self.w_array)))
        ax3.axvline(x=(self.w_lo+self.delta_w*(np.argsort(self.lgOm1_w[np.argmax(self.lgOm1[1:])])[-1])),  color="black", alpha=1  , linestyle="--", label="the 1st most probable period : "+str(np.round(2*np.pi/(self.w_lo+self.delta_w*(np.argsort(self.lgOm1_w[np.argmax(self.lgOm1[1:])])[-1])),3)))
        ax3.axvline(x=(self.w_lo+self.delta_w*(np.argsort(self.lgOm1_w[np.argmax(self.lgOm1[1:])])[-2])),  color="black", alpha=0.7, linestyle="--", label="the 2nd most probable period : "+str(np.round(2*np.pi/(self.w_lo+self.delta_w*(np.argsort(self.lgOm1_w[np.argmax(self.lgOm1[1:])])[-2])),3)))
        ax3.axvline(x=(self.w_lo+self.delta_w*(np.argsort(self.lgOm1_w[np.argmax(self.lgOm1[1:])])[-3])),  color="black", alpha=0.5, linestyle="--", label="the 3rd most probable period : "+str(np.round(2*np.pi/(self.w_lo+self.delta_w*(np.argsort(self.lgOm1_w[np.argmax(self.lgOm1[1:])])[-3])),3)))
        ax3.set_xlabel("w")
        ax3.set_ylabel("$lgP_{m_opt,w}$")
        ax3.legend(fontsize=8)
        fig.tight_layout()
        self.fig = fig
        if save_path is not None:
            self.save_diag(save_path)
        
