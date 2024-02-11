import sys
#add to pythonpath search modules the location of the gwskysim package
gwskysim_path = ['/Users/federicodesanti/desktop/close_encounters_thesis/gwskysim', 
                  '/mnt/c/Users/fdesa/desktop/close_encounter_thesis_project/gwskysim', 
                 #'C:\Users\fdesa\desktop\close_encounter_thesis_project\gwskysim-master'
                  '/home/gwmluser/work/gwskysim',
                  '/home/luciapapalini/Desktop/gwskysim',
		'/home/gwmluser/work/lpapalini/gwskysim'
                 ]
for path in gwskysim_path:
    sys.path.append(path)
#--------------------------------------------------------------------------------------------

from scipy.interpolate import interp1d


import datetime
from gwpy.time import from_gps
from gwpy.timeseries import TimeSeries as gwpy_TimeSeries
from gwpy.signal.filter_design import bandpass
#bp = filter_design.bandpass(50, 250, hdata.sample_rate)

from pycbc.filter import  matched_filter
from pycbc.psd import welch, interpolate



from gwskysim.sources.gwbinarysource import GWBinarySource
from gwskysim.sources.gwcolored_noise import GWColoredNoise
from gwskysim.detectors.gwdetector import GWDetector


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
seed = np.random.get_state()[1][0]
#print(seed)
class Injection():
    """Class that performs injection of CE into detectors"""
    
    def __init__(self, 
                 detectors = ['L1', 'H1', 'V1'],
                 asd_curves = None,
                 fs = 4096//2, 
                 noise_duration = 8,
                 whiten_kwargs = {'fftlength': 4, 'overlap':2}
                 ):
        self.fs = fs
        self.noise_duration = noise_duration
        self.noise_points   = noise_duration * fs
        self.whiten_kwargs  = whiten_kwargs
        self.len2smooth_factor     = 8

        #detector initialization
        self.detectors = dict()
        for det_name in detectors:
           self.detectors[det_name] = GWDetector(det_name)

        #asd initialization
        if asd_curves is not None:
            self.asd_curves = asd_curves
        else:
            self.asd_curves = {'L1': 'pop_config/L1_O3.txt', 
                               'H1': 'pop_config/H1_O3.txt',
                               'V1': 'pop_config/V1_O3.txt'}
            
        self.asd_interp   = dict()
        self.fmin     = dict()
        self.fmax     = dict()
        for det_name in detectors:
            f, asd = np.loadtxt(self.asd_curves[det_name], unpack=True)
            self.asd_interp[det_name] = interp1d(f, np.log10(asd))
            self.fmin[det_name] = f[0]
            self.fmax[det_name] = np.amax(f) - 1
            
            
        
        return
    
    @staticmethod
    def _crop_and_apply_timeshift(whitened_strain, time_shift, fs):
        middle_shifted = len(whitened_strain)//2 - int(time_shift*fs/2)
        return whitened_strain[middle_shifted - fs//2 : middle_shifted + fs//2]
    
    @staticmethod
    def _get_hp_and_hc(fs, m1, m2, pol, incl, dist, e0, p_0, t0_p, ra=0, dec=0):
        #CLOSE ENCOUNTER TEMPLATE GENERATION: taken from Nunzio's code 
        gw_source = GWBinarySource('CE {}-{} Msun'.format(m1, m2), ra, dec,
                                pol, incl, dist, e0, p_0, m1, m2, tp=t0_p) #we fix ra/dec since we generate unprojected waveforms
        tl_ = gw_source.t_edge(np.pi/4)
        tl_m = gw_source.t_edge(-np.pi/4)
        t_f = tl_m 
        duration = tl_- tl_m
        
        hp, hc = gw_source._import_template(np.linspace(t_f, t_f+duration, int(duration*fs)))
    
        return np.array(hp), np.array(hc)
    
    @staticmethod
    def _add_glitch(strain, times, fs):
        f0=150 #Hz
        h0 = np.random.uniform(1e-23,1e-20)
        tau = np.random.uniform(1e-3, 1e-2)
        
        t0 = times[len(times)//2 + int(np.random.uniform(-.5, .5)*fs)]
        
        aux    =  np.exp(-((times - t0) ** 2) / (2 * tau ** 2))
        glitch =  h0*np.sin(2 * np.pi * f0 * (times - t0)) * aux
        return strain + glitch
    
    
    @staticmethod
    def _project_strain_onto_detector(detector, hp, hc, ra, dec, pol, t0_p):
        fp, fc = detector.antenna_pattern_functions(ra, dec, pol, t0_p)
        return np.array(hp*fp + hc*fc)
    
    @staticmethod
    def _add_noise(det_name, strain, t0, noise_duration, fs, fmin, fmax, asd_interp, samples = None):
        d_colnoise = GWColoredNoise('Colored Noise', samples, asd_interp[det_name], fmin[det_name], fmax[det_name]) 
        noise = d_colnoise.generate_signal(t0, noise_duration, fs, detectors=[det_name])[0]
        
        return np.array(strain + noise), np.array(noise)
    
    
    @staticmethod
    def _whiten_strain(strain, fs, t0, asd=None, **whiten_kwargs):
        #asd is None for the template+noise case so we compute it 
        #to whiten the template we use the asd computed above
        
        gwpy_strain = gwpy_TimeSeries(strain, dt = 1/fs, t0 = t0)#.bandpass(10, 1000)
        
        if asd is  None: 
            asd = gwpy_strain.asd(**whiten_kwargs, method = 'welch') 

        return np.array(gwpy_strain.whiten(asd=asd)), asd
    
    @staticmethod
    def compute_SNR(strain, template, fs, t0):
        pycbc_strain = gwpy_TimeSeries(strain, dt = 1/fs, t0 = t0).to_pycbc()
        pycbc_template = gwpy_TimeSeries(template, dt = 1/fs, t0 = t0).to_pycbc()
        psd = interpolate(welch(pycbc_strain), 1.0 / pycbc_strain.duration) 
        
       
        # Calculate the complex (two-phase SNR)
        snr = matched_filter(pycbc_template.to_frequencyseries(), pycbc_strain.to_frequencyseries(), psd=psd, low_frequency_cutoff=3.0)
        # Remove regions corrupted by filter wraparound
        #snr = snr[len(snr) // 4: len(snr) * 3 // 4]
        plt.plot(abs(snr))
        #plt.loglog(psd.sample_frequencies, psd)              
        plt.show()
        return np.array(abs(snr))


    
    
    def _plot(self, whitened_strain, whitened_template, SNR, whitened):
        times = np.linspace(-0.5, 0.5, self.fs) 
        date  = from_gps(self.t0_p)
        plt.figure(figsize=(12, 5))
        #plt.figure()
        plt.rcParams['font.size']=18
        for i, det_name in enumerate(self.detectors):
            ax = plt.subplot(len(self.detectors), 1, i+1)
            plt.plot(times, whitened_strain[det_name],linewidth=2)
            plt.plot(times, whitened_template[det_name], linewidth = 2)
            if whitened:
                plt.ylabel('whitened strain')
            else:
                plt.ylabel('strain')
            if SNR is not None and whitened:
                plt.title(f"{det_name} - SNR {SNR[det_name].to_numpy()[0]:.2f}")
            else:
                plt.title(f"{det_name}")
            #if (i+1)<len(self.detectors):
            #    ax.set_xticklabels([])
            ax.minorticks_on()
            plt.xlim(min(times), max(times))
            #plt.ylabel('whitened strain')
            plt.xlabel(f'Time [s] from {date} UTC ({self.t0_p:.1f})')
        
        #plt.savefig('BHBH_CE.png', dpi=300)
        plt.show()
        return
    
    
    def __call__(self, m1, m2, dist, p_0, e0, t0_p, ra, dec, pol, incl, time_shift=0, plot=False, compute_SNR=False, return_template=False, whiten=True, add_glitch = False):
        
        #GPS times array
        times = np.linspace(-0.5, 0.5, self.noise_points) + t0_p
        self.t0_p = t0_p
        
        
        #generating hp and hc polarization
        hp, hc = self._get_hp_and_hc(self.fs, m1, m2, pol, incl, dist, e0, p_0, t0_p)
        
        
        whitened_strain   = pd.DataFrame()#dict()
        whitened_template = pd.DataFrame()
        SNR               = pd.DataFrame() if compute_SNR else None
        
        if add_glitch:
            det_name_glitch = np.random.choice(list(self.detectors.keys()))
        
        
        for det_name in self.detectors.keys():
            detector = self.detectors[det_name]
            
            det_strain = self._project_strain_onto_detector(detector, hp, hc, ra, dec, pol, t0_p)
            
            if plot or compute_SNR or return_template:
                len2smooth = len(det_strain)//self.len2smooth_factor
                hann_function        = np.hanning(2*len2smooth)
                template = det_strain.copy()
                template[:len2smooth]  = template[:len2smooth] * hann_function[:len2smooth]  #left 
                template[-len2smooth:] = template[-len2smooth:]* hann_function[-len2smooth:] #right
            #plt.plot(template-det_strain)
            #plt.show()
            #padding to match noise duration
            pad_left  = (self.noise_points - len(det_strain))//2
            pad_right = self.noise_points - len(det_strain) - pad_left
            det_strain = np.pad(det_strain, (pad_left, pad_right))
            
            #plt.plot(det_strain)
            #plt.show()
            if add_glitch:
                if det_name == det_name_glitch:
                    det_strain = self._add_glitch(det_strain, times, self.fs)
                    #plt.plot(det_strain)
                    #plt.show()
            
            noisy_strain, noise = self._add_noise(det_name, det_strain, times[0], 
                                           self.noise_duration, self.fs, self.fmin, self.fmax, 
                                           asd_interp = self.asd_interp, samples = None)
            if compute_SNR:
                snr = self.compute_SNR(strain=noisy_strain, template=np.pad(template, (0,pad_left+ pad_right)), fs=self.fs, t0=t0_p)
                #plt.plot(snr)
                #plt.show()
                snr = self._crop_and_apply_timeshift(snr, time_shift, self.fs)
                #snr = snr[len(snr)//2-int(0.2*self.fs):len(snr)//2+int(0.2*self.fs)]
                SNR[det_name] = [max(snr)]
                #print(SNR)
            
            
            if whiten:
                whitened, asd = self._whiten_strain(noisy_strain, self.fs, times[0],  **self.whiten_kwargs)
            else:
                whitened = noisy_strain #used for bilby
            cropped  = self._crop_and_apply_timeshift(whitened, time_shift, self.fs)
            whitened_strain[det_name] = cropped
            
            if plot or return_template:
                if whiten:
                    whitened_t, _ = self._whiten_strain(np.pad(template, (pad_left, pad_right)), self.fs, times[0], asd=asd, **self.whiten_kwargs)
                else:
                    whitened_t = np.pad(template, (pad_left, pad_right))
                cropped_t  = self._crop_and_apply_timeshift(whitened_t, time_shift, self.fs)
                whitened_template[det_name] = cropped_t
        if plot:
            self._plot(whitened_strain, whitened_template, SNR, whiten)
            
        if return_template:
            return whitened_template, SNR
        else:
            return whitened_strain, SNR
            
    
if __name__ == '__main__':
    import os 
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import multiprocessing as mp
    mode = 'BHBH'
    
    fs = 2048 if mode=='BHBH'  else  4096
    fs= 4096
    injector = Injection(fs = fs)
   
    SNR = pd.DataFrame()
    #for i in tqdm(range(1000)):
    
    def compute_SNR(i):
        m1 = np.random.uniform(10, 100)
        m2 = np.random.uniform(10, 100)
        dist = np.random.uniform(1e8, 1e9)
        p_0 = np.random.uniform(10, 25)
        e0 = np.random.uniform(0.7 , 0.95)
        ra = np.random.uniform(0, 2*np.pi)
        dec = np.arcsin(2*np.random.uniform()-1)
        pol = np.random.uniform(0, np.pi)
        incl = np.arccos(2*np.random.uniform()-1)
        time_shift = np.random.uniform(-0.5, 0.5)
        t0_p=1370692818
        _, snr = injector(m1, m2, dist, p_0, e0, t0_p, ra, dec, pol, incl, time_shift, plot = False, compute_SNR=True)
        
        return snr
    #strain, snr = injector(m1=100, m2=100,  dist=1e8, p_0=15, e0=0.7, t0_p=1370692818, ra=np.pi/3, dec=-np.pi/3, pol=0, incl=3.0, time_shift=0.0, plot = True, compute_SNR=True)
    '''
    N = 10_000
    with mp.Pool(os.cpu_count()) as p:
        for snr in tqdm(p.imap(compute_SNR, range(N)), total=N):
        
            SNR = pd.concat([SNR, snr])
    print(SNR)
    SNR.to_csv('SNR_'+mode+'.csv')    
    '''
    '''
    plt.figure()
    for det in SNR:
        plt.hist(SNR[det], bins = 'auto', label = det, alpha = 0.5, density=True, histtype='step')
        plt.hist(SNR[det], bins = 'auto', label = det, alpha = 0.5, density=True, histtype='stepfilled')

    plt.title(mode +' CE')
    plt.ylabel('density')
    plt.xlabel('SNR')
    plt.xlim(0, 15)
    plt.legend()
    plt.savefig('SNR_'+mode+'.png', dpi=200)
    plt.show()
    '''
    SNR = pd.read_csv('_SNR_'+mode+'.csv').drop(0); del SNR[SNR.columns[0]]
    
    #renaming
    rename_dict = {'L1':'LIGO Livingston', 'H1': 'LIGO Hanford', 'V1': 'Virgo'}
    
    SNR = SNR.rename(columns=rename_dict)
    
    
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    sns.histplot(SNR, element="step", stat='percent')
    plt.xlim(1, 18)
    plt.title(mode +' CE')
    plt.ylabel('counts [%]')
    plt.grid(True, alpha=0.3)
    #plt.xlabel('SNR')
    #plt.savefig('SNR_'+mode+'.png', dpi=200)
    #plt.show()
        #print(len(strain[det]))
        
    combined_SNR = np.array([SNR[det].to_numpy()**2 for det in SNR.keys()])
    combined_SNR = {'network': np.sqrt(np.sum(combined_SNR, axis=0))}
    
    plt.subplot(212)

    sns.histplot(combined_SNR, element="step", stat='percent')
    plt.xlim(3, 30)
    #plt.xticks(np.arange(3, 30, 3))
    #plt.title(mode +' CE')
    plt.ylabel('counts [%]')
    plt.xlabel('SNR')
    plt.grid(True, alpha=0.3)
    plt.savefig('SNR_'+mode+'.png', dpi=200)
    plt.show()
    
    
