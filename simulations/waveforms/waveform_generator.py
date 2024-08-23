import torch
import numpy as np
import multiprocess as mp
mp.set_start_method('spawn', force=True)  # Oppure 'fork' se su Unix


from tqdm import tqdm
from torch.nn.functional import pad
from .models import EffectiveFlyByTemplate, TEOBResumSDALI
from ...core.fft.windows import tukey

models_dict = {'EffectiveFlyBy': EffectiveFlyByTemplate, 
                'TEOBResumSDALI': TEOBResumSDALI}


class WaveformGenerator:
    """
    Waveform generator class that wraps the waveform models.

    Constructor Args:
    -----------------
        waveform_model: str
            Name of the waveform model to use. Available models are 'EffectiveFlyBy' and 'TEOBResumSDALI'.
        
        waveform_model_kwargs: dict
            kwargs to pass to the waveform's model constructor.
       
    """
    
    def __init__(self, 
                 waveform_model, 
                 fs = 2048,
                 duration = 4,
                 det_network:dict = None,
                 **waveform_model_kwargs):

        assert waveform_model in models_dict.keys(), f"Waveform model {waveform_model} not found. \
                                                       Available models are {models_dict.keys()}"
        
        self.fs = fs
        self.duration = duration
        self.det_network = det_network
        self.wvf_model   = models_dict[waveform_model](fs, **waveform_model_kwargs)
    
        return

    @property
    def name(self):
        return self.wvf_model.name
    
    @property
    def delta_t(self):
        return 1/self.fs
    
    @property
    def has_torch(self):
        return self.wvf_model.has_torch
    
    @property
    def duration(self):
        return self._duration
    @duration.setter
    def duration(self, value):
        self._duration = value
                

    def _resize_waveform(self, times, t_wvf, hp, hc):
        """
        Resize the waveform to the desired duration.
        """
        N = int(self.duration * self.fs)

        hp *= tukey(len(hp), alpha=0.1)
        hc *= tukey(len(hc), alpha=0.1)
        
        #signal is longer --> crop
        if hp.shape[-1] >= N:
            t_wvf = t_wvf[-N:]
            hp = hp[-N:]
            hc = hc[-N:]
        
        #signal is shorter --> symmetrically pad with zeros
        elif hp.shape[-1] < N:
            
            pad_l = (N - hp.shape[-1]) // 2
            pad_r = N - hp.shape[-1] - pad_l
            
            hp = pad(hp, (pad_l, pad_r))
            hc = pad(hc, (pad_l, pad_r))
            
            #extend the time array
            t_l = torch.linspace(-self.delta_t*pad_l, 0, pad_l, device=t_wvf.device) + t_wvf.min()
            t_r = torch.linspace(0, self.delta_t*pad_r,  pad_r, device=t_wvf.device) + t_wvf.max()
            t_wvf = torch.cat([t_l, t_wvf, t_r])

        tcoal = np.interp(0, t_wvf, times)
        '''
        import matplotlib.pyplot as plt
        #plt.plot(t, hp)
        #plt.axvline(len(t)//2, color='r')
        print(tcoal)
        plt.figure()
        plt.plot(times, hp)
        plt.axvline(tcoal, color='r')        
        plt.savefig('waveform.png')
        '''
        return hp, hc, tcoal


    def get_td_waveform(self, pars):
        """
        Computes the time domain waveform for a given set of parameters.

        Args:
        -----
            pars: dict or TensorDict instance
                Parameters of the waveform
        
        Returns:
        --------
             The output of the self.waveform_model __class__ method
        """        
        return self.wvf_model(pars)
    

    def _get_td_waveform_mp(self, i):

        """
        Compute the time domain waveform for a given set of parameters.
        This function is intended to be used with multiprocessing.

        Args:
        -----
            i: int
                Index of the parameters to use. (Assigned by the multiprocessing pool)
        
        Returns:
        --------
             The output of the self.waveform_model __class__ method
        """
        
        pars = self.parameters[i]
        
        return self.get_td_waveform(pars)
    


    def __call__(self, parameters, n_proc=None, project_onto_detectors=False):
        
        # Check if the model is a torch model so that
        # it can handle batches of parameters and / or if parameters are batched
        N = parameters.numel()
        if self.has_torch or N<=1:
            return self.wvf_model(parameters)
        
        # Otherwise, we exploit multiprocessing
        else:
            #define the analysis time segment array [-duration/2, duration/2]
            times = np.linspace(-self.duration/2, self.duration/2, int(self.duration*self.fs))
            
            n_proc = mp.cpu_count() if n_proc is None else n_proc
            with mp.Pool(n_proc) as p:
                self.parameters = parameters
                hps = []
                hcs = []
                tcoals = []
                
                for results in tqdm(p.imap(self._get_td_waveform_mp, range(N)), total=N, ncols = 100, ascii=' ='):
                    t = results['t']  #this is the time array referred to the waveform
                                      #it has 0 places in the merger
                    hp = results['hp']
                    hc = results['hc']

                    #resize the waveform to the desired duration and get the time of coalescence
                    hp, hc, tcoal = self._resize_waveform(times, t, hp, hc)
                    
                    tcoals.append(tcoal)
                    hps.append(hp)
                    hcs.append(hc)
                
                hps = torch.stack(hps)
                hcs = torch.stack(hcs)
                tcoals = torch.tensor(tcoals).unsqueeze(-1)

        if project_onto_detectors:
            #project the waveform onto the detectors
            projected_template = self.det_network.project_waveform(hps, hcs, 
                                                         parameters['ra'], 
                                                         parameters['dec'], 
                                                         parameters['polarization'])
            return projected_template, tcoals
        
        return hps, hcs, tcoals
        
        
        
        
        
        
        
        