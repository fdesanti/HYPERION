import torch
import multiprocessing as mp

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
                 **waveform_model_kwargs):

        assert waveform_model in models_dict.keys(), f"Waveform model {waveform_model} not found. \
                                                       Available models are {models_dict.keys()}"
        
        self.fs = fs
        self.duration = duration
        self.wvf_model = models_dict[waveform_model](fs, **waveform_model_kwargs)
    
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
           

    def _resize_waveform(self, t, hp, hc):
        """
        Resize the waveform to the desired duration.
        """
        N = int(self.duration * self.fs)

        hp *= tukey(len(hp), alpha=0.1)
        hc *= tukey(len(hc), alpha=0.1)
        
        #signal is longer --> crop
        if hp.shape[-1] >= N:
            t = t[-N:]
            hp = hp[-N:]
            hc = hc[-N:]
        
        #signal is shorter --> symmetrically pad with zeros
        elif hp.shape[-1] < N:
            
            pad_l = (N - hp.shape[-1]) // 2
            pad_r = N - hp.shape[-1] - pad_l
            
            hp = pad(hp, (pad_l, pad_r))
            hc = pad(hc, (pad_l, pad_r))
            
            #extend the time array
            t_l = torch.linspace(-self.delta_t*pad_l, 0, pad_l, device=t.device) + t.min()
            t_r = torch.linspace(0, self.delta_t*pad_r,  pad_r, device=t.device) + t.max()
            t = torch.cat([t_l, t, t_r])

        '''
        import matplotlib.pyplot as plt
        plt.plot(t, hp)
        #plt.axvline(len(t)//2, color='r')
        plt.show()
        '''
        
        return t, hp, hc


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
    


    def __call__(self, parameters, n_proc=None):
        
        # Check if the model is a torch model so that
        # it can handle batches of parameters and / or if parameters are batched
        N = parameters.numel()
        if self.has_torch or N<=1:
            return self.wvf_model(**parameters)
        
        # Otherwise, we exploit multiprocessing
        else:
            with mp.Pool(n_proc) as p:
                self.parameters = parameters
                
                hps = []
                hcs = []
                t_coals = []
                
                for results in tqdm(p.imap(self._get_td_waveform_mp, range(N)), total=N, ncols = 100, ascii=' ='):
                    t = results['t']
                    hp = results['hp']
                    hc = results['hc']

                    t, hp, hc = self._resize_waveform(t, hp, hc)
                    
                    t_coals.append(-t[len(t)//2])
                    hps.append(hp)
                    hcs.append(hc)
                
                hps = torch.stack(hps)
                hcs = torch.stack(hcs)
                t_coals = torch.tensor(t_coals).unsqueeze(-1)

        return hps, hcs, t_coals
        
        
        
        
        
        
        
        