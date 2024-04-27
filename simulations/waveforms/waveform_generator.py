import torch
import multiprocessing as mp

from tqdm import tqdm
from .models import EffectiveFlyByTemplate, TEOBResumSDALI

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
                 waveform_model:str, 
                 **waveform_model_kwargs):

        assert waveform_model in models_dict.keys(), f"Waveform model {waveform_model} not found. \
                                                       Available models are {models_dict.keys()}"

        self.wvf_model = models_dict[waveform_model](**waveform_model_kwargs)
        
        
        return

    @property
    def name(self):
        return self.wvf_model.name
    
    @property
    def has_torch(self):
        return self.wvf_model.has_torch
    


    def get_td_waveform(self, parameters, n_proc=None):
        
        # Check if the model is a torch model so that
        # it can handle batches of parameters and / or if parameters are batched
        if self.has_torch or parameters.numel()<=1:
            return self.wvf_model(**parameters)
        
        # Otherwise, we exploit multiprocessing
        else:
            with mp.Pool(n_proc) as p:
                results = list(tqdm(p.imap(self.wvf_model, parameters), total=len(parameters)))
            


        
        
        
        
        
        
        
        