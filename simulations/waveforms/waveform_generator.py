
import torch
from .models import EffectiveFlyByTemplate, TEOBResumSDALI

models_dict = {'EffectiveFlyBy': EffectiveFlyByTemplate, 
                'TEOBResumSDALI': TEOBResumSDALI}



class WaveformGenerator():
    
    def __init__(self, waveform_model, **waveform_model_kwargs):
        
        assert waveform_model in models_dict.keys(), f"Waveform model {waveform_model} not found. Available models are {models_dict.keys()}"

        self.wvf_model = models_dict[waveform_model](**waveform_model_kwargs)
        
        
        return

    @property
    def name(self):
        return self.wvf_model.name
    
    @property
    def has_cuda(self):
        return self.wvf_model.has_cuda
    


    def get_td_waveform(self, **waveform_parameters):
        return self.wvf_model.get_td_waveform(**waveform_parameters)