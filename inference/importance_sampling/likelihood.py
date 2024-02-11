import torch

from ...core.fft import rfft, rfftfreq


class GWLikelihood():
    """
    Gravitational Wave Likelihood class
    
    Args:
    -----
        waveform_generator : object
            Waveform generator object which returns ifo injected strain (e.g. the EffectiveFlyByTemplate class)
    
    """
    
    def __init__(self, waveform_generator):

        self.waveform_generator = waveform_generator
        
        return
    
    @property
    def fs(self):
        return self.waveform_generator.fs
    
    @property
    def duration(self):
        return self.waveform_generator.duration
    
    @property
    def frequencies(self):
        return rfftfreq(self.duration * self.fs, d=1/self.fs)
    

    def noise_log_Likelihood(self, strain, psd):
        """
        Computes the log Likelihood assuming strain contains only Gaussian noise
        
                L_ifo = - (2/T) * sum( |hf|^2 / PSD )
                L = sum L_ifo 
        
        where hf is the frequency domained strain
        
        Args:
        -----
            strain : dict
                Dictionary containing interferometer strain time series
                
            psd : dict
                Dictionary containing interferometer Power Spectral Densities
                
        Returns:
        --------
            logL : float
                Noise Log Likelihood 
        """
        logL = 0
        
        #print('NOISE LIKELIHOOD###############')
        for det_name in strain.keys():
            
            frequency_domain_strain = rfft(strain[det_name], fs = self.fs)
            
            #logL -= (2. / self.duration) * torch.sum(torch.conj(frequency_domain_strain) * frequency_domain_strain / (psd[det_name])) / self.fs

            T = len(strain[det_name]/self.fs)
            #logL -= (2. / self.duration) * torch.sum(torch.abs(frequency_domain_strain)** 2 / psd[det_name])
            #logL -= (2. / T) * sum(abs(frequency_domain_strain) ** 2 / (psd[det_name]))
            logL += self._gaussian_likelihood(frequency_domain_strain, 0, psd[det_name])
        return logL.real
    
    
    def _gaussian_likelihood(self, aa, bb, psd):
        
        const = torch.log(2*torch.pi*psd)
        
        product = (aa-bb)**2 #/ psd
        '''
        import matplotlib.pyplot as plt
        plt.plot(bb.numpy())
        plt.show()
        '''
        #print('aaaa - bbbbb', product+const)
        
        return -0.5 * torch.sum(product + const, -1)
    

    def log_Likelihood(self, strain, psd, waveform_parameters):
        """
        Computes the log Likelihood assuming strain contains a signal and Gaussian noise
        
                L_ifo = - (2/T) * sum( |hf - template_f|^2 / PSD )
                L = sum L_ifo 
        
        where hf, template_f are in frequency domain
        
        Args:
        -----
            strain : dict
                Dictionary containing interferometer strain time series
                
            psd : dict
                Dictionary containing interferometer Power Spectral Densities
                
            waveform_parameters : dict
                Dictionary containing parameters to be passed to the waveform generator.
                
        Returns:
        --------
            logL : float
                Log Likelihood 
        """
        logL = 0
        frequency_domain_template = self.get_frequency_domain_template(det_names=list(strain.keys()), parameters=waveform_parameters)
    
        for det_name in strain.keys():
            frequency_domain_strain = rfft(strain[det_name], fs = self.fs)
            logL += self._gaussian_likelihood(frequency_domain_strain, 
                                                              frequency_domain_template[det_name], 
                                                              psd[det_name]) 
            
        return logL.real
    
   
    
    def _single_detector_noise_logLikelihood(self, frequency_domain_strain, frequency_domain_template, psd):
        """
        Computes the log Likelihood on single detector
        
        Args:
        -----
            frequency_domain_strain : torch.Tensor
                Interferometer Strain frequencyseries
            
            frequency_domain_template : torch.Tensor
                Frequency domain template projected onto the interferometer
                
            psd : torch.tensor
                Detector Power Spectral Density
                
        Returns:
        --------
            logL_det : torch.Tensor
                Detector Log Likelihood. (The shape of the tensor is the same of frequency_domain_template)
             
        """
        '''
        import matplotlib.pyplot as plt
        plt.plot(frequency_domain_strain.cpu().numpy())
        plt.plot(frequency_domain_template.cpu().numpy()/psd.cpu().numpy())
        plt.show()
        '''
        logL_det = - (2. / self.fs) * torch.linalg.vecdot(frequency_domain_strain - frequency_domain_template,
                                                               (frequency_domain_strain - frequency_domain_template) / psd )
        
        #logL_det = - (2. / self.duration) * torch.sum(frequency_domain_strain - frequency_domain_template * \
        #                                                       (frequency_domain_strain - frequency_domain_template) / psd )

        #print(frequency_domain_strain.shape, frequency_domain_template.shape, psd.shape, logL_det.shape)
        return logL_det.real
    
    
    def get_frequency_domain_template(self, det_names, parameters):
        """
        Computes the frequency domained templates using the waveform_generator object
        
        Args:
        -----
            det_names : list
                List of detector names into which project waveform polarizations
                
            parameters : dict
                Dictionary with each key representing a gw parameter
                
        Returns:
        --------
            frequency_domain_template : dict of torch.Tensors
                Dictionary (with det_names as keys) containing the projected frequency domain template 
        
        """
        template = self.waveform_generator(**parameters)
        
        '''
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure()
            for d in det_names:
                plt.plot(template[d][i].cpu().numpy())
            plt.show()
        '''
        
        frequency_domain_template = dict()
        for det_name in det_names:
            hf  = rfft(template[det_name], fs = self.fs)
            dt = template[f"{det_name}_delay"]

            #take into account the time shift
            frequency_domain_template[det_name] = hf * torch.exp(-1j * 2 * torch.pi * self.frequencies * dt) 
        
        return frequency_domain_template
    
     
