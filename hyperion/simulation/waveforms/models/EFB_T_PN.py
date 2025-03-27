"""Effective Fly By - Time Domain Post Newtonian Expansion"""
# Here are grouped the set of time domain PN expansion functions for the close-encounters modelling 
# see N. Loutrel 2019 for details (arXiv:1909.02143)

import torch
import torch.nn as nn
from inspect import getmembers, isfunction

torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False

class PN_Expansion(nn.Module):
    """
    Class that manages the PN Expansion of the EFB-T waveform model
    (See  N. Loutrel 2019 for details - arXiv:1909.02143)
    """

    def __init__(self, nmax, kmax):
        super(PN_Expansion, self).__init__()

        self.functions = None #initializes the "functions" properties

        self.nmax = nmax
        self.kmax = kmax
        #self.eps_factor = eps_factor
    
    @property
    def functions(self):
        return self._functions
    
    @functions.setter
    def functions(self, dummy):
        self._functions = dict(getmembers(PN_Expansion, isfunction)) #dictionary containing the single functions as keys


    """Store sin and cosines as @properties so that they are computed only once"""
    #INCLINATION COSINE / SINE ====================================
    @property
    def cos_incl(self):
        return self._cos_incl
    @cos_incl.setter
    def cos_incl(self, incl):
        self._cos_incl = torch.cos(incl)
    
    @property
    def sin_incl(self):
        return self._sin_incl
    @sin_incl.setter
    def sin_incl(self, incl):
        self._sin_incl = torch.sin(incl)

    #POLARIZATION COSINE / SINE ====================================
    @property
    def cos_pol(self):
        return self._cos_pol
    @cos_pol.setter
    def cos_pol(self, pol):
        self._cos_pol = torch.cos(pol)
    
    @property
    def sin_pol(self):
        return self._sin_pol
    @sin_pol.setter
    def sin_pol(self, pol):
        self._sin_pol = torch.sin(pol)


    # Helper functions
    #@torch.compile(**torch_compile_kwargs)   
    def th_ci_si(self, incl):
        return 3 + self.cos_incl**2 - self.sin_incl**2
    
    #@torch.compile(**torch_compile_kwargs)
    def cb_sb(self, pol):
        return self.cos_pol**2 - self.sin_pol**2
    
    #@torch.compile(**torch_compile_kwargs)
    def zeros(self, phase, incl, pol):
        return torch.zeros(phase.shape, device=phase.device)
    
    #======================================================================================
    #=========================   POST NEWTONIAN EXPANSION    ==============================
    #======================================================================================

    #======================================================================================
    #                                                                                     #
    #                         Cosin-like PLUS functions C_+^(n,k)                         #
    #                                                                                     #
    #======================================================================================
    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_0_0(self, phase, incl, pol):
        first_ = 8*phase/(1 + torch.pow(phase, 2))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)
    
    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_0_1(self, phase, incl, pol):
        first_  = 9*self.cos_pol**2*self.th_ci_si(incl)
        second_ = -9*(1 + self.cos_incl**2*(-1 + self.sin_pol**2) + self.sin_incl**2 - self.sin_pol**2*(-3 + self.sin_incl**2))
        third_  = (16*self.cos_pol*self.sin_pol*self.th_ci_si(incl)*phase)/(1 + torch.pow(phase, 2))
        return (first_ + second_ + third_)/20

    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_0_2(self, phase, incl, pol):
        first_  = -9*self.cos_incl**2*(3 + 265*self.sin_pol**2)
        second_ = 2385*self.cos_pol**2*self.th_ci_si(incl)
        third_  = 2385*self.sin_pol**2*(-3 + self.sin_incl**2)
        fourth_ = 27*(1 + self.sin_incl**2) 
        fifth_  = (3392*self.cos_pol*self.sin_pol*self.th_ci_si(incl)*phase)/(1 + torch.pow(phase, 2))
        return (first_ + second_ + third_ + fourth_ + fifth_)/2800

    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_1_0(self, phase, incl, pol):
        first_  = 1/torch.pow(1 + torch.pow(phase, 2), 1.5)
        second_ = -15*self.sin_pol**2 + self.sin_incl**2 + 5*self.sin_pol**2*self.sin_incl**2 + torch.pow(phase, 2) 
        third_  = 3*self.sin_pol**2*torch.pow(phase, 2) + self.sin_incl**2*torch.pow(phase, 2) - self.sin_pol**2*self.sin_incl**2*torch.pow(phase, 2)  
        fourth_ = -self.cos_pol**2*self.th_ci_si(incl)*(-5 + torch.pow(phase, 2))
        fifth_  = self.cos_incl**2*(-1 - torch.pow(phase, 2) + self.sin_pol**2*(-5 + torch.pow(phase, 2)))
        return first_*(1 + second_ + third_ + fourth_ + fifth_)
    
    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_1_1(self, phase, incl, pol):
        first_  = -0.3/torch.pow(1 + torch.pow(phase, 2), 1.5)
        second_ = -15*self.sin_pol**2 + self.sin_incl**2 + 5*self.sin_pol**2*self.sin_incl**2 + torch.pow(phase, 2) 
        third_  = -21*self.sin_pol**2*torch.pow(phase, 2) + self.sin_incl**2*torch.pow(phase, 2) + 7*self.sin_pol**2*self.sin_incl**2*torch.pow(phase, 2)  
        fourth_ = self.cos_pol**2*self.th_ci_si(incl)*(5 + 7*torch.pow(phase, 2))
        fifth_  = -self.cos_incl**2*(1 + torch.pow(phase, 2) + self.sin_pol**2*(5 + 7*torch.pow(phase, 2)))
        return first_*(1 + second_ + third_ + fourth_ + fifth_)

    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_1_2(self, phase, incl, pol):
        first_  = 7.1429e-4/torch.pow(1 + torch.pow(phase, 2), 1.5)
        second_ = 3585*self.sin_pol**2 - 111*self.sin_incl**2 - 1195*self.sin_pol**2*self.sin_incl**2 - 111*torch.pow(phase, 2) 
        third_  = 6915**self.sin_pol**2*torch.pow(phase, 2) - 111*self.sin_incl**2*torch.pow(phase, 2) - 2305*self.sin_pol**2*self.sin_incl**2*torch.pow(phase, 2)  
        fourth_ = -5*self.cos_pol**2*self.th_ci_si(incl)*(239 + 461*torch.pow(phase, 2))
        fifth_  = self.cos_incl**2*(111*(1 + torch.pow(phase, 2)) + 5*self.sin_pol**2*(239 + 461*torch.pow(phase, 2)))
        return  first_*(-111 + second_ + third_ + fourth_ + fifth_)

    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_2_0(self, phase, incl, pol):
        first_ = -2/(1 + torch.pow(phase, 2))
        return first_*self.cb_sb(pol)*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_2_1(self, phase, incl, pol):
        first_ = -0.2/(1 + torch.pow(phase, 2))
        return first_*self.cb_sb(pol)*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_2_2(self, phase, incl, pol):
        first_  = 0.00714/(1 + torch.pow(phase, 2))
        second_ = 96*self.sin_pol**2 + self.sin_incl**2 - 32*self.sin_pol**2*self.sin_incl**2 + torch.pow(phase, 2) 
        third_  = 3**self.sin_pol**2*torch.pow(phase, 2) + self.sin_incl**2*torch.pow(phase, 2) - self.sin_pol**2*self.sin_incl**2*torch.pow(phase, 2)  
        fourth_ = -self.cos_pol**2*self.th_ci_si(incl)*(32 + torch.pow(phase, 2))
        fifth_  = self.cos_incl**2*(-1 - torch.pow(phase, 2) + self.sin_pol**2*(32 + torch.pow(phase, 2)))
        return first_*(1 + second_ + third_ + fourth_ + fifth_)

    #@torch.compile(**torch_compile_kwargs)   
    def cos_plus_4_2(self, phase, incl, pol):
        first_ = 0.0429/(1 + torch.pow(phase, 2))
        return first_*self.cb_sb(pol)*self.th_ci_si(incl)

    #======================================================================================
    #                                                                                     #
    #                         Sin-like PLUS functions S_+^(n,k)                           #
    #                                                                                     #
    #======================================================================================

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_1_0(self, phase, incl, pol):
        first_ = 8/(1 + torch.pow(phase, 2))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)
    
    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_1_1(self, phase, incl, pol):
        first_ = -8/(5*(1 + torch.pow(phase, 2)))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_1_2(self, phase, incl, pol):
        first_ = 4/(175*(1 + torch.pow(phase, 2)))
        second_ = -179 + 26*(1 + torch.pow(phase, 2))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)*second_

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_2_0(self, phase, incl, pol):
        first_ = 12/torch.pow(1 + torch.pow(phase, 2), 1.5)
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_2_1(self, phase, incl, pol):
        first_  = -2/(5*torch.pow(1 + torch.pow(phase, 2), 1.5))
        second_ = 19 + 22*torch.pow(phase, 2)
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)*second_

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_2_2(self, phase, incl, pol):
        first_  = -1/(350*torch.pow(1 + torch.pow(phase, 2), 1.5))
        second_ = 2251 + 2026*torch.pow(phase, 2)
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)*second_
    
    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_4_2(self, phase, incl, pol):
        first_ = 78/(35*torch.pow(1 + torch.pow(phase, 2), 1.5))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_5_1(self, phase, incl, pol):
        first_ = 12/(5*(1 + torch.pow(phase, 2)))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_5_2(self, phase, incl, pol):
        first_ = 36/(25*(1 + torch.pow(phase, 2)))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_6_0(self, phase, incl, pol):
        first_ = -4/torch.pow(1 + torch.pow(phase, 2), 1.5)
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_6_1(self, phase, incl, pol):
        first_ = -2/(5*torch.pow(1 + torch.pow(phase, 2), 1.5))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_plus_6_2(self, phase, incl, pol):
        first_ = -37/(70*torch.pow(1 + torch.pow(phase, 2), 1.5))
        return first_*self.cos_pol*self.sin_pol*self.th_ci_si(incl)


    #======================================================================================
    #                                                                                     #
    #                         Cosin-like CROSS functions C_x^(n,k)                        #
    #                                                                                     #
    #======================================================================================

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_0_0(self, phase, incl, pol):
        first_ = 16*phase/(1 + torch.pow(phase, 2))
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_0_1(self, phase, incl, pol):
        first_ = -2/(5*(1 + torch.pow(phase, 2)))
        second_ = -4*self.cos_pol**2*phase + 4*self.sin_pol**2*phase + 9*self.cos_pol*self.sin_pol*(1 + torch.pow(phase, 2))
        return first_*self.cos_incl*second_

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_0_2(self, phase, incl, pol):
        first_  = -53/(350*(1 + torch.pow(phase, 2)))
        second_ = -16*self.cos_pol**2*phase + 16*self.sin_pol**2*phase + 45*self.cos_pol*self.sin_pol*(1 + torch.pow(phase, 2))
        return first_*self.cos_incl*second_

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_1_0(self, phase, incl, pol):
        first_  = 1/(35*torch.pow(1 + torch.pow(phase, 2), 1.5))
        second_ = -1400 + 280*torch.pow(phase, 2)
        return first_*self.cos_pol*self.sin_pol*self.cos_incl*second_

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_1_1(self, phase, incl, pol):
        first_  = 1/(35*torch.pow(1 + torch.pow(phase, 2), 1.5))
        second_ = 420 + 588*torch.pow(phase, 2)
        return first_*self.cos_pol*self.sin_pol*self.cos_incl*second_
    
    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_1_2(self, phase, incl, pol):
        first_  = 1/(35*torch.pow(1 + torch.pow(phase, 2), 1.5))
        second_ = 239 + 461*torch.pow(phase, 2)
        return first_*self.cos_pol*self.sin_pol*self.cos_incl*second_

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_2_0(self, phase, incl, pol):
        first_ = 16/(1 + torch.pow(phase, 2))
        return first_*self.cos_pol*self.sin_pol*self.cos_incl

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_2_1(self, phase, incl, pol):
        first_ = 8/(5*(1 + torch.pow(phase, 2)))
        return first_*self.cos_pol*self.sin_pol*self.cos_incl

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_2_2(self, phase, incl, pol):
        first_ = 2/(35*(1 + torch.pow(phase, 2)))
        second_ = 32 + torch.pow(phase, 2)
        return first_*self.cos_pol*self.sin_pol*self.cos_incl*second_

    #@torch.compile(**torch_compile_kwargs)   
    def cos_cross_4_2(self, phase, incl, pol):
        first_ = -12/(35*(1 + torch.pow(phase, 2)))
        return first_*self.cos_pol*self.sin_pol*self.cos_incl


    #======================================================================================
    #                                                                                     #
    #                         Sin-like CROSS functions S_x^(n,k)                          #
    #                                                                                     #
    #======================================================================================

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_1_0(self, phase, incl, pol):
        first_ = 16/(1 + torch.pow(phase, 2))
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_1_1(self, phase, incl, pol):
        first_ = -16/(5*(1 + torch.pow(phase, 2)))
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_1_2(self, phase, incl, pol):
        first_  = 8/(175*(1 + torch.pow(phase, 2)))
        second_ = -179 + 26*torch.pow(phase, 2)
        return first_*self.cos_incl*self.cb_sb(pol)*second_

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_2_0(self, phase, incl, pol):
        first_ = 24/torch.pow(1 + torch.pow(phase, 2), 1.5)
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_2_1(self, phase, incl, pol):
        first_  = -4/(5*torch.pow(1 + torch.pow(phase, 2), 1.5))
        second_ = 19 + 22*torch.pow(phase, 2)
        return first_*self.cos_incl*self.cb_sb(pol)*second_

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_2_2(self, phase, incl, pol):
        first_  = -1/(175*torch.pow(1 + torch.pow(phase, 2), 1.5))
        second_ = 2251 + 2026*torch.pow(phase, 2)
        return first_*self.cos_incl*self.cb_sb(pol)*second_

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_4_2(self, phase, incl, pol):
        first_ = 156/(35*torch.pow(1 + torch.pow(phase, 2), 1.5))
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_5_1(self, phase, incl, pol):
        first_ = 24/(5*(1 + torch.pow(phase, 2)))
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_5_2(self, phase, incl, pol):
        first_ = 72/(25*(1 + torch.pow(phase, 2)))
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_6_0(self, phase, incl, pol):
        first_ = -8/torch.pow(1 + torch.pow(phase, 2), 1.5)
        return first_*self.cos_incl*self.cb_sb(pol)

    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_6_1(self, phase, incl, pol):
        first_ = -4/(5*torch.pow(1 + torch.pow(phase, 2), 1.5))
        return first_*self.cos_incl*self.cb_sb(pol)
    
    #@torch.compile(**torch_compile_kwargs)   
    def sin_cross_6_2(self, phase, incl, pol):
        first_ = -37/(35*torch.pow(1 + torch.pow(phase, 2), 1.5))
        return first_*self.cos_incl*self.cb_sb(pol)

    
    #======================================================================================
    #                                                                                     #
    #                         Hyperbolic Scale functions Ch and Sh                        #
    #                                                                                     #
    #======================================================================================

    @property
    def arcsinh_phase(self):
        return self._arcsinh_phase
    @arcsinh_phase.setter
    def arcsinh_phase(self, phase):
        self._arcsinh_phase = torch.arcsinh(phase)
        
    #@torch.compile(**torch_compile_kwargs)   
    def ch(self, k):
        """
        Composed hyperbolic function that defines the scale of binary waveform functions (see N. Loutrel 2019).
        """
        return torch.cosh((k/3)*self.arcsinh_phase)
    
    #@torch.compile(**torch_compile_kwargs)   
    def sh(self, k):
        """
        Composed hyperbolic function that defines the scale of binary waveform functions (see N. Loutrel 2019).
        """
        return torch.sinh((k/3)*self.arcsinh_phase)

    def _compute_function(self, cos_or_sin, plus_or_cross, n, k, phase, incl, pol):
        
        fname = '{}_{}_{}_{}'.format(cos_or_sin, plus_or_cross, k, n) 
        f = self.functions[fname] if fname in self.functions.keys() else self.functions['zeros']

        hyp_scale = self.ch(k) if cos_or_sin == 'cos' else self.sh(k)
        return f(self, phase, incl, pol) * hyp_scale
    
    def _set_quantities(self, phase, incl, pol):
        """
        Set the cos and sin of the inclination and polarization angles and the arcsinh of the phase
        """
        self.cos_incl = incl
        self.sin_incl = incl
        self.cos_pol  = pol
        self.sin_pol  = pol
        self.arcsinh_phase = phase

    def forward(self, phase, incl, pol, eps_factor):
        """"
        Computes the PN expansion of the EFB-T waveform model
        """
        self._set_quantities(phase, incl, pol)
        #Cos-like plus functions
        
        hp = torch.stack([ 
            torch.stack([ (eps_factor**n)*(self._compute_function('cos', 'plus', n, k, phase, incl, pol) + self._compute_function('sin', 'plus', n, k, phase, incl, pol))   for n in range(self.nmax)])
             for k in range(self.kmax)])
    
        hc = torch.stack([ 
            torch.stack([ (eps_factor**n)*(self._compute_function('cos', 'cross', n, k, phase, incl, pol) + self._compute_function('sin', 'cross', n, k, phase, incl, pol))   for n in range(self.nmax)])
             for k in range(self.kmax)])

        return torch.sum(hp, (0, 1)), torch.sum(hc, (0, 1))





if __name__=='__main__':
    from inspect import getmembers, isfunction  
    from tqdm import tqdm

    import torch._dynamo
    torch._dynamo.reset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_input(Nbatch):
        phase = torch.randn((Nbatch, 2048))
        incl = torch.randn((Nbatch, 1))
        pol = torch.randn((Nbatch,1))
        return phase, incl, pol



    
    with torch.device(device):
        with torch.no_grad():


            Nbatch = 123
            phase, incl, pol = get_input(Nbatch)
            
            p = PN_Expansion()
            #p = torch.compile(p)
            for _ in tqdm(range(1)):
                
                cos_like_plus, sin_like_plus, cos_like_cross, sin_like_cross = p(self, phase, incl, pol)
                #cos_like_plus, sin_like_plus, cos_like_cross, sin_like_cross = cos_like_plus.cpu(), sin_like_plus.cpu(), cos_like_cross.cpu(), sin_like_cross.cpu()
            print(cos_like_plus, '\n', sin_like_plus, '\n', cos_like_cross, '\n', sin_like_cross)
        
            print(cos_like_plus.shape, '\n', sin_like_plus.shape, '\n', cos_like_cross.shape, '\n', sin_like_cross.shape)
            #while True:
            #    a=0
            #print(p.functions)
            
            
