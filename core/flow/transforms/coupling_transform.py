""" Here it is implemented the Coupling transform for the flow."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_model.splines import unconstrained_rational_quadratic_spline

#torch.manual_seed(0)

class CouplingTransform(nn.Module):
    """Class that implements the full Coupling transform
    
    Args:
        - transform_layers: list of SplineCouplingLayers istances 
        
    Methods:
        - forward: it's used during training  to map x --> u, where u = prior(u)
        - inverse: it's used during inference to map u --> x, where u = prior(u)
    """
    
    def __init__(self, transform_layers):
        super().__init__()
        
        self._transforms = nn.ModuleList(transform_layers)
        #self._flow_neural_network = flow_neural_network
        
        return
    
    @staticmethod
    def _cascade(inputs, layers, embedded_strain):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for layer in layers:
            outputs, logabsdet = layer(outputs, embedded_strain)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, embedded_strain):
        """Forward pass (training)
        Args: 
            - inputs: tensor of shape [N batch, N posterior parameters] coming from the dataset
            - embedded_strain: tensor of shape [N batch, 3, len_embedded_strain] embedded_strain associated to each element of the dataset
                      to contrain the flow
        Reuturns:
            - _cascade() of the layers in self._transforms 
        """
        layers = self._transforms
        return self._cascade(inputs, layers, embedded_strain)

    def inverse(self, inputs, embedded_strain):
        """Inverse pass (inference)
        Args: 
            - inputs: tensor of shape [N prior samples, N posterior parameters] coming from the sampled prior
            - embedded_strain: tensor of shape [1, 3, len_embedded_strain] embedded_strain associated to which predict the posterior
            
        Reuturns:
            - _cascade() of the layers in self._transforms.inverse in reversed order 
        """
        layers = (transform.inverse for transform in self._transforms[::-1])
        if inputs.shape[0]!= embedded_strain.shape[0]:
            s = torch.cat([embedded_strain for _ in range(inputs.shape[0])], dim = 0)
        else:
            s = embedded_strain
        return self._cascade(inputs, layers, s)

    





class SplineCouplingLayer(nn.Module):
    """
    Class that implements the single Coupling Layer with RationalQuadraticSplines
    
    Args:
        - input_dim (int):  dimensionality of the parameter space
        - num_circular (int): number of angles in parameters (ie. number of elements to transform with circular splines)
        - neural_network (nn.Module):  the neural network that produces the knots and is trained with the flow
        - layer_index (int):  index of the layer in the whole coupling transform. Used to permute the masks accordingly   
    
    Methods:
        - _create_transform_mask:  produces a 1D binary tensor mask with the first num_identity = False and the rest num_transformed = True
        - _create_circular_mask:   produces a 1D binary tensor mask with the first num_non_circuar = False and the rest num_circular = True
        - _permute_mask:           permutes cyclically the given mask 
        
        - _coupling_transform :    implements the coupling transform with splines 
        - forward :                makes the forward pass (training)
        - inverse :                makes the inverse pass (training)
    
    """
    
    def __init__(self,
                 num_features       :int = 10,
                 num_circular       :int = 1,
                 layer_index        :int = None,
                 permutation        :str = 'random',
                 tail_bounds        :list = [1, 1],
                 neural_network     : nn.Module = None,
                 num_identity       = 5,
                 num_transformed    = 5,
                 ):
        super(SplineCouplingLayer, self).__init__()
        
        #assert isinstance(neural_network, nn.Module), 'A torch neural network module must be passed'
        
    
        self.num_features       = num_features
        self.num_identity       = num_identity
        self.num_transformed    = num_transformed
        self.tail_bounds        = tail_bounds
        self.num_circular       = num_circular
        self.layer_index        = layer_index
        
        self.permutation   = permutation
        assert permutation in ['static', 'random', 'cyclic'], f"mask_permutation must be in {['static', 'random', 'cyclic']} - given {permutation}"
        
        
        self.neural_network  = neural_network
        
    
        assert self.num_identity+self.num_transformed == num_features, 'Mismatch beetween the input shape and the parameter to transform'
        assert self.num_transformed==self.neural_network.out_features, 'Num transformed params in layer is different than in network, got %d vs %d'%(self.num_transformed,
                                                                                                                                                     self.neural_network.out_features)
        assert self.num_identity == self.neural_network.in_features, 'Num identity params in layer is different than in network, got %d vs %d' % (self.num_identity,
                                                                                                                                                  self.neural_network.in_features)

        if self.permutation == 'random':
            self.register_buffer("mask_permutation", torch.randperm(num_features))
        
        #CREATE TRANSFORM MASK --------------------------------
        self.register_buffer("transform_mask", self._create_mask(self.num_identity, self.num_transformed))
        
        #CREATE CIRCULAR MASK
        self.register_buffer("circular_mask", self._create_mask(self.num_features -self.num_circular, self.num_circular))

        return
    

    @staticmethod
    def _create_mask(num_false, num_true):
        """creates a binary mask tensor of shape 1 (like an array) with entries [num_false, num_true]"""
        false = torch.zeros(num_false, dtype = torch.bool)
        true  = torch.ones(num_true, dtype = torch.bool)
        return torch.cat([false, true], axis = 0)

    
    def _permute_mask(self, mask, inverse = False):
        """Random or Cyclyc mask permutation described by permutation: it's fixed by the layer """
        #inverse = False
        if not self.permutation == 'static':
            if self.permutation == 'random':
                if inverse:  #FROM NFLOWS RANDOM TRANSFORM
                    return mask[torch.argsort(self.mask_permutation)]
                else:
                    return mask[self.mask_permutation]
            else: #cyclic
                return torch.roll(mask, self.layer_index)
        else:#static means we do not apply any permutation to transform mask
            return mask
        
        
    def _coupling_transform(self, inputs, embedded_strain, inverse=False):
        """Implements the coupling transform in both forward/inverse pass
        
        Args:
            if inverse = False <---- forward pass (training):
                inputs : tensor of shape (N batch, N posterior parameters) coming from the training dataset
                embedded_strain : tensor of shape (N batch, 3, Length embedded_strain) embedded_strain coming from the training dataset
                
            if inverse = True <----- inverse pass (inference):
                inputs : tensor of shape (N prior samples, N posterior parameters) tensor of prior samples
                embedded_strain : tensor of shape (1, 3, Length embedded_strain) single embedded_strain to analyze
                
                
        Returns:
            - outputs: tensor of same shape as input of parameters mapped by the coupling layer
            - logabsdet: tensor of shape [input.shape[0], 1] in both forward/inverse mode

        """
        
         
        #PERMUTE MASKS
        self.transform_mask = self._permute_mask(self.transform_mask, inverse )
        self.identity_mask  = ~self.transform_mask
        
        
            

        
        #inputs has shape (Nbatch, 10)
        #print('----------> MASK CHECKS trans' ,self.transform_mask)
        #print('----------> MASK CHECKS circ' ,self.circular_mask)
        #print('----------> MASK CHECKS *' ,self.transform_mask*self.circular_mask)    #circular
        #print('----------> MASK CHECKS *~' ,self.transform_mask*~self.circular_mask)   #linear

        #inputs to map with the identity
        identity_inputs  = inputs[:, self.identity_mask, ...]   #has shape (Nbatch, 5)

        
        network_output = self.neural_network(identity_inputs, embedded_strain) #has shape (Nbatch,5, 3*num_bins-1)
        self.num_bins        = self.neural_network.num_bins
        
        
        #extract w, h, d from the neural network output
        widths      = network_output[:, :, :self.num_bins]                  #has shape (Nbatch, 5, num_bins)
        heights     = network_output[:, :,  self.num_bins:2*self.num_bins]  #has shape (Nbatch, 5, num_bins)
        derivatives = network_output[:, :,  2*self.num_bins:]               #has shape (Nbatch, 5, num_bins-1)
        #print('\n!!!!!!!!!!!!!!! widths shape', heights.shape)
        #print(derivatives)
        if hasattr(self.neural_network, "hidden_features"):
            widths /= np.sqrt(self.neural_network.hidden_features)
            heights /= np.sqrt(self.neural_network.hidden_features)
        
        
        
        #print('>>>>>>>>>>>> w.shape', widths.shape)
        
        
        #initialize the output
        outputs = torch.empty_like(inputs)                 #full of zeros of shape as input
        outputs[:, self.identity_mask]  = identity_inputs  #assigning identity features
        
        #TRANSFORM LINEAR ELEMENTS ---------------------------------
        if  any(self.transform_mask*~self.circular_mask):
            mask = self.transform_mask*~self.circular_mask     

            imax = len(torch.where(mask)[0])
            #print('linear imax', imax)
            spline_kwargs = {'tails': 'linear', 'tail_bound': self.tail_bounds[0]}

            lin_trans, lin_logabs = unconstrained_rational_quadratic_spline(inputs[:, mask], 
                                                                            widths[:, :imax, :],
                                                                            heights[:,:imax, :],
                                                                            derivatives[:,:imax, :-1],
                                                                            inverse = inverse, 
                                                                            **spline_kwargs)
            outputs[:, mask] = lin_trans
            logabsdet = lin_logabs
            
            
        #TRANSFORM CIRCULAR ELEMENTS ---------------------------------
        if any(self.transform_mask*self.circular_mask):
            mask = self.transform_mask*self.circular_mask   
            #print('>>>>>>>> mask shape', mask.shape)         
            imax = len(torch.where(mask)[0])
            #print('circular imax', imax)
            spline_kwargs = {'tails': 'circular', 'tail_bound': self.tail_bounds[1]}
            #transformed_inputs[:, mask], logabsdet[:, mask] 
            circ_trans, circ_logabs = unconstrained_rational_quadratic_spline(inputs[:, mask, ...], 
                                                                              widths[:,-imax:, :],
                                                                              heights[:,-imax:, :],
                                                                              derivatives[:,-imax:, :],
                                                                              inverse = inverse, 
                                                                              **spline_kwargs)
            outputs[:, mask] = circ_trans
            logabsdet = torch.cat([logabsdet, circ_logabs], dim = 1) #assuming that some linear transf have been made
        
        #sum over batch logabsdet 
        logabsdet = torch.sum(logabsdet, (1))                        #has shape [N batch]
        
        return outputs, logabsdet
    


    def forward(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=False)
        
        
    
    def inverse(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=True)
        
        
####################################################################################################################################
#==================================================================================================================================#     
####################################################################################################################################

    
    
class AffineConditionalLayer(nn.Module):
    
    def __init__(self,
                 num_features       :int = 8,
                 strain_features    :int = 256,
                 num_identity       :int = 4,
                 num_transformed    :int = 4,
                 
                 ):
        super(AffineConditionalLayer, self).__init__()
        
        self.num_features    = num_features
        self.num_identity    = num_identity
        self.num_transformed = num_transformed
        self.strain_features = strain_features

        
        
        #self.s = nn.Parameter(torch.zeros(num_transformed))
        #self.t = nn.Parameter(torch.zeros(num_transformed))
        #self.s_cond = nn.Parameter(torch.zeros(strain_features, num_transformed))
        #self.t_cond = nn.Parameter(torch.zeros(strain_features, num_transformed))
        s_activation = nn.Tanh()
        t_activation = nn.ELU()
        self.s_network = nn.Sequential(nn.LazyLinear(512), s_activation, 
                                       nn.LazyLinear(512), s_activation, 
                                       nn.LazyLinear(512), s_activation, 
                                       nn.Dropout(0.2), 
                                       nn.LazyLinear(num_transformed), s_activation)
        
        self.t_network = nn.Sequential(nn.LazyLinear(512), t_activation, 
                                       nn.LazyLinear(512), t_activation, 
                                       nn.LazyLinear(512), t_activation, 
                                       nn.Dropout(0.2), 
                                       nn.LazyLinear(num_transformed), t_activation)

        #self.scale_activation = lambda x : x#s(F.softplus(x) + 1e-3).clamp(0, 0.1)
        #self.scale_activation = lambda x : F.sigmoid(x+2) + 1e-3
    
        return

    
    def _coupling_transform(self, inputs, embedded_strain, inverse):

        
        #initialize the output
        outputs = torch.empty_like(inputs)                   #full of zeros of shape as input
        outputs[:, :self.num_identity] = inputs[:, :self.num_identity] #untransformed output
        

        #s = self.s + embedded_strain @ self.s_cond
        #t = self.t + embedded_strain @ self.t_cond
        x = torch.cat([inputs[:, :self.num_identity], embedded_strain], dim=1)
        s = self.s_network(x)
        
        t = self.t_network(x)
        if inverse:
            outputs[:, self.num_identity:] = (inputs[:, self.num_identity:] - t) * torch.exp(-s)
            logabsdet = -torch.sum(s, dim=(1))
            
        else:
            outputs[:, self.num_identity:] = inputs[:, self.num_identity:] * torch.exp(s) + t
            logabsdet = torch.sum(s, dim=(1))


        return outputs, logabsdet
    
    
    
    def forward(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=False)
        
        
    
    def inverse(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=True)
        
        
    

















################################################
################ TESTING #######################
################################################ 
if __name__ == '__main__':
    from neural_network.network_architecture import ConvResNet

    print('Testing...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, n_det, len_embedded_strain = 256, 3, 4096
    embedded_strain_input_shape = (batch_size, n_det, len_embedded_strain)

    embedded_strain               = torch.rand(batch_size, n_det, len_embedded_strain).to(device)
    untransformed_params = torch.rand(batch_size, 5).to(device)

    #model = ConvBlock(input_shape, use_separable_conv=False, pooling='max')
    model = ConvResNet((n_det, len_embedded_strain), num_bins= 10, transformed_params_size=5)
    model = model.to(device)
    print('ok resnet')
    
    
    coupling = SplineCouplingLayer(neural_network = model)
    
    print('ok coupling')
    
    new_mask = coupling._permute_mask(coupling.transform_mask)
    print(new_mask)

