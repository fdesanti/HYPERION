from .permutation import RandomPermutation
from .coupling_transform import CouplingTransform
from .affine_coupling_layer import AffineCouplingLayer
from .spline_coupling_layer import SplineCouplingLayer

coupling_layer_dict = {'affine': AffineCouplingLayer,
                       'spline': SplineCouplingLayer
                       }

