"""MiniTorch Module

This package contains modules and utilities for building and training neural networks using the MiniTorch library.
It includes support for tensors, automatic differentiation, datasets handling, optimization routines, and various neural network layers.

Included Submodules:
- testing: Tools and classes for testing mathematical properties and variables.
- datasets: Data loading and transformations for machine learning models.
- optim: Optimizers for training models.
- tensor: Core tensor operations and tensor data structures.
- nn: Components for building neural networks such as layers and containers.
- fast_conv: Fast convolution operations.
- tensor_ops: Operations specific to tensors including mathematical operations.
- scalar: Operations and functionalities for scalar values.
- scalar_functions: Scalar-based mathematical functions.
- module: Base classes and utilities for defining and managing network modules.
- autodiff: Automatic differentiation capabilities for gradient computations.
- cuda_ops: CUDA operations for GPU acceleration.
- fast_ops: Optimized operations leveraging hardware acceleration.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .nn import *  # noqa: F401,F403
from .fast_conv import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403
