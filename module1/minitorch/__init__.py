"""minitorch: A minimalistic deep learning framework for educational purposes.

This package provides essential modules for autodifferentiation, optimization,
and building neural network models. It includes the following components:

- `autodiff`: Core functionality for automatic differentiation.
- `scalar`: Scalar operations and classes for tracking computation graphs.
- `scalar_functions`: Basic scalar mathematical functions for use in computation graphs.
- `optim`: Optimization algorithms, including parameter updates.
- `datasets`: Predefined datasets for testing and experimentation.
- `testing`: Tools for testing and validating models and functions.
- `module`: Base class for neural network modules and layers.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
