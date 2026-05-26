from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # Ensure the input tensor is contiguous
    if not input._tensor.is_contiguous():
        input = input.contiguous()
    # Compute the new height and width
    new_height = height // kh
    new_width = width // kw

    # Reshape the input tensor
    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)

    # Move the kernel dimensions to the last axis
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Flatten the kernel dimensions into a single axis
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform tiled average pooling on a 2D tensor."""
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=-1)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:  ### used in the backward function of Max
    """Compute the argmax along a given dimension as a 1-hot tensor."""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    """Max Function for backpropagation."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute the forward pass for the Max function.

        Args:
        ----
            ctx : Context
                The context object to save values for backpropagation.
            input : Tensor
                The input tensor on which to apply the max operation.
            dim : Tensor
                The dimension along which to compute the max.

        Returns:
        -------
            Tensor: The result of applying the Max function.

        """
        dimInt = int(dim[0])
        ctx.save_for_backward(input, dimInt)
        return max_reduce(input, dimInt)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the backward pass for the Max function.

        Args:
        ----
            ctx : Context
                The context object containing saved values for backpropagation.
            grad_output : Tensor
                The gradient of the output tensor from the forward pass.

        Returns:
        -------
            Tuple[Tensor, float]
                The gradient with respect to the input tensor and a placeholder (0.0).

        """
        input, dimInt = ctx.saved_values
        return (grad_output * argmax(input, dimInt), 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    """Apply the Max function to the input tensor along the specified dimension.

    Args:
    ----
        input : Tensor
            The input tensor on which to apply the max operation.
        dim : int
            The dimension along which to compute the max.

    Returns:
    -------
        Tensor: The result of applying the Max function.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a given dimension."""
    input = input.exp()
    t = input.sum(dim)
    input = input / t
    return input


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log softmax using the log-sum-exp trick."""
    t = input.exp()
    t = t.sum(dim)
    t = t.log()
    return input - t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform tiled max pooling on a 2D tensor."""
    batch, channel, height, width = input.shape
    input, tile_h, tile_w = tile(input, kernel)
    input = max(input, 4)
    input = input.view(batch, channel, tile_h, tile_w)
    return input


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout based on random noise."""
    if not ignore:
        bit_tensor = rand(input.shape, input.backend) > rate
        input = bit_tensor * input
    return input
