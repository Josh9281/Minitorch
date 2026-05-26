from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    x_plus_epsilon = list(vals)
    x_plus_epsilon[arg] = x_plus_epsilon[arg] + epsilon
    x_minus_epsilon = list(vals)
    x_minus_epsilon[arg] = x_minus_epsilon[arg] - epsilon
    return (f(*x_plus_epsilon) - f(*x_minus_epsilon)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Return True if the variable is a leaf."""
        ...

    def is_constant(self) -> bool:
        """Return True if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute derivatives."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()  # Keep track of visited variables
    result = []  # To store the final topological order

    # Helper function to recursively visit nodes
    def dfs(var: Variable) -> None:
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            dfs(parent)
        result.append(var)

    # Start the DFS traversal from the given variable
    dfs(variable)

    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    """
    # TODO: Implement for Task 1.4.
    # Dictionary to store derivatives for each variable
    derivatives_dict = {variable.unique_id: deriv}

    sorted_vars = topological_sort(variable)

    # Iterate through the topological order and calculate the derivatives
    for curr_var in sorted_vars:
        if curr_var.is_leaf():
            continue

        # Get the derivatives of the current variable
        var_n_der = curr_var.chain_rule(derivatives_dict[curr_var.unique_id])

        # Accumulate the derivative for each parent of the current variable
        for var, deriv in var_n_der:
            if var.is_leaf():
                var.accumulate_derivative(deriv)
            else:
                if var.unique_id not in derivatives_dict:
                    derivatives_dict[var.unique_id] = deriv
                else:
                    derivatives_dict[var.unique_id] += deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors for backpropagation."""
        return self.saved_values
