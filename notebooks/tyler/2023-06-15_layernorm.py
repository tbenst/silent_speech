##
from typing import Tuple
import torch

def layer_norm(
    x: torch.Tensor, dim: Tuple[int] = None, eps: float = 0.00001
) -> torch.Tensor:
    """
    Layer normalization as described in https://arxiv.org/pdf/1607.06450.pdf.
    
    Supports inputs of any shape, where first dimension is the batch. Does not
    apply elementwise affine transformation.
    
    https://stackoverflow.com/questions/59830168/layer-normalization-in-pytorch
    """
    if dim is None:
        # all except batch
        dim = tuple(range(1, len(x.shape)))
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.var(x, dim=dim, keepdim=True, correction=0)
    return (x - mean) / torch.sqrt(var + eps)


def test_that_results_match() -> None:
    dims = (1, 2)
    X = torch.normal(0, 1, size=(3, 3, 3))

    indices = torch.tensor(dims)
    normalized_shape = torch.tensor(X.size()).index_select(0, indices)
    orig_layer_norm = torch.nn.LayerNorm(normalized_shape)

    y = orig_layer_norm(X)
    y_hat = layer_norm(X)

    assert torch.allclose(y, y_hat), f"y: {y}\ny_hat: {y_hat}"
    
test_that_results_match()
##
