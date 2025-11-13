import torch
from .temporalgcn import TGCN
from .temporalgcn import TGCN2

import sys
import os
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)
from utils.device_set import device_set


class A3TGCN(torch.nn.Module):
    r"""An implementation of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(A3TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = device_set()
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor | None = None,
        H: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        H_sequence_outputs = [] 
        
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        
        H_previous = H 
        
        for period in range(self.periods):
            X_current = X[:, :, :, period]
            H_current = self._base_tgcn(X_current, edge_index, edge_weight, H_previous)
            H_previous = H_current 
            H_sequence_outputs.append(probs[period] * H_current)

        H_accum = sum(H_sequence_outputs)

        return H_accum



class A3TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,  
        periods: int, 
        batch_size:int, 
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super(A3TGCN2, self).__init__()

        self.in_channels = in_channels  # 2
        self.out_channels = out_channels # 32
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,  
            batch_size=self.batch_size,
            improved=self.improved,
            cached=self.cached, 
            add_self_loops=self.add_self_loops)

        device = device_set()
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward( 
        self, 
        X: torch.FloatTensor,
        edge_index: torch.LongTensor, 
        edge_weight: torch.FloatTensor | None = None,
        H: torch.FloatTensor | None  = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        H_sequence_outputs = [] 
        
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        
        H_previous = H 
        
        for period in range(self.periods):
            X_current = X[:, :, :, period]
            H_current = self._base_tgcn(X_current, edge_index, edge_weight, H_previous)
            H_previous = H_current 
            H_sequence_outputs.append(probs[period] * H_current)

        H_accum = sum(H_sequence_outputs)

        return H_accum