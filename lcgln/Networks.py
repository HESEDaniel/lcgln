import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator

from utils.utils import View


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())


class LCGLN(nn.Module):
    
    def __init__(
        self,
        x_dim,
        y_dim,
        u_dim,
        z_dim,
        output_dim=1,
        act_fn='SOFTPLUS',
        **kwargs
    ):
        super(LCGLN, self).__init__()
        
        if act_fn.upper()=='ELU':
            self.act_fn = nn.ELU()
        elif act_fn.upper()=='SOFTPLUS':
            self.act_fn = nn.Softplus()
        else:
            raise LookupError()
        
        # Input
        #   Upstream
        self.x_to_u = nn.Linear(x_dim, u_dim)
        #   Downstream
        self.x_to_ydim = nn.Linear(x_dim, y_dim)
        self.y_to_zdim = nn.Linear(y_dim, z_dim)
        self.x_to_zdim = nn.Linear(x_dim, z_dim)
        
        # Hidden (for later use)
        #   Upstream
        self.u_to_u = nn.Linear(u_dim, u_dim)
        #   Downstream
        #       1st Term
        self.u_to_zdim = nn.Linear(u_dim, z_dim)
        self.zdim_to_z = nn.Linear(z_dim, z_dim, bias=False)
        #       2nd Term
        self.u_to_ydim = nn.Linear(u_dim, y_dim)
        self.ydim_to_z = nn.Linear(y_dim, z_dim, bias=False)
        #       3rd Term
        self.u_to_z = nn.Linear(u_dim, z_dim)
        
        # Output
        #   Downstream
        #       1st Term
        self.out_u_to_zdim = nn.Linear(u_dim, z_dim)
        self.out_zdim_to_out = nn.Linear(z_dim, output_dim, bias=False)
        #       2nd Term
        self.out_u_to_ydim = nn.Linear(u_dim, y_dim)
        self.out_ydim_to_out = nn.Linear(y_dim, output_dim, bias=False)
        #       3rd Term
        self.out_u_to_out = nn.Linear(u_dim, output_dim)

        
    def forward(self, x, y):
        # Input
        #   Upstream
        u1 = self.x_to_u(x)
        u1 = self.act_fn(u1)
        #   Downstream
        xz1 = self.x_to_zdim(x)
        yz1 = self.y_to_zdim(y)
        z1 = self.act_fn(xz1 + yz1)
        
        # Hidden
        # no hid
        
        # Output
        #   Downstream
        #       1st Term
        uzdim = self.out_u_to_zdim(u1)
        uzdim = torch.clamp_min(uzdim,0) * z1
        uzz2 = self.out_zdim_to_out(uzdim)
        #       2nd Term
        uydim = self.out_u_to_ydim(u1)
        uydim *= y
        # uydim = uydim * y.repeat(uydim.shape[0],1)
        uyz2 = self.out_ydim_to_out(uydim)
        #       3rd Term
        uz2 = self.out_u_to_out(u1)
        
        out = self.act_fn(uzz2 + uyz2 + uz2)
        
        return out
    
class dense_nn(nn.Module):
    def __init__(
        self, 
        num_features, 
        num_targets, 
        num_layers, 
        intermediate_size=10, 
        activation='relu', 
        output_activation='relu'
    ):
        super(dense_nn, self).__init__()
        
        if num_layers > 1:
            if intermediate_size is None:
                intermediate_size = max(num_features, num_targets)
            if activation == 'relu':
                activation_fn = torch.nn.ReLU
            elif activation == 'sigmoid':
                activation_fn = torch.nn.Sigmoid
            else:
                raise Exception('Invalid activation function: ' + str(activation))
            net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
            for _ in range(num_layers - 2):
                net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
                net_layers.append(activation_fn())
            if not isinstance(num_targets, tuple):
                net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
            else:
                net_layers.append(torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1)))
                net_layers.append(View(num_targets))
        else:
            if not isinstance(num_targets, tuple):
                net_layers = [torch.nn.Linear(num_features, num_targets)]
            else:
                net_layers = [torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)), View(num_targets)]

        if output_activation == 'relu':
            net_layers.append(torch.nn.ReLU())
        elif output_activation == 'sigmoid':
            net_layers.append(torch.nn.Sigmoid())
        elif output_activation == 'tanh':
            net_layers.append(torch.nn.Tanh())
        elif output_activation == 'softmax':
            net_layers.append(torch.nn.Softmax(dim=-1))
        elif output_activation == 'softplus':
            net_layers.append(torch.nn.Softplus())
            
        self.dense = nn.Sequential(*net_layers)
        
    def forward(self, x):
        return self.dense(x)