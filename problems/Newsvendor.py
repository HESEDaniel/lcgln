from problems.PThenO import PThenO
# import quandl
import datetime as dt
import pandas as pd
import os
import torch
import random
import pdb
import numpy as np
import cvxpy as cp
import itertools

import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
from qpth.qp import QPFunction

class SolveNewsvendor(nn.Module):
    """ Solve newsvendor scheduling problem """
    def __init__(self, params, eps=1e-2):
        super(SolveNewsvendor, self).__init__()
        k = len(params['d'])
        self.Q = torch.diag(torch.Tensor(
            [params['c_quad']] + [params['b_quad']]*k + [params['h_quad']]*k))
        self.p = torch.Tensor(
            [params['c_lin']] + [params['b_lin']]*k + [params['h_lin']]*k)
        self.G = torch.cat([
            torch.cat([-torch.ones(k,1), -torch.eye(k), torch.zeros(k,k)], 1),
            torch.cat([torch.ones(k,1), torch.zeros(k,k), -torch.eye(k)], 1),
            -torch.eye(1 + 2*k)], 0)
        self.h = torch.Tensor(
            np.concatenate([-params['d'], params['d'], np.zeros(1+ 2*k)]))
        self.one = torch.Tensor([1])
        self.eps_eye = eps * torch.eye(1 + 2*k).unsqueeze(0)

        if torch.cuda.is_available():
            self.Q = self.Q.cuda()
            self.p = self.p.cuda()
            self.G = self.G.cuda()
            self.h = self.h.cuda()
            self.one = self.one.cuda()
            self.eps_eye = self.eps_eye.cuda()

    def forward(self, y):
        nBatch, k = y.size()

        Q_scale = torch.cat([torch.diag(torch.cat(
            [self.one, y[i], y[i]])).unsqueeze(0) for i in range(nBatch)], 0)
        Q = self.Q.unsqueeze(0).expand_as(Q_scale).mul(Q_scale)
        # p_scale = torch.cat([torch.ones(nBatch, 1, device=DEVICE), y, y], 1)
        p_scale = torch.cat([torch.ones(nBatch, 1, device=y.device), y, y], 1)
        p = self.p.unsqueeze(0).expand_as(p_scale).mul(p_scale)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        e = torch.DoubleTensor()
        # if USE_GPU:
        #     e = e.cuda()

        out = QPFunction(verbose=False)\
            (Q.double(), p.double(), G.double(), h.double(), e, e).float()

        return out[:,:1]
    
class Newsvendor(PThenO):
    """Newsvendor problem from Task-based End-to-end model learning in stochastic optimization.
    
    """

    def __init__(
        self,
        num_train_instances=2000,  # number of instances to use from the dataset to train
        num_test_instances=1000,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for test
        rand_seed=0,  # for reproducibility
    ):
        super(Newsvendor, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        train_seed, test_seed = random.randrange(2**32), random.randrange(2**32)
        self.params = self.init_newsvendor_params()
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        # Load train and test labels
        self.Xs_train, self.Ys_train, self.Xs_test, self.Ys_test = self._load_instances(num_train_instances, num_test_instances, test_seed, train_seed, self.params)

        # Split data into train/val/test
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])
        
        # Create functions for optimisation

        self.opt = self._create_cvxpy_problem(self.params)


        # Undo random seed setting
        self._set_seed()
        

    def _load_instances(self, num_train_instances, num_test_instances, test_seed, train_seed, params):
        Theta_true_lin= self.init_theta_true(
            params, with_seed=True)

        # Test data. Set with_seed=True to replicate paper test data.
        X_test, Y_test = self.gen_data(num_test_instances, params, Theta_true_lin, 
                                  with_seed=test_seed)

        # Train data.
        X, Y = self.gen_data(num_train_instances, params, Theta_true_lin, with_seed=train_seed)

        # Split into instances


        return X, Y, X_test, Y_test
    
    def init_newsvendor_params(self):
        params = {}

        # Ordering costs
        params['c_lin'] = 10
        params['c_quad'] = 2.0

        # Over-order penalties
        params['b_lin'] = 30
        params['b_quad'] = 14

        # Under-order penalties
        params['h_lin'] = 10
        params['h_quad'] = 2

        # Discrete demands
        params['d'] = np.array([1, 2, 5, 10, 20]).astype(np.float32)

        # Number of features
        params['n'] = 20

        return params
    
    def init_theta_true(self, params, with_seed=False):

        Theta_true_lin = np.random.randn(params['n'], len(params['d']))

        return Theta_true_lin
    
    def gen_data(self, m, params, Theta_true_lin, with_seed):
        np.random.seed(with_seed)
        X  = np.random.randn(m, params['n'])

        PY = np.exp(X.dot(Theta_true_lin))
        PY = PY / np.sum(PY, axis=1)[:, None]

        # Generate demand realizations
        Y  = np.where(np.cumsum(np.random.rand(m)[:, None]
                    < np.cumsum(PY, axis=1), axis=1) == 1)[1]
        Y  = np.eye(len(params['d']))[Y, :]

        np.random.seed(None)

        return torch.tensor(X).float(), torch.tensor(Y).float()
    
    def _create_cvxpy_problem(self, params):
        newsvendor_solve = SolveNewsvendor(params)
        return newsvendor_solve

    def get_obj(self, py):
        z = cp.Variable(1)
        d = self.params['d']
        f = (self.params['c_lin']*z + 0.5*self.params['c_quad']*cp.square(z) +
            py @ (self.params['b_lin'] * cp.pos(d-z) + 
                    0.5 * self.params['b_quad'] * cp.square(cp.pos(d-z)) +
                    self.params['h_lin'] * cp.pos(z-d) +
                    0.5 * self.params['h_quad'] * cp.square(cp.pos(z-d)) ))
        fval = cp.Problem(cp.Minimize(f), [z >= 0]).solve()
        return fval, torch.tensor(z.value).float()

    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs],  [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs],  [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test,  [None for _ in range(len(self.Ys_test))]

    def get_modelio_shape(self):
        return self.params['n'], len(self.params['d'])

    def get_decision(self, Y, **kwargs):
        return self.opt(Y)

    def get_objective(self, Y, Z, sample = False,**kwargs):
        Y = np.array(Y.cpu()) @ self.params['d']
        Z = Z.detach().cpu().numpy()
        if sample != True:
            Y = Y.reshape(Z.shape[0],Z.shape[1])
        obj = (self.params['c_lin'] * Z + 0.5 * self.params['c_quad'] * (Z**2) + 
                self.params['b_lin'] * np.maximum(Y-Z, 0) + 
                0.5 * self.params['b_quad'] * np.maximum(Y-Z, 0)**2 + 
                self.params['h_lin'] * np.maximum(Z-Y, 0) +
                0.5 * self.params['h_quad'] * np.maximum(Z-Y, 0)**2)
        return torch.tensor(obj)
   
    def get_output_activation(self):
        return 'softmax'

