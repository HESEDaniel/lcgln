from PThenO import PThenO
# import quandl
import datetime as dt
import pandas as pd
import os
import torch
import random
import pdb
import cvxpy as cp
import itertools
from cvxpylayers.torch import CvxpyLayer
# quandl.ApiConfig.api_key = '3Uxzq4TZV5V9RghuRYsY'


class PortfolioOpt(PThenO):
    """A class that implements the Portfolio Optimization problem from
    Wang, Kai, et al. "Automatically learning compact quality-aware surrogates
    for optimization problems." Advances in Neural Information Processing
    Systems 33 (2020): 9586-9596.
    
    The code is largely adapted from: https://github.com/guaguakai/surrogate-optimization-learning/"""

    def __init__(
        self,
        num_train_instances=200,  # number of *days* to use from the dataset to train
        num_test_instances=200,  # number of *days* to use from the dataset to test
        num_stocks=50,  # number of stocks per instance to choose from
        val_frac=0.2,  # fraction of training data reserved for test
        rand_seed=0,  # for reproducibility
        alpha=1,  # risk aversion constant
        data_dir="data",  # directory to store data
    ):
        super(PortfolioOpt, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)

        # Load train and test labels
        self.num_stocks = num_stocks
        self.Xs, self.Ys, self.covar_mat = self._load_instances(data_dir, num_stocks)

        # Split data into train/val/test
        #   Sanity check and initialisations
        total_days = self.Xs.shape[0]
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        num_days = self.num_train_instances + self.num_test_instances
        assert self.num_train_instances + self.num_test_instances < total_days
        assert 0 < val_frac < 1
        self.val_frac = val_frac

        #   Creating "days" for train/valid/test
        idxs = list(range(num_days))
        num_val = int(self.val_frac * self.num_train_instances)
        self.train_idxs = idxs[:self.num_train_instances - num_val]
        self.val_idxs = idxs[self.num_train_instances - num_val:self.num_train_instances]
        self.test_idxs = idxs[self.num_train_instances:]
        assert all(x is not None for x in [self.train_idxs, self.val_idxs, self.test_idxs])

        # Create functions for optimisation
        # TODO: Try larger constant
        self.alpha = alpha
        self.opt = self._create_cvxpy_problem(alpha=self.alpha)
        # Undo random seed setting
        self._set_seed()

    def _load_instances(
        self,
        data_dir,
        stocks_per_instance,
        reg=0.1,
    ):
        # Get raw data
        feature_mat, target_mat, _, future_mat, _, dates, symbols = self._get_data(data_dir=data_dir)

        # Split into instances
        # Sample stocks in a day to define an instance
        total_stocks = len(symbols)
        stocks_subset = random.sample(range(total_stocks), stocks_per_instance)
        feature_mat = feature_mat[:, stocks_subset]
        target_mat = target_mat[:, stocks_subset].squeeze()
        future_mat =future_mat[:, stocks_subset]

        # Calculate covariances
        def computeCovariance(
            future_mat,
            correl=True  # Normalise covariance matrix to get correlation matrix
        ):
            # Normalize
            mean = future_mat.mean(dim=-1, keepdim=True)
            fm_norm = future_mat - mean  # normalised future matrix
            if correl == True:
                std = (future_mat.square().mean(dim=-1, keepdim=True) - mean.square()).sqrt()
                fm_norm = fm_norm / std

            # Compute covariance
            # TODO: See if things change if you get rid of num_samples
            num_samples = future_mat.shape[-1]
            spi = future_mat.shape[-2]  # stocks per instance
            covar_raw = [(fm_norm * fm_norm[..., i:i+1, :].repeat((*((1,) * (fm_norm.ndim - 2)), spi, 1))).sum(dim=-1) for i in range(spi)]
            covar_mat_unreg = torch.stack(covar_raw, dim=-1) / (num_samples - 1)

            # Add regularisation to make sure that the covariance matrix is positive-definite
            covar_mat = covar_mat_unreg + reg * torch.eye(spi)

            return covar_mat
        covar_mat = computeCovariance(future_mat)

        # Normalize features
        num_features = feature_mat.shape[-1]
        feature_mat_flat = feature_mat.reshape(-1, num_features)
        feature_mat = torch.div((feature_mat - torch.mean(feature_mat_flat, dim=0)), (torch.std(feature_mat_flat, dim=0) + 1e-12))

        return feature_mat.float(), target_mat.float(), covar_mat.float()

    def _get_price_feature_df(
        self,
        overwrite=False,
    ):
        """
        Loads raw historical price data if it exists, otherwise compute the file on the fly, this adds other timeseries
        features based on rolling windows of the price
        :return:
        """
        print("Loading dataset...")
        price_feature_df = pd.read_csv(self.price_feature_file, index_col=["Date", "Symbol"])


        return price_feature_df
    
    
    def _get_price_feature_matrix(self, price_feature_df):
        num_dates, num_assets = map(len, price_feature_df.index.levels)
        price_matrix = price_feature_df.values.reshape((num_dates, num_assets, -1))
        return price_matrix

    def _get_data(
        self,
        data_dir,
        start_date=dt.datetime(2004, 1, 1),
        end_date=dt.datetime(2017, 1, 1),
        collapse="daily",
        overwrite=False,
    ):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save constants
        self.start_date = start_date
        self.end_date = end_date
        self.collapse = collapse

        # Define data directories to write to
        self.raw_historical_price_file = os.path.join(data_dir, "raw_historical_prices_{}_{}_{}.csv".format(start_date.date(), end_date.date(), collapse))
        self.raw_symbol_file = os.path.join(data_dir, "raw_symbols.csv")
        self.price_feature_file = os.path.join(data_dir, "price_feature_mat_{}_{}_{}.csv".format(start_date.date(), end_date.date(), collapse))
        self.torch_file = os.path.join(data_dir, "price_data_{}_{}_{}.pt".format(start_date.date(), end_date.date(), collapse))

        # Load data if it exists
        print(self.torch_file)
        print("Loading pytorch data...")
        feature_mat, target_mat, feature_cols, future_mat, target_names, dates, symbols = torch.load(self.torch_file)
       
        return feature_mat, target_mat, feature_cols, future_mat, target_names, dates, symbols

    def _create_cvxpy_problem(
        self,
        alpha,
    ):
        x_var = cp.Variable(self.num_stocks)
        L_sqrt_para = cp.Parameter((self.num_stocks, self.num_stocks))
        p_para = cp.Parameter(self.num_stocks)
        constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
        objective = cp.Minimize(- p_para.T @ x_var + alpha * cp.sum_squares(L_sqrt_para @ x_var))
        problem = cp.Problem(objective, constraints)

        return CvxpyLayer(problem, parameters=[p_para, L_sqrt_para], variables=[x_var])   

    def get_train_data(self, **kwargs):
        return self.Xs[self.train_idxs], self.Ys[self.train_idxs], self.covar_mat[self.train_idxs]

    def get_val_data(self, **kwargs):
        return self.Xs[self.val_idxs], self.Ys[self.val_idxs], self.covar_mat[self.val_idxs]

    def get_test_data(self, **kwargs):
        return self.Xs[self.test_idxs], self.Ys[self.test_idxs], self.covar_mat[self.test_idxs]

    def get_modelio_shape(self):
        return self.Xs.shape[-1], 1

    def get_twostageloss(self):
        return 'mse'

    def _get_covar_mat(self, instance_idxs):
        return self.covar_mat.reshape((-1, *self.covar_mat.shape[2:]))[instance_idxs]

    def get_decision(self, Y, aux_data, max_instances_per_batch=1500, **kwargs):
        # Get the sqrt of the covariance matrix
        covar_mat = aux_data
        sqrt_covar = torch.linalg.cholesky(covar_mat)

        # Split Y into reasonably sized chunks so that we don't run into memory issues
        # Assumption Y is only 2D at max
        assert Y.ndim <= 2
        if Y.ndim == 2:
            results = []
            for start in range(0, Y.shape[0], max_instances_per_batch):
                end = min(Y.shape[0], start + max_instances_per_batch)
                result = self.opt(Y[start:end], sqrt_covar)[0]
                results.append(result)
            return torch.cat(results, dim=0)
        else:
            return self.opt(Y, sqrt_covar)[0]

    def get_objective(self, Y, Z, aux_data, **kwargs):
        # TODO: look at either torch.bmm or torch.matmul
        covar_mat = aux_data
        covar_mat_Z_t = (torch.linalg.cholesky(covar_mat) * Z.unsqueeze(dim=-2)).sum(dim=-1)
        quad_term = covar_mat_Z_t.square().sum(dim=-1)
        obj = - (Y * Z).sum(dim=-1) + self.alpha * quad_term
        return obj
    
    def get_output_activation(self):
        return 'tanh'


if __name__ == "__main__":
    problem = PortfolioOpt()
    X_train, Y_train, Y_train_aux = problem.get_train_data()

    Z_train = problem.get_decision(Y_train, aux_data=Y_train_aux)
    obj = problem.get_objective(Y_train, Z_train, aux_data=Y_train_aux)

    pdb.set_trace()