import argparse
import ast
import csv
import cvxpy as cp
import numpy as np
import os
import random
import torch

from copy import deepcopy
from cvxpylayers.torch import CvxpyLayer
from functools import partial
from time import time
from tqdm import tqdm

from losses import get_loss_fn, MSE
from Networks import ICNN
from PortfolioOpt import PortfolioOpt
from utils import print_metrics, init_if_not_saved, move_to_gpu


def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation='relu',
    output_activation='sigmoid',
):
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

    return torch.nn.Sequential(*net_layers)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['knapsack', 'portfolio'], default='portfolio')
    parser.add_argument('--loadnew', type=ast.literal_eval, default=False)
    parser.add_argument('--instances', type=int, default=400)
    parser.add_argument('--testinstances', type=int, default=400)
    parser.add_argument('--valfrac', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--intermediatesize', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--valfreq', type=int, default=50)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--wandb', type=bool, default=False)
    # parser.add_argument('--twostage', type=bool, default=True)
    #   DFL-specific
    parser.add_argument('--dflalpha', type=float, default=10)
    #   Sampling-specific
    parser.add_argument('--sampling', type=str, choices=['random', 'mbs'], default='mbs')
    parser.add_argument('--numsamples', type=int, default=2)
    parser.add_argument('--samplinglr', type=float, default=0.1)
    parser.add_argument('--samplingstd', type=float, default=0.1)
    #   Learning Losses
    parser.add_argument('--loss', type=str, choices=['gicnn', 'lancer', 'mse', 'dense', 'dfl', 'quad', 'eglwmse', 'egldq'], default='lancer')
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--losslr', type=float, default=0.001)
    #   ICNN-specific: Hyperparameters
    parser.add_argument('--icnnhid', type=int, default=2)
    parser.add_argument('--actfn', type=str, default='SOFTPLUS')
    parser.add_argument('--minmax', type=str, default='MIN')
    #   Domain-specific: Portfolio Optimization
    parser.add_argument('--stocks', type=int, default=50)
    parser.add_argument('--stockalpha', type=float, default=0.1)
    # parser.add_argument('--sampling_std', type=float, default=0.1)
    #   Domain-specific: Knapsack
    parser.add_argument('--numitems', type=int, default=50)
    parser.add_argument('--budget', type=int, default=1)
    args = parser.parse_args()
    
    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")
    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
    
    if args.problem == 'portfolio':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_stocks': args.stocks,
                            'alpha': args.stockalpha,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
        problem = init_problem(PortfolioOpt, problem_kwargs)
    
    # Get data [day,stock,feature]
    X_train, Y_train, Y_train_aux = problem.get_train_data()    # [200,50,28], [200,50], [200,50,50]
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()
    
    
    # Define model_theta
    print(f"Building dense Model...")
    ipdim, opdim = problem.get_modelio_shape()
    model_theta = dense_nn(
        num_features=ipdim,
        num_targets=opdim,
        num_layers=1,
        intermediate_size=500,
        output_activation=problem.get_output_activation(),
    )
    optimizer_theta = torch.optim.Adam(model_theta.parameters(), lr=0.005)
    
    # Define loss function
    loss_fn = dense_nn(
        num_features=2*Y_train.shape[1],
        num_targets=1,
        num_layers=2,
        intermediate_size=100,
        output_activation=torch.nn.Tanh(),
    )
    optimizer_loss = torch.optim.Adam(loss_fn.parameters(), lr=0.001)

    # Move everything to GPU, if available
    if torch.cuda.is_available():
        move_to_gpu(problem)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_theta = model_theta.to(device)
        loss_fn = loss_fn.to(device)
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        Y_train = Y_train.to(device)
        Y_train_aux = Y_train_aux.to(device)
        
        
    # LANCER
    print(f"Loading {args.loss} Loss Function...")
    time_start = time()
    sample_set = []
    for t in tqdm(range(args.numsamples)):
        samples = model_theta(X_train).squeeze()  # [200,50,28]->[200,50]
        opt = partial(problem.get_decision, isTrain=False, aux_data=Y_train_aux)
        obj = partial(problem.get_objective, aux_data=Y_train_aux)
        
        Z_opt = opt(Y_train)
        opt_objective = obj(Y_train, Z_opt)
        
        Zs = opt(samples, Z_init=Z_opt)
        objectives = obj(Y_train, Zs)
        objectives = opt_objective - objectives
        
        sample_set.append((Y_train, opt_objective, samples, objectives))
        # Update Surrogate Loss
        for smp in sample_set:
            optimizer_loss.zero_grad()
            # Assume smp has the format: (true_y, predicted_y, samples, objective)
            y, obj_y, yhat, obj_yhat = smp
            inputs = torch.cat((y, yhat), dim=1).to(device)
            target = obj_yhat.to(device)
            loss_output = loss_fn(inputs).squeeze()
            loss = MSE(loss_output, target)
            loss.backward(retain_graph=True)
            optimizer_loss.step()
        
        # Update Predictive Model
        for epoch in range(10):
            losses = []
            # DFL only
            # losses.append( loss_fn( torch.cat((Y_train, model_theta(X_train).squeeze()), dim=1) ) )
            # DFL + PFL
            losses.append( loss_fn( torch.cat((Y_train, model_theta(X_train).squeeze()), dim=1) ) + MSE(Y_train,model_theta(X_train).squeeze()))
            loss = torch.stack(losses).mean()   
            optimizer_theta.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_theta.step()
            
    time_elapsed = time() - time_start      

    print('Ours Training Method...')
    if args.minmax.upper()=="MIN":
        best = (float("inf"), None)
    elif args.minmax.upper()=="MAX":
        best = (float("-inf"), None)
    else:
        raise LookupError()
    
    # Document how well this trained model does
    print("\nBenchmarking Model...")
    # Print final metrics
    datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
    
    #   Document the optimal value
    Z_test_opt = problem.get_decision(Y_test, aux_data=Y_test_aux, isTrain=False)
    objectives = problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)
    obj_opt = objectives.mean().item()
    print(f"Optimal Decision Quality: {obj_opt}")
    
    Z_ours = problem.get_decision(model_theta(X_test.to('cuda')).squeeze(), aux_data=Y_test_aux.to('cuda'), isTrain=False)
    objectives_ours = problem.get_objective(Y_test.to('cuda'), Z_ours, aux_data=Y_test_aux.to('cuda'))
    obj_ours = objectives_ours.mean().item() 
    
    # #   Document OURS value
    print(f"Our Decision Quality: {obj_ours}")

    if args.problem.upper()=="KNAPSACK":
        #   Knapsack METRIC
        Z_test_worst = problem.get_decision(-Y_test, aux_data=Y_test_aux, isTrain=False)    # for worst decision
        obj_standard = problem.get_objective(Y_test, Z_test_worst, aux_data=Y_test_aux).mean().item()
    elif args.problem.upper()=="PORTFOLIO":
        #   Portfolio METRIC
        worst_indices = torch.argmin(Y_test, dim=1).unsqueeze(1)        # for worst decision
        Z_test_worst = torch.zeros_like(Y_test).scatter(1, worst_indices, 1)
        Z_test_worst = Z_test_worst.to(device)
        obj_standard = problem.get_objective(Y_test.to('cuda'), Z_test_worst.to('cuda'), aux_data=Y_test_aux.to('cuda')).mean().item()
    else:
        raise LookupError()
    
    #   Our METIRC for paper
    obj2print = (obj_ours-obj_standard) / (obj_opt-obj_standard)
    print(obj2print)
    
    #   Write csv file with results
    result_folder = 'results'
    filename = os.path.join(result_folder, f'{args.problem}-lancer.csv')
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        #   Portfolio
        if (args.problem.upper() == "PORTFOLIO"):
            headers = ['seed', 'num_stock', 'dflalpha', 'smp', 'smpstd', 'sample_num', \
                        'dense_hid_layer', 'dense_hid', 'opt_lr', 'epoch', \
                        'act_fn', 'icnn_hid', 'loss_lr', 'LOSS', 'OBJ', 'Time']
            if os.stat(filename).st_size == 0:
                
                writer.writerow(headers)
            writer.writerow([args.seed, args.stocks, args.dflalpha, args.sampling, args.samplingstd, args.numsamples, \
                                args.layers, args.intermediatesize, args.lr, epoch, \
                                args.actfn, args.icnnhid, args.losslr, args.loss, round(obj2print, 4), time_elapsed])
           