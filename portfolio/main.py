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
    parser.add_argument('--loadnew', type=ast.literal_eval, default=True)
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
    parser.add_argument('--numsamples', type=int, default=6)
    parser.add_argument('--samplinglr', type=float, default=0.1)
    parser.add_argument('--samplingstd', type=float, default=0.1)
    #   Learning Losses
    parser.add_argument('--loss', type=str, choices=['gicnn', 'icnn', 'mse', 'dense', 'dfl', 'quad', 'eglwmse', 'egldq'], default='dfl')
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--losslr', type=float, default=0.001)
    #   ICNN-specific: Hyperparameters
    parser.add_argument('--icnnhid', type=int, default=2)
    parser.add_argument('--actfn', type=str, default='SOFTPLUS')
    parser.add_argument('--minmax', type=str, default='MAX')
    # #   GICNN-specific: # Samples
    # parser.add_argument('--gicnnsmp', type=int, default=10000)
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
    
    print(f"Loading {args.loss} Loss Function...")
    time_start = time()
    loss_fn = get_loss_fn(
        args.loss,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        sampling_std=args.samplingstd,
        lr=args.losslr,
        serial=args.serial,
        dflalpha=args.dflalpha,
        icnn_hidden_num=args.icnnhid,
        icnn_actfn=args.actfn,
        minmax=args.minmax,
        sampling_lr=args.samplinglr,
    )
      
    # Load an ML model to predict the parameters of the problem
    print(f"Building dense Model...")
    ipdim, opdim = problem.get_modelio_shape()
    model = dense_nn(
        num_features=ipdim,
        num_targets=opdim,
        num_layers=args.layers,
        intermediate_size=args.intermediatesize,
        output_activation=problem.get_output_activation(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)     
    
    # Move everything to GPU, if available
    if torch.cuda.is_available():
        move_to_gpu(problem)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
    # Get data [day,stock,feature]
    X_train, Y_train, Y_train_aux = problem.get_train_data()    # [200,50,28], [200,50], [200,50,50]
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    print('Ours Training Method...')
    if args.minmax.upper()=="MIN":
        best = (float("inf"), None)
    elif args.minmax.upper()=="MAX":
        best = (float("-inf"), None)
    
    else:
        raise LookupError()

    for epoch in tqdm(range(2000)):
        if epoch % args.valfreq == 0:
            # Check if well trained by objective value
            datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val')]
            metrics = print_metrics(datasets, model, problem, args.loss, loss_fn, f"Iter {epoch},", args.wandb)           
            # Save model if it's the best one
            if args.minmax.upper()=="MIN":
                if best[1] is None or metrics['val']['objective'] < best[0]:
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0
            else:
                if best[1] is None or metrics['val']['objective'] > best[0]:
                    # print(epoch)
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0
                    
            if (args.earlystopping) and (steps_since_best > args.patience):
                break
    
        #################### TEST ####################
        # Learn
        losses = []
        for i in random.sample(range(len(X_train)), min(args.batchsize, len(X_train))):
            pred = model(X_train[i]).squeeze()
            losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i, Xs=X_train[i]))
        loss = torch.stack(losses).mean()   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps_since_best += 1
        
    if args.earlystopping:
        print("Early Stopping... Saving the Model...")
        model = best[1]
        # TODO: SAVE THE MODEL!!!!!!!!!
        
    time_elapsed = time() - time_start
    # Document how well this trained model does
    print("\nBenchmarking Model...")
    # Print final metrics
    datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
    # metric_ours = metrics2wandb(datasets, model, problem, "Final")
    metric_ours = print_metrics(datasets, model, problem, args.loss, loss_fn, "Final", args.wandb)   

    #   Document the optimal value
    Z_test_opt = problem.get_decision(Y_test, aux_data=Y_test_aux, isTrain=False)
    objectives = problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)
    obj_opt = objectives.mean().item()
    print(f"Optimal Decision Quality: {obj_opt}")
    
    #   Document OURS value
    obj_ours = metric_ours['test']['objective']
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
        obj_standard = problem.get_objective(Y_test, Z_test_worst, aux_data=Y_test_aux).mean().item()
    else:
        raise LookupError()
    
    #   Our METIRC for paper
    obj2print = (obj_ours-obj_standard) / (obj_opt-obj_standard)
    std2print = ((metric_ours['test']['all_objs']-obj_standard) / (obj_opt-obj_standard)).std().item()
    print(f"Relative Objective: {round(obj2print,4)} +- {round(std2print, 4)}")
    
    #   Write csv file with results
    result_folder = 'results'
    # os.makedirs(os.path.dirname(result_folder), exist_ok=True)
    filename = os.path.join(result_folder, f'{args.problem}-all.csv')
    filename_good = os.path.join(result_folder, f'{args.problem}-better.csv')
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        #   Knapsack
        if (args.problem.upper() == "KNAPSACK"):
            headers = ['seed', 'num_items', 'budget', 'smp', 'smpstd', 'sample_num', \
                        'dense_hid_layer', 'dense_hid', 'opt_lr', 'epoch', \
                        'act_fn', 'icnn_hid', 'loss_lr', 'LOSS', 'OBJ', 'Time']
            if os.stat(filename).st_size == 0:
                writer.writerow(headers)
            writer.writerow([args.seed, args.numitems, args.budget, args.sampling, args.samplingstd, args.numsamples, \
                                args.layers, args.intermediatesize, args.lr, epoch, \
                                args.actfn, args.icnnhid, args.losslr, args.loss, round(obj2print, 4), time_elapsed])
            #   files for well performed parameters
            if obj2print > 0.9:
                with open(filename_good, 'a', newline='') as goodfile:
                    writer_good = csv.writer(goodfile)
                    if os.stat(filename_good).st_size == 0:
                        writer_good.writerow(headers)
                    writer_good.writerow([args.seed, args.numitems, args.budget, args.sampling, args.samplingstd, args.numsamples, \
                                        args.layers, args.intermediatesize, args.lr, epoch, \
                                        args.actfn, args.icnnhid, args.losslr, args.loss, round(obj2print, 4), time_elapsed])
        #   Portfolio
        elif (args.problem.upper() == "PORTFOLIO"):
            headers = ['seed', 'num_stock', 'dflalpha', 'smp', 'smpstd', 'sample_num', \
                        'dense_hid_layer', 'dense_hid', 'opt_lr', 'epoch', \
                        'act_fn', 'icnn_hid', 'loss_lr', 'LOSS', 'OBJ', 'Time']
            if os.stat(filename).st_size == 0:
                
                writer.writerow(headers)
            writer.writerow([args.seed, args.stocks, args.dflalpha, args.sampling, args.samplingstd, args.numsamples, \
                                args.layers, args.intermediatesize, args.lr, epoch, \
                                args.actfn, args.icnnhid, args.losslr, args.loss, round(obj2print, 4), time_elapsed])
            #   files for well performed parameters
            if obj2print > 0.81:
                with open(filename_good, 'a', newline='') as goodfile:
                    writer_good = csv.writer(goodfile)
                    if os.stat(filename_good).st_size == 0:
                        writer_good.writerow(headers)
                    writer_good.writerow([args.seed, args.stocks, args.dflalpha, args.sampling, args.samplingstd, args.numsamples, \
                                        args.layers, args.intermediatesize, args.lr, epoch, \
                                        args.actfn, args.icnnhid, args.losslr, args.loss, round(obj2print, 4), time_elapsed])
