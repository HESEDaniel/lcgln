import argparse
import ast
import csv
import numpy as np
import os
import random
import torch
import datetime as dt
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from Dataloader import *
from losses import get_loss_fn
from BudgetAllocation import BudgetAllocation
from utils import print_metrics, init_if_not_saved, move_to_gpu, dense_nn
from constants import *

def metrics2wandb(
    datasets,
    model,
    problem,
    prefix="",
):
    metrics = {}
    for Xs, Ys, Ys_aux, partition in datasets:
        # Choose whether we should use train or test 
        isTrain = (partition=='train') and (prefix != "Final")
        pred = model(Xs).squeeze()
        Zs_pred = problem.get_decision(pred, aux_data=Ys_aux, isTrain=isTrain)
        objectives = problem.get_objective(Ys, Zs_pred, aux_data=Ys_aux)
        # Print
        objective = objectives.mean().item()
        metrics[partition] = {'objective': objective}
        


    return metrics


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['budgetallocation'], default='budgetallocation')
    parser.add_argument('--loadnew', type=ast.literal_eval, default=False)
    parser.add_argument('--instances', type=int, default=100)
    parser.add_argument('--testinstances', type=int, default=500)
    parser.add_argument('--valfrac', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--intermediatesize', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--valfreq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--wandb', type=bool, default=False)
    #   Learning Losses
    parser.add_argument('--loss', type=str, choices=['gicln', 'mse', 'dense', 'dfl', 'quad', 'eglwmse', 'egldq'], default='gicln')
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--sampling', type=str, choices=['random', 'mbs'], default='mbs')
    parser.add_argument('--numsamples', type=int, default=16)
    parser.add_argument('--samplinglr', type=float, default=0.05)
    parser.add_argument('--samplingstd', type=float, default=0.1)
    parser.add_argument('--losslr', type=float, default=0.001)

    #   ICLN-specific: Hyperparameters
    parser.add_argument('--iclnhid', type=int, default=2)
    parser.add_argument('--actfn', type=str, default='SOFTPLUS')
    parser.add_argument('--minmax', type=str, default='MAX')

    #   Domain-specific: BudgetAllocation
    parser.add_argument('--targets', type=int, default=10)
    parser.add_argument('--items', type=int, default=5)
    parser.add_argument('--BAbudget', type=int, default=2)
    parser.add_argument('--fake_targets', type=int, default=50)
    args = parser.parse_args()

    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")

    save_folder = os.path.join('results', str(args.loss), str(args.numsamples))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    results_file = os.path.join(save_folder, f"newsvendor_results.csv")

    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
    
    if args.problem == 'budgetallocation':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_targets': args.targets,
                            'num_items': args.items,
                            'budget': args.BAbudget,
                            'num_fake_targets': args.fake_targets,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
        problem = init_problem(BudgetAllocation, problem_kwargs)
    
    print(f"Loading {args.loss} Loss Function...")
    loss_fn = get_loss_fn(
        args.loss,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        sampling_std=args.samplingstd,
        lr=args.losslr,
        serial=args.serial,
        icln_hidden_num=args.iclnhid,
        icln_actfn=args.actfn,
        minmax=args.minmax,
        samplinglr=args.samplinglr
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

            # Stop if model hasn't improved for patience steps
            if (args.earlystopping) and (steps_since_best > args.patience):
                break
        
        #################### TEST ####################
        # Learn
        losses = []

        def news_loss_fn(pred, y, loss_fn):
            news_losses =[]
            for i in range(y.shape[0]):
                news_losses.append(loss_fn(pred[i], y[i], partition = 'train', index = i))
            return torch.stack(news_losses).mean() 
                
        train_dataloader = ICLN_data_loader(X_train, Y_train, batch_size=args.batchsize)
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X, y = batch
            pred = model(X).squeeze()
            loss = news_loss_fn(pred, y, loss_fn)
            loss.backward()
            optimizer.step()
        steps_since_best += 1
        ###############################################       
    if args.earlystopping:
        print("Early Stopping... Saving the Model...")
        model = best[1]      

    # Document how well this trained model does
    print("\nBenchmarking Model...")
    # Print final metrics
    datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
    metric_ours = print_metrics(datasets, model, problem, args.loss, loss_fn, "Final", args.wandb)   

    #   Document the optimal value
    Z_test_opt = problem.get_decision(Y_test, aux_data=Y_test_aux, isTrain=False)
    objectives = problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)
    obj_opt = objectives.mean().item()
    print(f"Optimal Decision Quality: {obj_opt}")
    
    #   Document OURS value
    obj_ours = metric_ours['test']['objective']
    print(f"Our Decision Quality: {obj_ours}")
    with open(results_file, 'a') as f:
        f.write('{},{},{},{}\n'.format('DQ_O:', obj_opt, 'DQ:', obj_ours))
    pass
