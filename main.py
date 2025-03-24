import argparse
import ast
import numpy as np
import random
import torch

from copy import deepcopy
from functools import partial
from time import time
from tqdm import tqdm

from lcgln.losses import get_loss_fn
from lcgln.Networks import *
from problems.BudgetAllocation import *
from problems.PortfolioOpt import *
from problems.Newsvendor import *
from utils.utils import *


def set_seed(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--problem', type=str, choices=['budget', 'portfolio', 'inventory'], default='portfolio')
    parser.add_argument('--loadnew', type=ast.literal_eval, default=False)
    #   Problem-specific: Budget Allocation
    parser.add_argument('--targets', type=int, default=10)
    parser.add_argument('--items', type=int, default=5)
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--faketargets', type=int, default=500)
    #   Problem-specific: Portfolio Optimization
    parser.add_argument('--stocks', type=int, default=50)
    parser.add_argument('--stockalpha', type=float, default=0.1)
    #   Dataset
    parser.add_argument('--instances', type=int, default=400)
    parser.add_argument('--testinstances', type=int, default=400)
    parser.add_argument('--valfrac', type=float, default=0.5)
    #   Sampling
    parser.add_argument('--sampling', type=str, choices=['mbs'], default='mbs')
    parser.add_argument('--numsamples', type=int, default=32)
    parser.add_argument('--samplinglr', type=float, default=1)
    parser.add_argument('--samplingnumlayers', type=int, default=2)
    parser.add_argument('--samplingintermediatesize', type=int, default=10)
    #   LCGLN
    parser.add_argument('--loss', type=str, choices=['lcgln'], default='lcgln')
    parser.add_argument('--losslr', type=float, default=0.001)
    # parser.add_argument('--lcglnhid', type=int, default=2)
    parser.add_argument('--lcglnactfn', type=str, default='SOFTPLUS')
    parser.add_argument('--lcglnbatchsize', type=int, default=128)
    #   Predictive model
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--intermediatesize', type=int, default=500)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--valfreq', type=int, default=50)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--predmodelbatchsize', type=int, default=1000)
    parser.add_argument('--n_iter', type=int, default=2000)
    
    args = parser.parse_args()
    

    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")
    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
   
    if args.problem == 'budget':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_targets': args.targets,
                            'num_items': args.items,
                            'budget': args.budget,
                            'num_fake_targets': args.faketargets,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
        minmax = 'MAX'
        problem = init_problem(BudgetAllocation, problem_kwargs)
        
    elif args.problem == 'portfolio':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_stocks': args.stocks,
                            'alpha': args.stockalpha,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
        minmax = 'MAX'
        problem = init_problem(PortfolioOpt, problem_kwargs)
    
    elif args.problem == 'inventory':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'rand_seed': args.seed,}
        minmax = 'MIN'
        problem = init_problem(Newsvendor, problem_kwargs)
       
    print(f"Loading {args.loss} Loss Function...")
    time_start = time()
    loss_fn = get_loss_fn(
        args.loss,
        problem,
        sampling_method=args.sampling,
        num_samples=args.numsamples,
        sampling_lr=args.samplinglr,
        sampling_model_num_layers=args.samplingnumlayers,
        sampling_model_intermediate_size=args.samplingintermediatesize,
        batch_size=args.lcglnbatchsize,
        lr=args.losslr,
        lcgln_actfn=args.lcglnactfn,
        minmax=minmax,
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
    if minmax.upper()=="MIN":
        best = (float("inf"), None)
    elif minmax.upper()=="MAX":
        best = (float("-inf"), None)
    else:
        raise LookupError()
    
    for epoch in tqdm(range(args.n_iter)):
        if epoch % args.valfreq == 0:
            # Check if well trained by objective value
            datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val')]
            metrics = print_metrics(datasets, model, problem, args.loss, loss_fn, f"Iter {epoch},")           
            # Save model if it's the best one
            if minmax.upper()=="MIN":
                if best[1] is None or metrics['val']['objective'] < best[0]:
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0
            else:
                if best[1] is None or metrics['val']['objective'] > best[0]:
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0
            # Stop if model hasn't improved for patience steps
            if (args.earlystopping) and (steps_since_best > args.patience):
                break

        # Learn
        losses = []
        for i in random.sample(range(len(X_train)), min(args.predmodelbatchsize, len(X_train))):
            pred = model(X_train[i]).squeeze()
            if args.problem.upper() in ['PORTFOLIO', 'INVENTORY']:
                losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i, Xs=X_train[i]))
            elif args.problem.upper() == 'BUDGET':
                losses.append(loss_fn(pred.flatten(), Y_train[i].flatten(), aux_data=Y_train_aux[i], partition='train', index=i, Xs=X_train[i]))
        
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps_since_best += 1
        
    if args.earlystopping:
        print("Early Stopping... Saving the Model...")
        model = best[1]
        
    time_elapsed = time() - time_start
    datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
    metric_ours = print_metrics(datasets, model, problem, args.loss, loss_fn, "Final")   

    if args.problem.upper() in ['PORTFOLIO', 'BUDGET']:
        Z_test_opt = problem.get_decision(Y_test, aux_data=Y_test_aux, isTrain=False)
        objectives = problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)
        obj_opt = objectives.mean().item()
    elif args.problem.upper() in ['INVENTORY']:
        obj_opt = np.mean([problem.get_obj(y)[0] for y in Y_test.cpu().numpy()])
    print(f"Optimal Decision Quality: {obj_opt}")

    obj_ours = metric_ours['test']['objective']
    print(f"Our Decision Quality: {obj_ours}")

    pass
