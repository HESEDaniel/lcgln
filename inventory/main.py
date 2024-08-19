import argparse
import ast
import torch

import datetime as dt

from copy import deepcopy
from functools import partial
from tqdm import tqdm

from constants import *
from losses import get_loss_fn, dense_nn
import os

from Newsvendor import Newsvendor
from Dataloader import *
from utils import print_metrics, init_if_not_saved, move_to_gpu
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['Newsvendor'], default='Newsvendor')
    parser.add_argument('--loadnew', type=ast.literal_eval, default=True)
    parser.add_argument('--instances', type=int, default=2000)
    parser.add_argument('--testinstances', type=int, default=1000)
    parser.add_argument('--valfrac', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--valfreq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=128)

    #   Learning Losses
    parser.add_argument('--loss', type=str, choices=['dense', 'GICLN', 'mse', 'dfl', 'quad', 'eglwmse', 'egldq'], default='egldq')
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--sampling', type=str, choices=['random', 'mbs', 'random_newsvendor'], default='mbs')
    parser.add_argument('--numsamples', type=int, default=16)
    parser.add_argument('--losslr', type=float, default=0.001)
    parser.add_argument('--lossbatchsize', type=int, default=256)
    parser.add_argument('--samplinglr', type=float, default=0.05)
    args = parser.parse_args()
    

    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")
    
    save_folder = os.path.join('results', str(args.loss), str(args.numsamples))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    now = dt.datetime.now()
    

    results_file = os.path.join(save_folder, f"newsvendor_results.csv")

    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
    if args.problem == 'Newsvendor':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'rand_seed': args.seed,}
        problem = init_problem(Newsvendor, problem_kwargs)
    
    
    print(f"Loading {args.loss} Loss Function...")
    loss_fn = get_loss_fn(
        args.loss,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        lr=args.losslr,
        lossbatchsize=args.lossbatchsize,
        sampling_lr=args.samplinglr
    )
        
    # Load an ML model to predict the parameters of the problem
    print(f"Building dense Model...")
    ipdim, opdim = problem.get_modelio_shape()
    model = dense_nn(
        num_features=ipdim,
        num_targets=opdim,
        num_layers=args.layers,
        intermediate_size=10,
        output_activation=problem.get_output_activation(),
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  

    
    # Move everything to GPU, if available
    if DEVICE == 'cuda':
        move_to_gpu(problem)
        model = model.to(DEVICE)

    # Get data [day,stock,feature]
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()


    #########################################################################
    print('Ours Training Method...')
    best = (float("inf"), None)
    steps_since_best = 0
    for epoch in tqdm(range(1000)):
        if epoch % args.valfreq == 0:
            # Check if well trained by objective value
            datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val')]
            # metrics = metrics2wandb(datasets, model, problem, f"Iter {epoch}")
            metrics = print_metrics(datasets, model, problem, args.loss, loss_fn, f"Iter {epoch},")           
            # Save model if it's the best one
            if best[1] is None or metrics['val']['objective'] < best[0]:
                best = (metrics['val']['objective'], deepcopy(model))
                steps_since_best = 0
            # Stop if model hasn't improved for patience steps
            if args.earlystopping and steps_since_best > args.patience:
                break

        
        #################### TEST ####################
        # Learn
        losses = []

        def news_loss_fn(pred, y, loss_fn):
            news_losses =[]
            for i in range(y.shape[0]):
                news_losses.append(loss_fn(pred[i], y[i], partition = 'train', index = 0))
            return torch.stack(news_losses).mean() 
                
        train_dataloader = ICNN_data_loader(X_train, Y_train, batch_size=args.batchsize)
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X).squeeze()
            loss = news_loss_fn(pred, y, loss_fn)
            # print(round(loss.item(), 4))
            loss.backward()
            optimizer.step()
        steps_since_best += 1
        
    if args.earlystopping:
        print("Early Stopping... Saving the Model...")
        model = best[1]
        
    # Document how well this trained model does
    print("\nBenchmarking Model...")
    # Print final metrics
    datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
    metric_ours = print_metrics(datasets, model, problem, args.loss, loss_fn, "Final")
    dq_train = metric_ours['test']['objective']
    with open(results_file, 'a') as f:
        f.write('{},{}\n'.format('DQ:', dq_train))
        

    #########################################################################
    pass
