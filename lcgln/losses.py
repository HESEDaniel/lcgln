import os
import pickle
import random
import time
import torch
import torch.nn as nn

from copy import deepcopy
from functools import partial
from tqdm import tqdm

from lcgln.dataloader import *
from lcgln.Networks import LCGLN, dense_nn
from utils.utils import find_saved_problem, starmap_with_kwargs
from torch.nn.functional import mse_loss

NUM_CPUS = os.cpu_count()


def MSE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).square().mean()

def MAE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).abs().mean()

def CE(Yhats, Ys, **kwargs):
    return torch.nn.BCELoss()(Yhats, Ys)

def MSE_Sum(
    Yhats,
    Ys,
    alpha=0.1,  # weight of MSE-based regularisation
    **kwargs
):
    """
    Custom loss function that the squared error of the _sum_
    along the last dimension plus some regularisation.
    Useful for the Submodular Optimisation problems in Wilder et. al.
    """
    # Check if prediction is a matrix/tensor
    assert len(Ys.shape) >= 2

    # Calculate loss
    sum_loss = (Yhats - Ys).sum(dim=-1).square().mean()
    loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(Yhats, Ys)
    return loss_regularised

def _sample_points(
    Y,  # The set of true labels
    problem,  # The optimisation problem at hand
    sampling,  # The method for sampling points
    num_samples,  # Number of points with which to fit model
    Y_aux=None,  # Extra information needed to solve the problem
    sampling_std=None,  # Standard deviation for the training data
    num_restarts=10,  # The number of times to run the optimisation problem for Z_opt
    sampling_model = None,
):
    # Sample points in the neighbourhood
    #   Find the rough scale of the predictions
    try:
        Y_std = float(sampling_std)
    except TypeError:
        Y_std = torch.std(Y) + 1e-5
    #   For sampling="random_3std"
    except ValueError:
        pass
    #   Generate points
    if sampling == 'random':
        #   Create some noise
        Y_noise = torch.distributions.Normal(0, Y_std).sample((num_samples, *Y.shape))
        #   Add this noise to Y to get sampled points
        Yhats = (Y + Y_noise)
    else:
        raise LookupError()
    #   Make sure that the points are valid predictions
    # if isinstance(problem, BudgetAllocation) or isinstance(problem, BipartiteMatching):
    #     Yhats = Yhats.clamp(min=0, max=1)  # Assuming Yhats must be in the range [0, 1]
    # elif isinstance(problem, RMAB):
    #     Yhats /= Yhats.sum(-1, keepdim=True)

    # Calculate decision-focused loss for points
    opt = partial(problem.get_decision, isTrain=False, aux_data=Y_aux)
    obj = partial(problem.get_objective, aux_data=Y_aux)

    # #   Calculate for 'true label'
    # best = None
    # assert num_restarts > 0
    # for _ in range(num_restarts):
    Z_opt = opt(Y)
    opt_objective = obj(Y, Z_opt)

    #     if best is None or opt_objective > best[1]:
    #         best = (Z_opt, opt_objective)
    # Z_opt, opt_objective = best

    #   Calculate for Yhats
    Zs = opt(Yhats, Z_init=Z_opt)
    objectives = obj(Y.unsqueeze(0).expand(*Yhats.shape), Zs)

    return (Y, opt_objective, Yhats, objectives)

def _learn_loss(
    problem,  # The problem domain
    dataset,  # The data set on which to train SL
    model_type,  # The model we're trying to fit
    num_iters=1000,  # Number of iterations over which to train model
    lr=1,  # Learning rate with which to train the model
    verbose=False,  # print training loss?
    train_frac=0.3,  # fraction of samples to use for training
    val_frac=0.3,  # fraction of samples to use for testing
    val_freq=1,  # the number of training steps after which to check loss on val set
    print_freq=5,  # the number of val steps after which to print losses
    patience=10,  # number of iterations to wait for the train loss to improve when learning
    lcgln_actfn='ELU',
    minmax='MAX',
    data_idx=None,
    batch_size=128,
    **kwargs
):
    """
    Function that learns a model to approximate the behaviour of the
    'decision-focused loss' from Wilder et. al. in the neighbourhood of Y
    """
    
    # Get samples from dataset
    Y, opt_objective, Yhats, objectives = dataset
    if minmax.upper()=="MIN":
        objectives = objectives - opt_objective
    else:
        objectives = opt_objective - objectives
    
    assert train_frac + val_frac < 1
    
    # Split train and test  
    train_idxs = range(0, int(train_frac * Yhats.shape[0]))
    val_idxs = range(int(train_frac * Yhats.shape[0]), int((train_frac + val_frac) * Yhats.shape[0]))
    test_idxs = range(int((train_frac + val_frac) * Yhats.shape[0]), Yhats.shape[0])

    Yhats_train, objectives_train, data_idx_train = Yhats[train_idxs], objectives[train_idxs], data_idx[train_idxs]
    Yhats_val, objectives_val, data_idx_val = Yhats[val_idxs], objectives[val_idxs], data_idx[val_idxs]
    Yhats_test, objectives_test, data_idx_test = Yhats[test_idxs], objectives[test_idxs], data_idx[test_idxs]

    # Load a model
    if model_type.upper() == 'LCGLN':
        half = int(Yhats.shape[-1]/2)
        model = LCGLN(x_dim=half, y_dim=half, u_dim=half, z_dim=half, act_fn=lcgln_actfn)
    else:
        raise LookupError()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Yhats_train, Yhats_val, Yhats_test = Yhats_train.to(device), Yhats_val.to(device), Yhats_test.to(device)
        objectives_train, objectives_val, objectives_test = objectives_train.to(device), objectives_val.to(device), objectives_test.to(device)
        model = model.to(device)

    # Fit a model to the points
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = (float("inf"), None)
    time_since_best = 0
    # Generate Dataloader
    train_dataloader = LCGLN_data_loader(Yhats_train, objectives_train, batch_size=batch_size)
    val_dataloader = LCGLN_data_loader(Yhats_val, objectives_val, batch_size=batch_size)

    time_since_best = 0
    for iter_idx in range(num_iters):
        train_loss_tracker = []
        val_loss_tracker = []
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X, y = batch
            X = X.to(device)
            pred = model(X[:, :half], X[:, half:])
            loss = mse_loss(pred, y)
            train_loss_tracker.append(loss.item())
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                X_val, y_val = batch
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                pred_val = model(X_val[ : , :half], X_val[ : , half: ]).flatten()
                loss_val = MSE(pred_val, y_val)
                val_loss_tracker.append(loss_val.item())
            
        if verbose and iter_idx % (val_freq * print_freq) == 0:
            print(f"Iter {iter_idx}, Train Loss MSE: {np.mean(train_loss_tracker)}")
            print(f"Iter {iter_idx}, Val Loss MSE: {np.mean(val_loss_tracker)}")
        
        if best[1] is None or np.mean(val_loss_tracker) < best[0]:
            best = (np.mean(val_loss_tracker), deepcopy(model))
            time_since_best = 0
        else:
            time_since_best += 1
            
        if time_since_best > patience:
            break

    model = best[1]
    
    return model, 0, 0


def _get_learned_loss(
    problem,
    model_type='LCGLN',
    folder='models',
    num_samples=32,
    sampling_method='mbs',
    sampling_model_num_layers=2,
    sampling_model_intermediate_size=500,
    minmax='MIN',
    sampling_lr=0,     ## added
    **kwargs
):
    datasets, SL_dataset, num_samples_needed = get_dataset_for_loss_fn(
        problem, folder, sampling_method, num_samples, sampling_lr, sampling_model_num_layers, sampling_model_intermediate_size
        )

    print("Learning Loss Functions...")

    #   Learn SL based on the sampled Yhats
    losses = {}
    for Ys, Ys_aux, partition in [datasets[0]]:
        Y_Yhats_ = [ torch.cat( (tmp[0].flatten().repeat(tmp[2].shape[0],1), tmp[2].flatten(1)), dim=1 ) for tmp in SL_dataset[partition] ]
        Y_Yhats = torch.vstack(Y_Yhats_)
        opt_objs = torch.vstack( [ tmp[1].repeat(tmp[2].shape[0],1) for tmp in SL_dataset['train'] ] )
        objs = torch.hstack( [ tmp[3] for tmp in SL_dataset['train'] ] ).unsqueeze(dim=1)
        data_idx = torch.tensor(np.repeat(range(Ys.shape[0]),SL_dataset['train'][0][3].shape[0]))
        idxs = random.sample(range(Y_Yhats.shape[0]), Ys.shape[0]*num_samples_needed)

        start_time = time.time()
        if problem.__class__.__name__ in ['PortfolioOpt', 'Newsvendor']: 
            losses[partition] = _learn_loss(problem, (Ys, opt_objs[idxs], Y_Yhats[idxs], objs[idxs]), model_type, minmax=minmax, data_idx=data_idx[idxs], **kwargs)
        elif problem.__class__.__name__=='BudgetAllocation':
            losses[partition] = _learn_loss(problem, (Ys[(np.array(idxs)%num_samples)], opt_objs[idxs], Y_Yhats[idxs], objs[idxs]), model_type, minmax=minmax, data_idx=data_idx[idxs], **kwargs)
        print(f"({partition}) Time taken to learn loss for {len(Ys)} instances: {round(time.time() - start_time, 2)} sec")
            
    # Return the loss function in the expected form
    def surrogate_decision_quality(Yhats, Ys, partition, index, Xs=None, **kwargs):
        return losses[partition][0](Ys, Yhats).flatten()
            
    return surrogate_decision_quality


def get_loss_fn(
    name,
    problem,
    **kwargs
):
    if name == 'mse':
        return MSE
    else:
        return _get_learned_loss(problem, name, **kwargs)


def get_dataset_for_loss_fn(
    problem, 
    folder='models',
    sampling_method='mbs', 
    num_samples=32, 
    sampling_lr=0, 
    sampling_model_num_layers=2, 
    sampling_model_intermediate_size=500
):
    
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    #   Get points in the neighbourhood of the Ys
    #       Try to load sampled points
    master_filename = os.path.join(folder, f"{problem.__class__.__name__}.csv")
    problem_filename, _ = find_saved_problem(master_filename, problem.__dict__)
    samples_filename_read = f"{problem_filename[:-4]}_{sampling_method}_{num_samples}_{sampling_lr}.pkl"

    # Check if there are enough stored samples
    num_samples_needed = num_extra_samples = num_samples
    if os.path.exists(samples_filename_read):
        with open(samples_filename_read, 'rb') as filehandle:
            num_existing_samples, SL_dataset_old = pickle.load(filehandle)
    else:
        num_existing_samples = 0
        SL_dataset_old = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}

    # Sample more points if needed
    num_samples_needed = num_samples
    num_extra_samples = max(num_samples_needed - num_existing_samples, 0)
    datasets = [entry for entry in zip([Y_train, Y_val], [Y_train_aux, Y_val_aux], ['train', 'val'])]
    if num_extra_samples > 0:
        SL_dataset = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}
        for Ys, Ys_aux, partition in datasets:
            #   Model-based Sampling from ...
            print(f'(smp_lr={sampling_lr}) Training mbs sampling model...')
            if partition=='train':
                Xs = X_train.clone()
            elif partition=='val':
                Xs = X_val.clone()
            else:
                raise TypeError
            ipdim, opdim = problem.get_modelio_shape()
            sampling_model = dense_nn(
                    num_features=ipdim,
                    num_targets=opdim,
                    num_layers=sampling_model_num_layers,
                    intermediate_size=sampling_model_intermediate_size,
                    output_activation=problem.get_output_activation(),
                )
            optimizer = torch.optim.Adam(sampling_model.parameters(), lr=sampling_lr)

            if problem.__class__.__name__=='PortfolioOpt':
                Yhats = Ys.clone().unsqueeze(dim=1)
            elif problem.__class__.__name__ in ['BudgetAllocation', 'Newsvendor']:
                Yhats = Ys.clone()
                
            for epoch in tqdm(range(num_extra_samples+1)):
                if problem.__class__.__name__ in ['PortfolioOpt', 'BudgetAllocation']:
                    loss = MSE(sampling_model(Xs).squeeze(), Ys)
                elif problem.__class__.__name__ in ['BudgetAllocation', 'Newsvendor']:
                    loss = nn.NLLLoss()(sampling_model(Xs).squeeze(), torch.where(Ys)[1].detach())
                if epoch>=2:
                    if problem.__class__.__name__=='PortfolioOpt':
                        Yhats = torch.cat((Yhats, sampling_model(Xs).transpose(1,2)), dim=1)
                    elif problem.__class__.__name__ in ['BudgetAllocation', 'Newsvendor']:
                        Yhats = torch.cat((Yhats, sampling_model(Xs)), dim=0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            opt = partial(problem.get_decision, isTrain=False, aux_data=Ys_aux)
            obj = partial(problem.get_objective, aux_data=Ys_aux)
            
            print('Solving opt problem for samples to generate dataset for training LCGLN...')
            if problem.__class__.__name__=='PortfolioOpt':
                Z_opt = opt(Ys)
                opt_objective = obj(Ys, Z_opt)
                for i, yhat in tqdm(enumerate(Yhats.transpose(0,1))):
                    if not i:
                        tmp_Zs = opt(yhat, Z_init=Z_opt)
                        Zs = tmp_Zs.unsqueeze(dim=1)
                        objectives = obj(Ys, tmp_Zs).unsqueeze(dim=1)
                    else:
                        tmp_Zs = opt(yhat, Z_init=Z_opt)
                        Zs = torch.cat((Zs, tmp_Zs.unsqueeze(dim=1)), dim=1)
                        objectives = torch.cat((objectives, obj(Ys, tmp_Zs).unsqueeze(dim=1)), dim=1)
                sampled_points = []
                for i in range(Ys.shape[0]):
                    sampled_points.append((Ys[i].detach(), opt_objective[i].detach(), Yhats[i].detach(), objectives[i].detach()))
            
            elif problem.__class__.__name__=='BudgetAllocation':
                for i in tqdm(range(Ys.shape[0])):
                    Z = opt(Ys[i])
                    objective = obj(Ys[i], Z).unsqueeze(dim=0)
                    if i == 0:
                        opt_objective = objective
                    else:
                        opt_objective = torch.cat([opt_objective, objective], dim=0)
                opt_objective = torch.unsqueeze(opt_objective,dim=1)

                for i, yhat in tqdm(enumerate(Yhats)):
                    if not i:
                        tmp_Zs = opt(yhat)
                        objectives = obj(Ys[i%Ys.shape[0]], tmp_Zs).unsqueeze(dim=0).unsqueeze(dim=0)
                    else:
                        tmp_Zs = opt(yhat)
                        objectives = torch.cat((objectives, obj(Ys[i%Ys.shape[0]], tmp_Zs).unsqueeze(dim=0).unsqueeze(dim=0)), dim=0)

                sampled_points = []
                for i in range(Ys.shape[0]):
                    sampled_points.append((Ys[i].detach(), opt_objective[i].squeeze().detach(), Yhats[torch.tensor(range(Yhats.shape[0]))%Ys.shape[0] == i].detach(), objectives[torch.tensor(range(Yhats.shape[0]))%Ys.shape[0] == i, 0].detach()))
            
            elif problem.__class__.__name__=='Newsvendor':
                get_obj = partial(problem.get_obj)
                for i in range(Ys.shape[0]):
                    Z = get_obj(Ys[i])[1]
                    objective = obj(Ys[i], Z, sample =True).unsqueeze(dim=0)
                    if i == 0:
                        opt_objective = objective
                    else:
                        opt_objective = torch.cat([opt_objective, objective], dim=0)

                for i, yhat in enumerate(Yhats):
                    if not i:
                        tmp_Zs = get_obj(yhat.detach())[1]
                        objectives = obj(Ys[i%Ys.shape[0]], tmp_Zs, sample = True).unsqueeze(dim=0)
                    else:
                        tmp_Zs = get_obj(yhat.detach())[1]
                        objectives = torch.cat((objectives, obj(Ys[i%Ys.shape[0]], tmp_Zs, sample = True).unsqueeze(dim=0)), dim=0)

                sampled_points = []
                for i in range(Ys.shape[0]):
                    sampled_points.append((Ys[i].detach(), opt_objective[i].squeeze().detach(), Yhats[torch.tensor(range(Yhats.shape[0]))%Ys.shape[0] == i].detach(), objectives[torch.tensor(range(Yhats.shape[0]))%Ys.shape[0] == i, 0].detach()))

            # Use them to augment existing sampled points
            for idx, (Y, opt_objective, Yhats, objectives) in enumerate(sampled_points):
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)

        #   Augment with new data
        for Ys, Ys_aux, partition in datasets:
            for idx, Y in enumerate(Ys):
                # Get old samples
                Y_old, opt_objective_old, Yhats_old, objectives_old = SL_dataset_old[partition][idx]
                Y_new, opt_objective_new, Yhats_new, objectives_new = SL_dataset[partition][idx]

                # Combine entries
                opt_objective = opt_objective_new if opt_objective_old is None else max(opt_objective_new, opt_objective_old)
                Yhats = Yhats_new if Yhats_old is None else torch.cat((Yhats_old, Yhats_new), dim=0)
                objectives = objectives_new if objectives_old is None else torch.cat((objectives_old, objectives_new), dim=0)

                # Update
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)
        num_existing_samples += num_extra_samples
        
        # Save dataset
        samples_filename_write = f"{problem_filename[:-4]}_{sampling_method}_{num_samples}_{sampling_lr}.pkl"
        with open(samples_filename_write, 'wb') as filehandle:
            pickle.dump((num_existing_samples, SL_dataset), filehandle)
            print(f'Saved in {samples_filename_write}')
    else:
        print("Loading from Saved Sample Data...")
        SL_dataset = SL_dataset_old
        
    return datasets, SL_dataset, num_samples_needed