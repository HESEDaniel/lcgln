import pickle
import torch
from copy import deepcopy
import os
import random
import time
from tqdm import tqdm
from functools import partial
from functools import reduce
from torch.nn.functional import mse_loss
from constants import *
from Dataloader import *
import numpy as np
from Networks import GlobalICLN, EGLWeightedMSE, EGLDirectedQuadratic, Quadratic
import warnings
warnings.filterwarnings("ignore")
from utils import find_saved_problem, dense_nn
import torch.nn as nn

def MSE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).square().mean()
    


def _learn_loss(
    problem,  # The problem domain
    dataset,  # The data set on which to train SL
    model_type, # The model we're trying to fit
    num_iters=2000,  # Number of iterations over which to train model
    lr=0.01,  # Learning rate with which to train the model
    train_frac=0.6,  # fraction of samples to use for training
    val_frac=0.2,  # fraction of samples to use for testing
    val_freq=1,
    verbose=False,
    print_freq=5,
    patience=10,  # number of iterations to wait for the train loss to improve when learning
    icln_actfn='ELU',
    data_idx=None,
    **kwargs
):

    """
    Function that learns a model to approximate the behaviour of the
    'decision-focused loss' from Wilder et. al. in the neighbourhood of Y
    """
    # Get samples from dataset
    Y, opt_objective, Yhats, objectives = dataset
    objectives =  objectives - opt_objective

    # Split train and test  
    train_idxs = range(0, int(train_frac * Yhats.shape[0]))
    val_idxs = range(int(train_frac * Yhats.shape[0]), int((train_frac + val_frac) * Yhats.shape[0]))
    test_idxs = range(int((train_frac + val_frac) * Yhats.shape[0]), Yhats.shape[0])

    if model_type.upper()=="GICLN" or model_type.upper() == 'EGLWMSE' or model_type.upper() == 'EGLDQ': 
        Yhats_train, objectives_train, data_idx_train = Yhats[train_idxs], objectives[train_idxs], data_idx[train_idxs]
        Yhats_val, objectives_val, data_idx_val = Yhats[val_idxs], objectives[val_idxs], data_idx[val_idxs]
        Yhats_test, objectives_test, data_idx_test = Yhats[test_idxs], objectives[test_idxs], data_idx[test_idxs]
    else:
        Yhats_train, objectives_train = Yhats[train_idxs], objectives[train_idxs]
        Yhats_val, objectives_val = Yhats[val_idxs], objectives[val_idxs]
        Yhats_test, objectives_test = Yhats[test_idxs], objectives[test_idxs]

    # Load a model
    if model_type == 'dense':
        model = dense_nn(
        num_features=Yhats.shape[-1],
        num_targets=1,
        num_layers=4,
        output_activation='relu'
            )
    elif model_type.upper() =='QUAD':
        model = Quadratic(Y, **kwargs)
    elif model_type.upper() == 'GICLN':
        tmp = int(Yhats.shape[-1]/2)
        model = GlobalICLN(x_dim=tmp, y_dim=tmp, u_dim=tmp, z_dim=tmp, act_fn=icln_actfn)
    elif model_type.upper() == 'EGLWMSE':
        model = EGLWeightedMSE(problem.get_train_data()[0], Y, **kwargs)
    elif model_type.upper() =='EGLDQ':
        model = EGLDirectedQuadratic(problem.get_train_data()[0], Y, **kwargs)
    else:
        raise LookupError()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Yhats_train, Yhats_val, Yhats_test = Yhats_train.to(device), Yhats_val.to(device), Yhats_test.to(device)
        objectives_train, objectives_val, objectives_test = objectives_train.to(device), objectives_val.to(device), objectives_test.to(device)
        model = model.to(device)

    # Use GPU if available
    if DEVICE == 'cuda':
        Yhats_train, Yhats_val, Yhats_test = Yhats_train.to(DEVICE), Yhats_val.to(DEVICE), Yhats_test.to(DEVICE)
        objectives_train, objectives_val, objectives_test = objectives_train.to(DEVICE), objectives_val.to(DEVICE), objectives_test.to(DEVICE)
        model = model.to(DEVICE)

    # Fit a model to the points
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = (float("inf"), None)
    time_since_best = 0
    if model_type.upper() == 'ICNN':
        # Generate Dataloader
        batchsize=128
        train_dataloader = ICNN_data_loader(Yhats_train, objectives_train, batch_size=batchsize)
        for _ in range(num_iters):
            # Get performance on val dataset
            pred_val = model(Yhats_val.float())
            loss_val = MSE(pred_val, objectives_val)
            # Save model if it's the best one
            if best[1] is None or loss_val.item() < best[0]:
                best = (loss_val.item(), deepcopy(model))
                time_since_best = 0
            # Stop if model hasn't improved for patience steps
            if time_since_best > patience:
                break
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                X, y = batch
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(X.float())
                loss = mse_loss(pred, y.float())
                # print(round(loss.item(), 4))
                loss.backward()
                optimizer.step()
            time_since_best += 1
    elif model_type.upper() == 'GICLN':
        # Generate Dataloader
        batchsize = 128
        train_dataloader = ICNN_data_loader(Yhats_train, objectives_train, batch_size=batchsize)
        for _ in range(num_iters):
            # Get performance on val dataset
            pred_val = model(Yhats_val[:,:5].float(),Yhats_val[:,5:].float())
            loss_val = MSE(pred_val, objectives_val)
            # Save model if it's the best one
            if best[1] is None or loss_val.item() < best[0]:
                best = (loss_val.item(), deepcopy(model))
                time_since_best = 0
            # Stop if model hasn't improved for patience steps
            if time_since_best > patience:
                break
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                X, y = batch
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(X[:,:5].float(), X[:,5:].float())
                loss = mse_loss(pred.to(torch.float32), y.to(torch.float32))
                # print(round(loss.item(), 4))
                loss.backward()
                optimizer.step()
            time_since_best += 1
    else:
        for iter_idx in range(num_iters):
            # Define update step using "closure" function
            def loss_closure():
                optimizer.zero_grad()
                if model_type.upper()=='EGLWMSE' or model_type.upper()=='EGLDQ':
                    pred = model(data_idx_train, Yhats_train).flatten()
                else:
                    pred = model(Yhats_train.float()).flatten()

                loss = MSE(pred, objectives_train)
                loss.backward()
                # print(round(loss.item(), 4))
                return loss

            # Perform validation
            if iter_idx % val_freq == 0:
                # Get performance on val dataset
                if model_type.upper()=='EGLWMSE' or model_type.upper()=='EGLDQ':
                    pred_val = model(data_idx_val, Yhats_val).flatten()

                else:
                    pred_val = model(Yhats_val.float()).flatten()
                loss_val = MSE(pred_val, objectives_val)

                # Print statistics
                if verbose and iter_idx % (val_freq * print_freq) == 0:
                    print(f"Iter {iter_idx}, Train Loss MSE: {loss_closure().item()}")
                    print(f"Iter {iter_idx}, Val Loss MSE: {loss_val.item()}")
                # Save model if it's the best one
                if best[1] is None or loss_val.item() < best[0]:
                    best = (loss_val.item(), model)
                    time_since_best = 0
                # Stop if model hasn't improved for patience steps
                if time_since_best > patience:
                    break

            # Make an update step
            optimizer.step(loss_closure)
            time_since_best += 1        
    model = best[1]


    return model, 0, 0

def _divide_data(problem, p_theta, i):
    Yhats = p_theta.to(DEVICE)
    Y = torch.eye(5)[i].unsqueeze(dim=0)
    opt_objective = torch.tensor(problem.get_obj(Y)).to(DEVICE)
    Zs = problem.opt(Yhats)
    objectives = problem.get_objective(Y, Zs, sample = True).to(DEVICE)
    SL_dataset= (Y.to(DEVICE), opt_objective, Yhats, objectives)
    return SL_dataset

def _get_learned_loss(
    problem,
    model_type='weightedmse',
    folder='models',
    num_samples=10000,
    sampling='random',
    lossbatchsize=32,
    save=False,
    saved=False,
    sampling_lr = 0,
    **kwargs
):
    if sampling == 'random_newsvendor':
        if saved == True:
            with open('Glosses_50000', 'rb') as f:
                losses = pickle.load(f)
            if model_type.upper() == 'GICLN':
                def surrogate_decision_quality(Yhats, Ys, **kwargs):
                    Yhats = torch.hstack((Ys,Yhats))
                    return losses(Yhats[:5].to(torch.float32),Yhats[5:].to(torch.float32)).flatten() 
                return surrogate_decision_quality
            else:
                def surrogate_decision_quality(Yhats, Ys, **kwargs):
                    return losses[(Ys @ torch.tensor([0,1,2,3,4]).double().to(DEVICE)).long()](Yhats[:,:5],Yhats[:,5:]).flatten() 
                return surrogate_decision_quality
        else:
            train_maes, test_maes, opt_obj = [], [], []
            sample_data = torch.rand(num_samples,5)
            p_theta = sample_data / (sample_data.sum(axis = 1).unsqueeze(dim=1))
            exp_p_theta = p_theta @ problem.params['d']
            if model_type.upper() == 'GICLN':
                for i in range(5):
                    values, data_indices = torch.topk((exp_p_theta - problem.params['d'][i]).abs(),int(num_samples/25),largest = False)
                    Y_dataset, opt_objective, Yhats, objectives = _divide_data(problem, p_theta[data_indices], i)
                    if i == 0:
                        Y_dataset_repeat = Y_dataset.repeat(Yhats.shape[0],1)
                        opt_objective_repeat = opt_objective.repeat(objectives.shape[0],1)
                        Yhats_repeat = Yhats
                        objectives_repeat = objectives
                    else:
                        Y_dataset_repeat = torch.vstack((Y_dataset_repeat,Y_dataset.repeat(Yhats.shape[0],1)))
                        opt_objective_repeat = torch.vstack((opt_objective_repeat,opt_objective.repeat(objectives.shape[0],1)))
                        Yhats_repeat = torch.vstack((Yhats_repeat,Yhats))
                        objectives_repeat = torch.vstack((objectives_repeat, objectives))
                indices = torch.randperm(Y_dataset_repeat.shape[0]).to(DEVICE)
                Y_dataset_repeat = torch.index_select(Y_dataset_repeat, dim=0, index=indices)
                opt_objective_repeat = torch.index_select(opt_objective_repeat, dim=0, index=indices)
                Yhats_repeat = torch.index_select(Yhats_repeat, dim=0, index=indices)
                objectives_repeat = torch.index_select(objectives_repeat, dim=0, index=indices)
                losses_and_stats = _learn_loss(problem, (Y_dataset_repeat, opt_objective_repeat, torch.hstack((Y_dataset_repeat, Yhats_repeat)), objectives_repeat), model_type,lossbatchsize, **kwargs)
                learned_loss, train_mae, test_mae = losses_and_stats
                losses = learned_loss
                if save == True:
                    with open('Glosses_10000_1', 'wb') as f:
                        pickle.dump(losses, f, pickle.HIGHEST_PROTOCOL)
                def surrogate_decision_quality(Yhats, Ys, **kwargs):
                    Yhats = torch.hstack((Ys,Yhats))
                    return losses(Yhats[:5].to(torch.float32),Yhats[5:].to(torch.float32)).flatten()
                return surrogate_decision_quality
            else:
                losses = []
                for i in range(5):
                    values, data_indices = torch.topk((exp_p_theta - problem.params['d'][i]).abs(),int(num_samples/25),largest = False)
                    Y_dataset, opt_objective, Yhats, objectives = _divide_data(problem, p_theta[data_indices], i)
                    indices = torch.randperm(Yhats.shape[0]).to(DEVICE)
                    Yhats = torch.index_select(Yhats, dim=0, index=indices)
                    objectives = torch.index_select(objectives, dim=0, index=indices)
                    losses_and_stats = _learn_loss(problem, (Y_dataset.T, opt_objective, Yhats, objectives), model_type,lossbatchsize, **kwargs)
                    # Parse and log results
                    learned_loss, train_mae, test_mae = losses_and_stats
                    train_maes.append(train_mae)
                    test_maes.append(test_mae)
                    opt_obj.append(opt_objective)
                    losses.append(learned_loss)
                if save == True:
                    with open('Glosses_1', 'wb') as f:
                        pickle.dump(losses, f, pickle.HIGHEST_PROTOCOL)
                def surrogate_decision_quality(Yhats, Ys, **kwargs):
                    return losses[(Ys @ torch.tensor([0,1,2,3,4]).double().to(DEVICE)).long()](Yhats).flatten()
                return surrogate_decision_quality
    else:
        # Learn Losses
    #   Get Ys
        X_train, Y_train, Y_train_aux = problem.get_train_data()
        X_val, Y_val, Y_val_aux = problem.get_val_data()

        #   Get points in the neighbourhood of the Ys
        #       Try to load sampled points
        master_filename = os.path.join(folder, f"{problem.__class__.__name__}.csv")
        problem_filename, _ = find_saved_problem(master_filename, problem.__dict__)
        if sampling == 'mbs':
            samples_filename_read = f"{problem_filename[:-4]}_{sampling}_{sampling_lr}_{num_samples}.pkl"
        else:
            samples_filename_read = f"{problem_filename[:-4]}_{sampling}.pkl"

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
                if sampling=='mbs':
                    print(f'(smp_lr={sampling_lr}) Training dense model for mbs inside losses.py...')
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
                            num_layers=2,
                            intermediate_size=10,
                            output_activation=problem.get_output_activation(),
                        )
                    optimizer = torch.optim.Adam(sampling_model.parameters(), lr=sampling_lr)
            
                    Yhats = Ys.clone()
                    for epoch in range(num_extra_samples+1):
                        loss = nn.NLLLoss()(sampling_model(Xs).squeeze(), torch.where(Ys)[1].detach())
                        sampling_freq = max(1, int(300/num_extra_samples))

                        if epoch>=2:
                            print(loss)

                            Yhats = torch.cat((Yhats, sampling_model(Xs)), dim=0)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    opt = partial(problem.get_decision, isTrain=False, aux_data=Ys_aux)
                    obj = partial(problem.get_objective, aux_data=Ys_aux)
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

            for Ys, Ys_aux, partition in datasets:
                for idx, Y in enumerate(Ys):
                    # Get old samples
                    Y_old, opt_objective_old, Yhats_old, objectives_old = SL_dataset_old[partition][idx]
                    Y_new, opt_objective_new, Yhats_new, objectives_new = SL_dataset[partition][idx]
                    assert torch.isclose(Y_old, Y).all()
                    assert torch.isclose(Y_new, Y).all()

                    # Combine entries
                    opt_objective = opt_objective_new if opt_objective_old is None else max(opt_objective_new, opt_objective_old)
                    Yhats = Yhats_new if Yhats_old is None else torch.cat((Yhats_old, Yhats_new), dim=0)
                    objectives = objectives_new if objectives_old is None else torch.cat((objectives_old, objectives_new), dim=0)

                    # Update
                    SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)
            num_existing_samples += num_extra_samples

                    # Save dataset
            samples_filename_write = f"{problem_filename[:-4]}_{sampling}_{sampling_lr}_{num_samples}.pkl"
            with open(samples_filename_write, 'wb') as filehandle:
                pickle.dump((num_existing_samples, SL_dataset), filehandle)
        else:
            print("Loading from Saved Sample Data...")
            SL_dataset = SL_dataset_old

    print("Learning Loss Functions...")

    #   Learn SL based on the sampled Yhats
    train_maes, test_maes, avg_dls = [], [], []
    losses = {}
    if model_type.upper()=='GICLN' or model_type.upper()=='EGLWMSE' or model_type.upper()=='EGLDQ' or model_type.upper()=='DENSE':
        for Ys, Ys_aux, partition in [datasets[0]]:
            
            Y_Yhats_ = [ torch.cat( (tmp[0].flatten().repeat(tmp[2].shape[0],1), tmp[2].flatten(1)), dim=1 ) for tmp in SL_dataset[partition] ]
            Y_Yhats = torch.vstack(Y_Yhats_)
            opt_objs = torch.vstack( [ tmp[1].repeat(tmp[2].shape[0],1) for tmp in SL_dataset['train'] ] )
            objs = torch.hstack( [ tmp[3] for tmp in SL_dataset['train'] ] ).unsqueeze(dim=1)
            data_idx = torch.tensor(np.repeat(range(Ys.shape[0]),SL_dataset['train'][0][3].shape[0]))

            idxs = random.sample(range(Y_Yhats.shape[0]), Ys.shape[0]*num_samples_needed)

            
            start_time = time.time()

            losses[partition] = _learn_loss(problem, (Ys, opt_objs[idxs], Y_Yhats[idxs], objs[idxs]), model_type, data_idx=data_idx[idxs], **kwargs)

            print(f"({partition}) Time taken to learn loss for {len(Ys)} instances: {round(time.time() - start_time, 2)} sec")
    else:
        for Ys, Ys_aux, partition in [datasets[0]]:

            for idx, (Y, Y_aux) in enumerate(zip(Ys, Ys_aux)):
                Y_dataset, opt_objective, _, objectives = SL_dataset[partition][idx]

                avg_dls.append((opt_objective - objectives).abs().mean().item())

            # Get num_samples_needed points
            idxs = random.sample(range(num_existing_samples), num_samples_needed)


            # Learn a loss
            start_time = time.time()
            losses_and_stats = [_learn_loss(problem, (Y_dataset, opt_objective, Yhats[idxs], objectives[idxs]), model_type, **kwargs) for Y_dataset, opt_objective, Yhats, objectives in tqdm(SL_dataset[partition])]

            print(f"({partition}) Time taken to learn loss for {len(Ys)} instances: {round(time.time() - start_time, 2)} sec")

            # Parse and log results
            losses[partition] = []
            for learned_loss, train_mae, test_mae in losses_and_stats:
                train_maes.append(train_mae)
                test_maes.append(test_mae)
                losses[partition].append(learned_loss)
            
    # Return the loss function in the expected form
    def surrogate_decision_quality(Yhats, Ys, partition, index, **kwargs):
        if model_type.upper() == 'GICLN':
            return losses[partition][0](Ys, Yhats).flatten()
        elif model_type.upper() == 'DENSE':
            return losses[partition][0](torch.cat([Ys.float(), Yhats.float()],dim=0)).flatten()
        elif (model_type.upper() == 'ICLN') or (model_type.upper() == 'ICLN++'):
            return losses[partition][index](Yhats).flatten()
        elif model_type.upper() == 'EGLWMSE' or model_type.upper() == 'EGLDQ':
            return losses[partition][0](index, Yhats).flatten()
        else:
            return losses[partition][index](Yhats).flatten() - SL_dataset[partition][index][1]
    return surrogate_decision_quality


def get_loss_fn(
    name,
    problem,
    num_samples,
    samplinglr=0,
    **kwargs
):
    if name == 'mse':
        return MSE
    else:
        return  _get_learned_loss(problem, name, num_samples = num_samples, **kwargs)

