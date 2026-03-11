""" Training script to train the cryolithe models using real data and corresponding reconstruction obtained from 
self-supervised methods.
"""


"""
This script is used to train the model using the real data and projection pairs 
"""
import os
import timeit
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from ml_collections import config_dict as cd
import matplotlib.pyplot as plt

from .models import get_model

from .filter_models import get_filter_model
from .datasets.real_volumes import RealVolumes
from .trainers import TrainerReal,TrainerRealVolume, TrainerRealWavelet


import wandb


def train_model_real(configs, path, load_checkpoint = False, seed = 0, device ='cpu'):


    DEVICE = device



    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="vinith-unibas",
        # Set the wandb project where this run will be logged.
        project="cryolithe-icecream",
        # Track hyperparameters and run metadata.
        config= configs.to_dict()
    )

    PATH = path
    LOAD_CHECKPOINT = load_checkpoint
    print(PATH)
    #LOAD_CHECKPOINT = False
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print(f"Folder '{PATH}' created.")
    else:
        print(f"Folder '{PATH}' already exists.")


    if not os.path.exists(PATH+ 'Train/'):
        os.makedirs(PATH+ 'Train/')
        print(f"Folder '{PATH+ 'Train/'}' created.")
    else:
        print(f"Folder '{PATH+ 'Train/'}' already exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        torch.cuda.set_device(DEVICE)
        device = torch.device("cuda:"+str(DEVICE))
    print(device)
    # Load the configs from the folder if loading from check point
    if LOAD_CHECKPOINT:
        print("Loading configs")
        configs = cd.ConfigDict(json.load(open(PATH + 'config.json')))
        print("Epochs:"+str(configs.training.num_epochs))
    # loading the data
    if hasattr(configs.training,'ramp_lr') is False:
        print('Adding type as ramp_lr')
        configs.training.ramp_lr = None
    if hasattr(configs.training,'filter_2D_lr') is False:
        print('Adding type as ramp_lr')
        configs.training.filter_2D_lr = None


    if hasattr(configs.training,'ramp_weight_decay') is False:
        print('Adding type as ramp_weight_decay')
        configs.training.ramp_weight_decay = configs.training.weight_decay

        
    if hasattr(configs.training,'discrete_sampling') is False:
        print('Adding type as discrete_sampling')
        configs.training.discrete_sampling = False
    if hasattr(configs.training,'get_projection_prefiltered') is False:
        print('Adding type as get_projection_prefiltered')
        configs.training.get_projection_prefiltered = False
    if hasattr(configs.training,'scale_data') is False:
        print('Adding type as get_projection_prefiltered')
        configs.training.scale_data = False
    if hasattr(configs, 'deform_volume') is False:
        print('Adding deform_volume as False')
        configs.deform_volume = False
    if hasattr(configs.training, 'use_volumetric_trainer') is False:
        print('Adding use_volumetric_trainer as False')
        configs.training.use_volumetric_trainer = False
    if hasattr(configs.training, 'use_wavelet_trainer') is False:
        print('Adding use_wavelet_trainer as False')
        configs.training.use_wavelet_trainer = False
    if hasattr(configs.training , 'reduction') is False:
        print('Adding reduction as mean')
        configs.training.reduction = 'mean'


    # Model parameters
    if hasattr(configs.model,'type') is False:
        print('Adding attribute type as mlp')
        configs.model.type = 'mlp'
    N_PROJECTIONS = configs.data.n_projections
    model_input_projections = N_PROJECTIONS
    if configs.training.get_projection_prefiltered:
        model_input_projections = 2*N_PROJECTIONS

    print("using model:"+configs.model.type )
    model = get_model(n_projections = model_input_projections, **configs.model).to(DEVICE)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"---> Number of trainable parameters in volume net: {num_param}")


        
    if configs.data.type == 't20':
        train_dataset = RealVolumes(**configs.train_dataset)
        valid_dataset = RealVolumes(**configs.valid_dataset)

    # TODO: there is an inconsisitency in loading data if preload is true look at training loop
    # if configs.deform_volume:
    #     # vol_deformer = VolumeDeformer(**configs.volume_deformer)
    # else:
    vol_deformer = None
    #projection_simulator = ProjectionSimulator(**configs.data)


    # generate validation data
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


    # valid_data = next(iter(valid_vol_loader))
    # print('Generating Validation Data')
    # valid_data = projection_simulator.simulate_batch(valid_data, 
    #                                                 vol_deformer= vol_deformer,
    #                                                 downsample = configs.training.downsample,
    #                                                 downsample_factors = configs.training.downsample_factor,
    #                                                 scale_data= configs.training.scale_data,
    #                                                 device= DEVICE)
    # valid_loader = DataLoader(valid_data, batch_size=configs.training.batch_size, shuffle=False)



    train_list = []
    train_list.append({'params':model.parameters()})
    # TODO : May be using a learnable polynomial ramp to account for different resolutions
    if configs.filter_projections:
        if configs.ramp.use_pretrained:
            checkpoint = torch.load(configs.ramp.pretrain_path + '/checkpoint.pth',map_location=torch.device(device))
            ramp = checkpoint['ramp']
        else:
            ramp  = get_filter_model(configs.ramp.type,**configs.ramp.model).to(DEVICE)
            if configs.ramp.use_learnable_ramp:
                if configs.training.ramp_lr is None:
                    train_list.append({'params':ramp.parameters(),'weight_decay':configs.training.ramp_weight_decay })
                else:
                    train_list.append({'params':ramp.parameters(), 'lr': configs.training.ramp_lr, 'weight_decay':configs.training.ramp_weight_decay})
    else:
        ramp = None


    if configs.use_2D_filters:
        filter_2D =  get_filter_model(configs.filter_2D.type,**configs.filter_2D.model).to(DEVICE)
        if configs.training.filter_2D_lr is None:
            train_list.append({'params':filter_2D.parameters()})
        else:
            train_list.append({'params':filter_2D.parameters(), 'lr': configs.training.filter_2D_lr})
    else: 
        filter_2D = None


    patch_scale = torch.FloatTensor([configs.training.patch_scale_init]).to(DEVICE).clone()
    if configs.training.learn_patch_scale:
        patch_scale = Variable(patch_scale,requires_grad=True)
        train_list.append({'params':patch_scale})

    # Training parameters

    if configs.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(train_list, lr=configs.training.lr,
                                    weight_decay=configs.training.weight_decay)
    elif configs.training.optimizer == 'sgd':
        optimizer = torch.optim.SGD(train_list, lr=configs.training.lr,
                                    weight_decay=configs.training.weight_decay)
    elif configs.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(train_list, lr=configs.training.lr,
                                    weight_decay=configs.training.weight_decay)
        

    if hasattr(configs.training, 'lr_scheduler_type') and configs.training.lr_scheduler_type == 'linear_warmup_cosine_annealing':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=configs.training.lr_warmup_epochs,
                                                  max_epochs=configs.training.num_epochs,
                                                  eta_min=configs.training.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=configs.training.lr_decay_epochs,
                                                        gamma=configs.training.lr_decay)
    if configs.training.loss == 'MSE':
        criteria = torch.nn.MSELoss(reduction = configs.training.reduction)
    elif configs.training.loss == 'l1':
        criteria = torch.nn.L1Loss(reduction = configs.training.reduction)
    elif configs.training.loss == 'huber':
        criteria = torch.nn.SmoothL1Loss(reduction = configs.training.reduction)
    else:
        raise NotImplementedError(f"Loss {configs.training.loss} not implemented")
    REPEATS = configs.training.repeat
    EPOCHS = configs.training.num_epochs
    NORM_ORD = configs.training.patch_ord
    start = 0

    wandb.watch(model, criterion=criteria, log="all", log_freq=10)

    # Check what trainer class to use
    if configs.training.use_volumetric_trainer:
        trainer = TrainerRealVolume(config = configs, 
                    model = model, 
                    ramp = ramp,
                    patch_scale = patch_scale,
                    filter_2D = filter_2D,
                    optimizer = optimizer, 
                    criterion = criteria, 
                    scheduler = scheduler, 
                    device = DEVICE,
                    )
    elif configs.training.use_wavelet_trainer:
        trainer = TrainerRealWavelet(config = configs, 
                    model = model, 
                    ramp = ramp,
                    patch_scale = patch_scale,
                    filter_2D = filter_2D,
                    optimizer = optimizer, 
                    criterion = criteria, 
                    scheduler = scheduler, 
                    device = DEVICE,
        )
    else:
        trainer = TrainerReal(config = configs, 
                        model = model, 
                        ramp = ramp,
                        patch_scale = patch_scale,
                        filter_2D = filter_2D,
                        optimizer = optimizer, 
                        criterion = criteria, 
                        scheduler = scheduler, 
                        device = DEVICE,
                        )


    # Load checkpoint
    if LOAD_CHECKPOINT:
        start = trainer.load_checkpoint(PATH+'checkpoint.pth')
        valid_loss = trainer.valid_loss
        valid_loss_epoch = trainer.valid_loss[-1]
        #start = checkpoint['epoch']+1


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                pin_memory=True, num_workers=4)
    print('Starting training')
    for epc in range(start,EPOCHS):
        start_epoch = timeit.default_timer()
        trainer.train_step(train_loader, device = DEVICE, wandb_run = run)
        train_loss = trainer.train_loss
        train_loss_epoch = train_loss[-1]
        stop = timeit.default_timer()


        if (epc%configs.training.save_every == 0):
            trainer.validate(valid_loader, device=DEVICE, wandb_run = run)
            
            valid_loss = trainer.valid_loss
            valid_loss_epoch = trainer.valid_loss[-1]
            
            plt.figure(1)
            plt.clf()
            plt.semilogy(train_loss)
            plt.savefig(os.path.join(PATH,'Train/loss'))


            # saving 

            trainer.save_checkpoint(epc, PATH+'checkpoint.pth')
            # Backup checkpoint if the first one fails
            trainer.save_checkpoint(epc, PATH+'checkpoint_BP.pth')
            # save the best model
            if valid_loss_epoch == min(trainer.valid_loss):
                trainer.save_checkpoint(epc, PATH+'checkpoint_best.pth')
                # Backup checkpoint if the first one fails
                trainer.save_checkpoint(epc, PATH+'checkpoint_best_BP.pth')

            # Save config file as json
            with open(PATH+'config.json', 'w') as f:
                config_dict = configs.to_dict()
                json.dump(config_dict, f, indent=2)

            print('Config file saved')

            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(PATH+'checkpoint.pth')
            run.log_artifact(artifact)


        print('Epoch: {} || Training loss: {:0.4} || Evaluation loss: {:0.4} || Elapsed time: {}'.format(str(epc).zfill(6), np.mean(train_loss_epoch), valid_loss[-1], str(np.round(stop - start_epoch,2)).zfill(6)))

    run.finish()
