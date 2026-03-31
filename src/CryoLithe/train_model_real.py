"""Training script for CryoLithe models using real data and reconstructions."""

import json
import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_collections import config_dict as cd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .datasets.real_volumes import RealVolumes
from .filter_models import get_filter_model
from .models import get_model
from .trainers import TrainerReal, TrainerRealVolume, TrainerRealWavelet
from .schedulers import LinearWarmupCosineAnnealingLR

try:
    import wandb
except ImportError:
    wandb = None


def _ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")


def _prepare_output_paths(path):
    output_path = path
    train_path = os.path.join(output_path, "Train")
    print(output_path)
    _ensure_directory(output_path)
    _ensure_directory(train_path)
    return {
        "output_path": output_path,
        "train_path": train_path,
        "config_path": os.path.join(output_path, "config.json"),
        "checkpoint_path": os.path.join(output_path, "checkpoint.pth"),
        "checkpoint_backup_path": os.path.join(output_path, "checkpoint_BP.pth"),
        "checkpoint_best_path": os.path.join(output_path, "checkpoint_best.pth"),
        "checkpoint_best_backup_path": os.path.join(output_path, "checkpoint_best_BP.pth"),
        "loss_plot_path": os.path.join(train_path, "loss"),
    }


def _resolve_torch_device(device_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(device_id)
        device = torch.device("cuda:" + str(device_id))
    print(device)
    return device


def _load_configs_if_needed(configs, load_checkpoint, config_path):
    if load_checkpoint:
        print("Loading configs")
        configs = cd.ConfigDict(json.load(open(config_path)))
        print("Epochs:" + str(configs.training.num_epochs))
    return configs


def _apply_config_defaults(configs):
    defaults = (
        ("ramp_lr", None, "Adding type as ramp_lr"),
        ("filter_2D_lr", None, "Adding type as filter_2D_lr"),
        ("ramp_weight_decay", configs.training.weight_decay, "Adding type as ramp_weight_decay"),
        ("discrete_sampling", False, "Adding type as discrete_sampling"),
        ("get_projection_prefiltered", False, "Adding type as get_projection_prefiltered"),
        ("scale_data", False, "Adding type as get_projection_prefiltered"),
        ("use_volumetric_trainer", False, "Adding use_volumetric_trainer as False"),
        ("use_wavelet_trainer", False, "Adding use_wavelet_trainer as False"),
        ("reduction", "mean", "Adding reduction as mean"),
        ("use_wandb", False, "Adding use_wandb as False"),
        ("wandb_project", "cryolithe", "Adding wandb_project as cryolithe"),
        ("wandb_entity", None, "Adding wandb_entity as None"),
    )

    for attribute, value, message in defaults:
        if hasattr(configs.training, attribute) is False:
            print(message)
            setattr(configs.training, attribute, value)

    if hasattr(configs, "deform_volume") is False:
        print("Adding deform_volume as False")
        configs.deform_volume = False

    if hasattr(configs.model, "type") is False:
        print("Adding attribute type as mlp")
        configs.model.type = "mlp"

    return configs


def _init_wandb(configs):
    use_wandb = configs.training.use_wandb
    if use_wandb and wandb is None:
        print("wandb requested but not installed. Continuing with wandb disabled.")
        use_wandb = False

    if use_wandb is False:
        return None

    entity = getattr(configs.training, "wandb_entity", None)
    project = getattr(configs.training, "wandb_project", "cryolithe")

    return wandb.init(
        entity=entity,
        project=project,
        config=configs.to_dict(),
    )


def _build_datasets(configs):
    if configs.data.type == "t20":
        train_dataset = RealVolumes(**configs.train_dataset)
        valid_dataset = RealVolumes(**configs.valid_dataset)
        return train_dataset, valid_dataset

    raise NotImplementedError(f"Dataset type {configs.data.type} not implemented")


def _build_model_components(configs, model_device, runtime_device):
    n_projections = configs.data.n_projections
    model_input_projections = n_projections
    if configs.training.get_projection_prefiltered:
        model_input_projections = 2 * n_projections

    print("using model:" + configs.model.type)
    model = get_model(n_projections=model_input_projections, **configs.model).to(model_device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"---> Number of trainable parameters in volume net: {num_param}")

    train_list = [{"params": model.parameters()}]

    ramp = _build_ramp(configs, train_list, model_device, runtime_device)
    filter_2d = _build_filter_2d(configs, train_list, model_device)
    patch_scale = _build_patch_scale(configs, train_list, model_device)

    return model, ramp, filter_2d, patch_scale, train_list


def _build_ramp(configs, train_list, model_device, runtime_device):
    if configs.filter_projections is False:
        return None

    if configs.ramp.use_pretrained:
        checkpoint = torch.load(
            configs.ramp.pretrain_path + "/checkpoint.pth",
            map_location=torch.device(runtime_device),
        )
        return checkpoint["ramp"]

    ramp = get_filter_model(configs.ramp.type, **configs.ramp.model).to(model_device)
    if configs.ramp.use_learnable_ramp:
        ramp_group = {
            "params": ramp.parameters(),
            "weight_decay": configs.training.ramp_weight_decay,
        }
        if configs.training.ramp_lr is not None:
            ramp_group["lr"] = configs.training.ramp_lr
        train_list.append(ramp_group)
    return ramp


def _build_filter_2d(configs, train_list, model_device):
    if configs.use_2D_filters is False:
        return None

    filter_2d = get_filter_model(configs.filter_2D.type, **configs.filter_2D.model).to(model_device)
    filter_group = {"params": filter_2d.parameters()}
    if configs.training.filter_2D_lr is not None:
        filter_group["lr"] = configs.training.filter_2D_lr
    train_list.append(filter_group)
    return filter_2d


def _build_patch_scale(configs, train_list, model_device):
    patch_scale = torch.FloatTensor([configs.training.patch_scale_init]).to(model_device).clone()
    if configs.training.learn_patch_scale:
        patch_scale = Variable(patch_scale, requires_grad=True)
        train_list.append({"params": patch_scale})
    return patch_scale


def _build_optimizer(configs, train_list):
    if configs.training.optimizer == "adam":
        return torch.optim.Adam(
            train_list,
            lr=configs.training.lr,
            weight_decay=configs.training.weight_decay,
        )
    if configs.training.optimizer == "sgd":
        return torch.optim.SGD(
            train_list,
            lr=configs.training.lr,
            weight_decay=configs.training.weight_decay,
        )
    if configs.training.optimizer == "adamw":
        return torch.optim.AdamW(
            train_list,
            lr=configs.training.lr,
            weight_decay=configs.training.weight_decay,
        )
    raise NotImplementedError(f"Optimizer {configs.training.optimizer} not implemented")


def _build_scheduler(configs, optimizer):
    if (
        hasattr(configs.training, "lr_scheduler_type")
        and configs.training.lr_scheduler_type == "linear_warmup_cosine_annealing"
    ):
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=configs.training.lr_warmup_epochs,
            max_epochs=configs.training.num_epochs,
            eta_min=configs.training.lr_min,
        )

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=configs.training.lr_decay_epochs,
        gamma=configs.training.lr_decay,
    )


def _build_criterion(configs):
    if configs.training.loss == "MSE":
        return torch.nn.MSELoss(reduction=configs.training.reduction)
    if configs.training.loss == "l1":
        return torch.nn.L1Loss(reduction=configs.training.reduction)
    if configs.training.loss == "huber":
        return torch.nn.SmoothL1Loss(reduction=configs.training.reduction)
    raise NotImplementedError(f"Loss {configs.training.loss} not implemented")


def _build_trainer(configs, model, ramp, patch_scale, filter_2d, optimizer, criterion, scheduler, device_id):
    trainer_kwargs = {
        "config": configs,
        "model": model,
        "ramp": ramp,
        "patch_scale": patch_scale,
        "filter_2D": filter_2d,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "device": device_id,
    }

    if configs.training.use_volumetric_trainer:
        return TrainerRealVolume(**trainer_kwargs)
    if configs.training.use_wavelet_trainer:
        return TrainerRealWavelet(**trainer_kwargs)
    return TrainerReal(**trainer_kwargs)


def _watch_model(run, model, criterion):
    if run is not None:
        wandb.watch(model, criterion=criterion, log="all", log_freq=10)


def _load_trainer_checkpoint(trainer, load_checkpoint, checkpoint_path):
    if load_checkpoint:
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        return start_epoch, trainer.valid_loss
    return 0, []


def _save_training_outputs(trainer, configs, paths, epoch, train_loss, run):
    trainer.save_checkpoint(epoch, paths["checkpoint_path"])
    trainer.save_checkpoint(epoch, paths["checkpoint_backup_path"])

    valid_loss_epoch = trainer.valid_loss[-1]
    if valid_loss_epoch == min(trainer.valid_loss):
        trainer.save_checkpoint(epoch, paths["checkpoint_best_path"])
        trainer.save_checkpoint(epoch, paths["checkpoint_best_backup_path"])

    plt.figure(1)
    plt.clf()
    plt.semilogy(train_loss)
    plt.savefig(paths["loss_plot_path"])

    with open(paths["config_path"], "w") as handle:
        json.dump(configs.to_dict(), handle, indent=2)

    print("Config file saved")

    if run is not None:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(paths["checkpoint_path"])
        run.log_artifact(artifact)


def _run_training_loop(trainer, train_loader, valid_loader, configs, paths, start_epoch, run, device_id):
    print("Starting training")
    valid_loss = trainer.valid_loss if trainer.valid_loss else [float("nan")]

    for epoch in range(start_epoch, configs.training.num_epochs):
        start_time = timeit.default_timer()
        trainer.train_step(train_loader, device=device_id, wandb_run=run)
        train_loss = trainer.train_loss
        train_loss_epoch = train_loss[-1]
        stop_time = timeit.default_timer()

        if epoch % configs.training.save_every == 0:
            trainer.validate(valid_loader, device=device_id, wandb_run=run)
            valid_loss = trainer.valid_loss
            _save_training_outputs(trainer, configs, paths, epoch, train_loss, run)

        print(
            "Epoch: {} || Training loss: {:0.4} || Evaluation loss: {:0.4} || Elapsed time: {}".format(
                str(epoch).zfill(6),
                np.mean(train_loss_epoch),
                valid_loss[-1],
                str(np.round(stop_time - start_time, 2)).zfill(6),
            )
        )


def train_model_real(configs, load_checkpoint=False):
    seed = configs.seed if hasattr(configs, "seed") else 42
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    path = configs.get("output_dir")
    assert path is not None, "output_dir must be specified in the training config"

    device_id = configs.get("device", 0)
    paths = _prepare_output_paths(path)
    runtime_device = _resolve_torch_device(device_id)
    configs = _load_configs_if_needed(configs, load_checkpoint, paths["config_path"])
    configs = _apply_config_defaults(configs)
    run = _init_wandb(configs)

    train_dataset, valid_dataset = _build_datasets(configs)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    model, ramp, filter_2d, patch_scale, train_list = _build_model_components(
        configs,
        device_id,
        runtime_device,
    )
    optimizer = _build_optimizer(configs, train_list)
    scheduler = _build_scheduler(configs, optimizer)
    criterion = _build_criterion(configs)
    _watch_model(run, model, criterion)

    trainer = _build_trainer(
        configs,
        model,
        ramp,
        patch_scale,
        filter_2d,
        optimizer,
        criterion,
        scheduler,
        device_id,
    )
    start_epoch, _ = _load_trainer_checkpoint(trainer, load_checkpoint, paths["checkpoint_path"])

    _run_training_loop(
        trainer,
        train_loader,
        valid_loader,
        configs,
        paths,
        start_epoch,
        run,
        device_id,
    )

    if run is not None:
        run.finish()
