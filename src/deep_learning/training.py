"""Development early stopping and fixed-epoch all-data final training."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .architectures import build_architecture
from .datasets import ScaledFold
from .determinism import set_deterministic_seed


@dataclass(frozen=True)
class TrainResult:
    scaled_prediction: float
    best_epoch: int
    fit_seconds: float
    forecast_seconds: float
    training_sample_n: int
    validation_sample_n: int
    early_stopping_used: bool


def _batches(count: int, batch_size: int):
    for start in range(0,count,batch_size): yield slice(start,min(count,start+batch_size))


def _fit_epochs(model: nn.Module, fold: ScaledFold, epochs: int, learning_rate: float,
                weight_decay: float, batch_size: int, clip: float) -> None:
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    loss_fn=nn.MSELoss(); model.train()
    for _ in range(epochs):
        for batch in _batches(len(fold.train_y),batch_size):
            optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(fold.train_X[batch]),fold.train_y[batch])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),clip); optimizer.step()


def development_train(fold: ScaledFold, validation_scaled_target: float, architecture: str,
                      size: dict[str,Any], dropout: float, learning_rate: float, weight_decay: float,
                      seed: int, settings: dict[str,Any]) -> TrainResult:
    """Train with one historical validation forecast and restore the best epoch."""
    set_deterministic_seed(seed,int(settings["torch_num_threads"])); model=build_architecture(architecture,size,dropout)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay); loss_fn=nn.MSELoss()
    best_loss=np.inf; best_state=None; best_epoch=0; stale=0; started=time.perf_counter()
    for epoch in range(1,int(settings["maximum_epochs"])+1):
        model.train()
        for batch in _batches(len(fold.train_y),int(settings["batch_size"])):
            optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(fold.train_X[batch]),fold.train_y[batch])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(settings["gradient_clip_norm"])); optimizer.step()
        model.eval()
        with torch.no_grad(): prediction=float(model(fold.forecast_X).item())
        validation_loss=(prediction-validation_scaled_target)**2
        if validation_loss < best_loss-1e-12:
            best_loss=validation_loss; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else: stale+=1
        if epoch>=int(settings["minimum_epochs"]) and stale>=int(settings["early_stopping_patience"]): break
    if best_state is None: raise RuntimeError("Development training never produced a finite validation state.")
    model.load_state_dict(best_state); fit_seconds=time.perf_counter()-started; model.eval(); forecast_started=time.perf_counter()
    with torch.no_grad(): prediction=float(model(fold.forecast_X).item())
    return TrainResult(prediction,best_epoch,fit_seconds,time.perf_counter()-forecast_started,len(fold.train_y),1,True)


def final_fixed_epoch_train(fold: ScaledFold, architecture: str, size: dict[str,Any], dropout: float,
                            learning_rate: float, weight_decay: float, epochs: int, seed: int,
                            settings: dict[str,Any]) -> TrainResult:
    """Train on every supervised row for a fixed development-derived epoch count."""
    set_deterministic_seed(seed,int(settings["torch_num_threads"])); model=build_architecture(architecture,size,dropout)
    started=time.perf_counter(); _fit_epochs(model,fold,epochs,learning_rate,weight_decay,
                                             int(settings["batch_size"]),float(settings["gradient_clip_norm"]))
    fit_seconds=time.perf_counter()-started; model.eval(); forecast_started=time.perf_counter()
    with torch.no_grad(): prediction=float(model(fold.forecast_X).item())
    return TrainResult(prediction,epochs,fit_seconds,time.perf_counter()-forecast_started,len(fold.train_y),0,False)


def final_fixed_epoch_fit(fold: ScaledFold, architecture: str, size: dict[str,Any], dropout: float,
                          learning_rate: float, weight_decay: float, epochs: int, seed: int,
                          settings: dict[str,Any]) -> tuple[TrainResult, nn.Module]:
    """Fit a frozen final configuration and return both result and model state.

    This is the checkpoint-aware counterpart to :func:`final_fixed_epoch_train`.
    It deliberately uses every supervised row, no validation holdout, and no
    early-stopping callback.
    """
    set_deterministic_seed(seed,int(settings["torch_num_threads"])); model=build_architecture(architecture,size,dropout)
    started=time.perf_counter(); _fit_epochs(model,fold,epochs,learning_rate,weight_decay,
                                             int(settings["batch_size"]),float(settings["gradient_clip_norm"]))
    fit_seconds=time.perf_counter()-started; model.eval(); forecast_started=time.perf_counter()
    with torch.no_grad(): prediction=float(model(fold.forecast_X).item())
    result=TrainResult(prediction,epochs,fit_seconds,time.perf_counter()-forecast_started,
                       len(fold.train_y),0,False)
    return result, model
