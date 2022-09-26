import weakref
from typing import Callable

import pyro.ops.jit
import torch
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import (
    get_dependent_plate_dims,
    is_validation_enabled,
    torch_sum,
)
from pyro.util import check_if_enumerated, warn_if_nan


class Loss(ELBO):
    def __init__(self, loss_fn: Callable, *args, **kwargs):
        self.loss_fn = loss_fn
        super().__init__(*args, **kwargs)

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles
        many samples/particles.
        """
        loss_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(
            model, guide, args, kwargs
        ):
            loss_particle = self.loss_fn(
                model_trace, guide_trace, args, kwargs
            )

            loss_particles.append(loss_particle)

        if is_vectorized:
            loss_particles = loss_particles[0]
        else:
            loss_particles = torch.stack(loss_particles)

        loss = -loss_particles.sum().item() / self.num_particles
        warn_if_nan(loss, "loss")
        return loss

    def loss_and_grads(self, model, guide, jit=False, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used
        to form the gradient estimator. Performs backward on the latter.
        Num_particle many samples are used to form the estimators.
        """
        loss_particles = []
        surrogate_loss_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1
        tensor_holder = None

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(
            model, guide, args, kwargs
        ):
            surrogate_loss_particle = self.loss_fn(
                model_trace, guide_trace, args, kwargs
            )
            loss_particle = surrogate_loss_particle.detach()

            if is_identically_zero(loss_particle):
                if tensor_holder is not None:
                    loss_particle = torch.zeros_like(tensor_holder)
                    surrogate_loss_particle = torch.zeros_like(tensor_holder)
            else:  # loss_particle is not None
                if tensor_holder is None:
                    tensor_holder = torch.zeros_like(loss_particle)
                    # change types of previous `loss_particle`s
                    for i in range(len(loss_particles)):
                        loss_particles[i] = torch.zeros_like(tensor_holder)
                        surrogate_loss_particles[i] = torch.zeros_like(
                            tensor_holder
                        )

            loss_particles.append(loss_particle)
            surrogate_loss_particles.append(surrogate_loss_particle)

        if tensor_holder is None:
            return 0.0

        if is_vectorized:
            loss_particles = loss_particles[0]
            surrogate_loss_particles = surrogate_loss_particles[0]
        else:
            loss_particles = torch.stack(loss_particles)
            surrogate_loss_particles = torch.stack(surrogate_loss_particles)

        loss_val = loss_particles.sum(dim=0, keepdim=True) / self.num_particles
        surrogate_loss_val = (
            -surrogate_loss_particles.sum() / self.num_particles
        )
        if jit:
            return loss_val, surrogate_loss_val
        # collect parameters to train from model and guide
        trainable_params = any(
            site["type"] == "param"
            for trace in (model_trace, guide_trace)
            for site in trace.nodes.values()
        )

        if trainable_params and getattr(
            surrogate_loss_particles, "requires_grad", False
        ):

            surrogate_loss_val.backward()
        loss = -loss_val.item()
        warn_if_nan(loss, "loss")
        return loss


class JITLoss(Loss):
    def __init__(self, *args, **kwargs):
        self._loss_and_surrogate_loss = None
        super().__init__(*args, **kwargs)

    def loss_and_surrogate_loss(self, model, guide, *args, **kwargs):
        kwargs["_pyro_model_id"] = id(model)
        kwargs["_pyro_guide_id"] = id(guide)
        if getattr(self, "_loss_and_surrogate_loss", None) is None:
            # build a closure for loss_and_surrogate_loss
            weakself = weakref.ref(self)

            @pyro.ops.jit.trace(
                ignore_warnings=self.ignore_jit_warnings,
                jit_options=self.jit_options,
            )
            def loss_and_surrogate_loss(*args, **kwargs):
                kwargs.pop("_pyro_model_id")
                kwargs.pop("_pyro_guide_id")
                return weakself().loss_and_grads(
                    model, guide, jit=True, *args, **kwargs
                )

            self._loss_and_surrogate_loss = loss_and_surrogate_loss

        return self._loss_and_surrogate_loss(*args, **kwargs)

    def differentiable_loss(self, model, guide, *args, **kwargs):
        loss, surrogate_loss = self.loss_and_surrogate_loss(
            model, guide, *args, **kwargs
        )

        warn_if_nan(loss, "loss")
        return loss + (surrogate_loss - surrogate_loss.detach())

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss, surrogate_loss = self.loss_and_surrogate_loss(
            model, guide, *args, **kwargs
        )
        surrogate_loss.backward()
        loss = loss.item()

        warn_if_nan(loss, "loss")
        return loss


def _log_prob_loss(model_trace, guide_trace, *args, **kwargs):
    loss_particle = 0
    sum_dims = get_dependent_plate_dims(model_trace.nodes.values())
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample" and site["is_observed"]:
            loss_particle = loss_particle + torch_sum(
                site["log_prob"], sum_dims
            )

    return loss_particle


class LogLikelihoodLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(loss_fn=_log_prob_loss, *args, **kwargs)


class JITLogLikelihoodLoss(JITLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(loss_fn=_log_prob_loss, *args, **kwargs)
