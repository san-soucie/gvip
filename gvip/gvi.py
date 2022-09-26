import weakref

import pyro.ops.jit
import torch
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import is_validation_enabled
from pyro.util import check_if_enumerated, warn_if_nan

from .divergence import Divergence
from .loss import Loss


class GeneralizedVariationalLoss(ELBO):
    def __init__(self, divergence: Divergence, loss_fn: Loss, *args, **kwargs):
        self.loss_fn = loss_fn
        self.divergence = divergence
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
        return self.loss_fn.loss(
            model, guide, *args, **kwargs
        ) + self.divergence.loss(model, guide, *args, **kwargs)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        return self.loss_fn.loss_and_grads(
            model, guide, *args, **kwargs
        ) + self.divergence.loss_and_grads(model, guide, *args, **kwargs)


class JITGeneralizedVariationalLoss(GeneralizedVariationalLoss):
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
                (
                    loss_fn_loss,
                    loss_fn_surrogate_loss,
                ) = weakself().loss_fn.loss_and_grads(
                    model, guide, *args, **kwargs
                )
                (
                    divergence_loss,
                    divergence_surrogate_loss,
                ) = weakself().divergence.loss_and_grads(
                    model, guide, *args, **kwargs
                )
                return (
                    loss_fn_loss + divergence_loss,
                    loss_fn_surrogate_loss + divergence_surrogate_loss,
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
