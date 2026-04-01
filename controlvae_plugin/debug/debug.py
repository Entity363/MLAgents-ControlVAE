from ControlVAECore.Utils.radam import RAdam




def dump_actor(tag, actor):
    print(
        f"{tag} | actor_id={id(actor)} "
        f"| post_mu_ptr={actor.encoder.post.mu.weight.data_ptr()} "
        f"| prior_mu_ptr={actor.encoder.prior.mu.weight.data_ptr()} "
        f"| post_mu={actor.encoder.post.mu.weight.abs().mean().item():.12e} "
        f"| prior_mu={actor.encoder.prior.mu.weight.abs().mean().item():.12e}"
    )

import traceback
import torch


def watch_param(param: torch.nn.Parameter, name: str):
    orig_zero_ = torch.Tensor.zero_
    orig_copy_ = torch.Tensor.copy_
    orig_fill_ = torch.Tensor.fill_
    orig_set_ = torch.Tensor.set_

    def is_target(t):
        try:
            return t.data_ptr() == param.data.data_ptr()
        except Exception:
            return False

    def zero__hook(self, *args, **kwargs):
        if is_target(self):
            print(f"\n[WATCH] zero_ on {name}")
            traceback.print_stack(limit=25)
        return orig_zero_(self, *args, **kwargs)

    def copy__hook(self, src, *args, **kwargs):
        if is_target(self):
            print(f"\n[WATCH] copy_ into {name}")
            print("src abs mean:", src.abs().mean().item() if hasattr(src, "abs") else type(src))
            traceback.print_stack(limit=25)
        return orig_copy_(self, src, *args, **kwargs)

    def fill__hook(self, *args, **kwargs):
        if is_target(self):
            print(f"\n[WATCH] fill_ on {name}")
            traceback.print_stack(limit=25)
        return orig_fill_(self, *args, **kwargs)

    def set__hook(self, *args, **kwargs):
        if is_target(self):
            print(f"\n[WATCH] set_ on {name}")
            traceback.print_stack(limit=25)
        return orig_set_(self, *args, **kwargs)

    torch.Tensor.zero_ = zero__hook
    torch.Tensor.copy_ = copy__hook
    torch.Tensor.fill_ = fill__hook
    torch.Tensor.set_ = set__hook


import os
import mlagents
import mlagents_envs
import importlib.metadata as md

print("mlagents file:", mlagents.__file__)
print("mlagents_envs file:", mlagents_envs.__file__)
print("mlagents version:", md.version("mlagents"))
print("mlagents_envs version:", md.version("mlagents-envs"))