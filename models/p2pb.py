from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_pytorch import EMA
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from models.train_utils import DiffusionModel

from .loss import get_loss


def space_indices(num_steps: int, count: int):
    """
    Generate a list of indices that evenly space out over a given number of steps.

    Args:
        num_steps (int): The total number of steps.
        count (int): The number of indices to generate.

    Returns:
        list: A list of indices that evenly space out over the given number of steps.
    """
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def extract(a: Tensor, t: int, x_shape: Tuple[int, ...]):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def unsqueeze_xdim(z: Tensor, xdim: Tuple[int, ...]) -> Tensor:
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def compute_gaussian_product_coef(sigma1: float, sigma2: float):
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def make_beta_schedule(n_timestep: int = 1000, linear_start: float = 1e-4, linear_end: float = 2e-2):
    scale = 1000 / n_timestep
    linear_start *= scale
    linear_end *= scale
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


# heavily adapted from https://github.com/NVlabs/I2SB
class P2PB(DiffusionModel):
    def __init__(self, cfg: Dict, model: torch.nn.Module):
        super().__init__()
        # setup config
        device = cfg.gpu if cfg.gpu is not None else torch.device("cuda")
        self.device = device
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.ot_ode = cfg.diffusion.ot_ode
        self.cfg = cfg
        self.cond_x1 = cfg.diffusion.cond_x1 if "cond_x1" in cfg.diffusion else False
        self.add_x1_noise = cfg.diffusion.add_x1_noise if "add_x1_noise" in cfg.diffusion else False
        self.objective = cfg.diffusion.objective if "objective" in cfg.diffusion else "pred_noise"
        self.weight_loss = cfg.diffusion.weight_loss if "weight_loss" in cfg.diffusion else False
        self.symmetric = cfg.diffusion.symmetric if "symmetric" in cfg.diffusion else True
        self.loss_multiplier = cfg.diffusion.loss_multiplier if "loss_multiplier" in cfg.diffusion else 1.0
        snr_clip = cfg.diffusion.snr_clip if "snr_clip" in cfg.diffusion else False
        straight_cfg = cfg.diffusion.get("straightness", {})
        straight_train_cfg = straight_cfg.get("train", {})
        straight_sample_cfg = straight_cfg.get("sample", {})

        self.straightness_enabled = bool(straight_cfg.get("enabled", False))
        self.straight_vm_enabled = bool(straight_train_cfg.get("use_vm_head", True))
        self.straight_lambda_vm_mse = float(straight_train_cfg.get("lambda_vm_mse", 0.0))
        self.straight_lambda_vm_cos = float(straight_train_cfg.get("lambda_vm_cos", 0.0))
        self.straight_lambda_line = float(straight_train_cfg.get("lambda_line", 0.0))
        self.straight_warmup_steps = int(straight_train_cfg.get("warmup_steps", 0))
        self.straight_min_t_ratio = float(np.clip(straight_train_cfg.get("min_t_ratio", 0.6), 0.0, 1.0))
        self.straight_sample_enabled = bool(straight_sample_cfg.get("enabled", False))
        self.straight_sample_gamma = float(straight_sample_cfg.get("guidance_gamma", 0.0))
        self.straight_use_vm_direction = bool(straight_sample_cfg.get("use_vm_direction", True))

        # load model
        self.model = model.to(device)
        self.ema = EMA(self.model, beta=0.999) if cfg.model.ema else None
        self.vm_head = None
        if self.straightness_enabled and self.straight_vm_enabled:
            vm_hidden_dim = int(straight_train_cfg.get("vm_hidden_dim", 64))
            self.vm_head = nn.Sequential(
                nn.LazyConv1d(vm_hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(vm_hidden_dim, 3, kernel_size=1),
            ).to(device)

        # create betas
        betas = make_beta_schedule(
            n_timestep=cfg.diffusion.timesteps,
            linear_start=cfg.diffusion.beta_start,
            linear_end=cfg.diffusion.beta_end,
        )

        if self.symmetric:
            betas = np.concatenate(
                [
                    betas[: cfg.diffusion.timesteps // 2],
                    np.flip(betas[: cfg.diffusion.timesteps // 2]),
                ]
            )

        self.noise_levels = (
            torch.linspace(
                cfg.diffusion.t0,
                cfg.diffusion.T,
                cfg.diffusion.timesteps,
                dtype=torch.float32,
            ).to(device)
            * cfg.diffusion.timesteps
        )

        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)
        self.calculate_loss = get_loss(cfg.diffusion.get("loss_type", "mse"))

        alphas_cumprod = np.cumprod(1 - betas)
        snr = alphas_cumprod / (1 - alphas_cumprod)

        snr = torch.from_numpy(snr).to(device)
        maybe_clipped_snr = snr.clone()

        if snr_clip:
            maybe_clipped_snr.clamp_(max=5.0)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        # in noise prediction the loss weight is 1 except we clip the maximum SNR
        if self.objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        # in x0 prediction the loss weight is set to the backward std
        elif self.objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)

        self.register_buffer("_straight_train_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.latest_straightness_terms = {}

    def get_std_fwd(self, step: int, xdim: Tuple[int, ...] = None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def compute_pred_x0_from_eps(self, step: int, xt: Tensor, net_out: Tensor, clip_denoise: bool = False):
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])

        if std_fwd.ndim != net_out.ndim:
            std_fwd = std_fwd.squeeze(-1)

        pred_x0 = xt - std_fwd * net_out

        if clip_denoise:
            pred_x0.clamp_(-3.0, 3.0)
        return pred_x0

    def compute_gt(self, step: int, x0: Tensor, xt: Tensor) -> Tensor:
        if self.objective == "pred_noise":
            std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
            gt = (xt - x0) / std_fwd
            return gt.detach()
        elif self.objective == "pred_x0":
            return x0.detach()

    def q_sample(self, step: int, x0: Tensor, x1: Tensor) -> Tensor:
        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1

        if not self.ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)

        return xt.detach()

    def p_posterior(self, nprev: int, n: int, x_n: Tensor, x0: Tensor) -> Tensor:
        """Calculates the posterior p(x_{t-1} | x_t, x_0)

        Args:
            nprev (int): Time step t-1
            n (int): Time step t
            x_n (Tensor): Latent state x_t
            x0 (Tensor): Clean sample x_0

        Returns:
            Tensor: Latent state x_{t-1}
        """
        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not self.ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def _straightness_warmup_scale(self) -> float:
        if not self.straightness_enabled:
            return 0.0
        if self.straight_warmup_steps <= 0:
            return 1.0
        current_step = int(self._straight_train_step.item())
        return min(1.0, current_step / float(self.straight_warmup_steps))

    def _predict_vm_direction(self, latent_features: Optional[Tensor]) -> Optional[Tensor]:
        if self.vm_head is None or latent_features is None:
            return None
        return self.vm_head(latent_features)

    @staticmethod
    def _collinearity_loss(direction: Tensor, target_direction: Tensor) -> Tensor:
        direction = direction.transpose(1, 2)
        target_direction = F.normalize(target_direction.transpose(1, 2), dim=-1, eps=1e-8)

        proj = torch.sum(direction * target_direction, dim=-1, keepdim=True) * target_direction
        perp = direction - proj

        perp_norm_sq = torch.sum(perp**2, dim=-1)
        direction_norm_sq = torch.sum(direction**2, dim=-1)
        return (perp_norm_sq / (direction_norm_sq + 1e-8)).mean(dim=1)

    def _compute_straightness_losses(
        self,
        steps: Tensor,
        xt: Tensor,
        pred: Tensor,
        x0: Tensor,
        x1: Tensor,
        vm_direction: Optional[Tensor],
    ) -> Tensor:
        target_direction = (x0 - x1).detach()
        step_ratio = steps.float() / float(max(self.timesteps - 1, 1))
        active_mask = (step_ratio >= self.straight_min_t_ratio).float()
        active_count = active_mask.sum()
        if active_count.item() < 1:
            self.latest_straightness_terms = {}
            return torch.tensor(0.0, device=xt.device)

        masked_mean = lambda x: (x * active_mask).sum() / (active_count + 1e-8)

        if self.objective == "pred_noise":
            pred_x0 = self.compute_pred_x0_from_eps(steps, xt, pred, clip_denoise=False)
        else:
            pred_x0 = pred

        denoise_direction = pred_x0 - xt

        straight_loss = torch.tensor(0.0, device=xt.device)
        loss_terms = {"active_ratio": active_mask.mean().detach()}

        if vm_direction is not None and self.straight_lambda_vm_mse > 0:
            vm_mse_per_sample = ((vm_direction - target_direction) ** 2).mean(dim=(1, 2))
            vm_mse = masked_mean(vm_mse_per_sample)
            straight_loss = straight_loss + self.straight_lambda_vm_mse * vm_mse
            loss_terms["vm_mse"] = vm_mse.detach()

        if vm_direction is not None and self.straight_lambda_vm_cos > 0:
            vm_cos = F.cosine_similarity(
                vm_direction.transpose(1, 2),
                target_direction.transpose(1, 2),
                dim=-1,
                eps=1e-8,
            )
            vm_cos_loss_per_sample = (1.0 - vm_cos).mean(dim=-1)
            vm_cos_loss = masked_mean(vm_cos_loss_per_sample)
            straight_loss = straight_loss + self.straight_lambda_vm_cos * vm_cos_loss
            loss_terms["vm_cos"] = vm_cos_loss.detach()

        if self.straight_lambda_line > 0:
            line_loss_per_sample = self._collinearity_loss(denoise_direction, target_direction)
            line_loss = masked_mean(line_loss_per_sample)
            straight_loss = straight_loss + self.straight_lambda_line * line_loss
            loss_terms["line"] = line_loss.detach()

        self.latest_straightness_terms = {k: v.item() for k, v in loss_terms.items()}
        return straight_loss

    def _apply_straightness_guidance(self, xt: Tensor, xt_prev: Tensor, line_direction: Tensor) -> Tensor:
        gamma = float(np.clip(self.straight_sample_gamma, 0.0, 1.0))
        if gamma <= 0:
            return xt_prev

        displacement = (xt_prev - xt).transpose(1, 2)
        line_direction = F.normalize(line_direction.transpose(1, 2), dim=-1, eps=1e-8)

        displacement_proj = torch.sum(displacement * line_direction, dim=-1, keepdim=True) * line_direction
        displacement_guided = (1.0 - gamma) * displacement + gamma * displacement_proj
        return xt + displacement_guided.transpose(1, 2)

    def sample_ddpm(
        self,
        steps: List[int],
        pred_x0_fn: callable,
        x1: Tensor,
        x_cond: bool = None,
        log_steps: List[int] = None,
        verbose: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """DDPM Sampling

        Args:
            steps (List[int]): Index of steps used in sampling.
            pred_x0_fn (callable): Function to predict x0 from xt.
            x1 (Tensor): The noisy prior sample.
            x_cond (bool, optional): Conditioning sample. Defaults to None.
            log_steps (List[int], optional): Steps to log. Defaults to None.
            verbose (bool, optional): Print progress. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor]: Trajectory, Predicted x0
        """
        if self.add_x1_noise:
            x1 = x1 + torch.randn_like(x1)

        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"
            pred_x0, vm_direction = pred_x0_fn(xt, step, x1=x1, x_cond=x_cond)
            xt_prev = self.p_posterior(prev_step, step, xt, pred_x0)

            if self.straight_sample_enabled and self.straight_sample_gamma > 0:
                line_direction = vm_direction if (vm_direction is not None and self.straight_use_vm_direction) else (
                    pred_x0 - x1
                )
                xt_prev = self._apply_straightness_guidance(xt, xt_prev, line_direction)

            xt = xt_prev

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach())
                xs.append(xt.detach())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    @torch.no_grad()
    def ddpm_sampling(
        self,
        x1: Tensor,
        x_cond: Tensor = None,
        clip_denoise: bool = False,
        sampling_steps: int = None,
        log_count: int = 10,
        verbose: bool = True,
        use_ema: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """DDPM Sampling

        Args:
            x1 (Tensor): Noisy prior sample.
            x_cond (Tensor, optional): Conditioning sample. Defaults to None.
            clip_denoise (bool, optional): Clips the predictions. Defaults to False.
            sampling_steps (int, optional): Number of sampling steps. Defaults to None.
            log_count (int, optional): Number of samples to log. Defaults to 10.
            verbose (bool, optional): Print progress. Defaults to True.
            use_ema (bool, optional): Use EMA model. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: Chain trajectory, Predicted x0
        """

        sampling_steps = sampling_steps or self.timesteps - 1
        assert 0 < sampling_steps < self.timesteps == len(self.betas)
        steps = space_indices(self.timesteps, sampling_steps + 1)

        # create log steps
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        if verbose:
            logger.info(f"[DDPM Sampling] T={self.timesteps}, {sampling_steps=}!")

        self.model.eval()

        def pred_x0_fn(xt, step, x1, x_cond=None):
            step = torch.full((xt.shape[0],), step, device=self.device, dtype=torch.long)
            noise_levels = self.noise_levels[step].detach()
            use_vm_direction = (
                self.straight_sample_enabled
                and self.straight_use_vm_direction
                and self.vm_head is not None
                and not (use_ema and self.ema is not None)
            )
            need_latent = use_vm_direction

            if self.cond_x1:
                if x_cond is not None:
                    x_cond = torch.cat([x1, x_cond], dim=1)
                else:
                    x_cond = x1

            if use_ema and self.ema is not None:
                out = self.ema(xt, noise_levels, x_cond=x_cond, return_latent=need_latent)
            else:
                out = self.model(xt, noise_levels, x_cond=x_cond, return_latent=need_latent)

            vm_direction = None
            if need_latent:
                out, latent_features = out
                vm_direction = self._predict_vm_direction(latent_features)

            if self.objective == "pred_noise":
                pred_x0 = self.compute_pred_x0_from_eps(step, xt, out, clip_denoise=clip_denoise)
            elif self.objective == "pred_x0":
                pred_x0 = out

            return pred_x0, vm_direction

        xs, pred_x0 = self.sample_ddpm(
            steps,
            pred_x0_fn,
            x1,
            x_cond=x_cond,
            log_steps=log_steps,
            verbose=verbose,
        )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        self.model.train()

        return xs, pred_x0

    @torch.no_grad()
    def sample(
        self,
        x_cond: Optional[Tensor] = None,
        x_start: Optional[Tensor] = None,
        clip: bool = False,
        use_ema: bool = False,
        verbose: bool = True,
        log_count: int = 10,
        steps: int = None,
    ) -> Dict:
        if self.cfg.diffusion.sampling_strategy == "DDPM":
            xs, x0s = self.ddpm_sampling(
                x1=x_start,
                x_cond=x_cond,
                clip_denoise=clip,
                sampling_steps=self.cfg.diffusion.sampling_timesteps if steps is None else steps,
                verbose=verbose,
                use_ema=use_ema,
                log_count=log_count,
            )
            data = {
                "x_chain": xs,
                "x_pred": xs[:, 0, ...],
                "x_start": x_start,
            }
        return data

    def loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred = pred.to(self.device)
        gt = gt.to(self.device)
        loss = self.calculate_loss(pred, gt)
        # loss = loss * extract(self.loss_weight, torch.zeros(pred.shape[0]), loss.shape)  # SNR weighted loss
        loss = loss.mean()
        return loss

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        x_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward step

        Args:
            x0 (Tensor): Clean sample.
            x1 (Tensor): Noisy sample.
            x_cond (Optional[Tensor], optional): Additional conditioning. Defaults to None.

        Returns:
            Tensor: Loss
        """
        steps = torch.randint(0, self.timesteps, (x0.shape[0],)).to(self.device)

        if self.add_x1_noise:
            x1 = x1 + torch.randn_like(x1)

        xt = self.q_sample(steps, x0, x1)
        gt = self.compute_gt(steps, x0, xt)

        if self.cond_x1:
            if x_cond is not None:
                x_cond = torch.cat([x1, x_cond], dim=1)
            else:
                x_cond = x1

        noise_levels = self.noise_levels[steps].detach()
        need_latent = self.straightness_enabled and self.vm_head is not None
        model_out = self.model(xt, noise_levels, x_cond=x_cond, return_latent=need_latent)

        latent_features = None
        if need_latent:
            pred, latent_features = model_out
        else:
            pred = model_out

        loss = self.calculate_loss(pred, gt)
        if self.weight_loss:
            loss = loss * extract(self.loss_weight, steps, loss.shape)
        loss = loss.mean()
        loss = loss * self.loss_multiplier

        if self.training and self.straightness_enabled:
            self._straight_train_step += 1
            warmup_scale = self._straightness_warmup_scale()
            if warmup_scale > 0:
                vm_direction = self._predict_vm_direction(latent_features)
                straight_loss = self._compute_straightness_losses(steps, xt, pred, x0, x1, vm_direction)
                loss = loss + warmup_scale * straight_loss

        return loss
