import torch
from torch.nn import Module

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from .common import *
from .encoders import *
from .diffusion import *
from .flow import *

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x=x, beta=t, context=extras["context"], infer=True)

class FlowMatch(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusionUnet = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual)
        self.path = AffineProbPath(scheduler=CondOTScheduler())

    def flow_loss(self, x, context):
        """
        Args:
            x:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        device = x.device
        batch_size = x.shape[0]

        x0 = torch.randn_like(x).to(device) #src dist at x0 (noise) normal dist
        x1 = x                             #trgt dist at x1 (actual)

        t = torch.rand(batch_size).to(device) #uniform dist
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        v_pred = self.diffusionUnet(path_sample.x_t, path_sample.t, context=context)
        v_star = path_sample.dx_t

        loss = F.mse_loss(v_pred, v_star)
        return loss


    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size = x.size(0)
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)

       # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        flow_match_loss = self.flow_loss(x, z)

        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = flow_match_loss
        loss = kl_weight*(loss_entropy + loss_prior) + flow_match_loss

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss

    def sample(self, w, num_points, flexibility, point_dim=3, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)

        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        x_init = torch.randn((batch_size, num_points, point_dim), dtype=torch.float32, device=z.device)
        ts = torch.tensor([0.0, 1.0], dtype=torch.float, device=z.device)

        tmp_model = WrappedModel(self.diffusionUnet)
        solver = ODESolver(velocity_model=tmp_model)
        x1 = solver.sample(x_init=x_init, time_grid=ts, step_size=0.01, method="rk4", return_intermediates=True, context=z)

        return x1[-1] #get result at ts=1.0
