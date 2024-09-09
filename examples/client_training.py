import os
import json
import time
import math
import tqdm
import numpy as np
import imageio
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from datasets.traj import generate_interpolated_path, get_ordered_poses
from flwr.client import NumPyClient
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import (
    get_positional_encodings,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    SpotLessModule,
)
from gsplat.rendering import rasterization
from datasets.colmap import Parser

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, torch.optim.Optimizer]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means3d", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    return splats, optimizers


class GaussianFlowerClient(NumPyClient):
    def __init__(self, client_id, trainset, valloader, cfg, runner, device="cuda"):
        super().__init__()
        self.client_id = client_id
        self.cfg = cfg
        self.runner = runner
        self.trainset = trainset
        self.valloader = valloader
        self.device = device

        self.client_result_dir = os.path.join(self.cfg.result_dir, f"client_{self.client_id}")
        os.makedirs(self.client_result_dir, exist_ok=True)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.runner.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.cfg.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))

        self.spotless_optimizers = []
        self.mlp_spotless = cfg.semantics and not cfg.cluster
        if self.mlp_spotless:
            self.spotless_module = SpotLessModule(
                num_classes=1, num_features=cfg.mlp_spotless_num_feats
            ).cuda()
            self.spotless_optimizers = [
                torch.optim.Adam(
                    self.spotless_module.parameters(),
                    lr=1e-3,
                )
            ]
            self.spotless_loss = lambda p, minimum, maximum: torch.mean(
                torch.nn.ReLU()(p - minimum) + torch.nn.ReLU()(maximum - p)
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
            "hist_err": torch.zeros((cfg.bin_size,)),
            "avg_err": 1.0,
            "lower_err": 0.0,
            "upper_err": 1.0,
            "sqrgrad": torch.zeros(n_gauss, device=self.device),
        }

    def setup_directories(self, round):
        # Dumping results - one for each round.
        self.ckpt_dir = f"{self.client_result_dir}/{round}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{self.client_result_dir}/{round}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{self.client_result_dir}/{round}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{self.client_result_dir}/{round}/tb")

    def get_parameters(self, config=None):
        if config:
            print(f"Recieved {config} from server")
        # Return the model parameters from the client to server
        return [param.cpu().numpy() for _, param in self.spotless_module.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.spotless_module.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.spotless_module.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Starting client training; info from server: ", config)

        round_id = config.get('round', -1)
        self.train(round_id)
        new_parameters = self.get_parameters()
        return new_parameters, len(self.trainset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        round_id = config.get('round', -1)
        metrics = self.eval(round=round_id, return_metric=True)
        psnr = metrics['PSNR'].item() if 'PSNR' in metrics else 0.0
        ssim = metrics['SSIM'].item() if 'SSIM' in metrics else 0.0
        lpips = metrics['LPIPS'].item() if 'LPIPS' in metrics else 0.0
        num_samples = len(self.trainset)  # Number of samples might be different depending on your setup

        additional_metrics = {
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips,
        }

        return 0.0, num_samples, additional_metrics
    def train(self, round):
        self.setup_directories(round)
        with open(f"{self.client_result_dir}/{round}/cfg.json", "w") as f:
            json.dump(vars(self.cfg), f)

        max_steps = self.cfg.max_steps
        init_step = 0

        schedulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=False, # True, Experimental - memory efficiency optimization
            pin_memory=False, # True, Experimental - memory efficiency optimization
        )

        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))

        for step in pbar:
            if not self.cfg.disable_viewer:
                while self.runner.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.runner.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(self.device)  # [1, 4, 4]
            Ks = data["K"].to(self.device)  # [1, 3, 3]
            pixels = data["image"].to(self.device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(self.device)
            if self.cfg.depth_loss:
                points = data["points"].to(self.device)  # [1, M, 2]
                depths_gt = data["depths"].to(self.device)  # [1, M]

            height, width = pixels.shape[1:3]

            # sh schedule
            sh_degree_to_use = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if self.cfg.depth_loss else "RGB",
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if self.cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=self.device)
                colors = colors + bkgd * (1.0 - alphas)

            info["means2d"].retain_grad()  # used for running stats
            rgb_pred_mask = None

            # loss
            if self.cfg.loss_type == "l1":
                rgbloss = F.l1_loss(colors, pixels)
            else:
                # robust loss
                error_per_pixel = torch.abs(colors - pixels)
                pred_mask = self.runner.robust_mask(
                    error_per_pixel, self.running_stats["avg_err"]
                )
                if self.cfg.semantics:
                    sf = data["semantics"].to(self.device)
                    if self.cfg.cluster:
                        # cluster the semantic feature and mask based on cluster voting
                        sf = nn.Upsample(
                            size=(colors.shape[1], colors.shape[2]),
                            mode="nearest",
                        )(sf).squeeze(0)
                        pred_mask = self.runner.robust_cluster_mask(pred_mask, semantics=sf)
                    else:
                        # use spotless mlp to predict the mask
                        sf = nn.Upsample(
                            size=(colors.shape[1], colors.shape[2]),
                            mode="bilinear",
                        )(sf).squeeze(0)
                        pos_enc = get_positional_encodings(
                            colors.shape[1], colors.shape[2], 20
                        ).permute((2, 0, 1))
                        sf = torch.cat([sf, pos_enc], dim=0)
                        sf_flat = sf.reshape(sf.shape[0], -1).permute((1, 0))
                        self.spotless_module.eval()
                        pred_mask_up = self.spotless_module(sf_flat)
                        pred_mask = pred_mask_up.reshape(
                            1, colors.shape[1], colors.shape[2], 1
                        )
                        # calculate lower and upper bound masks for spotless mlp loss
                        lower_mask = self.runner.robust_mask(
                            error_per_pixel, self.running_stats["lower_err"]
                        )
                        upper_mask = self.runner.robust_mask(
                            error_per_pixel, self.running_stats["upper_err"]
                        )
                log_pred_mask = pred_mask.clone()
                if self.cfg.schedule:
                    # schedule sampling of the mask based on alpha
                    alpha = np.exp(self.cfg.schedule_beta * np.floor((1 + step) / 1.5))
                    pred_mask = torch.bernoulli(
                        torch.clip(
                            alpha + (1 - alpha) * pred_mask.clone().detach(),
                            min=0.0,
                            max=1.0,
                        )
                    )
                rgbloss = (pred_mask.clone().detach() * error_per_pixel).mean()
            ssimloss = 1.0 - self.runner.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = rgbloss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda
            if self.cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.cfg.scene_scale
                loss += depthloss * self.cfg.depth_lambda

            loss.backward()

            if self.mlp_spotless:
                self.spotless_module.train()
                spot_loss = self.spotless_loss(
                    pred_mask_up.flatten(), upper_mask.flatten(), lower_mask.flatten()
                )
                reg = 0.5 * self.spotless_module.get_regularizer()
                spot_loss = spot_loss + reg
                spot_loss.backward()

            # Pass the error histogram for capturing error statistics
            info["err"] = torch.histogram(
                torch.mean(torch.abs(colors - pixels), dim=-3).clone().detach().cpu(),
                bins=self.cfg.bin_size,
                range=(0.0, 1.0),
            )[0]

            # Experimental - memory efficiency optimization
            del renders, alphas, colors, depths
            torch.cuda.empty_cache()

            desc = f"client={self.client_id} | round={round} | loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if self.cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if self.cfg.pose_opt and self.cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if self.cfg.tb_every > 0 and step % self.cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/rgbloss", rgbloss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                if self.cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if self.cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # update running stats for prunning & growing
            if step < self.cfg.refine_stop_iter:
                self.update_running_stats(info)
                
                if step > self.cfg.refine_start_iter and step % self.cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= self.cfg.grow_grad2d
                    is_small = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            <= self.cfg.grow_scale3d * self.cfg.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=self.device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < self.cfg.prune_opa
                    if step > self.cfg.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                                torch.exp(self.splats["scales"]).max(dim=-1).values
                                > self.cfg.prune_scale3d * self.cfg.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                        if self.cfg.ubp:
                            not_utilized = self.running_stats["sqrgrad"] < self.cfg.ubp_thresh
                            is_prune = is_prune | not_utilized
                            dis_prune = is_prune | not_utilized
                        n_prune = is_prune.sum().item()
                        self.refine_keep(~is_prune)
                        print(
                            f"Step {step}: {n_prune} GSs pruned. "
                            f"Now having {len(self.splats['means3d'])} GSs."
                        )

                        # reset running stats
                        self.running_stats["grad2d"].zero_()
                    if self.cfg.ubp:
                        self.running_stats["sqrgrad"].zero_()
                    self.running_stats["count"].zero_()

                if step % self.cfg.reset_every == 0 and self.cfg.loss_type != "robust":
                    self.reset_opa(self.cfg.prune_opa * 2.0)
                if step == self.cfg.reset_sh and self.cfg.loss_type == "robust":
                    self.reset_sh()
            # Turn Gradients into Sparse Tensor before running optimizer
            if self.cfg.sparse_grad:
                assert self.cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )
            print(self.ckpt_dir)
            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # for optimizer in self.pose_optimizers:
            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)
            # for optimizer in self.app_optimizers:
            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)
            for optimizer in self.spotless_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Save the mask image
            if step > max_steps - 200 and self.cfg.semantics:
                st_interval = time.time()
                rgb_pred_mask = (
                    (log_pred_mask > 0.5).repeat(1, 1, 1, 3).clone().detach()
                )
                canvas = (
                    torch.cat([pixels, rgb_pred_mask, colors], dim=2)
                    .squeeze(0)
                    .cpu()
                    .detach()
                    .numpy()
                )
                imname = image_ids.cpu().detach().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/train_{imname}.png",
                    (canvas * 255).astype(np.uint8),
                )
                global_tic += time.time() - st_interval

            # save checkpoint
            if step in [i - 1 for i in self.cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {
                    "mem": mem,
                    "elapsed_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set at the end of entire training
            if step in [i - 1 for i in self.cfg.eval_steps] or step == max_steps - 1:
                self.eval(step, round)
                # self.render_traj(step, round)

            self.log_memory_usage(step, 'training')

            if not self.cfg.disable_viewer:
                self.runner.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.runner.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.runner.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int=0, round: int=0, return_metric: bool=False):
        """Entry for evaluation."""
        print("Running evaluation for round: ", round)
        self.setup_directories(round)
        cfg = self.cfg
        device = self.device

        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(self.valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.runner.psnr(colors, pixels))
            metrics["ssim"].append(self.runner.ssim(colors, pixels))
            metrics["lpips"].append(self.runner.lpips(colors, pixels))

        ellipse_time /= len(self.valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"For round: {round} "
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        if not return_metric:
            with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"val/{k}", v, step)
            self.writer.flush()
        else:
            # returning metrics with list to get weighted avg
            return {'PSNR': psnr, 'SSIM': ssim, 'LPIPS': lpips}
    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reset_sh(self, value: float = 0.001):
        """Utility function to reset SH specular coefficients."""
        colors = torch.clamp(
            self.splats["shN"], max=torch.abs(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "shN":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(colors)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None or k.find("err") != -1:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            if k == "sqrgrad":
                v_new = torch.ones_like(
                    v_new
                )  # the new ones are assumed to have high utilization in the start
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if k.find("err") != -1:
                continue
            if k == "sqrgrad":
                self.running_stats[k] = torch.cat(
                    (v, torch.ones_like(v[sel]))
                )  # new ones are assumed to have high utilization
            else:
                self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if k.find("err") != -1:
                continue
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)

        # Not needed part-----
        if self.cfg.app_opt:
            # TODO: define app_module sep as well
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            ubp=self.cfg.ubp,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        return render_colors, render_alphas, info

    @torch.no_grad()
    def update_running_stats(self, info: Dict):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        if cfg.ubp:
            sqrgrads = info["means2d"].sqrgrad.clone()
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size

        self.running_stats["hist_err"] = (
                0.95 * self.running_stats["hist_err"] + info["err"]
        )
        mid_err = torch.sum(self.running_stats["hist_err"]) * cfg.robust_percentile
        self.running_stats["avg_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(self.running_stats["hist_err"], 0) >= mid_err)[0][
                0
            ]
        ]

        lower_err = torch.sum(self.running_stats["hist_err"]) * cfg.lower_bound
        upper_err = torch.sum(self.running_stats["hist_err"]) * cfg.upper_bound

        self.running_stats["lower_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(self.running_stats["hist_err"], 0) >= lower_err)[
                0
            ][0]
        ]
        self.running_stats["upper_err"] = torch.linspace(0, 1, cfg.bin_size + 1)[
            torch.where(torch.cumsum(self.running_stats["hist_err"], 0) >= upper_err)[
                0
            ][0]
        ]

        if cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )
            if cfg.ubp:
                self.running_stats["sqrgrad"].index_add_(
                    0, gs_ids, torch.sum(sqrgrads, dim=-1)
                )
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )
            if cfg.ubp:
                self.running_stats["sqrgrad"].index_add_(
                    0, gs_ids, torch.sum(sqrgrads[sel], dim=-1)
                )

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if k.find("err") != -1:
                continue
            if k == "sqrgrad":
                self.running_stats[k] = torch.cat(
                    (v, torch.ones_like(v[sel]))
                )  # new ones are assumed to have high utilization
            else:
                self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def render_traj(self, step: int, round: int=0):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        K = torch.from_numpy(list(self.runner.parser.Ks_dict.values())[0]).float().to(device)
        camtoworlds = get_ordered_poses(self.runner.parser.camtoworlds)

        camtoworlds = generate_interpolated_path(
            camtoworlds[::20].copy(), 40, spline_degree=1, smoothness=0.3
        )  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]
        camtoworlds = camtoworlds * np.reshape([1.1, 1.1, 1, 1], (1, 4, 1))

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        width, height = list(self.runner.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i: i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.gif", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.gif")

    def log_memory_usage(self, step, context):
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        print(f"[{context}] Step {step}: Allocated memory: {allocated_memory:.3f} GB")
