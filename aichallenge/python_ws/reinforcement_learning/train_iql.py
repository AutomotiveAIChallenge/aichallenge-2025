#!/usr/bin/env python3
import math
from pathlib import Path
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model.policy import PolicyNet
from src.model.qnet import QNet
from src.model.vnet import VNet
from src.data.concat_dataset import MultiSequenceDataset


# =============================================================
# Utility
# =============================================================

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def expectile_loss(diff, tau: float):
    """ L = |tau - I(diff < 0)| * diff^2 """
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return (w * diff.pow(2)).mean()


def huber(x, delta=1.0):
    absx = x.abs()
    return torch.where(absx <= delta, 0.5 * x.pow(2), delta * (absx - 0.5 * delta))


def build_optimizer(params, lr, wd):
    return optim.Adam(params, lr=lr, weight_decay=wd)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# =============================================================
# Reward Provider
# =============================================================

class RewardProvider:
    """
    1) HDF5に /reward があればそれを使う
    2) なければ簡易報酬を合成:
         r = w_v * max(0, vx) - w_s * |steer| - w_a * |accel|
    """
    def __init__(self, dataset, w_v=1.0, w_s=0.05, w_a=0.01):
        self.dataset = dataset
        self.w_v, self.w_s, self.w_a = w_v, w_s, w_a

        self.has_reward = any("reward" in d.h5 for d in getattr(dataset, "datasets", []))
        self.can_synthesize = any(
            ("velocity" in d.h5 and "control" in d.h5)
            for d in getattr(dataset, "datasets", [])
        )

    def compute(self, batch):
        if self.has_reward and "reward" in batch:
            return batch["reward"].float()

        if self.can_synthesize:
            vx = batch.get("velocity", torch.zeros_like(batch["obs"]))[..., 0]
            steer = batch["action"][..., 0].abs()
            accel = batch["action"][..., 1].abs()
            r = self.w_v * torch.clamp(vx, min=0.0) - self.w_s * steer - self.w_a * accel
            return r.float()

        return torch.zeros(batch["obs"].shape[0], device=batch["obs"].device)


# =============================================================
# IQL Trainer
# =============================================================

class IQLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        # ---- Dataset ----
        self.dataset = MultiSequenceDataset(
            root_dir=Path(cfg["dataset"]["path"]),
            downsample_dim=cfg["dataset"]["downsample_dim"],
            normalize=True,
            require_next=True,
            max_seq=cfg["dataset"].get("max_seq", None)
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        obs_dim = cfg["dataset"]["downsample_dim"]
        act_dim = cfg["model"]["action_dim"]
        hidden = cfg["model"]["hidden_dim"]

        # ---- Networks ----
        self.policy = PolicyNet(obs_dim, hidden_dim=hidden, action_dim=act_dim).to(self.device)
        self.q1 = QNet(obs_dim, action_dim=act_dim, hidden_dim=hidden).to(self.device)
        self.q2 = QNet(obs_dim, action_dim=act_dim, hidden_dim=hidden).to(self.device)
        self.value = VNet(obs_dim, hidden_dim=hidden).to(self.device)
        self.value_target = VNet(obs_dim, hidden_dim=hidden).to(self.device)
        self.value_target.load_state_dict(self.value.state_dict())

        # ---- Optimizers ----
        lr = cfg["train"]["lr"]
        wd = cfg["train"]["weight_decay"]
        self.opt_policy = build_optimizer(self.policy.parameters(), lr, wd)
        self.opt_q1 = build_optimizer(self.q1.parameters(), lr, wd)
        self.opt_q2 = build_optimizer(self.q2.parameters(), lr, wd)
        self.opt_v = build_optimizer(self.value.parameters(), lr, wd)

        # ---- Hyperparams ----
        algo = cfg["algo"]
        self.gamma = algo["gamma"]
        self.tau = algo["expectile_tau"]
        self.beta = algo["awbc_beta"]
        self.clip_weight = algo["awbc_clip"]
        self.target_momentum = algo["target_momentum"]
        self.target_update_interval = algo["target_update_interval"]
        self.act_scale = algo.get("action_scale", 1.0)
        self.act_bias = algo.get("action_bias", 0.0)

        # ---- Optional: BC初期化 ----
        init_path = algo.get("init_policy", None)
        if init_path and Path(init_path).exists():
            self.policy.load_state_dict(torch.load(init_path, map_location=self.device), strict=False)
            print(f"✅ Loaded pretrained BC policy from: {init_path}")
        else:
            print("⚠️ No pretrained BC policy specified or file missing.")

        # ---- Reward ----
        self.rprovider = RewardProvider(
            self.dataset,
            w_v=cfg["reward"].get("w_v", 1.0),
            w_s=cfg["reward"].get("w_s", 0.05),
            w_a=cfg["reward"].get("w_a", 0.01),
        )

        # ---- Logging ----
        self.save_dir = Path(cfg["train"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("inf")

    # =============================================================
    # Helper
    # =============================================================
    def _update_target(self):
        with torch.no_grad():
            for p, tp in zip(self.value.parameters(), self.value_target.parameters()):
                tp.data.mul_(self.target_momentum).add_(p.data * (1 - self.target_momentum))

    def _scale_action(self, a):
        return (a - self.act_bias) / max(1e-6, self.act_scale)

    # =============================================================
    # Main training
    # =============================================================
    def train(self):
        epochs = self.cfg["train"]["epochs"]
        save_every = self.cfg["train"]["save_interval"]

        best_ckpt = self.save_dir / "iql_best.pth"
        last_ckpt = self.save_dir / "iql_last.pth"
        global_step = 0

        for epoch in range(1, epochs + 1):
            metrics = {"Q": [], "V": [], "Pi": [], "Adv": []}

            for batch in tqdm(self.loader, desc=f"[Epoch {epoch}/{epochs}]"):
                batch = to_device(batch, self.device)
                s, a, s2, d = batch["obs"], batch["action"], batch["next_obs"], batch["done"].float()
                r = self.rprovider.compute(batch)

                # --- (1) Value update ---
                with torch.no_grad():
                    q1_sa = self.q1(s, self._scale_action(a))
                    q2_sa = self.q2(s, self._scale_action(a))
                    q_min = torch.minimum(q1_sa, q2_sa)
                v_s = self.value(s)
                loss_v = expectile_loss(v_s - q_min, self.tau)
                self.opt_v.zero_grad()
                loss_v.backward()
                self.opt_v.step()

                # --- (2) Q update ---
                with torch.no_grad():
                    v_next = self.value_target(s2)
                    td_target = r.unsqueeze(1) + self.gamma * (1 - d.unsqueeze(1)) * v_next
                q1, q2 = self.q1(s, self._scale_action(a)), self.q2(s, self._scale_action(a))
                loss_q = huber(q1 - td_target).mean() + huber(q2 - td_target).mean()
                self.opt_q1.zero_grad(); self.opt_q2.zero_grad()
                loss_q.backward(); self.opt_q1.step(); self.opt_q2.step()

                # --- (3) Policy update ---
                with torch.no_grad():
                    adv = (torch.minimum(self.q1(s, self._scale_action(a)), self.q2(s, self._scale_action(a))) - self.value(s)).squeeze(1)
                    weights = torch.clamp(torch.exp(adv / max(1e-6, self.beta)), max=self.clip_weight)
                mu, _ = self.policy(s)
                a_scaled = torch.clamp(self._scale_action(a), -1, 1)
                loss_pi = (weights.unsqueeze(1) * nn.functional.smooth_l1_loss(mu, a_scaled, reduction="none")).mean()
                self.opt_policy.zero_grad(); loss_pi.backward(); self.opt_policy.step()

                if global_step % self.target_update_interval == 0:
                    self._update_target()
                global_step += 1

                metrics["Q"].append(loss_q.item())
                metrics["V"].append(loss_v.item())
                metrics["Pi"].append(loss_pi.item())
                metrics["Adv"].append(adv.mean().item())

            # --- epoch summary ---
            q, v, pi, adv = map(np.mean, [metrics["Q"], metrics["V"], metrics["Pi"], metrics["Adv"]])
            print(f"Epoch {epoch} | Q:{q:.4f} V:{v:.4f} Pi:{pi:.4f} Adv:{adv:.4f}")

            torch.save({
                "policy": self.policy.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                "cfg": self.cfg,
            }, last_ckpt)

            if q < self.best_metric:
                self.best_metric = q
                torch.save(self.policy.state_dict(), best_ckpt)
                print(f"⭐ New best checkpoint saved: {best_ckpt}")

            if epoch % save_every == 0:
                torch.save(self.policy.state_dict(), self.save_dir / f"iql_policy_epoch{epoch:03d}.pth")

        torch.save(self.policy.state_dict(), self.save_dir / "iql_policy_final.pth")
        print(f"✅ Training completed. Saved to {self.save_dir}")
        self.dataset.close()


# =============================================================
# Entry
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train_iql.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = IQLTrainer(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        trainer.dataset.close()


if __name__ == "__main__":
    main()
