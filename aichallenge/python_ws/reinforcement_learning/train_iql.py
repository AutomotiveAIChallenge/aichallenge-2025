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


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def expectile_loss(diff, tau: float):
    """
    diff = V(s) - stop_grad(Q_min(s,a))
    L = |tau - I(diff < 0)| * diff^2
    """
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return (w * diff.pow(2)).mean()


def huber(x, delta=1.0):
    absx = x.abs()
    return torch.where(absx <= delta, 0.5 * x.pow(2), delta * (absx - 0.5 * delta))


def build_optimizer(params, lr, wd):
    return optim.Adam(params, lr=lr, weight_decay=wd)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# =============================================================
# Reward provider（HDF5にrewardが無い場合のフォールバック）
# =============================================================

class RewardProvider:
    """
    1) HDF5に /reward がある → それを使う（推奨）
    2) なければ velocity と control があれば簡易報酬:
       r = w_v * max(0, vx) - w_s * |steer| - w_a * |accel|
    3) それもなければゼロ
    """
    def __init__(self, dataset, w_v=1.0, w_s=0.05, w_a=0.01):
        self.dataset = dataset
        self.w_v = w_v
        self.w_s = w_s
        self.w_a = w_a

        # dataset 内の各 sequence が reward を持っているかを判定
        self.has_reward = False
        try:
            # MultiSequenceDataset は内部に SequenceDatasets を持つ想定
            for d in getattr(dataset, "datasets", []):
                if "reward" in d.h5:
                    self.has_reward = True
                    break
        except Exception:
            pass

        # velocity/pose の有無（簡易合成で使用）
        self.can_synthesize = False
        try:
            for d in getattr(dataset, "datasets", []):
                if "velocity" in d.h5 and "control" in d.h5:
                    self.can_synthesize = True
                    break
        except Exception:
            pass

    def compute(self, batch):
        if self.has_reward and ("reward" in batch):
            return batch["reward"].float()

        # synthesize if possible
        if self.can_synthesize and ("velocity" in batch) and ("action" in batch):
            vx = batch["velocity"][..., 0]  # (B,)
            steer = batch["action"][..., 0].abs()
            accel = batch["action"][..., 1].abs()
            r = self.w_v * torch.clamp(vx, min=0.0) - self.w_s * steer - self.w_a * accel
            return r.float()

        # fallback
        return torch.zeros(batch["obs"].shape[0], dtype=torch.float32, device=batch["obs"].device)


# =============================================================
# IQL Trainer
# =============================================================

class IQLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        # Dataset（RLなので require_next=True）
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

        # Networks
        self.policy = PolicyNet(obs_dim, hidden_dim=hidden, action_dim=act_dim).to(self.device)
        self.q1 = QNet(obs_dim, action_dim=act_dim, hidden_dim=hidden).to(self.device)
        self.q2 = QNet(obs_dim, action_dim=act_dim, hidden_dim=hidden).to(self.device)
        self.value = VNet(obs_dim, hidden_dim=hidden).to(self.device)
        self.value_target = VNet(obs_dim, hidden_dim=hidden).to(self.device)
        self.value_target.load_state_dict(self.value.state_dict())

        # Optims
        lr = cfg["train"]["lr"]
        wd = cfg["train"]["weight_decay"]
        self.opt_policy = build_optimizer(self.policy.parameters(), lr, wd)
        self.opt_q1 = build_optimizer(self.q1.parameters(), lr, wd)
        self.opt_q2 = build_optimizer(self.q2.parameters(), lr, wd)
        self.opt_v = build_optimizer(self.value.parameters(), lr, wd)

        # Hyperparams
        self.gamma = cfg["algo"]["gamma"]
        self.tau = cfg["algo"]["expectile_tau"]       # 例: 0.7
        self.beta = cfg["algo"]["awbc_beta"]          # 例: 0.5
        self.clip_weight = cfg["algo"]["awbc_clip"]   # 例: 100.0
        self.target_momentum = cfg["algo"]["target_momentum"]  # 例: 0.995
        self.target_update_interval = cfg["algo"]["target_update_interval"]

        # 行動スケール（必要なら [-1,1] に合わせる）
        self.act_scale = cfg["algo"].get("action_scale", 1.0)
        self.act_bias = cfg["algo"].get("action_bias", 0.0)

        # Reward provider
        self.rprovider = RewardProvider(self.dataset,
                                        w_v=cfg["reward"].get("w_v", 1.0),
                                        w_s=cfg["reward"].get("w_s", 0.05),
                                        w_a=cfg["reward"].get("w_a", 0.01))

        # Logging / ckpt
        self.save_dir = Path(cfg["train"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("inf")  # TD誤差などで最小をbestに

    def _update_target(self):
        with torch.no_grad():
            for p, tp in zip(self.value.parameters(), self.value_target.parameters()):
                tp.data.mul_(self.target_momentum).add_(p.data * (1.0 - self.target_momentum))

    def _scale_action(self, a):
        # a_raw → a_scaled（policy出力と揃える用）
        return (a - self.act_bias) / max(1e-6, self.act_scale)

    def train(self):
        epochs = self.cfg["train"]["epochs"]
        log_every = self.cfg["train"]["log_interval"]
        save_every = self.cfg["train"]["save_interval"]

        global_step = 0
        last_ckpt = self.save_dir / "iql_last.pth"
        best_ckpt = self.save_dir / "iql_best.pth"

        for epoch in range(1, epochs + 1):
            metrics = {"loss_q": [], "loss_v": [], "loss_pi": [], "adv_mean": []}

            for it, batch in enumerate(tqdm(self.loader, desc=f"[Epoch {epoch}/{epochs}]")):
                batch = to_device(batch, self.device)

                s = batch["obs"]          # (B, N)
                a = batch["action"]       # (B, 2)
                s2 = batch["next_obs"]
                d = batch["done"].float()

                # optional attachments (for reward synth)
                if "velocity" in batch:
                    pass  # kept
                # compute reward
                r = self.rprovider.compute(batch)  # (B,)

                # =========================
                # 1) Value update (expectile)
                # =========================
                with torch.no_grad():
                    q1_sa = self.q1(s, self._scale_action(a))
                    q2_sa = self.q2(s, self._scale_action(a))
                    q_min = torch.minimum(q1_sa, q2_sa)  # (B,1)
                v_s = self.value(s)
                diff = v_s - q_min
                loss_v = expectile_loss(diff, tau=self.tau)

                self.opt_v.zero_grad()
                loss_v.backward()
                self.opt_v.step()

                # =========================
                # 2) Critic (Q) update (TD)
                # =========================
                with torch.no_grad():
                    v_s2_tgt = self.value_target(s2)  # (B,1)
                    td_target = r.unsqueeze(1) + self.gamma * (1.0 - d.unsqueeze(1)) * v_s2_tgt

                q1 = self.q1(s, self._scale_action(a))
                q2 = self.q2(s, self._scale_action(a))
                loss_q1 = huber(q1 - td_target).mean()
                loss_q2 = huber(q2 - td_target).mean()
                loss_q = loss_q1 + loss_q2

                self.opt_q1.zero_grad()
                self.opt_q2.zero_grad()
                loss_q.backward()
                self.opt_q1.step()
                self.opt_q2.step()

                # =========================
                # 3) Policy update (AWBC)
                # =========================
                with torch.no_grad():
                    adv = (torch.minimum(self.q1(s, self._scale_action(a)),
                                         self.q2(s, self._scale_action(a))) - self.value(s)).squeeze(1)
                    weights = torch.clamp(torch.exp(adv / max(1e-6, self.beta)), max=self.clip_weight)

                mu, _ = self.policy(s)       # policyは [-1,1] 出力を想定
                a_scaled = torch.clamp(self._scale_action(a), -1.0, 1.0)
                loss_pi = (weights.unsqueeze(1) * nn.functional.smooth_l1_loss(mu, a_scaled, reduction='none')).mean()

                self.opt_policy.zero_grad()
                loss_pi.backward()
                self.opt_policy.step()

                # =========================
                # 4) Target update
                # =========================
                if (global_step % self.target_update_interval) == 0:
                    self._update_target()

                # log
                metrics["loss_q"].append(loss_q.item())
                metrics["loss_v"].append(loss_v.item())
                metrics["loss_pi"].append(loss_pi.item())
                metrics["adv_mean"].append(adv.mean().item())

                global_step += 1

            # ---- epoch end: metrics & ckpt ----
            avg_q = float(np.mean(metrics["loss_q"]))
            avg_v = float(np.mean(metrics["loss_v"]))
            avg_pi = float(np.mean(metrics["loss_pi"]))
            adv_m = float(np.mean(metrics["adv_mean"]))
            print(f"Epoch {epoch} | Q:{avg_q:.4f} V:{avg_v:.4f} Pi:{avg_pi:.4f} Adv:{adv_m:.4f}")

            # save last
            torch.save({
                "policy": self.policy.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                "cfg": self.cfg,
            }, last_ckpt)

            # define “best” as the smallest critic loss（お好みで指標変更OK）
            metric_for_best = avg_q
            if metric_for_best < self.best_metric:
                self.best_metric = metric_for_best
                torch.save({
                    "policy": self.policy.state_dict(),
                    "q1": self.q1.state_dict(),
                    "q2": self.q2.state_dict(),
                    "value": self.value.state_dict(),
                    "value_target": self.value_target.state_dict(),
                    "cfg": self.cfg,
                }, best_ckpt)
                print(f"⭐ New best checkpoint (critic) saved: {best_ckpt}")

            if (epoch % save_every) == 0:
                torch.save(self.policy.state_dict(), self.save_dir / f"iql_policy_epoch{epoch:03d}.pth")

        # final dumps
        torch.save(self.policy.state_dict(), self.save_dir / "iql_policy_final.pth")
        print(f"✅ Training completed. Artifacts in: {self.save_dir}")

        # tidy
        self.dataset.close()


# =============================================================
# Entry
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="IQL training for LiDAR-only (1080/540/270)")
    parser.add_argument("--config", default="./config/train_iql.yaml", type=str, help="Path to YAML config file")
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
