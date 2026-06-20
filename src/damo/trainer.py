from typing import Dict, Any, Mapping, List
import torch, torch.nn.functional as F
import numpy as np
import sys, os
import logging

import damo.utils as utils
from damo.utils.ensure_types import ensure_device, ensure_numpy, ensure_torch

log = logging.getLogger(__name__)

class Trainer:
    def __init__(
            self,
            cfg,
            model,
            train_loader,
            val_loader,
            device = "cuda",
            logger = None,
    ):
        self.device = ensure_device(device)
        self.model = model.to(self.device)
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            weight_decay=cfg["optimizer"]["weight_decay"]
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

        self.in_data_keys = cfg["in_data_keys"]
        self.gt_data_keys = cfg["gt_data_keys"]
        self.gt_data_loss = cfg["gt_data_loss"]

        self.max_epochs = cfg["max_epochs"]
        self.amp = cfg["amp"]
        self.grad_clip_norm = cfg["grad_clip_norm"]
        self.log_interval = cfg["log_interval"]
        self.val_interval = cfg["val_interval"]
        self.train_sample_interval = cfg["train_sample_interval"]
        self.val_sample_interval = cfg["val_sample_interval"]
        self.global_step = 0

        self.best_val = float("inf")

    def _forward_loss(self, batch):
        in_data, gt_data = self._load_batch(batch)

        loss_dict = {}
        loss_total = 0.0

        with torch.amp.autocast("cuda", enabled=(self.device.type=="cuda" and self.amp)):
            model_out = self.model(**in_data)

            for k, loss_cfg in self.gt_data_loss.items():
                fn_name = loss_cfg["fn"]
                loss_fn = utils.torch_utils.get_loss_fn(fn_name)

                pred = model_out[loss_cfg["pred_key"]]
                gt = gt_data[loss_cfg["gt_key"]]
                mask = gt_data.get(loss_cfg["mask_key"], None)

                loss = loss_fn(pred=pred, target=gt, target_mask=mask)

                loss_dict[k] = loss
                loss_total += loss_cfg["weight"] * loss

            loss_dict["total"] = loss_total

        return {
            "in_data": in_data,
            "gt_data": gt_data,
            "model_out": model_out,
            "loss": loss_dict,
        }

    def _load_batch(self, batch):
        def to_device(x):
            if torch.is_tensor(x):
                return x.to(self.device, non_blocking=True)
            return x

        in_data = {
            k: to_device(batch[k])
            for k in self.in_data_keys
        }
        gt_data = {
            k: to_device(batch[k])
            for k in self.gt_data_keys
        }

        return in_data, gt_data

    def _log_loss(self, loss_dict, epoch, is_training=True):
        if self.logger is None:
            log.warning("logger not initialized")

        category = "train" if is_training else "val"

        losses = {}
        for k, v in loss_dict.items():
            if k == "total":
                continue

            log_key = f"{category}/loss/{k}"
            if torch.is_tensor(v):
                losses[log_key] = v.detach().cpu().item()
            else:
                losses[log_key] = float(v)

        loss_log = {
            f"{category}/loss/total": loss_dict["total"],
            **losses,
            "epoch": float(epoch)
        }
        self.logger.log(loss_log, step=self.global_step)

    def train_epoch(self, epoch: int):
        self.model.train()

        pbar = utils.make_tqdm_pbar(self.train_loader, desc=f"Train | Epoch {epoch}: ")
        sample = None

        for bix, batch in enumerate(pbar, 1):
            outs = self._forward_loss(batch)

            self.optim.zero_grad()
            self.scaler.scale(outs["loss"]["total"]).backward()

            if self.grad_clip_norm:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optim)
            self.scaler.update()

            self.global_step += 1
            if self.global_step % self.log_interval == 0:
                self._log_loss(outs["loss"], epoch, is_training=True)
                log.info(f"[E{epoch:03d} S{self.global_step}] loss={outs['loss']['total'].item():.4f}")

            if sample is None:
                sample = {
                    "in_data": outs["in_data"],
                    "gt_data": outs["gt_data"],
                    "model_out": outs["model_out"],
                }

        return sample

    @torch.no_grad()
    def val_epoch(self, epoch: int):
        self.model.eval()

        pbar = utils.make_tqdm_pbar(self.val_loader, desc=f"Val | Epoch {epoch}: ")
        sample = None

        epoch_loss_dict = {}

        for bix, batch in enumerate(pbar, 1):
            outs = self._forward_loss(batch)

            for k, v in outs["loss"].items():
                if k in epoch_loss_dict:
                    epoch_loss_dict[k] += v.detach().cpu().item()
                else:
                    epoch_loss_dict[k] = v.detach().cpu().item()

            if sample is None:
                sample = {
                    "in_data": outs["in_data"],
                    "gt_data": outs["gt_data"],
                    "model_out": outs["model_out"],
                }

        for k in epoch_loss_dict.keys():
            epoch_loss_dict[k] /= max(1, len(self.val_loader))

        self._log_loss(epoch_loss_dict, epoch, is_training=False)
        val_loss = epoch_loss_dict["total"]
        log.info(f"[VAL] epoch={epoch} loss={val_loss:.4f}")

        return sample, val_loss

    def _save_sample(self, sample, tag, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        payload = {}
        for k0, subdict in sample.items():
            for k1, v in subdict.items():
                payload[f"{k0}/{k1}"] = utils.ensure_numpy(v)

        payload = {k: v for k, v in payload.items() if v is not None}

        path = os.path.join(out_dir, f"{tag}_step{self.global_step:06d}.npz")
        np.savez_compressed(path, **payload)

        return path

    def fit(self):
        val_count = 0

        for epoch in range(1, self.max_epochs + 1):
            train_sample = self.train_epoch(epoch)

            if epoch % self.train_sample_interval == 0:
                self._save_sample(train_sample, f"epc{epoch}_train", out_dir="samples")

            if epoch % self.val_interval == 0:
                val_count += 1
                val_sample, val_loss = self.val_epoch(epoch)

                if val_count % self.val_sample_interval == 0:
                    self._save_sample(val_sample, f"epc{epoch}_val", out_dir="samples")

                if val_loss < self.best_val:
                    self.best_val = val_loss

                    utils.save_ckpt(self.model, step=self.global_step, path="ckpt_best.pt")

            utils.save_ckpt(self.model, step=self.global_step, path="ckpt_last.pt")
