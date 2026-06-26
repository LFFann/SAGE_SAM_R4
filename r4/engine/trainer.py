from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from Model.deploy_unet import DeployUNet
from r4.calibration import PromptReliabilityCalibrator, SAMUtilityScheduler
from r4.data.dataset_2d import SegmentationDataset2D, resolve_dataset_root
from r4.data.paired_sampler import paired_batches
from r4.data.split import create_train_calibration_split
from r4.engine.checkpoint import export_deploy_payload, save_checkpoint
from r4.engine.evaluator import evaluate
from r4.engine.logger import OneLineProgress, append_jsonl, setup_logger
from r4.losses.boundary_losses import boundary_bce_loss
from r4.losses.sam_adapter_losses import gated_soft_sam_loss, sam_ce_dice_loss, sam_student_kd_loss
from r4.losses.set_valued_losses import set_valued_supervision_loss
from r4.losses.supervised import supervised_loss
from r4.models.dual_temporal_teacher import DualTemporalTeacher
from r4.models.promptable_sam_mentor import PromptableSAMMentor
from r4.models.real_sam_wrapper import RealSAMWrapper
from r4.ssl.adaptive_ultrasound_augmentation import make_weak_strong_views
from r4.ssl.online_sam_relation import online_sam_student_relation_loss
from r4.ssl.target_builder import build_set_valued_targets


class SAGESAMR4Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["checkpoints", "predictions", "visualizations", "calibration"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.output_dir)
        train_cfg = config["train"]
        dev = train_cfg.get("device", "cpu")
        if dev == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but unavailable; falling back to CPU")
            dev = "cpu"
        self.device = torch.device(dev)
        data_cfg = config["data"]
        model_cfg = config["model"]
        self.num_classes = int(data_cfg["num_classes"])
        self.ignore_index = int(data_cfg.get("ignore_index", 255))
        self.student = DeployUNet(
            in_channels=int(data_cfg.get("in_channels", 3)),
            num_classes=self.num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
            use_boundary_head=bool(model_cfg.get("use_boundary_head", True)),
            complementary_dropout_p=float(model_cfg.get("complementary_dropout_p", 0.2)),
        ).to(self.device)
        self.dual_teacher = DualTemporalTeacher(
            self.student,
            fast_decay=config["teacher"].get("fast_ema_decay", 0.99),
            slow_decay=config["teacher"].get("slow_ema_decay", 0.999),
            use_bn_eval=config["teacher"].get("use_bn_eval_for_teacher", True),
        ).to(self.device)

        sam_cfg = config.get("sam", {})
        self.use_sam = bool(sam_cfg.get("use_sam", False))
        self.mentor: PromptableSAMMentor | None = None
        if self.use_sam:
            wrapper = RealSAMWrapper(
                sam_cfg["model_type"],
                sam_cfg["checkpoint"],
                sam_cfg.get("device", str(self.device)),
                sam_cfg.get("image_size", 1024),
                in_channels=data_cfg.get("in_channels", 3),
                num_classes=data_cfg.get("num_classes", 3),
                train_peft=sam_cfg.get("train_peft", True),
                peft_type=sam_cfg.get("peft_type", "adapter"),
                train_mask_decoder=sam_cfg.get("train_mask_decoder", True),
                train_prompt_encoder=not sam_cfg.get("freeze_prompt_encoder", True),
                train_last_n_blocks=sam_cfg.get("train_last_n_blocks", 0),
                max_trainable_ratio=sam_cfg.get("max_trainable_ratio", 0.05),
            )
            if not wrapper.sam_is_real():
                raise RuntimeError("SAM did not load as a real model")
            self.mentor = PromptableSAMMentor(
                wrapper,
                num_classes=self.num_classes,
                config=sam_cfg,
                in_channels=data_cfg.get("in_channels", 3),
            ).to(self.device)

        cal_cfg = config.get("calibration", config.get("conformal", {}))
        self.calibrator = PromptReliabilityCalibrator(
            self.num_classes,
            start_iter=cal_cfg.get("start_iter", cal_cfg.get("calibrator_start_iter", 500)),
            update_every=cal_cfg.get("update_every", 250),
            momentum=cal_cfg.get("momentum", 0.8),
            min_pixels_per_class=cal_cfg.get("min_pixels_per_class", 128),
        )
        self.sam_utility = SAMUtilityScheduler(
            max_weight=sam_cfg.get("losses", {}).get("sam_student_kd_weight", sam_cfg.get("semantic_kd_max_weight", 0.15)),
            ema_decay=sam_cfg.get("utility_ema_decay", 0.9),
            disable_after_no_gain=sam_cfg.get("disable_semantic_kd_after_no_gain", 3),
        )
        self.optimizer = self._build_optimizer()
        self.trainable_parameters = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad]
        self.amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.amp)
        self.best_metrics = {"avg_dice": -1.0, "avg_hd95": float("inf")}
        self._log_trainability()
        self._build_data()

    def _build_optimizer(self):
        train_cfg = self.config["train"]
        base_lr = float(train_cfg.get("lr", 1e-3))
        groups = [{"params": list(self.student.parameters()), "lr": base_lr, "name": "student"}]
        sam_cfg = self.config.get("sam", {})
        if self.use_sam and self.mentor is not None:
            groups.extend(self.mentor.optimizer_param_groups(base_lr, sam_cfg))
            if sam_cfg.get("train_peft", True) and not any(str(g.get("name", "")).startswith("sam_") for g in groups):
                raise RuntimeError("sam.train_peft=true but optimizer has no SAM parameter group")
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=train_cfg.get("weight_decay", 1e-4))

    def _log_trainability(self):
        group_names = [str(g.get("name", f"group{idx}")) for idx, g in enumerate(self.optimizer.param_groups)]
        self.logger.info("optimizer_param_groups=%s", group_names)
        if self.mentor is None:
            return
        report = self.mentor.trainability_report()
        self.logger.info(
            "sam_trainability total=%s trainable=%s ratio=%.6f prompt_generator=%s modules=%s",
            report.get("total_sam_params"),
            report.get("trainable_sam_params"),
            report.get("trainable_sam_ratio", 0.0),
            report.get("trainable_prompt_generator_params"),
            report.get("trainable_sam_modules"),
        )
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "sam_trainability", **report})

    def _build_data(self):
        cfg = self.config["data"]
        root = resolve_dataset_root(
            cfg["root"],
            cfg.get("dataset_name"),
            cfg.get("labeled_subdir", "labeled"),
            cfg.get("image_dir_name", "image"),
        )
        self.config["data"]["resolved_root"] = str(root)
        self.logger.info("dataset_root=%s", root)
        common = dict(
            root=root,
            num_classes=cfg["num_classes"],
            image_size=cfg["image_size"],
            image_dir_name=cfg.get("image_dir_name", "image"),
            mask_dir_name=cfg.get("mask_dir_name", "mask"),
            ignore_index=cfg.get("ignore_index", 255),
        )
        labeled_all = SegmentationDataset2D(split=cfg.get("labeled_subdir", "labeled"), has_mask=True, **common)
        train_idx, cal_idx, shared = create_train_calibration_split(
            labeled_all.records,
            cfg.get("calibration_ratio", 0.15),
            cfg.get("calibration_min_images", 4),
            cfg.get("calibration_split_seed", 2026),
        )
        if shared:
            self.logger.warning("Calibration split shares labeled samples because labeled count is too small")
        self.labeled_ds = Subset(labeled_all, train_idx)
        self.calibration_ds = Subset(labeled_all, cal_idx)
        self.unlabeled_ds = SegmentationDataset2D(split=cfg.get("unlabeled_subdir", "unlabeled"), has_mask=False, **common)
        self.val_ds = SegmentationDataset2D(split=cfg.get("val_subdir", "val"), has_mask=True, **common)
        self.test_ds = SegmentationDataset2D(split=cfg.get("test_subdir", "test"), has_mask=True, **common)
        train_cfg = self.config["train"]
        self.labeled_loader = DataLoader(
            self.labeled_ds,
            batch_size=train_cfg.get("batch_size_labeled", 2),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
            drop_last=False,
        )
        self.unlabeled_loader = DataLoader(
            self.unlabeled_ds,
            batch_size=train_cfg.get("batch_size_unlabeled", 2),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
            drop_last=False,
        )
        self.val_loader = DataLoader(self.val_ds, batch_size=self.config.get("eval", {}).get("batch_size", 1), shuffle=False, num_workers=0)
        self.calibration_loader = DataLoader(self.calibration_ds, batch_size=self.config.get("eval", {}).get("batch_size", 1), shuffle=False, num_workers=0)

    def fit_calibrator(self):
        self.logger.info("Prompt reliability calibration is online; skipping random pre-training fit")
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "online_calibrator_waiting", "iteration": 0})

    def dry_run(self):
        batch_l = next(iter(self.labeled_loader))
        batch_u = next(iter(self.unlabeled_loader))
        result = self.train_one_iter(batch_l, batch_u, iteration=0, update=False)
        self.logger.info("dry-run ok: %s", result)
        return result

    def train(self, max_iterations: int | None = None):
        max_iter = int(max_iterations or self.config["train"].get("max_iterations", 1))
        pair_iter = paired_batches(self.labeled_loader, self.unlabeled_loader)
        progress = OneLineProgress(max_iter)
        self.logger.info("train_start max_iterations=%d output_dir=%s", max_iter, self.output_dir)
        for iteration in range(1, max_iter + 1):
            batch_l, batch_u = next(pair_iter)
            logs = self.train_one_iter(batch_l, batch_u, iteration=iteration, update=True)
            progress.update(
                iteration,
                loss=logs["loss_total"],
                sup=logs["loss_sup"],
                set=logs["loss_set"],
                lr=logs["lr"],
                sam=logs["sam_valid_ratio"],
            )
            if iteration % int(self.config["train"].get("log_every", 20)) == 0 or iteration == 1 or iteration == max_iter:
                append_jsonl(self.output_dir / "metrics.jsonl", {"iteration": iteration, "phase": "train", **logs})
                self.logger.info(
                    "train iter=%d loss=%.6f sup=%.6f set=%.6f sam_sup=%.6f sam_kd=%.6f prompt_q=%.4f adapter_grad=%.6f lr=%.6g",
                    iteration,
                    logs["loss_total"],
                    logs["loss_sup"],
                    logs["loss_set"],
                    logs["loss_sam_sup"],
                    logs["loss_sam_kd"],
                    logs["prompt_quality"],
                    logs["sam_adapter_grad_norm"],
                    logs["lr"],
                )
            if iteration % int(self.config["train"].get("val_every", 250)) == 0 or iteration == max_iter:
                metrics = self.validate(iteration)
                progress.update(
                    iteration,
                    loss=logs["loss_total"],
                    sup=logs["loss_sup"],
                    set=logs["loss_set"],
                    dice=metrics["avg_dice"],
                    lr=logs["lr"],
                    sam=logs["sam_valid_ratio"],
                )
        progress.close()
        latest = self.output_dir / "checkpoints" / "latest.pth"
        export_deploy_payload(latest, self.output_dir / "checkpoints" / "deploy_student.pth")
        self.logger.info("train_end latest=%s deploy=%s", latest, self.output_dir / "checkpoints" / "deploy_student.pth")
        return latest

    def train_one_iter(self, batch_l, batch_u, iteration: int, update: bool = True):
        self.student.train()
        if self.mentor is not None:
            self.mentor.train()
        x_l = batch_l["image"].to(self.device)
        y_l = batch_l["mask"].to(self.device)
        x_u = batch_u["image"].to(self.device)
        aug_cfg = self.config.get("augmentation", {})
        x_u_w, x_u_s1, x_u_s2, _, _ = make_weak_strong_views(
            x_u,
            strong_kwargs=aug_cfg.get("strong", {}),
            weak_kwargs=aug_cfg.get("weak", {}),
        )
        with torch.no_grad():
            teacher_out = self.dual_teacher.predict_weak(x_u_w)
            student_w_prob = torch.softmax(self.student(x_u_w), dim=1) if self.use_sam else None

        pseudo_cfg = self.config.get("pseudo", {})
        sam_loss_cfg = self.config.get("sam", {}).get("losses", {})
        structure_cfg = self.config.get("structure", {})
        sam_l = {"valid": False}
        sam_u = {"valid": False}
        with autocast(device_type=self.device.type, enabled=self.amp):
            out_l = self.student(x_l, return_features=True)
            loss_sup, sup_logs = supervised_loss(out_l["logits"], y_l, self.num_classes, self.ignore_index)

            loss_sam_sup = x_l.new_tensor(0.0)
            loss_sam_unsup = x_l.new_tensor(0.0)
            loss_kd = x_l.new_tensor(0.0)
            loss_relation = x_l.new_tensor(0.0)
            loss_boundary = x_l.new_tensor(0.0)
            if self.use_sam and self.mentor is not None:
                sam_l = self.mentor.forward_labeled(x_l, y_l)
                loss_sam_sup = sam_ce_dice_loss(sam_l["sam_prob"], y_l, self.num_classes, self.ignore_index)
                sam_u = self.mentor.forward_unlabeled(x_u_w, teacher_out["mean_prob"], student_w_prob)

            targets = build_set_valued_targets(teacher_out, sam_u, self.calibrator, pseudo_cfg)
            out_s1 = self.student(x_u_s1, return_features=True)
            out_s2 = self.student(x_u_s2, return_features=True, feature_dropout="complementary")
            ssl1 = set_valued_supervision_loss(out_s1["logits"], targets, pseudo_cfg.get("rank_margin", 0.5))
            ssl2 = set_valued_supervision_loss(out_s2["logits"], targets, pseudo_cfg.get("rank_margin", 0.5))
            ramp = min(1.0, iteration / max(1, int(self.config["train"].get("unsup_ramp_iterations", 1))))
            loss_unsup = (
                pseudo_cfg.get("singleton_weight", 1.0) * (ssl1["loss_singleton"] + ssl2["loss_singleton"]) * 0.5
                + pseudo_cfg.get("set_weight", 0.5) * (ssl1["loss_set"] + ssl2["loss_set"]) * 0.5
                + pseudo_cfg.get("rank_weight", 0.1) * (ssl1["loss_rank"] + ssl2["loss_rank"]) * 0.5
                + pseudo_cfg.get("negative_weight", 0.1) * (ssl1["loss_negative"] + ssl2["loss_negative"]) * 0.5
                + pseudo_cfg.get("fuzzy_weight", 0.25) * (ssl1["loss_fuzzy"] + ssl2["loss_fuzzy"]) * 0.5
            )

            if self.use_sam and sam_u.get("valid"):
                loss_kd = sam_student_kd_loss(
                    out_s1["logits"],
                    sam_u["sam_prob"],
                    gate=targets["semantic_gate"],
                    temperature=float(self.config.get("sam", {}).get("kd_temperature", 1.0)),
                )
                loss_sam_unsup = gated_soft_sam_loss(
                    sam_u["sam_prob"],
                    targets["soft_target"],
                    gate=targets["sam_train_gate"],
                )
                if structure_cfg.get("use_online_relation", True):
                    loss_relation = online_sam_student_relation_loss(
                        out_s1["bottleneck"],
                        sam_u.get("sam_embedding"),
                        gate=targets["structure_gate"],
                        topk=structure_cfg.get("online_topk", self.config.get("sam", {}).get("topk_edges", 8)),
                        resolution=structure_cfg.get("relation_resolution", 16),
                    )
                if out_s1.get("boundary_logits") is not None:
                    boundary_target = sam_u["sam_boundary"].detach() * targets["structure_gate"].unsqueeze(1).float()
                    loss_boundary = boundary_bce_loss(out_s1["boundary_logits"], boundary_target)

            loss = (
                loss_sup
                + sam_loss_cfg.get("sam_sup_weight", self.config.get("sam", {}).get("sam_sup_weight", 0.5)) * loss_sam_sup
                + ramp
                * (
                    loss_unsup
                    + sam_loss_cfg.get("sam_unsup_weight", 0.2) * loss_sam_unsup
                    + sam_loss_cfg.get("sam_student_kd_weight", self.sam_utility.semantic_weight(iteration)) * loss_kd
                    + sam_loss_cfg.get("sam_relation_weight", pseudo_cfg.get("relation_weight", 0.05)) * loss_relation
                    + sam_loss_cfg.get("sam_boundary_weight", pseudo_cfg.get("boundary_weight", 0.05)) * loss_boundary
                )
            )

        sam_grad_norm = 0.0
        if update:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.config["train"].get("grad_clip_norm"):
                self.scaler.unscale_(self.optimizer)
                sam_grad_norm = self.mentor.sam_grad_norm() if self.mentor is not None else 0.0
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.config["train"]["grad_clip_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.dual_teacher.update_fast(self.student)
            if iteration % int(self.config["teacher"].get("slow_refresh_every", 500)) == 0:
                self.dual_teacher.refresh_slow(self.student)
                append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "slow_teacher_refresh", "iteration": iteration})
            self._maybe_update_prompt_calibrator(iteration, out_l, y_l, sam_l)

        prompt_quality = 0.0
        if sam_u.get("valid") and sam_u.get("prompt_quality") is not None:
            prompt_quality = float(sam_u["prompt_quality"].detach().mean())
        logs = {
            "loss_total": float(loss.detach()),
            "loss_sup": float(loss_sup.detach()),
            "loss_singleton": float(ssl1["loss_singleton"].detach()),
            "loss_set": float(ssl1["loss_set"].detach()),
            "loss_rank": float(ssl1["loss_rank"].detach()),
            "loss_negative": float(ssl1["loss_negative"].detach()),
            "loss_fuzzy": float(ssl1["loss_fuzzy"].detach()),
            "loss_relation": float(loss_relation.detach()),
            "loss_boundary": float(loss_boundary.detach()),
            "loss_sam_sup": float(loss_sam_sup.detach()),
            "loss_sam_unsup": float(loss_sam_unsup.detach()),
            "loss_sam_kd": float(loss_kd.detach()),
            "loss_sam_sem": float(loss_kd.detach()),
            "unsup_weight": ramp,
            "sam_semantic_weight": self.sam_utility.semantic_weight(iteration),
            "fast_slow_agreement": float(teacher_out["agreement"].detach()),
            "sam_valid_ratio": 1.0 if sam_u.get("valid") else 0.0,
            "prompt_quality": prompt_quality,
            "sam_adapter_grad_norm": sam_grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
            "gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if self.device.type == "cuda" else 0.0,
            **targets["stats"],
            **sup_logs,
        }
        return logs

    @torch.no_grad()
    def _maybe_update_prompt_calibrator(self, iteration: int, out_l: dict, y_l: torch.Tensor, sam_l: dict):
        if not (self.use_sam and sam_l.get("valid") and self.calibrator.should_update(iteration)):
            return
        student_prob_l = torch.softmax(out_l["logits"].detach(), dim=1)
        self.calibrator.update_from_batch(
            teacher_prob=student_prob_l,
            sam_prob=sam_l["sam_prob"].detach(),
            sam_iou=sam_l.get("sam_iou"),
            prompt_quality=sam_l.get("prompt_quality"),
            gt=y_l.detach(),
        )
        append_jsonl(
            self.output_dir / "diagnostics.jsonl",
            {
                "event": "prompt_reliability_update",
                "iteration": iteration,
                "teacher_q": self.calibrator.teacher_q.tolist(),
                "sam_q": self.calibrator.sam_q.tolist(),
                "prompt_q": self.calibrator.prompt_q.tolist(),
            },
        )

    def validate(self, iteration: int):
        metrics = evaluate(
            self.student,
            self.val_loader,
            self.num_classes,
            self.device,
            compute_hd95=self.config.get("eval", {}).get("compute_hd95", True),
            save_dir=None,
            ignore_index=self.ignore_index,
        )
        row = {"iteration": iteration, "phase": "val", **metrics}
        append_jsonl(self.output_dir / "metrics.jsonl", row)
        ckpt_dir = self.output_dir / "checkpoints"
        latest = save_checkpoint(
            ckpt_dir / "latest.pth",
            iteration=iteration,
            student=self.student,
            fast_teacher=self.dual_teacher.fast,
            slow_teacher=self.dual_teacher.slow,
            optimizer=self.optimizer,
            scaler=self.scaler,
            calibrator=self.calibrator,
            sam_utility=self.sam_utility,
            mentor=self.mentor,
            config=self.config,
            best_metrics=self.best_metrics,
        )
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "checkpoint_saved", "iteration": iteration, "path": str(latest)})
        if metrics["avg_dice"] >= self.best_metrics.get("avg_dice", -1):
            self.best_metrics["avg_dice"] = metrics["avg_dice"]
            save_checkpoint(
                ckpt_dir / "best_val_dice.pth",
                iteration=iteration,
                student=self.student,
                fast_teacher=self.dual_teacher.fast,
                slow_teacher=self.dual_teacher.slow,
                optimizer=self.optimizer,
                scaler=self.scaler,
                calibrator=self.calibrator,
                sam_utility=self.sam_utility,
                mentor=self.mentor,
                config=self.config,
                best_metrics=self.best_metrics,
            )
            append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "best_updated", "metric": "avg_dice", "iteration": iteration})
        hd = metrics.get("avg_hd95", float("inf"))
        hd_key = hd if not math.isnan(hd) else float("inf")
        if hd_key <= self.best_metrics.get("avg_hd95", float("inf")):
            self.best_metrics["avg_hd95"] = hd_key
            save_checkpoint(
                ckpt_dir / "best_val_hd95.pth",
                iteration=iteration,
                student=self.student,
                fast_teacher=self.dual_teacher.fast,
                slow_teacher=self.dual_teacher.slow,
                optimizer=self.optimizer,
                scaler=self.scaler,
                calibrator=self.calibrator,
                sam_utility=self.sam_utility,
                mentor=self.mentor,
                config=self.config,
                best_metrics=self.best_metrics,
            )
        self.logger.info("val iter=%d avg_dice=%.4f avg_iou=%.4f", iteration, metrics["avg_dice"], metrics["avg_iou"])
        return metrics
