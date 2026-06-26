from __future__ import annotations

import math
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from torch.amp import GradScaler as _AmpGradScaler
except ImportError:
    _AmpGradScaler = None

try:
    from torch.amp import autocast as _amp_autocast
except ImportError:
    _amp_autocast = None

try:
    from torch.cuda.amp import GradScaler as _CudaGradScaler
    from torch.cuda.amp import autocast as _cuda_autocast
except ImportError:
    _CudaGradScaler = None
    _cuda_autocast = None

from r4.calibration import PromptReliabilityCalibrator, SAMUtilityScheduler
from r4.data.dataset_2d import SegmentationDataset2D, resolve_dataset_root
from r4.data.paired_sampler import paired_batches
from r4.data.split import create_train_calibration_split
from r4.engine.checkpoint import export_deploy_payload, safe_load, save_checkpoint
from r4.engine.evaluator import evaluate
from r4.engine.logger import OneLineProgress, append_jsonl, setup_logger
from r4.engine.model_factory import build_deploy_model
from r4.losses.boundary_losses import boundary_bce_loss
from r4.losses.sam_adapter_losses import gated_soft_sam_loss, sam_ce_dice_loss, sam_student_kd_loss
from r4.losses.set_valued_losses import set_valued_supervision_loss
from r4.losses.supervised import supervised_loss
from r4.models.dual_temporal_teacher import DualTemporalTeacher
from r4.models.promptable_sam_mentor import PromptableSAMMentor
from r4.models.real_sam_wrapper import RealSAMWrapper
from r4.ssl.adaptive_ultrasound_augmentation import make_weak_strong_views
from r4.ssl.correlation_propagation import correlation_propagation_loss, propagate_correlation_targets
from r4.ssl.online_sam_relation import online_sam_student_relation_loss
from r4.ssl.target_builder import build_set_valued_targets


class _NoOpGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def make_grad_scaler(device_type: str, enabled: bool):
    enabled = bool(enabled) and str(device_type) == "cuda"
    if _AmpGradScaler is not None:
        try:
            return _AmpGradScaler(str(device_type), enabled=enabled)
        except TypeError:
            return _AmpGradScaler(enabled=enabled)
    if _CudaGradScaler is not None:
        return _CudaGradScaler(enabled=enabled)
    return _NoOpGradScaler()


def amp_autocast(device_type: str, enabled: bool):
    enabled = bool(enabled) and str(device_type) == "cuda"
    if not enabled:
        return nullcontext()
    if _amp_autocast is not None:
        try:
            return _amp_autocast(device_type=str(device_type), enabled=True)
        except TypeError:
            return _amp_autocast(enabled=True)
    if _cuda_autocast is not None:
        return _cuda_autocast(enabled=True)
    return nullcontext()


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
        self.student = build_deploy_model(config).to(self.device)
        self.logger.info("deploy_model=%s backbone=%s", self.student.__class__.__name__, model_cfg.get("deploy_backbone", "dual_fusion"))
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
                lora_rank=sam_cfg.get("lora_rank", 4),
                lora_alpha=sam_cfg.get("lora_alpha", 8),
                adapter_dim=sam_cfg.get("adapter_dim", 32),
                adapter_scale=sam_cfg.get("adapter_scale", 1.0),
                max_trainable_ratio=sam_cfg.get("max_trainable_ratio", 0.05),
                use_mask_prompt=sam_cfg.get("prompt", {}).get("use_mask_prompt", True),
                use_box_prompt=sam_cfg.get("prompt", {}).get("use_box_prompt", True),
                use_point_prompt=sam_cfg.get("prompt", {}).get("use_point_prompt", True),
                use_negative_points=sam_cfg.get("prompt", {}).get("use_negative_points", True),
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
        calibrator_start_iter = cal_cfg.get("start_iter", cal_cfg.get("calibrator_start_iter", 500))
        if cal_cfg.get("update_only_after_warmup", False):
            calibrator_start_iter = max(int(calibrator_start_iter), int(train_cfg.get("warmup_iterations", 0)))
        self.calibrator = PromptReliabilityCalibrator(
            self.num_classes,
            start_iter=calibrator_start_iter,
            update_every=cal_cfg.get("update_every", 250),
            momentum=cal_cfg.get("momentum", 0.8),
            min_pixels_per_class=cal_cfg.get("min_pixels_per_class", 128),
            use_soft_gate=cal_cfg.get("use_soft_gate", True),
            min_participation_ratio=cal_cfg.get("min_participation_ratio", 0.0),
            max_quantile_clip=cal_cfg.get("max_quantile_clip", 0.97),
            temperature=cal_cfg.get("temperature", 0.05),
            coverage_target=cal_cfg.get("coverage_target"),
        )
        self.sam_utility = SAMUtilityScheduler(
            max_weight=sam_cfg.get("losses", {}).get("sam_student_kd_weight", sam_cfg.get("semantic_kd_max_weight", 0.15)),
            ema_decay=sam_cfg.get("utility_ema_decay", 0.9),
            disable_after_no_gain=sam_cfg.get("disable_semantic_kd_after_no_gain", 3),
        )
        self.optimizer = self._build_optimizer()
        self.trainable_parameters = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad]
        self.amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = make_grad_scaler(self.device.type, self.amp)
        self.grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation", 1)))
        self.best_metrics = {"avg_dice": -1.0, "avg_hd95": float("inf")}
        self.start_iteration = 0
        self.calibration_iter = None
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
            "sam_trainability total=%s trainable=%s ratio=%.6f lora=%s adapter=%s mask_decoder=%s prompt_generator_trainable=%s prompt_generator_params=%s modules=%s",
            report.get("total_sam_params"),
            report.get("trainable_sam_params"),
            report.get("trainable_sam_ratio", 0.0),
            report.get("lora_param_count"),
            report.get("adapter_param_count"),
            report.get("mask_decoder_trainable"),
            report.get("prompt_generator_trainable"),
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
        self.calibration_iter = cycle(self.calibration_loader)

    def fit_calibrator(self):
        self.logger.info("Prompt reliability calibration is online; skipping random pre-training fit")
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "online_calibrator_waiting", "iteration": 0})

    def dry_run(self):
        batch_l = next(iter(self.labeled_loader))
        batch_u = next(iter(self.unlabeled_loader))
        result = self.train_one_iter(batch_l, batch_u, iteration=0, update=False)
        self.logger.info("dry-run ok: %s", result)
        return result

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        payload = safe_load(checkpoint_path, map_location="cpu")
        student_state = payload["student"]
        if hasattr(self.student, "branch_a") and student_state and not any(str(k).startswith("branch_") for k in student_state):
            student_state = {f"branch_a.{k}": v for k, v in student_state.items()}
            self.logger.warning("Loaded legacy single-branch student weights into dual-fusion branch_a only")
        report = self.student.load_state_dict(student_state, strict=False)
        if report.missing_keys or report.unexpected_keys:
            self.logger.warning("student_resume_report missing=%s unexpected=%s", report.missing_keys, report.unexpected_keys)
        if payload.get("fast_teacher") is not None:
            self.dual_teacher.fast.load_state_dict(payload["fast_teacher"], strict=False)
        if payload.get("slow_teacher") is not None:
            self.dual_teacher.slow.load_state_dict(payload["slow_teacher"], strict=False)
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
            self._move_optimizer_state_to_device()
        if payload.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(payload["scaler"])
            except Exception as exc:
                self.logger.warning("Skipping scaler state from %s: %s", checkpoint_path, exc)
        if payload.get("calibrator") is not None:
            self.calibrator.load_state_dict(payload["calibrator"])
        if payload.get("sam_utility") is not None:
            self.sam_utility.load_state_dict(payload["sam_utility"])
        if self.mentor is not None and payload.get("mentor") is not None:
            report = self.mentor.load_trainable_state_dict(payload["mentor"])
            if report.get("missing") or report.get("unexpected"):
                self.logger.warning("mentor_resume_report=%s", report)
        if payload.get("best_metrics") is not None:
            self.best_metrics = payload["best_metrics"]
        self.start_iteration = int(payload.get("iteration", 0))
        append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "checkpoint_resumed", "iteration": self.start_iteration, "path": str(checkpoint_path)})
        self.logger.info("resumed checkpoint=%s iteration=%d", checkpoint_path, self.start_iteration)
        return payload

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def train(self, max_iterations: int | None = None):
        max_iter = int(max_iterations or self.config["train"].get("max_iterations", 1))
        pair_iter = paired_batches(self.labeled_loader, self.unlabeled_loader)
        progress = OneLineProgress(max_iter)
        last_val_metrics = {}
        self.logger.info("train_start max_iterations=%d output_dir=%s", max_iter, self.output_dir)
        if self.start_iteration >= max_iter:
            self.logger.warning("resume iteration %d already reaches max_iterations=%d", self.start_iteration, max_iter)
            return self.output_dir / "checkpoints" / "latest.pth"
        for iteration in range(self.start_iteration + 1, max_iter + 1):
            batch_l, batch_u = next(pair_iter)
            step_optimizer = iteration % self.grad_accum_steps == 0 or iteration == max_iter
            logs = self.train_one_iter(batch_l, batch_u, iteration=iteration, update=True, step_optimizer=step_optimizer)
            progress.update(
                iteration,
                loss=logs["loss_total"],
                sup=logs["loss_sup"],
                set=logs["loss_set"],
                lr=logs["lr"],
                sam=logs["sam_valid_ratio"],
                **last_val_metrics,
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
                last_val_metrics = {
                    "val_dice": metrics.get("avg_dice"),
                    "val_iou": metrics.get("avg_iou"),
                    "val_hd95": metrics.get("avg_hd95"),
                }
                progress.update(
                    iteration,
                    loss=logs["loss_total"],
                    sup=logs["loss_sup"],
                    set=logs["loss_set"],
                    lr=logs["lr"],
                    sam=logs["sam_valid_ratio"],
                    **last_val_metrics,
                )
        progress.close()
        latest = self.output_dir / "checkpoints" / "latest.pth"
        export_deploy_payload(latest, self.output_dir / "checkpoints" / "deploy_student.pth")
        self.logger.info("train_end latest=%s deploy=%s", latest, self.output_dir / "checkpoints" / "deploy_student.pth")
        return latest

    def train_one_iter(self, batch_l, batch_u, iteration: int, update: bool = True, step_optimizer: bool = True):
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
        with amp_autocast(self.device.type, self.amp):
            out_l = self.student(x_l, return_features=True)
            loss_sup_fusion, sup_logs = supervised_loss(out_l["logits"], y_l, self.num_classes, self.ignore_index)
            loss_sup_a = self._supervised_branch_loss(out_l, "logits_a", y_l)
            loss_sup_b = self._supervised_branch_loss(out_l, "logits_b", y_l)
            loss_cfg = self.config.get("losses", {})
            branch_sup_weight = float(loss_cfg.get("branch_sup_weight", 0.5))
            fusion_sup_weight = float(loss_cfg.get("fusion_sup_weight", 1.0))
            if "logits_a" in out_l and "logits_b" in out_l:
                loss_sup = fusion_sup_weight * loss_sup_fusion + branch_sup_weight * 0.5 * (loss_sup_a + loss_sup_b)
            else:
                loss_sup = loss_sup_fusion

            loss_sam_sup = x_l.new_tensor(0.0)
            loss_sam_unsup = x_l.new_tensor(0.0)
            loss_kd = x_l.new_tensor(0.0)
            loss_relation = x_l.new_tensor(0.0)
            loss_boundary = x_l.new_tensor(0.0)
            loss_corr = x_l.new_tensor(0.0)
            loss_conflict = x_l.new_tensor(0.0)
            if self.use_sam and self.mentor is not None:
                sam_l = self.mentor.forward_labeled(x_l, y_l)
                loss_sam_sup = sam_ce_dice_loss(sam_l["sam_prob"], y_l, self.num_classes, self.ignore_index)
                sam_u = self.mentor.forward_unlabeled(x_u_w, teacher_out["mean_prob"], student_w_prob)

            targets = build_set_valued_targets(teacher_out, sam_u, self.calibrator, pseudo_cfg)
            out_s1 = self.student(x_u_s1, return_features=True)
            out_s2 = self.student(x_u_s2, return_features=True, feature_dropout="complementary")
            ssl1 = set_valued_supervision_loss(out_s1["logits"], targets, pseudo_cfg.get("rank_margin", 0.5))
            ssl2 = set_valued_supervision_loss(out_s2["logits"], targets, pseudo_cfg.get("rank_margin", 0.5))
            branch_ssl = self._branch_ssl_loss(out_s1, out_s2, targets, pseudo_cfg)
            loss_conflict = 0.5 * (
                self._dual_consistency_loss(out_s1, targets) + self._dual_consistency_loss(out_s2, targets)
            )
            ramp = min(1.0, iteration / max(1, int(self.config["train"].get("unsup_ramp_iterations", 1))))
            loss_unsup = (
                pseudo_cfg.get("singleton_weight", 1.0) * (ssl1["loss_singleton"] + ssl2["loss_singleton"]) * 0.5
                + pseudo_cfg.get("set_weight", 0.5) * (ssl1["loss_set"] + ssl2["loss_set"]) * 0.5
                + pseudo_cfg.get("rank_weight", 0.1) * (ssl1["loss_rank"] + ssl2["loss_rank"]) * 0.5
                + pseudo_cfg.get("negative_weight", 0.1) * (ssl1["loss_negative"] + ssl2["loss_negative"]) * 0.5
                + pseudo_cfg.get("fuzzy_weight", 0.25) * (ssl1["loss_fuzzy"] + ssl2["loss_fuzzy"]) * 0.5
                + float(loss_cfg.get("branch_ssl_weight", 0.5)) * branch_ssl
            )
            if float(pseudo_cfg.get("correlation_weight", 0.0)) > 0.0 and out_s1.get("fusion_feature") is not None:
                sam_shape = None
                if sam_u.get("valid"):
                    sam_shape = sam_u.get("sam_boundary")
                    if sam_shape is None and sam_u.get("sam_prob") is not None:
                        sam_shape = sam_u["sam_prob"].detach().max(dim=1, keepdim=True).values
                propagated = propagate_correlation_targets(
                    out_s1["fusion_feature"],
                    torch.softmax(out_s1["logits"].detach(), dim=1),
                    sam_shape=sam_shape,
                    reliable_mask=targets["singleton_mask"] | targets["semantic_gate"],
                    resolution=structure_cfg.get("correlation_resolution", structure_cfg.get("relation_resolution", 16)),
                    topk=structure_cfg.get("correlation_topk", structure_cfg.get("online_topk", 8)),
                    temperature=structure_cfg.get("correlation_temperature", structure_cfg.get("relation_temperature", 0.2)),
                    min_weight=structure_cfg.get("correlation_min_weight", 0.15),
                )
                loss_corr = correlation_propagation_loss(out_s1["logits"], propagated)

            if self.use_sam and sam_u.get("valid"):
                sam_gate_weight = targets["sam_weight"] * targets["teacher_reliable_mask"].float()
                sam_utility_value = float((sam_u["sam_prob"].detach() * targets["teacher_only_soft_target"]).sum(dim=1).mean())
                self.sam_utility.update(sam_utility_value)
                loss_kd = sam_student_kd_loss(
                    out_s1["logits"],
                    sam_u["sam_prob"],
                    gate=sam_gate_weight,
                    temperature=float(self.config.get("sam", {}).get("kd_temperature", 1.0)),
                )
                loss_sam_unsup = gated_soft_sam_loss(
                    sam_u["sam_prob"],
                    targets["teacher_only_soft_target"],
                    gate=sam_gate_weight,
                )
                if structure_cfg.get("use_online_relation", True):
                    loss_relation = online_sam_student_relation_loss(
                        out_s1["bottleneck"],
                        sam_u.get("sam_embedding"),
                        gate=targets["structure_gate"],
                        boundary=sam_u.get("sam_boundary"),
                        topk=structure_cfg.get("online_topk", 8),
                        resolution=structure_cfg.get("relation_resolution", 16),
                        temperature=structure_cfg.get("relation_temperature", 0.2),
                        rank_weight=structure_cfg.get("relation_rank_weight", 0.25),
                    )
                if out_s1.get("boundary_logits") is not None:
                    boundary_target = sam_u["sam_boundary"].detach() * targets["structure_weight"].unsqueeze(1).float()
                    loss_boundary = boundary_bce_loss(out_s1["boundary_logits"], boundary_target)

            sam_scale = self._sam_self_reliance_scale(iteration)
            sam_sup_scale = 1.0 if self.config.get("sam", {}).get("keep_labeled_sam_sup", True) else sam_scale
            loss = (
                loss_sup
                + sam_sup_scale * sam_loss_cfg.get("sam_sup_weight", self.config.get("sam", {}).get("sam_sup_weight", 0.5)) * loss_sam_sup
                + ramp
                * (
                    loss_unsup
                    + float(loss_cfg.get("conflict_review_weight", 0.2)) * loss_conflict
                    + pseudo_cfg.get("correlation_weight", 0.0) * loss_corr
                    + sam_scale
                    * (
                        sam_loss_cfg.get("sam_unsup_weight", 0.2) * loss_sam_unsup
                        + sam_loss_cfg.get("sam_student_kd_weight", self.sam_utility.semantic_weight(iteration)) * loss_kd
                        + sam_loss_cfg.get("sam_relation_weight", pseudo_cfg.get("relation_weight", 0.05)) * loss_relation
                        + sam_loss_cfg.get("sam_boundary_weight", pseudo_cfg.get("boundary_weight", 0.05)) * loss_boundary
                    )
                )
            )

        sam_grad_norm = 0.0
        if update:
            if (iteration - 1) % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss / self.grad_accum_steps).backward()
            if step_optimizer and self.config["train"].get("grad_clip_norm"):
                self.scaler.unscale_(self.optimizer)
                sam_grad_norm = self.mentor.sam_grad_norm() if self.mentor is not None else 0.0
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.config["train"]["grad_clip_norm"])
            if step_optimizer:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.dual_teacher.update_fast(self.student)
                if iteration % int(self.config["teacher"].get("slow_refresh_every", 500)) == 0:
                    self.dual_teacher.refresh_slow(self.student)
                    append_jsonl(self.output_dir / "diagnostics.jsonl", {"event": "slow_teacher_refresh", "iteration": iteration})
                self._maybe_update_prompt_calibrator(iteration, fallback_out=out_l, fallback_y=y_l, fallback_sam=sam_l)

        prompt_quality = 0.0
        if sam_u.get("valid") and sam_u.get("prompt_quality") is not None:
            prompt_quality = float(sam_u["prompt_quality"].detach().mean())
        logs = {
            "loss_total": float(loss.detach()),
            "loss_sup": float(loss_sup.detach()),
            "loss_sup_fusion": float(loss_sup_fusion.detach()),
            "loss_sup_a": float(loss_sup_a.detach()),
            "loss_sup_b": float(loss_sup_b.detach()),
            "loss_singleton": float(ssl1["loss_singleton"].detach()),
            "loss_set": float(ssl1["loss_set"].detach()),
            "loss_rank": float(ssl1["loss_rank"].detach()),
            "loss_negative": float(ssl1["loss_negative"].detach()),
            "loss_fuzzy": float(ssl1["loss_fuzzy"].detach()),
            "loss_branch_ssl": float(branch_ssl.detach()),
            "loss_conflict_review": float(loss_conflict.detach()),
            "loss_correlation": float(loss_corr.detach()),
            "loss_relation": float(loss_relation.detach()),
            "loss_boundary": float(loss_boundary.detach()),
            "loss_sam_sup": float(loss_sam_sup.detach()),
            "loss_sam_unsup": float(loss_sam_unsup.detach()),
            "loss_sam_kd": float(loss_kd.detach()),
            "loss_sam_sem": float(loss_kd.detach()),
            "unsup_weight": ramp,
            "sam_self_reliance_scale": sam_scale,
            "sam_semantic_weight": self.sam_utility.semantic_weight(iteration),
            "sam_utility_ema": float(self.sam_utility.utility_ema),
            "sam_utility_disabled": 1.0 if self.sam_utility.disabled else 0.0,
            "fast_slow_agreement": float(teacher_out["agreement"].detach()),
            "sam_valid_ratio": 1.0 if sam_u.get("valid") else 0.0,
            "prompt_quality": prompt_quality,
            "sam_adapter_grad_norm": sam_grad_norm,
            "optimizer_step": 1.0 if (update and step_optimizer) else 0.0,
            "gradient_accumulation": float(self.grad_accum_steps),
            "lr": self.optimizer.param_groups[0]["lr"],
            "gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024) if self.device.type == "cuda" else 0.0,
            **targets["stats"],
            **sup_logs,
        }
        return logs

    def _supervised_branch_loss(self, out: dict, key: str, target: torch.Tensor):
        if key not in out:
            return out["logits"].new_tensor(0.0)
        loss, _ = supervised_loss(out[key], target, self.num_classes, self.ignore_index)
        return loss

    def _branch_ssl_loss(self, out_s1: dict, out_s2: dict, targets: dict, pseudo_cfg: dict):
        losses = []
        for out in (out_s1, out_s2):
            for key in ("logits_a", "logits_b"):
                if key not in out:
                    continue
                ssl = set_valued_supervision_loss(out[key], targets, pseudo_cfg.get("rank_margin", 0.5))
                losses.append(
                    pseudo_cfg.get("singleton_weight", 1.0) * ssl["loss_singleton"]
                    + pseudo_cfg.get("set_weight", 0.5) * ssl["loss_set"]
                    + pseudo_cfg.get("rank_weight", 0.1) * ssl["loss_rank"]
                    + pseudo_cfg.get("negative_weight", 0.1) * ssl["loss_negative"]
                    + pseudo_cfg.get("fuzzy_weight", 0.25) * ssl["loss_fuzzy"]
                )
        if not losses:
            return out_s1["logits"].new_tensor(0.0)
        return torch.stack(losses).mean()

    def _dual_consistency_loss(self, out: dict, targets: dict):
        if "logits_a" not in out or "logits_b" not in out:
            return out["logits"].new_tensor(0.0)
        target_prob = torch.softmax(out["logits"].detach(), dim=1)
        mask = (targets["singleton_mask"] | targets["ambiguous_mask"] | targets["conflict_mask"]).to(out["logits"].device)
        if mask.sum() == 0:
            return out["logits"].new_tensor(0.0)
        weight = targets.get("candidate_weight", mask.float()).to(out["logits"].device).float() * mask.float()
        denom = weight.sum().clamp_min(1e-6)

        def _ce(logits):
            per_pixel = -(target_prob * F.log_softmax(logits, dim=1)).sum(dim=1)
            return (per_pixel * weight).sum() / denom

        return 0.5 * (_ce(out["logits_a"]) + _ce(out["logits_b"]))

    def _sam_self_reliance_scale(self, iteration: int):
        sam_cfg = self.config.get("sam", {})
        max_iter = int(self.config["train"].get("max_iterations", 10**9))
        default_start = int(0.7 * max_iter)
        start = int(sam_cfg.get("self_reliance_start", default_start))
        if iteration <= start:
            return 1.0
        decay = float(sam_cfg.get("self_reliance_decay", 1.0))
        floor = float(sam_cfg.get("self_reliance_min_weight", 0.05))
        return max(floor, decay ** max(0, iteration - start))

    @torch.no_grad()
    def _maybe_update_prompt_calibrator(
        self,
        iteration: int,
        fallback_out: dict | None = None,
        fallback_y: torch.Tensor | None = None,
        fallback_sam: dict | None = None,
    ):
        if not (self.use_sam and self.mentor is not None and self.calibrator.should_update(iteration)):
            return
        cal_cfg = self.config.get("calibration", {})
        event_name = "prompt_reliability_update"
        if self.calibration_iter is not None and cal_cfg.get("use_calibration_split", True):
            batch_c = next(self.calibration_iter)
            x_c = batch_c["image"].to(self.device)
            y_c = batch_c["mask"].to(self.device)
            student_was_training = getattr(self.student, "training", None)
            mentor_was_training = getattr(self.mentor, "training", None)
            if hasattr(self.student, "eval"):
                self.student.eval()
            if hasattr(self.mentor, "eval"):
                self.mentor.eval()
            try:
                out_c = self.student(x_c, return_features=True)
                sam_c = self.mentor.forward_labeled(x_c, y_c)
            finally:
                if student_was_training and hasattr(self.student, "train"):
                    self.student.train()
                if mentor_was_training and hasattr(self.mentor, "train"):
                    self.mentor.train()
            event_name = "prompt_reliability_update_calibration_split"
        else:
            out_c = fallback_out
            y_c = fallback_y
            sam_c = fallback_sam or {}
        if out_c is None or y_c is None or not sam_c.get("valid"):
            return
        student_prob_l = torch.softmax(out_c["logits"].detach(), dim=1)
        self.calibrator.update_from_batch(
            teacher_prob=student_prob_l,
            sam_prob=sam_c["sam_prob"].detach(),
            sam_iou=sam_c.get("sam_iou"),
            prompt_quality=sam_c.get("prompt_quality"),
            gt=y_c.detach(),
        )
        append_jsonl(
            self.output_dir / "diagnostics.jsonl",
            {
                "event": event_name,
                "iteration": iteration,
                "calibration_batch_size": int(y_c.shape[0]),
                "teacher_q": self.calibrator.teacher_q.tolist(),
                "sam_q": self.calibrator.sam_q.tolist(),
                "sam_iou_q": self.calibrator.sam_iou_q.tolist(),
                "prompt_stability_q": self.calibrator.prompt_stability_q.tolist(),
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
