from __future__ import annotations

import torch


def soft_reliability(score: torch.Tensor, q: torch.Tensor, temperature: float = 0.05, max_q: float = 0.97):
    q = torch.clamp(q, max=float(max_q))
    return torch.sigmoid((score - q) / max(float(temperature), 1e-6))


class PromptReliabilityCalibrator:
    """Online class-wise reliability thresholds for SAM-assisted SSL."""

    def __init__(
        self,
        num_classes: int,
        start_iter: int = 500,
        update_every: int = 250,
        momentum: float = 0.8,
        min_pixels_per_class: int = 128,
        teacher_quantile: float = 0.30,
        sam_quantile: float = 0.30,
        agreement_quantile: float = 0.30,
        prompt_quantile: float = 0.30,
        sam_iou_quantile: float | None = None,
        use_soft_gate: bool = True,
        min_participation_ratio: float = 0.0,
        max_quantile_clip: float = 0.97,
        temperature: float = 0.05,
        coverage_target: float | None = None,
    ):
        self.num_classes = int(num_classes)
        self.start_iter = int(start_iter)
        self.update_every = int(update_every)
        self.momentum = float(momentum)
        self.min_pixels_per_class = int(min_pixels_per_class)
        self.teacher_q = torch.full((self.num_classes,), 0.50)
        self.sam_q = torch.full((self.num_classes,), 0.50)
        self.sam_iou_q = torch.full((self.num_classes,), 0.50)
        self.agreement_q = torch.full((self.num_classes,), 0.50)
        self.prompt_stability_q = torch.full((self.num_classes,), 0.50)
        self.prompt_q = self.prompt_stability_q
        self.teacher_quantile = float(teacher_quantile)
        self.sam_quantile = float(sam_quantile)
        self.sam_iou_quantile = float(sam_iou_quantile if sam_iou_quantile is not None else sam_quantile)
        self.agreement_quantile = float(agreement_quantile)
        self.prompt_quantile = float(prompt_quantile)
        self.use_soft_gate = bool(use_soft_gate)
        self.min_participation_ratio = float(min_participation_ratio)
        self.max_quantile_clip = float(max_quantile_clip)
        self.temperature = float(temperature)
        self.coverage_target = float(coverage_target if coverage_target is not None else min_participation_ratio)
        self.fitted = False

    def should_update(self, iteration: int):
        return iteration >= self.start_iter and (iteration - self.start_iter) % max(1, self.update_every) == 0

    @torch.no_grad()
    def update_from_batch(
        self,
        teacher_prob: torch.Tensor,
        sam_prob: torch.Tensor,
        sam_iou: torch.Tensor | None = None,
        prompt_quality: torch.Tensor | None = None,
        gt: torch.Tensor | None = None,
    ):
        teacher_prob = teacher_prob.detach().float().cpu()
        sam_prob = sam_prob.detach().float().cpu()
        if sam_iou is None:
            sam_iou = sam_prob.new_ones(sam_prob.shape[:2])
        sam_iou = sam_iou.detach().float().cpu()
        if prompt_quality is None:
            prompt_quality = sam_prob.new_ones(sam_prob.shape[:2])
        prompt_quality = prompt_quality.detach().float().cpu()
        teacher_arg = teacher_prob.argmax(dim=1)
        sam_arg = sam_prob.argmax(dim=1)
        agreement = (teacher_arg == sam_arg).float()

        new_teacher = self.teacher_q.clone()
        new_sam = self.sam_q.clone()
        new_sam_iou = self.sam_iou_q.clone()
        new_agreement = self.agreement_q.clone()
        new_prompt = self.prompt_stability_q.clone()
        for c in range(self.num_classes):
            if gt is not None:
                class_mask = gt.detach().cpu() == c
            else:
                class_mask = (teacher_arg == c) | (sam_arg == c)
            if int(class_mask.sum()) < self.min_pixels_per_class:
                continue
            image_mask = class_mask.reshape(class_mask.shape[0], -1).any(dim=1)
            new_teacher[c] = torch.quantile(teacher_prob[:, c][class_mask], self.teacher_quantile)
            new_sam[c] = torch.quantile(sam_prob[:, c][class_mask], self.sam_quantile)
            new_agreement[c] = torch.quantile(agreement[class_mask], self.agreement_quantile)
            if image_mask.any():
                new_sam_iou[c] = torch.quantile(sam_iou[:, c][image_mask], self.sam_iou_quantile)
                new_prompt[c] = torch.quantile(prompt_quality[:, c][image_mask], self.prompt_quantile)

        if self.fitted:
            keep = self.momentum
            self.teacher_q = keep * self.teacher_q + (1.0 - keep) * new_teacher
            self.sam_q = keep * self.sam_q + (1.0 - keep) * new_sam
            self.sam_iou_q = keep * self.sam_iou_q + (1.0 - keep) * new_sam_iou
            self.agreement_q = keep * self.agreement_q + (1.0 - keep) * new_agreement
            self.prompt_stability_q = keep * self.prompt_stability_q + (1.0 - keep) * new_prompt
        else:
            self.teacher_q = new_teacher
            self.sam_q = new_sam
            self.sam_iou_q = new_sam_iou
            self.agreement_q = new_agreement
            self.prompt_stability_q = new_prompt
            self.fitted = True
        self.prompt_q = self.prompt_stability_q
        return self

    def prediction_sets(self, probs: torch.Tensor):
        q = self.teacher_q.to(probs.device).view(1, -1, 1, 1)
        candidate = probs >= q
        empty = candidate.sum(dim=1) == 0
        if empty.any():
            arg = probs.argmax(dim=1, keepdim=True)
            candidate.scatter_(1, arg, True)
        return candidate, empty

    def gates(
        self,
        teacher_prob: torch.Tensor,
        sam_prob: torch.Tensor | None = None,
        sam_iou: torch.Tensor | None = None,
        prompt_quality: torch.Tensor | None = None,
    ):
        device = teacher_prob.device
        teacher_conf, teacher_arg = teacher_prob.max(dim=1)
        teacher_thresh = self._class_threshold_map(self.teacher_q, teacher_arg, teacher_conf)
        teacher_weight = soft_reliability(teacher_conf, teacher_thresh, self.temperature, self.max_quantile_clip)
        semantic_weight = teacher_weight.clone()
        sam_train_weight = teacher_weight.clone()
        structure_weight = teacher_weight.clone()
        agreement_ratio = teacher_prob.new_tensor(1.0)
        if sam_prob is not None:
            sam_conf, sam_arg = sam_prob.max(dim=1)
            sam_thresh = self._class_threshold_map(self.sam_q, sam_arg, sam_conf)
            sam_weight = soft_reliability(sam_conf, sam_thresh, self.temperature, self.max_quantile_clip)
            agree = teacher_arg == sam_arg
            agreement_ratio = agree.float().mean()
            agreement_thresh = self.agreement_q.to(device)[teacher_arg]
            agreement_gate = agree | (agreement_thresh < 0.5)
            disagreement_weight = teacher_prob.new_full(teacher_conf.shape, 0.15 if self.use_soft_gate else 0.0)
            agreement_weight = torch.where(agreement_gate, teacher_prob.new_ones(teacher_conf.shape), disagreement_weight)
            semantic_weight = torch.minimum(teacher_weight, sam_weight) * agreement_weight
            sam_train_weight = semantic_weight.clone()
            structure_weight = torch.minimum(teacher_weight, sam_weight) * agreement_weight
        if sam_iou is not None:
            iou_map = self._class_scores_to_map(sam_iou.to(device), teacher_arg)
            iou_thresh = self._class_threshold_map(self.sam_iou_q.clamp(max=0.95), teacher_arg, iou_map)
            iou_weight = soft_reliability(iou_map, iou_thresh, self.temperature, self.max_quantile_clip)
            sam_train_weight = torch.minimum(sam_train_weight, iou_weight)
            structure_weight = torch.minimum(structure_weight, iou_weight)
        if prompt_quality is not None:
            prompt_map = self._class_scores_to_map(prompt_quality.to(device), teacher_arg)
            prompt_thresh = self._class_threshold_map(self.prompt_stability_q, teacher_arg, prompt_map)
            prompt_weight = soft_reliability(prompt_map, prompt_thresh, self.temperature, self.max_quantile_clip)
            semantic_weight = torch.minimum(semantic_weight, prompt_weight)
            sam_train_weight = torch.minimum(sam_train_weight, prompt_weight)
            structure_weight = torch.minimum(structure_weight, prompt_weight)
        semantic_gate = semantic_weight >= 0.05 if self.use_soft_gate else semantic_weight >= 0.5
        sam_train_gate = sam_train_weight >= 0.05 if self.use_soft_gate else sam_train_weight >= 0.5
        structure_gate = structure_weight >= 0.05 if self.use_soft_gate else structure_weight >= 0.5
        return {
            "semantic_gate": semantic_gate,
            "sam_train_gate": sam_train_gate,
            "structure_gate": structure_gate,
            "semantic_weight": semantic_weight.clamp(0.0, 1.0),
            "sam_train_weight": sam_train_weight.clamp(0.0, 1.0),
            "structure_weight": structure_weight.clamp(0.0, 1.0),
            "teacher_weight": teacher_weight.clamp(0.0, 1.0),
            "teacher_sam_agreement": agreement_ratio,
        }

    def _class_threshold_map(self, q_vec: torch.Tensor, class_index: torch.Tensor, score: torch.Tensor):
        q = q_vec.to(score.device)[class_index].clamp(max=self.max_quantile_clip)
        target = max(self.min_participation_ratio, self.coverage_target)
        if target <= 0.0 or score.numel() == 0:
            return q
        hard_ratio = (score >= q).float().mean()
        if float(hard_ratio.detach()) >= min(target, 1.0):
            return q
        flat = score.detach().float().reshape(-1)
        fallback_q = torch.quantile(flat.cpu(), max(0.0, min(1.0, 1.0 - target))).to(score.device, score.dtype)
        return torch.minimum(q, fallback_q)

    @staticmethod
    def _class_scores_to_map(scores: torch.Tensor, class_index: torch.Tensor):
        if scores.ndim == 1:
            scores = scores.view(1, -1)
        if scores.ndim != 2:
            raise ValueError(f"class scores must be shaped BxC, got {tuple(scores.shape)}")
        flat = class_index.reshape(class_index.shape[0], -1)
        return scores.gather(1, flat).reshape_as(class_index)

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "start_iter": self.start_iter,
            "update_every": self.update_every,
            "momentum": self.momentum,
            "min_pixels_per_class": self.min_pixels_per_class,
            "teacher_q": self.teacher_q.tolist(),
            "sam_q": self.sam_q.tolist(),
            "sam_iou_q": self.sam_iou_q.tolist(),
            "agreement_q": self.agreement_q.tolist(),
            "prompt_stability_q": self.prompt_stability_q.tolist(),
            "prompt_q": self.prompt_stability_q.tolist(),
            "use_soft_gate": self.use_soft_gate,
            "min_participation_ratio": self.min_participation_ratio,
            "max_quantile_clip": self.max_quantile_clip,
            "temperature": self.temperature,
            "coverage_target": self.coverage_target,
            "fitted": self.fitted,
        }

    def load_state_dict(self, state):
        self.num_classes = int(state["num_classes"])
        self.start_iter = int(state.get("start_iter", self.start_iter))
        self.update_every = int(state.get("update_every", self.update_every))
        self.momentum = float(state.get("momentum", self.momentum))
        self.min_pixels_per_class = int(state.get("min_pixels_per_class", self.min_pixels_per_class))
        self.teacher_q = torch.tensor(state["teacher_q"]).float()
        self.sam_q = torch.tensor(state["sam_q"]).float()
        self.sam_iou_q = torch.tensor(state.get("sam_iou_q", state.get("sam_q", [0.5] * self.num_classes))).float()
        self.agreement_q = torch.tensor(state.get("agreement_q", [0.5] * self.num_classes)).float()
        self.prompt_stability_q = torch.tensor(
            state.get("prompt_stability_q", state.get("prompt_q", [0.5] * self.num_classes))
        ).float()
        self.prompt_q = self.prompt_stability_q
        self.use_soft_gate = bool(state.get("use_soft_gate", self.use_soft_gate))
        self.min_participation_ratio = float(state.get("min_participation_ratio", self.min_participation_ratio))
        self.max_quantile_clip = float(state.get("max_quantile_clip", self.max_quantile_clip))
        self.temperature = float(state.get("temperature", self.temperature))
        self.coverage_target = float(state.get("coverage_target", self.coverage_target))
        self.fitted = bool(state.get("fitted", True))
