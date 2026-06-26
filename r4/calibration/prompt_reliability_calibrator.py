from __future__ import annotations

import torch


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
        teacher_thresh = self.teacher_q.to(device)[teacher_arg]
        semantic_gate = teacher_conf >= teacher_thresh
        sam_train_gate = semantic_gate.clone()
        structure_gate = semantic_gate.clone()
        agreement_ratio = teacher_prob.new_tensor(1.0)
        if sam_prob is not None:
            sam_conf, sam_arg = sam_prob.max(dim=1)
            sam_thresh = self.sam_q.to(device)[sam_arg]
            agree = teacher_arg == sam_arg
            agreement_ratio = agree.float().mean()
            agreement_thresh = self.agreement_q.to(device)[teacher_arg]
            agreement_gate = agree | (agreement_thresh < 0.5)
            semantic_gate = semantic_gate & (sam_conf >= sam_thresh) & agreement_gate
            sam_train_gate = semantic_gate.clone()
            structure_gate = (sam_conf >= sam_thresh) & (teacher_conf >= teacher_thresh) & agreement_gate
        if sam_iou is not None:
            iou_map = self._class_scores_to_map(sam_iou.to(device), teacher_arg)
            iou_thresh = self.sam_iou_q.to(device)[teacher_arg].clamp(max=0.95)
            iou_gate = iou_map >= iou_thresh
            sam_train_gate = sam_train_gate & iou_gate
            structure_gate = structure_gate & iou_gate
        if prompt_quality is not None:
            prompt_map = self._class_scores_to_map(prompt_quality.to(device), teacher_arg)
            prompt_thresh = self.prompt_stability_q.to(device)[teacher_arg]
            prompt_gate = prompt_map >= prompt_thresh
            semantic_gate = semantic_gate & prompt_gate
            sam_train_gate = sam_train_gate & prompt_gate
            structure_gate = structure_gate & prompt_gate
        return {
            "semantic_gate": semantic_gate,
            "sam_train_gate": sam_train_gate,
            "structure_gate": structure_gate,
            "teacher_sam_agreement": agreement_ratio,
        }

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
        self.fitted = bool(state.get("fitted", True))
