from __future__ import annotations

import torch


class ClassConditionalConformalCalibrator:
    def __init__(self, num_classes: int, alpha: float = 0.1, min_pixels_per_class: int = 128, shrink_to_global: bool = True):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.min_pixels_per_class = int(min_pixels_per_class)
        self.shrink_to_global = shrink_to_global
        self.q_per_class = torch.full((self.num_classes,), 0.5)
        self.global_q = torch.tensor(0.5)
        self.fitted = False

    def fit(self, probs: torch.Tensor, masks: torch.Tensor):
        probs = probs.detach().float().cpu()
        masks = masks.detach().cpu()
        true_scores = []
        q = []
        for c in range(self.num_classes):
            pix = masks == c
            scores = 1.0 - probs[:, c][pix]
            if scores.numel() > 0:
                true_scores.append(scores)
            if scores.numel() >= self.min_pixels_per_class:
                q.append(torch.quantile(scores, min(0.999, 1.0 - self.alpha)))
            else:
                q.append(None)
        self.global_q = torch.quantile(torch.cat(true_scores), min(0.999, 1.0 - self.alpha)) if true_scores else torch.tensor(0.5)
        self.q_per_class = torch.stack([v if v is not None else self.global_q for v in q])
        self.fitted = True
        return self

    def prediction_sets(self, probs: torch.Tensor):
        q = self.q_per_class.to(probs.device).view(1, -1, 1, 1)
        candidate = (1.0 - probs) <= q
        empty = candidate.sum(dim=1) == 0
        if empty.any():
            arg = probs.argmax(dim=1, keepdim=True)
            candidate.scatter_(1, arg, True)
        low_reliability = empty
        return candidate, low_reliability

    def state_dict(self):
        return {"num_classes": self.num_classes, "alpha": self.alpha, "min_pixels_per_class": self.min_pixels_per_class, "q_per_class": self.q_per_class.tolist(), "global_q": float(self.global_q), "fitted": self.fitted}

    def load_state_dict(self, state):
        self.num_classes = int(state["num_classes"])
        self.alpha = float(state["alpha"])
        self.min_pixels_per_class = int(state.get("min_pixels_per_class", self.min_pixels_per_class))
        self.q_per_class = torch.tensor(state["q_per_class"]).float()
        self.global_q = torch.tensor(state.get("global_q", 0.5)).float()
        self.fitted = bool(state.get("fitted", True))
