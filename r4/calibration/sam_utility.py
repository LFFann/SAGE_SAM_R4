from __future__ import annotations


class SAMUtilityScheduler:
    def __init__(self, max_weight: float = 0.15, ema_decay: float = 0.9, disable_after_no_gain: int = 3):
        self.max_weight = float(max_weight)
        self.ema_decay = float(ema_decay)
        self.disable_after_no_gain = int(disable_after_no_gain)
        self.utility_ema = 0.0
        self.no_gain_count = 0
        self.disabled = False

    def update(self, utility: float):
        self.utility_ema = self.ema_decay * self.utility_ema + (1 - self.ema_decay) * float(utility)
        if utility <= 0:
            self.no_gain_count += 1
        else:
            self.no_gain_count = 0
        if self.no_gain_count >= self.disable_after_no_gain:
            self.disabled = True
        return self.disabled

    def semantic_weight(self, iteration: int):
        if self.disabled:
            return 0.0
        return max(0.0, self.max_weight * (1.0 + self.utility_ema))

    def state_dict(self):
        return dict(max_weight=self.max_weight, ema_decay=self.ema_decay, disable_after_no_gain=self.disable_after_no_gain, utility_ema=self.utility_ema, no_gain_count=self.no_gain_count, disabled=self.disabled)

    def load_state_dict(self, state):
        self.__dict__.update(state)

