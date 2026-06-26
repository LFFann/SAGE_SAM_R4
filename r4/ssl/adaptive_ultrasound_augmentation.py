from __future__ import annotations

from r4.data.transforms import strong_transform, weak_transform


def make_weak_strong_views(image, pseudo_mask=None, strong_kwargs=None, weak_kwargs=None):
    x_w, y_w = weak_transform(image, pseudo_mask, **(weak_kwargs or {}))
    x_s1, y_s1 = strong_transform(x_w.clone(), y_w.clone() if y_w is not None else None, **(strong_kwargs or {}))
    x_s2, y_s2 = strong_transform(x_w.clone(), y_w.clone() if y_w is not None else None, **(strong_kwargs or {}))
    return x_w, x_s1, x_s2, y_s1, y_s2

