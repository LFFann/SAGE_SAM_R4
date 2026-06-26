from __future__ import annotations

import torch

from r4.data.transforms import strong_transform


def test_strong_transform_preserves_spatial_coordinates_without_patch():
    image = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 100.0
    mask = torch.arange(4 * 5, dtype=torch.long).reshape(4, 5)
    out, out_mask = strong_transform(
        image.clone(),
        mask.clone(),
        gamma_range=(1.0, 1.0),
        gain_range=(1.0, 1.0),
        speckle_std_range=(0.0, 0.0),
        patch_mask_prob=0.0,
    )
    assert torch.equal(out, image.clamp_min(1e-4))
    assert torch.equal(out_mask, mask)


def test_strong_patch_mask_supports_chw_and_bchw():
    chw = torch.ones(3, 8, 8)
    bchw = torch.ones(2, 3, 8, 8)
    kwargs = dict(
        gamma_range=(1.0, 1.0),
        gain_range=(1.0, 1.0),
        speckle_std_range=(0.0, 0.0),
        patch_mask_prob=1.0,
        patch_mask_ratio=0.25,
    )
    out_chw, _ = strong_transform(chw, **kwargs)
    out_bchw, _ = strong_transform(bchw, **kwargs)
    assert out_chw.shape == chw.shape
    assert out_bchw.shape == bchw.shape
    assert (out_chw == 0).any()
    assert (out_bchw == 0).any()
