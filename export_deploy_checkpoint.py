from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r4.engine.checkpoint import export_deploy_payload, safe_load
from r4.engine.model_factory import build_deploy_model, minimal_deploy_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strip-boundary", action="store_true")
    args = p.parse_args()
    out = export_deploy_payload(args.checkpoint, args.output, strip_boundary=args.strip_boundary)
    payload = safe_load(out)
    model = build_deploy_model(minimal_deploy_config(payload))
    model.load_state_dict(payload["model"], strict=False)
    x = torch.randn(1, payload["in_channels"], payload["config_minimal"].get("image_size", 64), payload["config_minimal"].get("image_size", 64))
    _ = model(x)
    print(f"Exported dual-fusion deploy student: {out}")


if __name__ == "__main__":
    main()
