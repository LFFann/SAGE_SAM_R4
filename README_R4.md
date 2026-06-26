# SAGE-SAM R4

Calibrated Episodic SAM Mentoring with Set-valued Structural Co-evolution.

This directory is a standalone R4 implementation. It keeps the final deploy path aligned with training: the only exported model is `DeployUNet`. Fast/slow temporal teachers and SAM are training-time mentors only. Validation, testing, and deploy export call the student network without SAM, teacher, VNet, or structure cache.

## Directory

- `train_r4.py`, `validate_r4.py`, `test_r4.py`, `export_deploy_checkpoint.py`: main entry points.
- `Model/deploy_unet.py`: KnowSAM-style five-level UNet deploy student.
- `r4/models/dual_temporal_teacher.py`: fast EMA and slow temporal teachers, no gradients.
- `Model/sam`, `Model/common`, `Model/ImageEncoder`, `Model/prompt.py`: KnowSAM-copied SAM implementation used by default.
- `r4/models/real_sam_wrapper.py`: real SAM checkpoint guard and loader.
- `r4/calibration/class_conditional_conformal.py`: class-conditional conformal prediction sets.
- `r4/losses/set_valued_losses.py`: singleton, candidate-set, rank margin, and safe negative losses.
- `tools/`: dataset validation, real SAM verification, and SAM structure cache generation.
- `tests/`: CPU smoke and core unit tests.

## Data Format

```text
SampleData/<dataset_name>/
  labeled/image
  labeled/mask
  unlabeled/image
  val/image
  val/mask
  test/image
  test/mask
```

Masks must contain integer ids in `0..num_classes-1` or `ignore_index`.

## Commands

Check data:

```bash
python SAGE_SAM_R4/tools/validate_dataset.py --config SAGE_SAM_R4/configs/r4_3class_v100.yaml
```

Verify real SAM:

```bash
python SAGE_SAM_R4/tools/verify_real_sam.py --config SAGE_SAM_R4/configs/r4_3class_v100.yaml
```

Build SAM structure cache:

```bash
python SAGE_SAM_R4/tools/build_sam_structure_cache.py --config SAGE_SAM_R4/configs/r4_3class_v100.yaml --split unlabeled
```

CPU smoke:

```bash
python SAGE_SAM_R4/train_r4.py --config SAGE_SAM_R4/configs/r4_smoke_cpu.yaml --dry-run
python SAGE_SAM_R4/train_r4.py --config SAGE_SAM_R4/configs/r4_smoke_cpu.yaml --max-iterations 2
```

Train:

```bash
python SAGE_SAM_R4/train_r4.py --config SAGE_SAM_R4/configs/r4_3class_v100.yaml
```

Resume:

```bash
python SAGE_SAM_R4/train_r4.py --config outputs/SAGE_SAM_R4_3Class/resolved_config.yaml --resume outputs/SAGE_SAM_R4_3Class/checkpoints/latest.pth
```

Validate:

```bash
python SAGE_SAM_R4/validate_r4.py --config outputs/SAGE_SAM_R4_3Class/resolved_config.yaml --checkpoint outputs/SAGE_SAM_R4_3Class/checkpoints/best_val_dice.pth
```

Test:

```bash
python SAGE_SAM_R4/test_r4.py --config outputs/SAGE_SAM_R4_3Class/resolved_config.yaml --checkpoint outputs/SAGE_SAM_R4_3Class/checkpoints/best_val_dice.pth --save-pred
```

Export deploy student:

```bash
python SAGE_SAM_R4/export_deploy_checkpoint.py --checkpoint outputs/SAGE_SAM_R4_3Class/checkpoints/best_val_dice.pth --output outputs/SAGE_SAM_R4_3Class/checkpoints/deploy_student.pth
```

## SAM Checkpoint

Set `sam.checkpoint` in the YAML config. R4 now carries a local SAM implementation copied from KnowSAM under `SAGE_SAM_R4/Model/sam`, so `segment_anything` is not required on the server. When `sam.use_sam=true`, R4 intentionally fails if the checkpoint is missing or the loaded SAM has no real parameters. There is no silent fallback to `torch.softmax(student_logits)`.

For CPU smoke, `sam.use_sam=false`; this covers train/val/test/export without needing SAM.

## How To Confirm It Is Not Self-Distillation

- `RealSAMWrapper` first loads `SAGE_SAM_R4/Model/sam`, then falls back to `segment_anything` only if the local copy is absent.
- The wrapper loads the configured checkpoint, hashes it, records `sam_source`, and checks the SAM parameter count.
- `verify_real_sam.py` compares embeddings for two different images and raises if embeddings do not depend on image content.
- The SAM mentor output must contain `sam_prob` with `num_classes` channels before Stage B can consume it.
- Validation, testing, and export never instantiate `RealSAMWrapper`.

## Deploy Export

`export_deploy_checkpoint.py` exports only:

```python
{
  "model": student_state_dict,
  "num_classes": ...,
  "in_channels": ...,
  "model_name": "SAGE_SAM_R4_DeployStudent",
  "config_minimal": ...
}
```

The exporter rejects keys containing `sam`, `teacher`, `mentor`, `calibrator`, `optimizer`, `vnet`, or `relation`.

## Common Errors

- `FileNotFoundError: SAM checkpoint does not exist`: set `sam.checkpoint` to a real SAM `.pth`.
- SAM import error: keep `SAGE_SAM_R4/Model/sam`, `Model/common`, `Model/ImageEncoder`, and `Model/prompt.py` in the repository. `segment_anything` is only a fallback.
- Invalid mask ids: run `tools/validate_dataset.py` and fix masks outside `0..num_classes-1`.
- CUDA unavailable: use `--device cpu` for debugging, or run on a CUDA machine for V100 training.
