# SAGE-SAM R4-Final

SAGE-SAM R4-Final is a single-run calibrated SAM-guided dual co-training segmentation pipeline:

```text
DeployDualFusionSegmentor student
  Branch A: UNet-like local texture path
  Branch B: residual/dilated morphology path
  Fusion: HAM-lite disagreement-aware fusion logits
+ online promptable SAM mentor
+ trainable SAM LoRA + BottleneckAdapter PEFT parameters
+ trainable prompt generator
+ trainable SAM mask decoder
+ EMA weak-to-strong student learning
+ calibrated soft SAM participation with minimum coverage protection
+ singleton, set-valued, fuzzy-positive, safe-negative, and correlation-propagated supervision
+ online SAM-student boundary/relation consistency without disk cache
+ late self-reliance decay for SAM unsupervised/KD losses
```

The final deploy path is `DeployDualFusionSegmentor`, and validation/testing evaluate fusion logits by default. SAM, the prompt generator, EMA teachers, calibration state, and relation/correlation losses are training-time components only. Validation, testing, and deploy export do not instantiate SAM.

## Directory

- `train_r4.py`, `validate_r4.py`, `test_r4.py`, `export_deploy_checkpoint.py`: main entry points.
- `Model/deploy_dual_fusion.py`: dual-view deploy student with UNet branch, morphology branch, and HAM-lite fusion.
- `Model/deploy_unet.py`: UNet branch used inside the dual-fusion deploy student.
- `r4/models/real_sam_wrapper.py`: real SAM checkpoint guard plus prompted SAM forward.
- `r4/models/prompt_generator.py`: trainable one-vs-rest mask-prompt generator.
- `r4/models/promptable_sam_mentor.py`: online SAM co-learner used during training.
- `r4/models/sam_peft.py`: SAM freezing plus real LoRA and BottleneckAdapter injection for selected image-encoder blocks.
- `r4/calibration/prompt_reliability_calibrator.py`: online prompt-aware thresholds plus soft participation weights.
- `r4/data/`: dataset, paired semi-supervised iterator, split, and augmentation transforms.
- `r4/ssl/target_builder.py`: SAM-teacher singleton, ambiguous, conflict, and safe-negative targets.
- `r4/ssl/online_sam_relation.py`: batch-local top-k KL/rank SAM-student relation loss without cache.
- `r4/ssl/correlation_propagation.py`: low-resolution online correlation propagation for low-confidence pixels.
- `r4/losses/set_valued_losses.py`: singleton, set, fuzzy positive, rank, and safe-negative losses.
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

Export dual-fusion deploy student:

```bash
python SAGE_SAM_R4/export_deploy_checkpoint.py --checkpoint outputs/SAGE_SAM_R4_3Class/checkpoints/best_val_dice.pth --output outputs/SAGE_SAM_R4_3Class/checkpoints/deploy_student.pth
```

## SingleLoop Training Contract

R4 does not use multi-round training, offline pseudo-label generation, or structure cache in the main method. Each iteration performs:

```text
1. supervised fusion + branch losses on labeled data
2. supervised SAM adapter/decoder/prompt-generator loss on labeled data
3. EMA teacher prediction on weak unlabeled images
4. online prompted SAM forward with mask, box, positive point, and negative point prompts
5. SAM-teacher set-valued target construction with calibrated soft weights
6. spatially aligned strong dual-fusion student prediction with complementary dropout
7. set-valued, fuzzy-positive, safe-negative, conflict-review, correlation, KD, soft-gated SAM unsupervised, boundary, and online relation losses
8. one optimizer step for student + SAM PEFT + prompt generator + mask decoder
9. EMA teacher update and periodic prompt reliability calibration
```

## SAM Checkpoint

Set `sam.checkpoint` in the YAML config. R4 carries a local SAM implementation copied from KnowSAM under `SAGE_SAM_R4/Model/sam`, so `segment_anything` is not required on the server. When `sam.use_sam=true`, R4 fails if the checkpoint is missing, the loaded SAM has no parameters, or `sam.train_peft=true` leaves SAM without injected LoRA or BottleneckAdapter parameters.

For CPU smoke, `sam.use_sam=false`; this covers train/val/test/export without needing SAM.

## How To Confirm It Is Not Self-Distillation

- `RealSAMWrapper.forward_prompted()` must call `image_encoder`, `prompt_encoder`, and `mask_decoder`.
- The prompt encoder receives mask prompts plus configured box, positive point, and negative point prompts.
- `sam_prob` is assembled from SAM foreground masks and a background channel, not from `teacher_prob.clone()`.
- Training logs include `loss_sam_sup`, `loss_sam_unsup`, `loss_sam_kd`, `prompt_quality`, `sam_teacher_agreement`, and `sam_adapter_grad_norm`.
- Training logs include `sam_soft_weight_mean`, `sam_participation_ratio`, `per_class_sam_participation_ratio`, `loss_conflict_review`, and `loss_correlation`.
- Diagnostics include `total_sam_params`, `trainable_sam_params`, `trainable_sam_ratio`, `lora_param_count`, `adapter_param_count`, `mask_decoder_trainable`, `prompt_generator_trainable`, and trainable module names.
- `loss_sam_unsup` uses teacher-only reliable targets, while student SSL may use the fused SAM-teacher target.
- Changing the prompt changes the SAM mask decoder input and therefore the SAM output.
- Weak-to-strong consistency keeps spatial geometry fixed after weak-view target generation; strong augmentation is intensity/noise/patch masking only.
- Validation, testing, and export never instantiate `RealSAMWrapper`.

## Deploy Export

`export_deploy_checkpoint.py` exports only:

```python
{
  "model": dual_fusion_student_state_dict,
  "num_classes": ...,
  "in_channels": ...,
  "model_name": "SAGE_SAM_R4_DualFusionDeploy",
  "config_minimal": ...
}
```

The exporter rejects model keys containing `sam`, `teacher`, `mentor`, `calibrator`, or `optimizer`.

## Common Errors

- `FileNotFoundError: SAM checkpoint does not exist`: set `sam.checkpoint` to a real SAM `.pth`.
- `sam.train_peft=true but no LoRA/Adapter parameter was injected or found`: lower-level SAM blocks were not found or `peft_type` is inconsistent with the loaded SAM.
- `sam.peft_type contains 'adapter' but no Adapter parameter exists`: adapter injection failed; check `adapter_dim` and image-encoder block structure.
- `SAM trainable ratio exceeds hard limit`: lower `train_last_n_blocks` or freeze more SAM modules.
- Invalid mask ids: run `tools/validate_dataset.py` and fix masks outside `0..num_classes-1`.
- CUDA unavailable: use `--device cpu` for debugging, or run on a CUDA machine for V100 training.
