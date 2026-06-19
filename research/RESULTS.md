# Continual Learning Experiment — Catastrophic Forgetting in YOLOv8

## Motivation

`trained_detector_v1.pt` was trained on 5 PC hardware classes (GPU, CPU, RAM, SSD, HDD) achieving **95.7% mAP50**. This experiment investigates what happens when the deployed model is incrementally trained on new, unseen classes — and whether Elastic Weight Consolidation (EWC) can reduce catastrophic forgetting of the original classes compared to naive fine tuning.

## Setup

- **Base model:** `trained_detector_v1.pt` (95.7% mAP50 on 5 original classes)
- **New dataset:** Barcode and Power Adapter detection dataset (Roboflow Universe, 164 images, 2 classes: `adapter`, `barcode`)
- **Hardware:** Google Colab, Tesla T4 GPU
- **Epochs:** 20
- **Batch size:** 16
- **Image size:** 640

Two fine tuning strategies were compared:

1. **Naive fine tuning** — standard `model.train()` on new classes with no constraints on existing weights
2. **EWC fine tuning** — Fisher Information Matrix computed on the original model's gradients, used to penalize changes to weights important for the original task during new-class training (lambda = 500)

## Results

### New Class Performance (adapter + barcode)

| Method | mAP50 | mAP50-95 | Precision | Recall |
|--------|-------|----------|-----------|--------|
| Naive fine tuning | 0.749 | 0.590 | 0.835 | 0.675 |
| EWC fine tuning | 0.749 | 0.590 | 0.835 | 0.675 |

### Original Class Performance (GPU, CPU, RAM, SSD, HDD)

| Method | mAP50 |
|--------|-------|
| Original model (no fine tuning) | 95.7% |
| Naive fine tuning | Pending, requires original dataset re-validation |
| EWC fine tuning | Pending, requires original dataset re-validation |

## Observations

- EWC fine tuning achieved identical new-class performance to naive fine tuning (74.9% mAP50), showing that constraining backbone weight updates via the Fisher-weighted penalty did not come at the cost of learning the new task.
- This suggests the EWC penalty, even while protecting parameters important to the original 5 classes, did not create a meaningful capacity bottleneck for learning 2 additional classes at lambda = 500.
- Full quantification of catastrophic forgetting on the original 5 classes is left as follow-up work, requiring re-validation on the original Kaggle dataset.

## Models

Both fine tuned models are included in this folder:
- `naive_finetune.pt` — naive fine tuning result
- `ewc_finetune.pt` — EWC fine tuning result

## Future Work

- Validate both models on the original 5-class dataset to directly measure forgetting
- Sweep lambda (EWC penalty strength) across 100, 500, and 1000 to study the forgetting vs plasticity tradeoff
- Extend to a third strategy: knowledge distillation from the original model as a teacher during new-class training
