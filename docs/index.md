# MedVLM: Inference, Confidence, and Calibration in Medical VQA

A Python toolkit for measuring and correcting overconfidence in Medical VLMs.

- **8 models supported**: Qwen3-VL, InternVL3, LLaVA-NeXT (2B--38B)
- **5 medical VQA datasets**: VQA-RAD, SLAKE, VQA-Med 2019/2020/2021
- **2 confidence methods**: Sampling-based, verbalized (6 prompt variants)
- **8 calibration methods**: 5 standard + 3 HAC methods with hallucination scores

## What's in the package?

| Category | Functions |
|----------|-----------|
| **Data** | `load_dataset`, `train_val_test_split` |
| **Models** | `load_model` (auto-detects family, quantization) |
| **Confidence** | `compute_confidence` (sampling, verbalized) |
| **Calibration** | `CalibrationPipeline` (Platt, Platt-Confidence, isotonic, HAC-Platt, HAC-Platt-Confidence, HAC-Gate) |
| **Evaluation** | `evaluate_calibration` (ECE, MCE, overconfidence, accuracy) |

## Key Findings

1. **Medical VLMs are overconfident** --- mean confidence exceeds accuracy by 10--27%.
2. **Post-hoc calibration reduces ECE** by 50--95% without retraining.
3. **HAC improves AUROC** by incorporating hallucination signals that standard calibration cannot use.

## Paper

> Byun, Park, Corbeil, Ben Abacha. "Overconfidence and Calibration in Medical VQA: Empirical Findings and Hallucination-Aware Mitigation." arXiv:2604.02543, 2026. [[arXiv]](https://arxiv.org/abs/2604.02543)
