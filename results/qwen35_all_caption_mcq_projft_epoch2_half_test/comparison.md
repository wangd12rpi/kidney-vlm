# All-caption MCQ trainable-projector SFT: epoch-2 evaluation

## Evaluation scope

- Checkpoint: epoch 2 of
  `qwen35_all_caption_mcq_projft_cont_4ep_b2_ga8_20260729`.
- Test population: all 4,295 held-out `mcq` / `from_caption` questions.
- Evaluated sample: a deterministic seed-42 sample of 2,148 questions (50.01%),
  spanning every task category and modality combination.
- Prompt: no CoT; return only the exact text of one choice.
- Inference: deterministic, one repeat, live projectors, Qwen3.5-9B loaded in
  8-bit, batch size 12 on physical GPU 1.

## Overall result

| Model | Correct | Accuracy | Macro F1 | Parse failures |
|---|---:|---:|---:|---:|
| Submitted full-VQA OncoVLM, repeat 0 | 674 / 2,148 | 31.38% | — | — |
| All-caption MCQ projector FT, epoch 2 | 818 / 2,148 | **38.08%** | **37.93%** | 1 |

On the exact same questions, epoch 2 improves accuracy by **6.70 percentage
points**. It changes 452 questions from wrong to correct and 308 from correct
to wrong. A seed-42 question-level paired bootstrap gives a 95% interval of
**+4.19 to +9.22 points**; the exact McNemar p-value is
**1.97e-7**.

## By modality combination

| Modality combination | N | Epoch 2 | Submitted repeat 0 | Delta |
|---|---:|---:|---:|---:|
| All available | 1,017 | 39.23% | 32.84% | +6.39 pp |
| Pathology only | 977 | 35.93% | 29.38% | +6.55 pp |
| Radiology only | 154 | 44.16% | 34.42% | +9.74 pp |

## By caption task

| Task | N | Epoch 2 | Submitted repeat 0 | Delta |
|---|---:|---:|---:|---:|
| Genomics | 673 | 30.76% | 27.34% | +3.42 pp |
| Integrated | 716 | 46.23% | 36.17% | +10.06 pp |
| Pathology | 654 | 36.39% | 30.73% | +5.66 pp |
| Radiology | 105 | 40.00% | 28.57% | +11.43 pp |

## Comparison with the immediately preceding focused checkpoint

The deterministic half-sample contains 321 of the 642 focused
`all_available` / `pathology_findings` test questions.

| Model | Correct | Accuracy |
|---|---:|---:|
| Focused trainable-projector SFT | 93 / 321 | 28.97% |
| All-caption MCQ projector FT, epoch 2 | 119 / 321 | **37.07%** |

The gain on these paired questions is **+8.10 points** (66 wrong-to-correct
versus 40 correct-to-wrong). The paired bootstrap 95% interval is **+1.87 to
+14.33 points**, with exact McNemar p=`0.0148`.

## Interpretation

This is positive evidence that broadening caption-MCQ training across modality
combinations and tasks is helping, rather than the small focused dataset
dragging the model backward. The largest gains are currently on integrated and
radiology caption questions. Genomics improves more modestly and remains the
weakest task at 30.76%.

The result is an early checkpoint on a fixed 50% test sample, not the final
four-epoch result. Epoch selection should still use validation loss rather than
this test sample.
