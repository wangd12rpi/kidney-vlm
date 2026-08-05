# Trainable-projector focused SFT evaluation

## Evaluation scope

- Test set: 642 held-out `mcq` / `from_caption` / `all_available` /
  `pathology_findings` questions.
- Prompt: no CoT; return only the exact choice text.
- Inference: deterministic, one repeat, Qwen3.5-9B loaded in 8-bit.
- Projectors were evaluated live from the raw feature tensors. This preserves the
  training behavior for multi-slide cases by concatenating slide features before
  the pathology resampler.
- All 1,284 outputs across the two controlled models parsed successfully.

## Primary controlled comparison

| Model | Correct | Accuracy | Macro F1 | Parse failures |
|---|---:|---:|---:|---:|
| Focused no-CoT SFT, frozen projector | 175 / 642 | 27.26% | 27.24% | 0 |
| Focused no-CoT SFT, trainable projector | 190 / 642 | 29.60% | 29.56% | 0 |

The trainable-projector checkpoint improves accuracy by 2.34 percentage points.
In the paired comparison, 81 questions change from wrong to correct and 66
change from correct to wrong. A question-level paired bootstrap gives a 95%
interval of -1.40 to +6.07 percentage points; the exact McNemar p-value is
0.248. The observed gain is therefore directional but not statistically
conclusive on this test set.

The gain is +2.44 points among the 574 single-slide cases and +1.47 points among
the 68 multi-slide cases. This does not suggest that the aggregate improvement
is driven specifically by multi-slide handling.

## Comparison with earlier reported checkpoints

| Earlier result | Correct | Accuracy | New checkpoint delta |
|---|---:|---:|---:|
| Submitted full-VQA OncoVLM, repeat 0 on the same 642 IDs | 219 / 642 | 34.11% | -4.52 points |
| Frozen-projector image-step CoT SFT on the same 642 IDs | 175 / 642 | 27.26% | +2.34 points |
| Earlier focused frozen-projector no-CoT SFT on its 271 cached IDs | 79 / 271 | 29.15% | +2.95 points on those IDs |

Against the submitted full-VQA model, the paired difference is -4.52 points
(bootstrap 95% interval -8.72 to -0.31; exact McNemar p=0.0436). The focused
trainable-projector stage should therefore not replace the submitted full-VQA
checkpoint by itself.

## Interpretation boundary

The controlled focused models use the same task, answer-only SFT objective,
LoRA rank and learning rate, effective batch size, best epoch, prompt, and test
questions. They are not a perfect projector-only ablation: the older frozen
run trained on 3,873 cache-available rows for two epochs, while the trainable
run trained live on 3,877 rows and selected epoch 2 from a four-epoch schedule.
The older checkpoint was also trained from cached prefixes, although both are
evaluated here through the same live feature path.
