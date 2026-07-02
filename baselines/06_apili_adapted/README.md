# APILI-adapted

This baseline adapts [APILI: Attention-Based API Locating for Malware
Techniques](https://github.com/Irish-kw/Attention-Based-API-Locating-for-Malware-Techniques)
to the compact 50k malware classification datasets in this project.

APILI's original task is MITRE ATT&CK technique prediction and locating the API
calls associated with discovered techniques in dynamic execution traces. It is
not originally a direct malware detector. **APILI-adapted transfers the original
API locating / technique discovery approach to benign-vs-malware detection and
malware family classification.**

## Adaptation

- Input is the existing `api_seq` Windows API sequence.
- Dataset `split` values are used verbatim; samples are never re-split.
- A frozen BERT model provides semantic representations of API names.
- Separate API and resource attention branches retain APILI's attention idea.
- Because these datasets do not contain arguments, resources, or technique
  labels, each API name is converted into a simplified resource sentence.
- The technique prediction head is replaced by a benign/malware head.
- A malware-family auxiliary head is trained when family labels are available.
- The validation split selects the decision threshold; the test split is used
  only for final reporting.

The cloned official repository is preserved under `official_code/`. At commit
`84a18757896347f032f49aec4dd1907bbae7bb07`, the public repository mainly
contains prediction code and points to a separate "Training from sketch"
download. The adapted implementation is therefore maintained separately in
`train_apili_adapted.py`.

## Run

```bash
PYTHON=/path/to/python

"$PYTHON" train_apili_adapted.py --dataset quo_vadis
"$PYTHON" train_apili_adapted.py --dataset zenodo_11079764
```

The default semantic model is the frozen lightweight BERT checkpoint
`prajjwal1/bert-tiny`. Change it with `--bert-model` if a larger BERT checkpoint
is desired.

## Outputs

- `logs/quo_vadis.log`
- `logs/zenodo_11079764.log`
- `../../datasets_50k/quo_vadis/results/apili_adapted_metrics.json`
- `../../datasets_50k/zenodo_11079764/results/apili_adapted_metrics.json`
- One `apili_adapted` row appended to each dataset's `baseline_results.csv`
- Encoded arrays and checkpoints under
  `datasets_50k/<dataset>/artifacts/apili_adapted/`

The metrics JSON reports Accuracy, Precision, Recall, F1, and ROC AUC for both
validation and test splits, plus family metrics when the family head is active.

## Results

Both runs used seed 7, the existing dataset splits, `api_seq`, frozen
`prajjwal1/bert-tiny` semantics, and Apple MPS.

| Dataset | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quo Vadis test | 0.9387 | 0.9370 | 0.9405 | 0.9388 | 0.9869 |
| Zenodo test | 0.9360 | 0.9256 | 0.9483 | 0.9368 | 0.9806 |

The auxiliary family head was evaluated on malware samples with family labels.
Its test accuracy was 0.8104 on Quo Vadis (7 families) and 0.6419 on Zenodo
(124 training families). Detailed macro metrics are stored in the corresponding
metrics JSON files.
