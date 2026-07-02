# Main 50k Results: All Methods on Two Datasets

## Protocol

Both datasets contain 50,000 samples, balanced as 25,000 benign and 25,000
malware. Every method reads the same fixed split:

- train: 35,000
- validation: 7,500
- test: 7,500
- split seed: 7

All values below are test-set results. Complete machine-readable results are in
`all_method_results.csv`; per-run configurations and validation histories remain
in each dataset's `results/*_metrics.json`.

## Quo Vadis Malware Emulation

| Method | Implementation | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| DeepCapa-adapted | adapted fallback | **0.9649** | 0.9693 | **0.9603** | **0.9648** | **0.9953** |
| Ours | proposed | 0.9621 | 0.9707 | 0.9531 | 0.9618 | 0.9947 |
| GPT4-API-BERT-CNN | adapted | 0.9552 | **0.9723** | 0.9371 | 0.9544 | 0.9926 |
| MME-TextCNN | fallback reimplementation | 0.9380 | 0.9252 | 0.9531 | 0.9389 | 0.9890 |
| APILI-adapted | adapted | 0.9387 | 0.9370 | 0.9405 | 0.9388 | 0.9869 |
| API2Vec++ | baseline | 0.9265 | 0.9083 | 0.9488 | 0.9281 | 0.9824 |
| Nebula-Scratch | baseline | 0.9113 | 0.8916 | 0.9365 | 0.9135 | 0.9708 |
| DawnGNN-reimpl | reimplementation | 0.8741 | 0.8409 | 0.9229 | 0.8800 | 0.9458 |

## Zenodo 11079764

| Method | Implementation | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| Ours | proposed | **0.9512** | **0.9432** | **0.9603** | **0.9516** | **0.9874** |
| DeepCapa-adapted | adapted fallback | 0.9447 | 0.9364 | 0.9541 | 0.9452 | 0.9863 |
| APILI-adapted | adapted | 0.9360 | 0.9256 | 0.9483 | 0.9368 | 0.9806 |
| MME-TextCNN | fallback reimplementation | 0.9279 | 0.9184 | 0.9392 | 0.9287 | 0.9789 |
| GPT4-API-BERT-CNN | adapted | 0.9220 | 0.9135 | 0.9323 | 0.9228 | 0.9737 |
| API2Vec++ | baseline | 0.9052 | 0.8907 | 0.9237 | 0.9069 | 0.9611 |
| Nebula-Scratch | baseline | 0.8025 | 0.8363 | 0.7523 | 0.7921 | 0.8994 |
| DawnGNN-reimpl | reimplementation | 0.8215 | 0.8956 | 0.7277 | 0.8030 | 0.9189 |

## Family Classification

Only the four newly added methods report family classification. Detailed
results are stored in `family_classification_results.csv`.

| Dataset | Method | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro AUC |
|---|---|---:|---:|---:|---:|---:|
| Quo Vadis | GPT4-API-BERT-CNN | 0.8825 | 0.8244 | 0.7811 | 0.7982 | 0.9849 |
| Quo Vadis | DeepCapa-adapted | 0.8773 | 0.8423 | 0.8567 | **0.8440** | **0.9882** |
| Quo Vadis | APILI-adapted | 0.8104 | 0.7602 | 0.7272 | 0.7116 | 0.9768 |
| Quo Vadis | MME-TextCNN | 0.8459 | 0.8235 | 0.7913 | 0.7968 | 0.9828 |
| Zenodo | GPT4-API-BERT-CNN | 0.7355 | 0.1817 | 0.1163 | 0.1182 | 0.9182 |
| Zenodo | DeepCapa-adapted | 0.7173 | **0.3671** | **0.3068** | **0.3113** | **0.9682** |
| Zenodo | APILI-adapted | 0.6419 | 0.2636 | 0.2323 | 0.2249 | 0.9357 |
| Zenodo | MME-TextCNN | 0.6669 | 0.3348 | 0.2533 | 0.2571 | 0.9476 |

The GPT4-API-BERT-CNN family task includes benign as a class and evaluates all
7,500 test samples. The other adapted methods report family metrics on the
3,750 malware test samples only, so their family accuracy values are not
strictly comparable to GPT4-API-BERT-CNN.

## Reproduction Status

- Nebula-Scratch and API2Vec++ use the maintained local baseline runners.
- DawnGNN is the updated paper-style transition-graph implementation using
  documentation-style semantic node embeddings and a two-layer GAT.
- GPT4-API-BERT-CNN is adapted to the compact API-sequence datasets and uses
  DeepSeek-generated API descriptions with a BERT-CNN detector.
- DeepCapa-adapted is a fallback reimplementation because the datasets do not
  provide the original MITRE capability labels or compatible checkpoint.
- APILI-adapted transfers the API/resource attention design to binary and
  family classification because public training code and technique labels are
  unavailable.
- MME-TextCNN is a fallback reproduction because the official repository does
  not include training source or the original knowledge graph.

For a paper, retain these qualifiers in method names and experimental setup.
