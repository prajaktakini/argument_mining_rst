# RST-Augmented Argument Mining

**Multi-task R-GCN using RST discourse trees as typed graph edges.**

---

## Setup

```bash
# 1. Create environment
conda create -n rst_am python=3.10
conda activate rst_am

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install RST parser (DMRST)
python -m isanlp_rst download        # downloads pretrained checkpoint

# 4. Install spaCy English model
python -m spacy download en_core_web_sm
```

## Data Download

**IBM ArgKP** (Key Point Analysis, 27k pairs):
```
https://research.ibm.com/haifa/dept/vst/debating_data.shtml
→ "Key Point Analysis" section → download → extract to data/raw/ArgKP/
```

**UKP ConvArg** (Convincingness, 16k pairs):
```
https://github.com/UKPLab/acl2016-convincing-arguments
→ UKPConvArg1Strict-XML.zip → extract to data/raw/ConvArg/
```

Expected layout:
```
data/raw/
  ArgKP/
    train/arguments.csv, key_points.csv, labels.csv
    dev/  ...
    test/ ...
  ConvArg/
    UKPConvArg1Strict-XML/
      <topic>/
        *.xml
```

## Training

```bash
# Full model
python train.py

# With W&B logging
python train.py --use_wandb --run_name my_run

# Override hyperparameters
python train.py --epochs 20 --lr_rgcn 5e-4 --batch_size 32
```

## Ablations

```bash
# Run all five ablations automatically
bash scripts/run_ablations.sh

# Print comparison table
python scripts/compare_ablations.py
```

Individual ablations:
```bash
python train.py --ablation_untyped_edges   # collapse all RST edge types → 1
python train.py --ablation_no_nuclearity   # uniform node weights
python train.py --ablation_no_graph        # RoBERTa-only baseline
python train.py --ablation_single_task --active_task kp_match
```

## Project Structure

```
rst_am/
├── config.py              # all hyperparameters + RST→AM edge type mapping
├── train.py               # training loop + ablation CLI flags
├── evaluate.py            # metrics: macro-F1, mAP, Spearman ρ, MAE
├── data/
│   ├── rst_pipeline.py    # RST parser wrapper, EDU-span aligner, disk cache
│   └── datasets.py        # ArgKP + ConvArg loaders → AMGraphSample
├── models/
│   ├── encoder.py         # RoBERTa span encoder (freeze/unfreeze)
│   ├── graph_builder.py   # RSTGraph → PyG HeteroData
│   └── multitask.py       # RGCNLayer + four task heads + loss
└── scripts/
    ├── run_ablations.sh   # run all five ablations
    └── compare_ablations.py
```

## RST → AM Edge Type Mapping

| RST relation | AM edge type | Index |
|---|---|---|
| Evidence, Cause, Result, Justify, Purpose | SUPPORTS | 0 |
| Contrast, Antithesis | ATTACKS | 1 |
| Concession | CONCEDES | 2 |
| Elaboration, Background, Preparation | ELABORATES | 3 |
| Restatement, Summary | SUMMARISES | 4 |
| Condition | CONDITIONAL | 5 |
| Sequence, Joint, List, Same-unit | SEQUENCE | 6 |

Nuclearity salience weights: Nucleus=1.0, Satellite=0.6, Multinuclear=0.8

## Citation

If you use this code, please cite:
- Bar-Haim et al. (2020) *From Arguments to Key Points.* ACL.
- Habernal & Gurevych (2016) *Which Argument is More Convincing?* ACL.
- Liu et al. (2021) *DMRST: A Joint Framework for Document-Level Multilingual RST Parsing.* EMNLP.
- Chistova (2023) *End-to-End Argument Mining over Varying Rhetorical Structures.* ACL Findings.
