# Emergent Stack Representations in Modeling Counter Languages Using Transformers

Code for the paper [**Emergent Stack Representations in Modeling Counter Languages Using Transformers**](https://arxiv.org/abs/2502.01432) (arXiv: 2502.01432).

**Authors:** Utkarsh Tiwari\*, Aviral Gupta\*, Michael Hahn†
**Affiliations:** Birla Institute of Technology and Science, Pilani; Saarland University
**Correspondence:** {f20220052, f20220097}@pilani.bits-pilani.ac.in

## Overview

Transformer architectures are increasingly capable of modeling complex sequential patterns, but their inner workings remain largely opaque. This work investigates whether transformers trained on **counter languages** — formal languages modeled using counter machines with stack memory — develop internal representations that mimic stack-like structures.

We train transformer models on 4 counter languages as next-token predictors and probe their internal representations for stack depths at each input token. Our results show that these models learn stack-like representations, with high probing accuracy and near-random control task accuracy, confirming that the probes recover genuine structure rather than spurious correlations.

### Key Findings

- Probing classifiers (even linear probes) achieve high accuracy in predicting stack depth from transformer encoder representations
- Control tasks with randomized labels stay near chance level, yielding high **selectivity** (task accuracy − control accuracy)
- Probing accuracy is higher for Shuffle-k languages than Dyck-1, and increases with k
- Results are validated on a **Tracr**-compiled model with a known algorithm

## Languages Studied

| Language | Description | Stacks |
|----------|-------------|--------|
| **Dyck-1** | Well-formed parentheses `()` | 1 |
| **Shuffle-2** | Shuffle of two Dyck-1 languages with disjoint alphabets `()` and `[]` | 2 |
| **Shuffle-4** | Shuffle of four Dyck-1 languages | 4 |
| **Shuffle-6** | Shuffle of six Dyck-1 languages | 6 |

## Repository Structure

### Core Paper Experiments

```
aviral_transformer_formal_language/
├── src/                          # Model architecture and training code (from Bhattamishra et al., 2020)
│   ├── model.py                  # LanguageModel wrapper
│   ├── components/
│   │   └── transformers.py       # TransformerModel (encoder-only + LM head)
│   ├── dataloader.py             # Data loading and batching
│   └── utils/                    # Data generators (Dyck, Shuffle, etc.)
├── data/                         # Training/validation data for all languages
├── models/                       # Pretrained model checkpoints
│   └── testrun_dyck1_4h/         # Dyck-1 model (1-layer, 4-head, d_model=32)
├── tst.ipynb                     # Dyck-1 probing with ablation over probe depth (paper Fig. 3)
├── tstshuffle2.ipynb             # Shuffle-2 probing (Stack 1)
├── tstshuffle2_stack2.ipynb      # Shuffle-2 probing (Stack 2)
├── tstshuffle4.ipynb             # Shuffle-4 probing
├── tstshuffle6.ipynb             # Shuffle-6 probing
└── probing.py                    # Probing dataset utilities
```

### Additional Experiments

```
# aⁿbⁿ language (binary classification + probing)
anbn.py                  # Train transformer on aⁿbⁿ, probe for stack top, control task, selectivity ablation
anbn-ut.py               # Earlier version of anbn.py (runs on MPS)
anbn_ut.ipynb             # Notebook version with full probing pipeline and ablation
anbn_probe.ipynb          # aⁿbⁿ probing with activation visualization
anbnclaude.ipynb          # aⁿbⁿ with custom-built transformer (3-layer)

# Dyck-1 (balanced bracket classification + probing)
paren_1.py / paren_1.ipynb      # Train bracket classifier, Dyck-1
paren_2.py / paren_2.ipynb      # Train bracket classifier, extended to longer sequences
paren_probe.py / paren_probe.ipynb  # Probe trained bracket classifier for stack depth
tst.ipynb                        # Quick probing experiment on aviral's Dyck-1 model

# Control experiments
control/
├── ANBNTransformer_src.py   # ANBNTransformer model definition
└── anbncontrol.ipynb        # Control experiment on aⁿbⁿ model

# Multi-parenthesis (Dyck-k with multiple bracket types)
multipar/
├── mp.py                    # Model and training for ()[]{} classification
├── multi_paren.ipynb        # Training with global pooling classifier
├── trans.ipynb              # Earlier attempt (has debug prints)
└── trans2.ipynb             # Revised multi-bracket training

# Tracr verification
tracr_clone/                 # Tracr (compiled transformers) for probing validation
└── tracr/examples/
    ├── probe2.ipynb                              # Probing on Tracr-compiled Dyck-1 model
    └── Visualize_Tracr_Models.ipynb              # Visualization

# Reference material
part51_balanced_bracket_classifier/               # ARENA 3.0 bracket classifier exercises (reference)
```

## Method

### 1. Language Model Training

We use an encoder-only transformer with a linear decoder layer (language-modeling head), following [Bhattamishra et al. (2020)](https://arxiv.org/abs/2009.11264). The model is trained on the **next-token prediction** task: given a prefix, predict the set of valid next characters.

**Architecture:** 1-layer transformer, d_model=32, d_ffn=64, 4 attention heads, no positional encoding
**Training:** RMSProp optimizer, lr=5×10⁻³, batch size 32, 25 epochs, MSE loss

### 2. Probing Setup

For each token in a sequence, we:
1. Extract the encoder's hidden representation (dimension d_model)
2. Compute the ground-truth stack depth by simulating a stack
3. Train a feed-forward probing classifier to predict stack depth from the hidden state

**Probing classifiers:** Fully connected networks ranging from linear to 6 layers with ReLU activations (hidden size 128, dropout 0.2, Xavier initialization, Adam optimizer, lr=0.001, batch size 32, 10 epochs, cross-entropy loss).

For Shuffle-k languages, we train **k separate probes** — one per stack.

### 3. Control Task

Following [Hewitt and Liang (2019)](https://arxiv.org/abs/1909.03368), we create a control task by **randomizing the target labels** for the same input representations. The **selectivity** (task accuracy − control accuracy) measures whether the probe recovers genuine structure.

### 4. Tracr Verification

We compile a Dyck-1 recognizer using [Tracr](https://github.com/google-deepmind/tracr) from a RASP program and apply the same probing setup. Since the Tracr model uses a known algorithm, successful probing validates the methodology.

## Results

Probing accuracy across probe architectures (from paper Figure 3):

| Language | Linear | 2 layers | 3 layers | 4 layers | 6 layers | Control |
|----------|--------|----------|----------|----------|----------|---------|
| Dyck-1 | 57.4 | 54.9 | 57.2 | 63.5 | 64.9 | ~4.6 |
| Shuffle-2 (Stack 1) | 75.5 | 83.4 | 86.8 | 88.7 | 88.1 | ~6.8 |
| Shuffle-4 (Stack 1) | 90.8 | 94.7 | 96.5 | 97.5 | 97.0 | ~7.2 |
| Shuffle-6 (Stack 1) | 95.5 | 98.0 | 98.9 | 99.1 | 98.7 | ~9.2 |

High task accuracy with near-random control accuracy indicates genuine stack-like representations.

## Setup

### Prerequisites

- Python 3.6+
- PyTorch 2.0+

### Installation

```bash
git clone https://github.com/ut21/counter-language-probing.git
cd counter-language-probing

# Install dependencies for the aviral framework
cd aviral_transformer_formal_language
pip install -r requirements.txt
cd ..
```

### Data

The training data for Dyck-1 and Shuffle-k languages is included under `aviral_transformer_formal_language/data/`. To generate additional datasets:

```bash
cd aviral_transformer_formal_language
python generate_data.py -lang Shuffle -num_par 4 -dataset Shuffle-4 \
  -lower_window 2 -upper_window 50 -training_size 10000 -test_size 2000 -bins 2 -len_incr 50
```

### Running the Probing Experiments

The main experiments are in Jupyter notebooks. To reproduce the paper's Dyck-1 results:

```bash
cd aviral_transformer_formal_language
jupyter notebook tst.ipynb
```

**Note:** When loading pretrained checkpoints with PyTorch 2.6+, use `weights_only=False`:
```python
torch.load(chkpt_path, map_location='cpu', weights_only=False)
```

## Citation

```bibtex
@article{tiwari2025emergent,
  title={Emergent Stack Representations in Modeling Counter Languages Using Transformers},
  author={Tiwari, Utkarsh and Gupta, Aviral and Hahn, Michael},
  journal={arXiv preprint arXiv:2502.01432},
  year={2025}
}
```

## Acknowledgments

The transformer training framework is based on [Bhattamishra et al. (2020)](https://github.com/satwik77/Transformer-Formal-Languages). The Tracr verification uses [Tracr](https://github.com/google-deepmind/tracr) by Lindner et al. (2023). The balanced bracket classifier reference material is from [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0).

## License

MIT
