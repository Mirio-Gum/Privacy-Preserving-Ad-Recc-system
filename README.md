# Privacy-Preserving Federated Recommendation

An empirical study of differential privacy, secure aggregation and non-IID data in a federated matrix factorisation recommender, trained on MovieLens 100K.

This is the code behind my MSc Cybersecurity dissertation at Swansea University (2026). The full write-up is in [`docs/dissertation.pdf`](docs/dissertation.pdf), and the poster is in [`docs/poster.pdf`](docs/poster.pdf).

[![Open Baseline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/federated-mf-privacy/blob/main/notebooks/Baseline.ipynb)
[![Open Securing_Baseline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/federated-mf-privacy/blob/main/notebooks/Securing_Baseline.ipynb)

## What this is about

In federated learning a user's ratings never leave their device — the client sends a model update instead. For matrix factorisation, though, that update is a function of the ratings, and the gradient for the item matrix can be worked backwards to recover them. Hence the update has to be protected too.

Here we protect it with two mechanisms that answer different questions, and we measure what they cost in accuracy:

- **Differentially private SGD (DP-SGD)** — per-example gradient clipping plus calibrated Gaussian noise, so no single rating can move the model much.
- **Secure aggregation** — pairwise additive masks that cancel in the sum, so the server only ever sees the total and never an individual client's update.

Along with this we train on realistic, uneven data (a Dirichlet non-IID split with FedProx to limit client drift), and we test why the privacy cost turns out to be so small.

## Notebooks

| Notebook | What it does |
|---|---|
| [`notebooks/Baseline.ipynb`](notebooks/Baseline.ipynb) | Centralised matrix factorisation benchmark, then plain FedAvg with an IID split. No privacy or security layer. This sets the reference accuracy. |
| [`notebooks/Securing_Baseline.ipynb`](notebooks/Securing_Baseline.ipynb) | The full stack on top of the baseline: Dirichlet non-IID partition, FedProx, DP-SGD (`torch.func`), pairwise-mask secure aggregation, temporal realism, RDP privacy accounting, and the hidden-layer sweep. |

Run `Baseline.ipynb` first if you want the reference numbers; the second notebook is self-contained and does not import from it.

## How it works

**Data.** `u.data` from MovieLens 100K — 943 users, 1,682 items, 100,000 ratings (1–5). Random 80/20 train/test split (`random_state=42`). Matrix factorisation only needs the user ID, item ID and rating, so the other MovieLens files are not used.

**Model.** Biased matrix factorisation, `r̂ = b₀ + b_u + b_i + p_u · q_i`, with 64-dimensional embeddings (168,626 parameters). `TunableRecommender` generalises it: `hidden_layers=0` is exactly the matrix factorisation model, and `hidden_layers ≥ 1` replaces the dot product with that many shared dense layers (moving towards an NCF-style model).

**Federation.** 10 simulated clients, 300 rounds, 5 local epochs per round. All clients take part every round; in the secured notebook a client that has no data yet at that point in time sits the round out.

| Setting | Baseline | Secured |
|---|---|---|
| Partition | IID (users shuffled into 10 groups) | Dirichlet over items, α = 0.5 |
| Local optimiser | Adam, lr 0.01 | Manual SGD step, lr 0.01, batch size 256 |
| Drift control | none | FedProx, μ = 0.1 |
| Privacy | none | DP-SGD, clip norm 1.0, noise multiplier σ (default 4.5) |
| Aggregation | FedAvg | Secure aggregation (weighted FedAvg result) |
| Data availability | all data from round 0 | Ratings appear as their real timestamps are reached |

**DP-SGD.** Per-example gradients come from `torch.func` (`vmap` over `grad`), are clipped to `clip_norm`, summed, given noise with standard deviation `noise_multiplier × clip_norm`, and averaged. Opacus is used only for the Rényi-DP accountant, with δ = 10⁻⁵.

**Secure aggregation.** Every pair of clients shares one random mask per parameter; one adds it, the other subtracts it. Each client multiplies its update by its FedAvg weight *before* masking, and the server does a plain sum. The order matters: weighting after masking breaks the cancellation whenever client weights differ. This was a real bug found during development, and the proof is in Appendix A of the dissertation.

## Results

Test RMSE on the held-out 20,000 ratings.

**Reference points** (`Baseline.ipynb`)

| Setup | RMSE |
|---|---|
| Centralised matrix factorisation | 0.9323 |
| FedAvg, IID split, no privacy | 0.9910 |

**Privacy–utility sweep** (full stack, 300 rounds, δ = 10⁻⁵) — `noise_multiplier` is the only thing changed between rows.

| Noise multiplier σ | ε | Test RMSE |
|---|---|---|
| 0.5 | 629.18 | 1.0958 |
| 1.0 | 80.28 | 1.0960 |
| 2.0 | 23.23 | 1.0958 |
| 3.0 | 13.22 | 1.0964 |
| 4.0 | 9.16 | 1.0960 |
| 4.5 | 7.92 | 1.0968 |

The budget drops by nearly two orders of magnitude while RMSE moves by about 0.001 — within the run-to-run variation. Each user has rated only a handful of items, so their gradient is nonzero on just a few embedding rows. Clipping costs them almost nothing, and most of the noise lands on rows they never touched.

**Is that specific to matrix factorisation?** We test it by turning the `hidden_layers` dial at a fixed ε = 7.92. RMSE stays flat for 0–2 hidden layers, then steps up by roughly 0.035 from 3 layers on and plateaus — a threshold, not a slope (dissertation Figure 5.1). The saved output in `Securing_Baseline.ipynb` covers 6–10 layers:

| Hidden layers | ε | Test RMSE |
|---|---|---|
| 6 | 7.92 | 1.1307 |
| 7 | 7.92 | 1.1413 |
| 8 | 7.92 | 1.1344 |
| 9 | 7.92 | 1.1388 |
| 10 | 7.92 | 1.1404 |

`clip_norm` was held at 1.0 at every depth, so heavier clipping in deeper networks is not ruled out.

## Running it

Both notebooks were developed on Google Colab with a GPU (PyTorch 2.11, CUDA 12.8).

1. Click one of the Colab badges above, or upload the notebook to Colab yourself.
2. Switch the runtime to a GPU (Runtime → Change runtime type).
3. Run all cells. The MovieLens zip is downloaded by the notebook itself — nothing needs to be added to the repo.

Locally, you need a recent Python 3 and:

```
pip install torch numpy pandas scikit-learn opacus
```

(`opacus` is only needed by `Securing_Baseline.ipynb`.)

**To reproduce the sweeps** in `Securing_Baseline.ipynb`, edit the last code cell:

- noise sweep → change `noise_multiplier=4.5` (and the matching value in the `compute_epsilon(...)` call);
- hidden-layer sweep → change the list in `for HIDDEN_LAYERS in [6, 7, 8, 9, 10]`. Use `[0]` for plain matrix factorisation.

Each 300-round run takes a while — rounds start at about a second and grow to around ten as client data accumulates.

## Limitations

- **Simulation only.** All clients run on one machine. The masks are drawn from a random generator with no key agreement, so this shows the arithmetic of secure aggregation, not a deployable protocol.
- **No dropout tolerance.** Every client is assumed to finish every round it is selected for. The privacy accounting assumes that full participation, so it would need re-deriving for a real deployment.
- **Local accounting.** ε comes from the Rényi-DP accountant applied to each client's local DP-SGD, using the average client size.
- **No attack evaluation.** The project measures the accuracy cost of the defences; it does not run an attack (such as membership inference) against them.
- **Artificial federation.** MovieLens was collected centrally and is split into clients by us. It is also denser than real recommender data, and 10 clients is closer to cross-silo than cross-device learning.

## Repository layout

```
.
├── README.md
├── LICENSE
├── notebooks/
│   ├── Baseline.ipynb
│   └── Securing_Baseline.ipynb
└── docs/
    ├── dissertation.pdf
    └── poster.pdf
```

## Data

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/), provided by GroupLens Research at the University of Minnesota. The dataset is not included here; the notebooks download it directly from GroupLens. Please follow their usage terms.

> F. M. Harper and J. A. Konstan, "The MovieLens datasets: History and context," *ACM Transactions on Interactive Intelligent Systems*, vol. 5, no. 4, 2015.

## Citing

If this is useful to you:

> B. Das, "Privacy-Preserving Federated Recommendation: An Empirical Study of Differential Privacy, Secure Aggregation and Non-IID Data," MSc dissertation, Swansea University, 2026.

## Licence

The code is released under the MIT licence (see `LICENSE`). The dissertation and poster in `docs/` are © Bodhisattwa Das, all rights reserved.
