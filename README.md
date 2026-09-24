[README_1.md](https://github.com/user-attachments/files/32614725/README_1.md)
# Open-circuit fault diagnosis in a three-level T-type converter

Model, trained network and MATLAB scripts for the paper:

> R. Almasri and M. D. Siddique, "Open-Circuit Fault Diagnosis in a Three-Level
> T-Type Converter: Reproduction, Baseline Comparison and Noise Robustness",
> *Journal of Undergraduate Research International*, submitted 2026.

Department of Electrical Engineering, King Fahd University of Petroleum &
Minerals, Dhahran 31261, Saudi Arabia. Correspondence: s202452160@kfupm.edu.sa

## Requirements

MATLAB R2024a or later (developed on R2025b) with Simulink, Simscape Electrical
and Deep Learning Toolbox. The Statistics and Machine Learning Toolbox is **not**
required — no script uses `gscatter` or `confusionmat`.

## Files

| File | Produces |
|---|---|
| `new_t_type_three_level.slx` | the converter model, instrumented for open-circuit fault injection at any of the nine gate signals |
| `collect_data_paper.m` | the dataset of 2,500 records (Section 3.2, Table 1). About 3 h |
| `train_cnn.m` | the reproduction at 100.00 ± 0.00 % and the per-class recall (Section 4.1). About 15 min |
| `noise_sweep.m` | the nine-feature baseline and the noise sweep (Section 4.4, Fig. 4; Supplementary Table S2). About 20 min |
| `compare_configs_noisy2.m` | the input-representation comparison and error structure at 30 % noise, ten initialisations (Supplementary S4, S5, Tables S3–S5). About 38 min |
| `diagnose_live.m` | end-to-end demonstration: fresh simulation, then the predicted switch |
| `make_fig5.m` | Fig. 5, the sliding-window trace, and `fig5_data.mat`. About 15 s |
| `latency_sweep.m` | Fig. 6, detection latency over nine switches × twelve fault instants, and `latency_results.mat`. About 2 min |
| `diagnosis_net.mat` | the trained network used by `diagnose_live.m`, `make_fig5.m` and `latency_sweep.m` (configuration A, 2 % noise augmentation) |

## The dataset

`ttype_dataset_v3.mat` (110 MB) exceeds GitHub's 100 MB per-file limit and is not
in this repository. It is available from the corresponding author on request.

It is also reproducible: run `collect_data_paper.m` against
`new_t_type_three_level.slx`. The random seed is fixed at 42, so the regenerated
dataset is identical. Expect about 3 hours; `parsim` with four workers reduces
this to roughly 25 minutes.

The file contains `X` (2500 × 3000), `Y`, the operating conditions in `cond`, and
the stratified 1,000 / 500 / 1,000 train / validation / test split.

## Running anything

Put the script, `new_t_type_three_level.slx`, `diagnosis_net.mat` and
`ttype_dataset_v3.mat` in one folder and `cd` MATLAB into it. Parameters are set
at the top of each file — edit them there, not at the command line, because the
`clear` on the first line wipes anything set from the prompt.

## A note on figures

MATLAB R2025a and later export graphics in the desktop's colour theme, so on a
dark desktop `exportgraphics` writes figures with a black axes background.
`make_fig5.m` and `latency_sweep.m` force `fig.Theme = 'light'` and set every
colour explicitly to avoid this.

## Verifying a regenerated dataset

Before trusting regenerated data, check it against the expected physics:

```matlab
load('ttype_dataset_v3.mat')
size(X)                                 % 2500  3000
histcounts(Y, -0.5:1:9.5)               % ten 250s
any(~isfinite(X(:)))                    % 0
[min(X(:)) max(X(:))]                   % about -215 .. +215

% the load must really vary: this must slope upward, not be a flat cloud
pk = max(abs(X),[],2);  r0 = find(Y==0);
scatter([cond(r0).Pload], pk(r0))

% power factor must be 0.95-0.985 with ZERO runs above 0.99
pf = [cond.Pload]./sqrt([cond.Pload].^2 + [cond.Qload].^2);
[min(pf) max(pf)], sum(pf > 0.99)

% three-wire check on a faulted run: must be ~1e-9 A, not tens of amps
r = find(Y==1); W = reshape(X(r(1),:), 1000, 3);
max(abs(sum(W,2)))
```

Expected signatures: an open-circuit fault in an outer switch suppresses one
half-cycle of its own phase current — S1 and S2 act on phase a, S3 and S4 on
phase b, S5 and S6 on phase c — while a clamp-switch fault (S7, S8, S9) distorts
its phase around the zero crossings without removing a half-cycle.
