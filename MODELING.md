# Context-Aware, Leakage-Audited Job-Outcome Modeling on FIFE Batch Logs

This document is the living record of what we have built, what we found, and the
evidence behind each claim. It is meant to be updated as experiments continue (see the
[Changelog](#changelog) at the end).

**Thesis.** A job-outcome model is only useful if every feature it reads is knowable *at the moment
the prediction is made*, and if it is evaluated *out-of-time*. Most of the apparent skill in naive
job-failure models is leakage or within-period memorization. Our contribution is a **leakage-audited,
prediction-time-honest benchmark** that (a) separates predictions by when they fire (submission vs
match time), (b) removes outcome-correlated features and missingness, (c) evaluates on a temporal
split, and (d) reframes fault attribution to the split that is actually separable and actionable —
**Hardware/infrastructure vs Payload/user fault**.

**Terminology — fault vs. failure (used consistently throughout).** Following standard dependability
usage:

- A **failure** is the observable *outcome*: a job that did not complete successfully (`ExitCode != 0`,
  or a signal, or removed). `Failed` is this binary; "failure rate" always means the fraction of jobs
  with this outcome.
- A **fault** is the *attributed cause* of a failure — the payload/user side (**Payload fault**:
  code, inputs, or configuration) or the infrastructure (**Hardware fault**). A fault, when it
  manifests, *produces* a failed job; a payload fault is not itself a "failure."

So there are two distinct tasks: **failure prediction** (`Failed` — will the job fail?) and **fault
attribution** (`hw_fault` — given a failure, was the fault Hardware or Payload?). We do *not* say
"Payload failure" or "Hardware failure"; those are faults.

---

## 1. Dataset

| Duration         | Jobs       | Batches    | Users | Sites | Failed    | % Failed |
| ---------------- | ---------- | ---------- | ----- | ----- | --------- | -------- |
| 01/2024–06/2024 | 42,720,512 | 18,198,133 | 412   | 46    | 7,638,662 | 17.88%   |

Modeling uses the processed monthly parquets (`processed_job_data_0[1-6].parquet`, 41,079,913 rows
with the columns retained). The ~1.6M gap from the 42.72M summary total is **jobs that were queued
but never started, which were removed during preprocessing.** Consequently the modeled population is
conditioned on jobs that eventually ran: never-started removals (removed-while-idle) are out of
scope, and the submission-time `wait_time` target is defined only for jobs that reach a start — i.e.
"how long until this job runs," not "will it ever run." Failure is defined as `ExitCode != 0`
(non-null) **or** non-null `ExitSignal` **or** `JobStatus == 3` (removed). Data is Fermilab FIFE (glideinWMS pilots across ~46
grid sites); jobs are organized into POMS campaigns/stages for production and standalone submission
batches (`ClusterId`) for user jobs.

---

## 2. Prediction-time framing (central design)

Every target is assigned to the earliest moment it could be predicted, and may only use features
known by then. This is enforced physically — the two families have disjoint feature matrices.

| Family                    | Fires when                      | May use                                                                                                                                                                                         | Targets                                         |
| ------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Submission-time** | job enters the queue, pre-match | requested resources, campaign/owner/image identity, submission-clock cyclical time, leak-free trailing campaign rates, queue-depth-at-submit                                                    | `wait_time` (regression); (holds, deprecated) |
| **Match-time**      | job matched to a pilot slot     | everything above**plus** matched site/entry/resource/node, provisioned resources, execution-start cyclical time, running concurrency, leak-free trailing **site/entry/node health** | `Failed`, `hw_fault` (Hardware vs Payload)  |

**Feature-timing classification (abridged).**

- *Submission-known:* `Group`, `Owner`, `POMS4_CAMPAIGN_*`, `SingularityImage`, `Request{Cpus,Disk,Memory,Slots}`, `ExecutableSize`, `JOB_EXPECTED_MAX_LIFETIME`, `TotalSubmitProcs`, `proxy_hours_left`, `stage_type`, **qdate** cyclical, trailing rates keyed on campaign/owner/group.
- *Match-known only:* `MATCH_*` (site/entry/queue/resource), `MachineAttr*`, `*Provisioned` (provisioned ≠ requested), **jstart** cyclical, running concurrency, trailing node/site/entry health.

---

## 3. Label construction

### 3.1 Fault types — the attributed cause of a failure

Every failed job is attributed to a fault, defined by **what action fixes the underlying fault**,
not by blame (renamed from Application/User/System on 2026-07-17; see `README.md` for the full
code/signal tables, which are the authority). The fine-grained cascade produces three fault
categories, of which the first two are both **Payload faults**:

- **Application** (payload fault) — the payload code broke; the fix is in the code or its input
  data. All crash signals (SIGSEGV/SIGABRT/SIGILL/SIGBUS/SIGFPE/SIGTRAP/SIGPIPE) and all runtime
  art/C++ exception exit codes (2–14, 21–23, 28, 31–33, 66–89) map here. Resubmitting the identical
  job reproduces it.
- **Submission** (payload fault) — the request/config was wrong; the fix is the submission. Only
  exit codes `9, 65, 90, 91, 124, 126, 127, 130, 131, 137` (config/env/PATH/FHiCL/timeout) and the
  corresponding hold reasons.
- **Hardware** (infrastructure fault) — the infrastructure failed the job; nothing about the job
  changes. `129` (SIGHUP, lost session), `143` (SIGTERM, drain/maintenance), `JobRouter`/unmatched
  removals.

Label cascade (first match wins): `ExitSignal` table → `ExitCode` table → `LastHoldReasonCode` →
`RemoveReason` rules. Removed-by-user (`condor_rm (by user`) with no other evidence is excluded.

**Predictive fault attribution collapses to `hw_fault` = Hardware vs Payload** (Application +
Submission), because Application and Submission are inseparable at match time (§7.2). The three-way
split remains only as the descriptive cascade that *identifies* which failures are Hardware.

**Population (2024 H1, typed failures ≈ 7.68M):** **Payload 84%** (Application 67% + Submission 17%),
**Hardware 16%.** So of all jobs (18–24% fail depending on the period), the large majority of
failures are payload faults (user code / inputs / configuration) and only ~16% of failures — on the
order of **~3–4% of all jobs** — are Hardware faults. A high raw failure rate is therefore mostly a
statement about user payloads, not infrastructure health; separating the two is exactly why the
fault-vs-failure distinction matters for the paper.

### 3.2 Hold types (analysis only; deprecated as a prediction target)

Over held jobs, excluding benign spooling/manual/output-transfer: **memory/disk** (code 34, 26-mem/disk),
**runtime** (26-runtime), **node failure** (6, 35), **other policy**. Evidence-driven findings:

- The "other policy" bucket is **95% code-26 `Shadows/limit N/5`** (restart-budget exhaustion) + 5%
  code-3 PeriodicHold — a real "excessive restarts" mode, not a catch-all.
- Spooling (code 16) is **20.9M jobs (~91% of held jobs)**, benign, near-baseline failure rate — it
  swamps any hold-type classifier unless capped.

Holds were dropped as a modeling target (see §7); the recent hold *rate* is retained as a trailing
feature for failure prediction.

---

## 4. Leakage audit (the methodological core)

Each of these was measured, not assumed. All are excluded from the honest models.

| Leak                                   | Evidence                                                                                                                                                                                | Action                                                     |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `JobStatus`                          | directly encodes the`Failed` label                                                                                                                                                    | excluded                                                   |
| `LastHoldReasonCode`                 | outcome-revealing                                                                                                                                                                       | excluded (kept only to build labels)                       |
| `JobPrio`                            | adjustable in reaction to job state                                                                                                                                                     | excluded                                                   |
| `JobCurrentWaitTime`                 | **= (JobCurrentStartDate − QDate) exactly (corr 1.000)**; median wait by `NumJobStarts`: 1→1,817 s, 2→22,409 s, 3→43,927 s — bakes in post-hold requeue time               | excluded (now a*target*, using first-start − submit)    |
| Month-of-year cyclical                 | top feature (17–18% gain); per-site node-fail rate**corr(Jan, June) = 0.11**; `owner+month` temporal ROC **0.583 < owner-only 0.724** (month actively hurts out-of-time) | excluded                                                   |
| `Drain`                              | only values are`false`/null; **null → 51.1% failure vs false → 17.7%** — missingness is a fingerprint of an abnormal run                                                     | dropped                                                    |
| Match-attribute missingness            | all`MATCH_*` categoricals share a **2.3% null rate → 49.9% failure vs 17.9% populated**                                                                                        | match tasks scoped to non-null match attrs (matched 97.7%) |
| Duplicate/contemporaneity memorization | **34.8%** of jobs sit in near-pure `(owner, month)` groups; full submission model **random ROC 0.968 → temporal 0.665**                                                  | temporal split + trailing*state* (not identity) features |

**Standing guardrail:** a missingness-vs-failure audit cell screens every candidate feature (null
rate, failure rate null vs populated); features whose nulls fail far more are flagged. It cleared the
campaign fields (66% null, ~0 gap) and provisioned columns (near-zero null count).

---

## 5. Feature engineering

### 5.1 Leak-free trailing state features (validated)

For each job, the rate of an outcome (fail / hold / hardware-fault) over same-key jobs that
**completed strictly before this job's reference time** (submission for the submission family, start
for the match family), within a rolling window. A job's own completion is explicitly excluded, so it
can never see its own outcome even under data-clock quirks.

- **Validation:** exact match to a leak-free brute-force reference (0 count/rate difference), 0
  self-exclusion failures, 98–99% coverage.
- **Key granularity hierarchy:** `site` (≈44) ≈ `resource` (≈46) < `entry` (≈104) < `node` (`LastRemoteHost`, ≈57k+). `_hw` variants are hardware-fault rates at each zoom level.
- **Window ablation:** **1 min and 5 min windows carry the gain**; 15s/30s are too sparse (~2% gain even at 89% coverage) and 15m/30m/1h smear the signal (~0%). Only 1m/5m (+1h anchor, node 15m/1h) are kept.
- **Queue state:** `idle_queue_depth` (jobs submitted-but-not-yet-started at submit time, leak-free) and campaign trailing **mean queue-wait** are the top wait-time predictors.

### 5.2 Other features

Campaign-aware grouping (`POMS4_CAMPAIGN_ID`, else `ClusterId`); running concurrency via a validated
event-sweep; cyclical time for both clocks (qdate + jstart, **no month**); `stage_type` from the
campaign-stage regex; `proxy_hours_left`.

---

## 6. Evaluation protocol and the memorization decomposition

Splits, from optimistic to honest, holding model/features fixed:

- **Random** — unseen rows but not unseen conditions.
- **Campaign-aware** — whole campaigns (or `ClusterId` batches) to one side; removes duplicate memorization, keeps contemporaneity.
- **Temporal** — train months 1–4, test 5–6; the deployment-honest protocol.

**`Failed` across protocols (same model/features):** ROC **0.96 random → 0.88 campaign-aware → 0.78
leak-free temporal.** Most of the optimism is contemporaneity, not row duplication.

**Feature-ablation, memorization check** (temporal, train Apr+May → test Jun; target `Failed`):

| feature set                                | random ROC      | temporal ROC    |
| ------------------------------------------ | --------------- | --------------- |
| owner only                                 | 0.778           | 0.724           |
| owner + month                              | 0.827           | **0.583** |
| shape only (resources, no identity/month)  | 0.878           | 0.736           |
| shape + image                              | 0.881           | **0.744** |
| full (owner+campaign+month+shape+cyclical) | **0.968** | 0.665           |

Reading: **pure job characteristics generalize best out-of-time (0.744)**, beating the identity-laden
full model (0.665) which overfits. Identity + month is memorization.

---

## 7. Results (latest: leak-free, temporal, 20% subset, 3 GBMs)

Config: XGBoost / LightGBM / CatBoost, 200 trees, depth 8, lr 0.1; `scale_pos_weight` on binaries;
balanced weights on multiclass. Match tasks scoped to matched jobs.

### 7.1 Failure detection and attribution (match-time)

| task                                               | model    | ROC-AUC         | PR-AUC | recall @0.5 | recall @F1-max | notes                                                 |
| -------------------------------------------------- | -------- | --------------- | ------ | ----------- | -------------- | ----------------------------------------------------- |
| `Failed`                                         | CatBoost | **0.784** | 0.486  | 0.50        | 0.57           | top features`site_fail_5m`/`site_fail_1m`         |
| `hw_fault` (Hardware vs Payload, among failures) | CatBoost | **0.678** | 0.320  | 0.56        | 0.73           | top features`entry_fail_1m` (29%), site/node health |

- **`hw_fault` correctly leans on infrastructure health** (`entry_fail_1m`, `MATCH_EXP_JOB_GLIDEIN_Site`, `node_fail_1h`, `node_hw_1h`, `site_hw_1h`) — the signal that separates a systemic fault from a payload bug. Confusion @0.5: Payload 0.64 correct, Hardware 0.56 recall.
- Modest ceiling is honest: a healthy node can fail mid-job with no trailing warning.

### 7.2 Why Application vs Submission was abandoned

- **68% of campaigns emit both Application and Submission failures** (minority class >10%); **51% of App/Submission failed jobs live in these mixed campaigns.** Same features, different label → inseparable at match time (the distinguishing exit code *is* the label).
- Top Application code: 1 (general art error, 429k). Top Submission code: 127 (command-not-found / broken PATH, 262k = 45% of Submission).
- Predictive 3-class fault attribution (`fault_type`) peaked at macro-F1 ~0.44 with Submission F1 0.26; collapsing to **Hardware vs Payload** (`hw_fault`) is separable, honest, and matches the paper's user-vs-infrastructure thesis.

### 7.3 Wait-time regression (submission-time)

Heavy-tailed target (started jobs): p50 **40 min**, p75 2.9 h, p90 8.4 h, p99 40.6 h, mean 3.2 h.
Report robust/relative metrics, not raw MAE.

| model    | R²(log)       | median AE                 | within-2× | MAE <10 min | MAE 10 min–2 h | MAE >2 h |
| -------- | -------------- | ------------------------- | ---------- | ----------- | --------------- | -------- |
| CatBoost | **0.50** | **966 s (~16 min)** | 0.42       | 543 s       | 1,982 s         | 14,846 s |

- **Typical (median) error is ~16 min, not the ~1 h the raw MAE (3,881 s) implies** — the MAE is inflated by the >2 h tail. The model gets the order of magnitude right ~40% within 2×; very short waits are proportionally hardest.
- Top drivers: campaign trailing **mean queue-wait**, `idle_queue_depth`, requested resources.

### 7.4 Library choice

A wash across all tasks (spread ≤0.02–0.03). CatBoost holds a slight edge on every target in the
latest run. Pick on operational grounds (GPU, categorical handling, latency).

---

## 8. Key findings (evidence-based)

1. **Prediction-time discipline changes the conclusions.** Naive `Failed` ROC 0.96 is ~0.18 leakage/memorization; the deployment-honest number is **0.78**.
2. **Job characteristics carry genuine, transferable signal** (temporal ROC 0.744 from resource shape + image, no identity); **identity + month is memorization** (0.583 out-of-time).
3. **Hardware faults are genuinely — if modestly — predictable without leakage:** F1 0.44 from job shape alone (temporal), ROC 0.68 with match-time node/site health.
4. **Application vs Submission is inherently unpredictable** at match time (68% mixed campaigns); the actionable, separable split is **Hardware vs Payload**.
5. **Node-failure holds do not straddle the campaign-aware split** (incidents are single-campaign: 0% span >1 campaign, though 85% span >1 batch); their within-period optimism comes from **site × time** memorization (cross-month site-fail-rate corr 0.11), which only a temporal split exposes.
6. **Trailing *state* generalizes where identity does not:** "this node has failed N jobs in the last 5 min" transfers across time; "this is node X" does not.
7. **Missingness is leakage.** `Drain` and all match attributes fail ~50% when null — an outcome fingerprint, not a match-time cause.

---

## 9. Notebooks and reproducibility

| Notebook                                       | Role                                                                                                        |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `scripts/hold-failure-analysis.ipynb`        | hold/outcome analysis; 3-GBM comparison; feature-importance study                                           |
| `scripts/hold-failure-modeling.ipynb`        | **current** leakage-clean staged model (Failed + hw_fault + wait_time), missingness audit, all charts |
| `scripts/concurrency-context-modeling.ipynb` | concurrency/stage ablation                                                                                  |
| `scripts/temporal-model-comparison.ipynb`    | 3-GBM temporal benchmark + distillation                                                                     |

Standard chart suite: per-library training-loss lines; results tables (pandas, not markdown);
confusion matrices; ROC curves; decision-threshold curves; feature-importance heatmaps.
Models saved to `models/staged/`. Full-data GPU runs are done externally to compare against the
subset results.

### 9.1 Multi-modal / neural scaffolding (`mlp/`)

A separate GPU workstream (server-side, driven from `scripts/attribution-modeling.ipynb`) tests whether a
learned temporal branch beats the hand-crafted trailing windows. Runs on the **raw** HTCondor
history dumps (`/exp/nova/.../fifebatch-history-v3-YYYY.MM.DD-HHHH.parquet`, years 2022–2025), read
only. The loader (`mlp/data.py`) is schema-agnostic — standard→raw column-name resolution plus
magnitude-based seconds-vs-ms timestamp detection — and reuses the exact validated label logic and
leak-free `trailing_rate` helper. Components:

- `analyze.py` — dataset GB/counts + failure/wait analysis (run first to confirm the raw schema).
- `sequences.py` — per-job leak-free **same-site recent-history** sequence (outcome + hold-reason
  category + Δt per step); a step is included only if it *completed before* the job's execution
  start, so the job never sees its own outcome. Hold-reason text enters only as prior-jobs context.
- `model.py` — tabular branch (embedded categoricals + numerics) fused with a GRU over the
  sequence; multi-task heads for `Failed`, `hw_fault`, `wait_time`.
- `train.py` — temporal-split training; reports ROC-AUC / R²(log) to compare directly with the GBMs.

Validated locally (loader, labels, sequence assembly, model forward/backward/loss all pass on the
processed parquets). **Expectation:** the tabular MLP alone trails CatBoost; the test is the
sequence branch's marginal value — the honest ceilings (Failed 0.78, hw_fault 0.68) are information-
limited, not model-limited.

---

## 10. Open questions / future work

- **Full-data GPU run** — the sub-minute and node-level windows are sparse on the 20% subset; they may firm up `hw_fault` at scale.
- **LightGBM quantile objective** for calibrated P50/P90 wait estimates instead of a point regression.
- **Node-failure annotation** — node-incident clustering (≥k failures on one host in a window) as derived ground truth; per-entry eviction rates. (`MATCH_GLIDEIN_ToDie`/`ToRetire` are ≥95% null, so pilot-lifetime is unavailable.)
- **Multi-modal / temporal-sequence model** — a 1D-CNN/LSTM over leak-free recent per-site load sequences vs the fixed-window tabular summary, *if* tabular plateaus. Note the hold-reason text is an outcome (leakage as a same-job input; usable only as prior trailing context).
- **Queue-state modeling over time** to strengthen submission-time predictions.

---

## 11. Planned experiments (justification for the framework)

Each framework claim needs an experiment that quantifies it. All on 2025 (106M jobs), same
model/features held fixed except the variable under test.

- **E1 — Split protocol (justifies temporal / campaign-aware evaluation).** One model+feature set,
  three splits: random → campaign-aware → temporal. Metric gap = optimism from duplicate
  memorization + contemporaneity. *Prior 2024 evidence:* Failed ROC 0.96 → 0.88 → 0.78.
- **E2 — Prediction-time admissibility (justifies excluding inadmissible features).** From the
  leak-free temporal model, add back one inadmissible feature class at a time (`JobStatus`,
  `LastHoldReasonCode`, `JobCurrentWaitTime`, `Drain`/match-attr missingness, month) and measure the
  spurious lift. Shows each exclusion removed leakage, not signal.
- **E3 — Feature ablation (justifies the added features).** Temporal split, cumulative:
  baseline (requested resources + identity) → +cyclical → +leak-free trailing site/node/entry health
  → +queue-state (idle depth, wait-mean) → +learned sequence. Marginal ROC / R² per addition, per task.
- **E4 — Fault-attribution separability (justifies Hardware-vs-Payload).** (a) campaign-mixing:
  fraction of campaigns emitting both Application and Submission (2024: 68%) ⇒ App/Sub inseparable;
  (b) 3-class fault attribution (Submission F1 ~0.26) vs `hw_fault` binary ROC.
- **E5 — Model comparison (headline).** Same features/split: 3 BDTs (XGB/LGBM/CatBoost) tabular,
  MLP tabular, MLP tabular+sequence. Answers (i) BDT library wash, (ii) does the learned sequence
  branch beat the hand-crafted 1m/5m windows. BDT baseline is `mlp/train_bdt.py`; MLP is `train.py`,
  both consuming the same `data.py` features + temporal split.
- **E6 — Honest wait metrics (justifies robust reporting).** Raw MAE vs median-AE vs within-2× vs
  per-bucket on the heavy-tailed target; show MAE is tail-inflated (2024: MAE ~1 h vs median-AE ~16 min).
- **E7 — Cross-year generalization.** Train 2024 → test 2025 (and reverse); tests temporal robustness
  across a full year. *(2025 is notably different: 24.4% failure vs 18.7%, 91/9 vs 84/16 Payload/Hardware.)*
- **E8 (optional) — Relational/graph features.** Leak-free DAG-sibling (`DAGManJobId`/`ParentJobId`)
  recent-outcome feature; test marginal value before any GNN (DAG-cascade columns are ~all null, so
  a rich GNN is not data-supported).

## 12. Analysis findings (2025 data)

Full 2025 = 106.1M jobs; Jan+Feb subset (17.4M jobs, 344 files, 21.9 GB) used for iteration.
2025 differs from 2024: full-year failure rate **24.4%** (Jan+Feb 16.4%), fault split **~88% Payload
/ 12% Hardware** (2024 was 84/16), 88.5% matched (full-year). Analysis notebook:
`scripts/attribution-modeling.ipynb` (seaborn; streaming/memory-safe). Key patterns:

- **Failures are highly concentrated** (Lorenz): the top ~10% of campaigns and of users account for
  ~80% of all failures; sites are even more concentrated. → Direct justification for campaign-aware
  splitting and the memorization concern (identity is a strong but non-generalizing signal).
- **Hardware faults are bursty, site-localized incidents** (site × week heatmap): a site is clean
  for weeks, then spikes (e.g., Michigan/Wisconsin/SU-ITS in specific weeks) while FermiGrid carries
  a persistent moderate rate. → Justifies leak-free trailing node/site-health features + temporal split.
- **Fault composition varies sharply by experiment/group:** ~90% Payload for sbnd/nova/icarus vs
  33–45% Hardware for gm2/minerva/spinquest. → The Hardware-vs-Payload split is meaningful per-tenant.
- **Cyclical structure** in failure rate (day-of-week × submit-hour heatmap) and in wait time
  (median flat ~12 min, p90 swings 50–410 min by hour) → supports cyclical time features, honest
  wait metrics.
- **Failure drivers:** failure rate rises with requested memory/lifetime and with `NumJobStarts`
  (restarts); exit code 1 (art/framework) dominates non-zero codes.

## 13. Distributional fits (FermiGrid, Feb–Aug 2025)

Two generative fits on the anonymised extract. Both are on FermiGrid only (87.98% of
executed workload) and exclude GlideinWMS pilots (`Cmd == ./glidein_startup.sh`).
Notebook: `scripts/notebooks/data-analysis.ipynb`, cells `fitsite1` and `fgmix`.

### 13.1 Inter-failure intervals — two-component exponential

6,074,133 intervals; 3,515,718 zero-length dropped (57.9%); 300,153 in the [10, 300] s
fit window. Model `c + b·exp(−t/λ₂) + a·exp(−t/λ)`, scipy `curve_fit` (TRF, Poisson
weights, errors scaled by √(χ²/ndof)):

| parameter | value | ± | t | p |
| --- | --- | --- | --- | --- |
| a | 4,799.34 | 146.66 | 32.73 | 1.6e-98 |
| b | 69,535.60 | 1,614.37 | 43.07 | 8.5e-127 |
| λ₂ (fast) | 7.7514 s | 0.1109 | 69.90 | 2.0e-181 |
| λ (slow) | 37.9573 s | 0.5560 | 68.27 | 1.2e-178 |
| c | 37.53 | 1.44 | 26.00 | 3.3e-77 |

**χ²/ndof = 3.88** (χ² = 1,106.1, ndof = 285). Two time constants, **7.75 s** and
**37.96 s** (rates 0.129 and 0.0263 s⁻¹), plus a constant floor. A cross-check on raw
elapsed time moves no parameter by more than 0.002%. Failures are an arrival process,
which a mixture of exponentials fits naturally.

### 13.2 Queue wait — three-component lognormal mixture

48,998,479 jobs with wait > 0; 1,204,319 (2.46%) above a 1-day cap treated as outliers;
**47,794,160 workflow jobs fitted**. The same two-exponential form used for failure
intervals gives χ²/ndof ≈ 104 here — waits are a queueing *outcome*, not an arrival
process, so the functional form does not transfer.

| component | weight | median | σ (log) | reading |
| --- | --- | --- | --- | --- |
| 0 | 0.167 | 28 s | 1.566 | matched into an already-idle pilot slot |
| 1 | 0.450 | 779 s (13.0 min) | 1.073 | waiting for an occupied slot to turn over |
| 2 | 0.383 | 8,988 s (2.50 h) | 1.064 | waiting for new pilot provisioning |

KS = **0.0081** vs **0.0500** for a single lognormal. BIC sweep over k = 1…8:

```
 k          BIC       dBIC       KS
 1    2,256,260              0.0500
 2    2,217,773    -38,487   0.0192
 3    2,207,860     -9,913   0.0081
 4    2,206,092     -1,768   0.0092
 5    2,204,452     -1,641   0.0074
 6    2,202,254     -2,198   0.0072
 7    2,201,115     -1,139   0.0069
 8    2,201,149         34   0.0069
```

**k = 3 is chosen at the BIC elbow, not at the BIC minimum** (which is k = 7). With a
500k fit sample the complexity penalty (ln n ≈ 13/parameter) is negligible against the
likelihood gain, so BIC keeps buying components that chase the truncation edge; ΔBIC
collapses from −38k to ~−1.7k per step after k = 3 and KS stops moving. The sweep is
printed by the cell so the choice is auditable rather than asserted.

Fit quality across the axis: obs/exp is **1.003 at the median bin**, 0.790 at the first
and **1.614 at the last** — the mixture's right tail falls faster than the data's. That
last-bin excess is not a cap artifact: only 25 jobs sit at exactly 86,400 s, no value in
that bin repeats more than ~165 times, and the distribution continues smoothly past the
cap (42,417 jobs in the next hour). It is a real limitation of the lognormal tail, and
part of why BIC wanted more components.

### 13.3 What the wait fit implies for E2

- **Log wait is unimodal**, not trimodal: one mode at 1,310 s, no valleys, component
  separations of only 2.35σ and 1.85σ (bimodality needs ≳2σ). A hard mixture-of-experts
  router would therefore buy nothing. The problem is variance, not multimodality.
- **Variance decomposition:** log-space total 5.337 (sd 2.310), within-component 1.361
  (sd 1.167), **between-component share 0.745**. Knowing a job's regime would leave a
  conditional sd of ~1.17 in log space. Caveat: this share rises mechanically with k
  (0.75 at k=3, 0.89 at k=5, 0.96 at k=8), so it is a property of *this* decomposition,
  not a physical ceiling.
- **A point estimate is the wrong output object.** With that residual spread the honest
  deliverable is an interval; `e2dist` adds quantile and AFT heads.
- **Reference floor.** The best possible *constant* prediction scores within-2x ≈ 0.25.
  Current models span 0.266–0.297 and R²_log ≈ 0.267 (CatBoost best), so every feature
  in `Xsub` is currently worth at most ~+0.05 within-2x over a single number. E2 now
  prints two no-feature reference predictors before the model numbers, because the model
  figures are not interpretable without that floor.
- **Error bins are now the fitted regimes.** Component dominance crosses at **106 s**
  and **2,857 s**, rounded to 2 min and 45 min. k = 3 components give exactly three
  regions — `inst` / `turn` / `prov` — replacing the hand-picked <10 m / 10 m–2 h /
  >2 h bins. A fourth reporting row, `park` (>1 d), is **not** a fitted component: it
  is everything past the 1 d cap applied to the fit's population, about which the fit
  says nothing. The data runs smoothly through that cap (25 jobs at exactly 86,400 s,
  42,417 in the following hour), so the cap is a modelling choice rather than a
  boundary in the workload. It is kept as its own row only because its errors are
  ~20× larger (MAE 229,689 s vs 10,920 s in `prov`), and pooling would let ~2% of
  jobs dominate the regime that matters.

### 13.4 Trailing windows, chosen per key

The failure fit sets the timescale trailing health features should be measured on.
But the fit is **site-level**, and that does not transfer: each key has its own
arrival process, so each was fitted separately (same form, same [10, 300] s range,
same procedure).

| key | λ₂ (fast) | λ (slow) | χ²/ndof | median gap |
| --- | --- | --- | --- | --- |
| site | 7.75 s | 37.96 s | 3.88 | 1 s |
| campaign | 8.04 s | 45.79 s | 4.53 | — |
| **node** | **fit fails** | **fit fails** | **1449** | **415 s** |

**Site and campaign** cluster on tens of seconds. A trailing window's value is the
ratio of clustered excess to steady background inside it,

    burst(W) = a·λ·(1 − e^(−W/λ)) + b·λ₂·(1 − e^(−W/λ₂)),   bg(W) = c·W

which falls monotonically with W:

| W | 60 s | 120 s | 300 s | 900 s | 3600 s |
| --- | --- | --- | --- | --- | --- |
| site burst/bg | 304 | 158 | 64 | 21 | 5 |
| campaign burst/bg | 86 | 46 | 19 | 6 | 2 |
| fraction of slow burst captured | 79% | 96% | 100% | 100% | 100% |

Past ~120 s the window already holds the whole burst, so longer windows add
background and nothing else. 300 s and 900 s are dropped — identical coverage to
120 s at 4.7× and 14× worse signal-to-background. What remains is 60 s (burst) and
3600 s (the fit's constant term `c`, i.e. slow drift).

**Node is a different process entirely and the two-exponential form does not fit it
at any range.** Over the site's [10, 300] s the fit degenerates outright (λ₂ = −1.0e6
± 1.9e8 s, negative amplitudes, χ²/ndof 1449). A scan of candidate ranges puts the
best two-exponential at [60, 3600] s — λ₂ = 60.56 ± 0.70 s, λ = 703.0 ± 19.9 s — and
it is still rejected at **χ²/ndof = 80.2**, against 3.88 for the site fit on the same
functional form (figure: `output/interfailure_fit_node_FermiGrid.pdf`, notebook cell
`cded44c7`).

**Exposure.** Unlike FermiGrid as a whole, which is never empty (the site fit removes
0 s of idle), a single slot is idle between jobs: **57.6% of total within-node gap
time is idle**, and 41.0% of intervals contain some. Raw elapsed time therefore badly
overstates node exposure, mostly in the tail — p75 drops 29,952 → 18,330 s and p90
191,877 → 105,524 s under the correction. A slot is serial, so its active time needs
no concurrency sweep: it is the total runtime of that slot's jobs in the interval.
The cell fits active time and cross-checks against raw.

Correcting for it does **not** rescue the fit — χ²/ndof 88.2 on active time against
80.2 on raw, with the time constants moving ~4% (λ₂ 60.6 → 58.1 s, λ 703.0 → 688.0 s).
The model fails on the *shape* of the node distribution, not on the choice of time
measure. The window choice is likewise robust: 3600 s covers 66.2% of active gaps
against 64.5% of raw ones.

The residuals also show a distinct bump near **1,900–2,100 s** that no smooth
two-exponential can produce, and it **survives the active-time correction**, so it is
not an idle artifact. Its cause is not established; a glidein lifetime or retirement
boundary (`GLIDEIN_ToDie`/`ToRetire`) is the obvious thing to check.

Node failures are ~10⁴ times more spread out — mean gap 76,158 s against 7.2 s at
site level:

| | p10 | p25 | p50 | p75 | p90 |
| --- | --- | --- | --- | --- | --- |
| node gap | 49 s | 86 s | **415 s** | 29,952 s | 191,877 s |
| site gap | 1 s | 1 s | **1 s** | 3 s | 12 s |

The [10, 300] s fit range covers only the extreme left tail of the node
distribution, which is why the fit degenerated. A 60 s window contains a prior
failure on the same node just **15%** of the time, so the site-derived window is the
wrong instrument for node rates. Node windows are set from the gap distribution
directly: 3600 s covers 64.5% and 86400 s covers 83.2%.

    TRAIL_WINDOWS_BY_KEY = {
        "site_fail": [60, 3600],   "site_hw": [60, 3600],   "camp_fail": [60, 3600],
        "node_fail": [3600, 86400], "node_hw": [3600, 86400],
    }

Caveat: both distributions are over-dispersed relative to a single exponential
(CV 8.2 site, 4.0 node), driven by the tail beyond the fit range. The two-exponential
description is a statement about the burst regime it was fitted on, not the whole
distribution.

### 13.5 Trailing statistics: measured redundancy

The fits constrain windows, not which key a rate is computed over. That axis was
measured on the archived 46-column matrix (2M-row sample):

- **`entry_fail` is redundant with `site_fail`** — r = **0.995** (15 m), **0.996**
  (60 m). GlideinWMS entries nest inside sites, so this is redundancy by
  construction. Dropped.
- The same statistic at two windows correlates at r = 0.81–0.93, and the 12 trailing
  features need only **5 principal components for 90%** of their variance.
- Univariate |r| with `Failed` (60 m): node_fail 0.586, site_fail 0.540,
  entry_fail 0.541, camp_fail 0.383. Against `hw` the ordering inverts —
  **site_hw 0.040 is the largest of all twelve**, node_hw 0.005.
- **The `*_hw` rates are kept** despite near-zero correlation with `Failed`: they top
  the hardware column, are small only because hardware faults are 0.8% prevalent, and
  are the only hardware-specific trailing signal E3 has.

Net: **24 planned trailing columns → 10**; Xmatch 58 → **44 columns**, 14.9 →
**11.3 GB**. Derived from the fits and a redundancy measurement *before* any model
trains on the new matrix, so it cannot be a post-hoc rationalisation of one run's
feature importances.

## 14. Dataset accounting and provenance (verified 2026-09-17)

Measured on the 340-column extract at `/media/storage0/allison/BatchSystem-2025`,
Feb 1 – Aug 31 2025 (212 days). Several figures in the paper were wrong and are corrected
in `paper/02_related_work.tex` (`tab:datasets`) and `paper/03_methodology.tex`
(`tab:accounting`).

| quantity | value | previously reported |
| --- | --- | --- |
| queue records | 64,088,358 | 64,025,075 |
| GlideinWMS pilots | 5,145,404 (8.03%) | not separated |
| workflow jobs | 58,942,954 | — |
| never started | 2,837,833 (4.43%) | — |
| started at least once | 56,105,121 | — |
| named sites | **43** | 45 |
| telemetry fields | **340** | 475 |
| size | 15.6 GB (Parquet/zstd) | 80.9, units unstated |
| users / batches | 395 / 20,110,451 | 395 / 20.07M |

- **Federation is highly concentrated.** Of 56,071,537 workflow runs with a named site,
  FermiGrid carries 49,333,440 (**87.98%**) and the 42 remaining sites 6,738,097 (12.02%).
  A 44th site level is the null site: pilots and scheduler-universe jobs that never match.
- **Pilots are defined by `Cmd == ./glidein_startup.sh`**, not by `group_opportunistic`.
  The group proxy is wrong in both directions — it misses 30,324 real pilots and wrongly
  claims 10,424 non-pilots. **No pilot carries a matched site** (0 of 5,209,738 across all
  71.2M raw rows), so any site-sliced statistic excludes them by construction.
- **Label revision moved the fault split.** Under the revised rules (SIGTERM from
  `condor_rm` is no longer hardware evidence) the failed population is 8,025,995 with
  **6.52% hardware** (523,281) / 93.48% payload (7,502,714), against 8.35% before.
- **Boundary-job effect measured.** Splitting on label-observation time rather than
  submission time moves **22,370 jobs, 0.052%** of the training set. The previously
  reported 125,525 was not reproducible. This directly answers R2 and shows Δ cannot be a
  boundary artifact.

### 14.1 Wait-time semantics (checked, not assumed)

- `wait_s = JobStartDate − QDate` is **already per-submission**. `JobStartDate` is the
  *first* start, not the latest: across all 1,247,639 multi-start jobs
  `JobStartDate < JobCurrentStartDate`, never equal. Restarts (1.25M of 62.6M, 2.0%) do
  not inflate it. Had it been the latest start the tail would have been badly wrong —
  median wait to first start is 3,665 s vs 23,403 s to last match on rematched jobs.
- **Holds contribute nothing**: `NumSystemHolds > 0` on 0 of 54,014,638 FermiGrid jobs in
  the (0, 1 d] window.

### 14.2 Pilot → payload mapping is not available

Every payload-side `MATCH_GLIDEIN_*` identity field is the literal string `"Unknown"`
and pilot-side `GlideinName` is the constant `'gfactory_instance'`, so the intended
`MATCH_GLIDEIN_ClusterId` ↔ pilot `ClusterId` join is unpopulated. Only `LastRemoteHost`
is shared, and it is present on just 850,018 of 5,209,738 pilots (16%), needing a
hostname + time-interval join. It would not help E2 regardless: which pilot a payload
landed in is known only *after* the match, and the match is the event E2 predicts the
time until. The useful version is aggregate pilot supply at `QDate`, which needs no join.

## 15. Current direction

1. **E2 as an interval predictor.** `e2dist` with `--head quantile` (multi-quantile
   XGBoost) or `--head aft` (`survival:aft`, normal loss = conditional lognormal, and the
   only head that can consume never-ran jobs as right-censored). AFT quantiles are built
   from the *residual* sd, not `aft_loss_distribution_scale` — that hyperparameter is not
   a fitted spread, and using it gave 0.58 coverage against a nominal 0.80.
2. **Reference floors and regime-stratified error** in every E2 report.
3. **Cold start / subgroups** as post-hoc analysis over saved predictions
   (`eval.helper.cold_start_report`), not a pipeline flag: seen vs never-seen entities on
   the same test period and the same fitted model, which is a cleaner memorization probe
   than the random-vs-temporal gap.
4. **Uncertainty**: `--seeds` reports mean ± sd grouped by split; `--cutoff` reruns the
   temporal protocol at several dates, with results keyed by cutoff so sweeps accumulate.
5. **Split basis is fixed**, not configurable — always label-observation time.
6. **W&B tracking** for the 2-split × N-model × 3-task grid (`config/wandb.yaml`).

## Changelog

- **2026-09-18 (b)** — **Per-key trailing windows; load.py de-duplication.** The site fit
  does not transfer to other trailing keys, so each was fitted separately: campaign
  matches site (λ 8.04/45.79 s, χ²/ndof 4.53) but **node fails outright** (χ²/ndof 1449,
  negative amplitudes). Node failures are ~10⁴× more spread out (median gap 415 s vs 1 s),
  and a 60 s window holds a prior same-node failure only 15% of the time. Windows are now
  per key: site/campaign [60, 3600], node [3600, 86400], recorded in `schema_meta.json`.
  Fixed a bug this surfaced: the rate loop was a cross product over all windows × all
  specs, which with per-key windows would compute 18 rates for 10 named columns and
  misalign them silently; it is now driven by `TRAIL_PAIRS` with a count assertion.
  Separately removed duplicated processing between `load.py` and `feat-engineering.ipynb`:
  the window filter was applied twice from two independent thresholds, and `failed`/`hw`
  were re-derived from `fault_type` instead of reading load.py's columns. See §13.4–13.5.
- **2026-09-18** — **Trailing features cut from the fit.** Used the inter-failure fit to
  choose trailing windows instead of choosing them by hand: the burst/background ratio
  inside a window falls monotonically with W and the burst is fully contained by ~120 s,
  so 300 s and 900 s add only background (4.7× and 14× worse SNR than 60 s for identical
  coverage). `TRAIL_WINDOWS = [60, 3600]` — one burst window, one for the fit's constant
  term. Separately measured the statistic axis: `entry_fail` duplicates `site_fail`
  (r = 0.995/0.996, entries nest inside sites) and is dropped; the `*_hw` rates are kept
  because they carry the largest |r| with the hardware target despite tiny absolute
  values. `TRAIL_NAMES` 6 → 5. Net 24 → 10 trailing columns, Xmatch 58 → 44, 14.9 → 11.3 GB.
  See §13.4–13.5.
- **2026-09-17** — **Distributional fits + dataset re-accounting.** Added §13–15. Fitted the
  FermiGrid inter-failure intervals (two-component exponential, λ 7.75 s / 37.96 s,
  **χ²/ndof 3.88**) and the queue wait (**3-component lognormal**, medians 28 s / 779 s /
  2.50 h, KS 0.0081 vs 0.0500 for a single lognormal, k chosen at the BIC elbow not the
  minimum). The wait fit drove three pipeline changes: error bins re-cut to the fitted
  regime crossovers (106 s, 2,857 s → `inst`/`turn`/`prov`/`park`), no-feature reference
  predictors printed before every E2 result (a constant already scores within-2x 0.25),
  and distributional heads (`e2dist`, quantile + AFT). Re-measured the dataset against the
  new 340-column extract and corrected the paper: **43 named sites (not 45)**, 340
  telemetry fields (not 475), 64,088,358 records, FermiGrid **87.98%** of sited runs.
  Pilots are now defined by `Cmd == ./glidein_startup.sh` (the `group_opportunistic` proxy
  was wrong by ~41k jobs both ways) and carry no site at all. The boundary-job effect is
  **22,370 jobs (0.052%)**, replacing an unreproducible 125,525. Confirmed `wait_s` is
  already per-submission (`JobStartDate` is the first start, verified on all 1,247,639
  multi-start jobs) and that holds inflate no waits (0 of 54M). Pilot→payload joins are
  unavailable (`MATCH_GLIDEIN_*` is literally `"Unknown"`). Infrastructure: W&B logging
  with per-epoch curves, `--seeds`/`--cutoff` for variance and sensitivity, cold-start
  moved to post-hoc, `--split-basis` removed (label-observation is now unconditional), and
  all model/data/prediction paths centralised in `eval/paths.py` as
  `models/<experiment>/<model>/` on scratch.
- **2026-08-02 (b)** — **Multi-scale trailing windows.** Added a 15-minute window alongside the
  1-hour one for all six trailing rates (`TRAIL_WINDOWS = [900, 3600]` knob in B1; `Xmatch`
  40→46 cols, names `trail15m_*` / `trail60m_*`). Rationale: failures are bursty and busy keys
  (site/entry/campaign) support fresh short-window estimates, while node keys are too sparse at
  15 min — so both scales are provided and the trees arbitrate per key, rather than replacing 1 h.
  E3's trailing step is pinned by name to the 1-hour site/camp/node trio for comparability with
  prior runs (verified identical in smoke). `Xsub` unchanged (27). Cache col-check forces a B1
  rebuild. Whether the short window helps is measurable post-rerun via feature importances.
- **2026-08-02** — **MLP-wait nan fix + concurrency features (B1 v2).** The wait MLP diverged
  (`nan` loss from epoch 1) because `Xsub`'s campaign-trailing-wait feature was raw seconds (p99
  ~237k) — above the fp16 max (65,504), so it overflowed to `inf` under AMP autocast; BDTs were
  unaffected (scale-invariant). Fix: all seconds/count-valued features are now `log1p`-scaled
  (monotone → tree-neutral). Also added the **concurrent-running metrics** the paper's telemetry
  table promised but the matrices lacked: leak-free per-site + total running counts at the match
  clock (`Xmatch` 38→40) and total running at submit (`Xsub` 26→27) — counted only over jobs that
  both start and complete (no time-index drift), self-excluded, both validated exactly against
  brute force. New `FI` cell after B1 prints the full named feature inventory, targets, and split.
  B1 manifest now carries a column-count version, so the old 38-col cache rebuilds instead of
  silently reusing. **All E-cells + MLP must be rerun**; reported paper numbers correspond to the
  38-col matrices until then. Local smoke: MLP wait went from `nan` to R²(log) 0.24 at 1 CPU epoch.
- **2026-08-01** — **Complete 7-month BDT results** (Feb–Aug 2025, train Feb–Jun / test Jul–Aug,
  64.0M jobs, all models on the identical full training set). E5: Failed — CatBoost **0.825
  ROC / 0.599 PR / 0.534 F1** (P 0.64 @ R 0.46), XGB/LGBM ~0.767; hw_fault — XGB best **0.799**
  (P 0.61 @ R 0.24); wait — CatBoost **R²(log) 0.26, med-AE 30.7 min, within-2× 0.29**. E1:
  **0.972→0.767 ROC (Δ=0.205), 0.901→0.518 PR (Δ=0.384)** — naive protocol overstates the usable
  P–R range ~2×. E2: +JobStatus +0.088 / +LHR +0.099 ROC (+0.165/+0.172 PR); month −0.050. E3:
  base **0.548** → +cyc 0.569 → +3 trailing **0.773** (≥ full matrix 0.767) — three state features
  carry all transferable signal. E4: 163/367 campaigns (44%) mixed; 3-class macro-F1 0.53, Sub F1
  0.36. E6 buckets: 1,056 / 3,221 / 34,562 s (33× spread). Wait is the least temporally stable
  task (0.26 vs 0.44 on the shorter adjacent split). Paper updated: Results §4 (new model×task
  metric table, all numbers), Discussion §5 rewritten, intro headline (Δ=0.205/0.384), methodology
  label-construction stats (44%/0.53/0.36). **MLP cell not yet run** — table row held as comment.
- **2026-07-30 (c)** — **Full metric suite** across all experiment cells. New BT helpers:
  `cls_metrics` (ROC-AUC, PR-AUC, plus F1/precision/recall/balanced-accuracy/MCC at an operating
  threshold chosen by max-F1 on a ≤2M-row **training** sample and frozen for test — never chosen on
  test; Brier/log-loss omitted since `scale_pos_weight` deliberately distorts calibration) and
  `reg_metrics` (R²(log), median-AE, P90-AE, raw MAE, within-2×). E5 reports the full suite per
  library/task; E1 now reports Δ in ROC **and** PR **and** F1 terms (local smoke: PR-Δ 0.54 vs ROC-Δ
  0.21 — the random-split inflation is far larger in PR space, worth a sentence in the paper); E2/E3
  report ROC+PR+F1 per step; E4 adds per-class precision/recall; MLP gets the same treatment with
  its own train-sample threshold. All metrics also append to `results.jsonl`.
- **2026-07-30 (b)** — **CatBoost OOM fix + crash-proof results.** The uncapped E5 run OOM-killed the
  kernel in the CatBoost fit (XGBoost streamed fine; LightGBM bins in place from the caller's
  pointer): CatBoost **double-buffers** — it builds its own full internal Pool copy of the training
  matrix while the caller's ~7 GB float32 slice is still alive, plus quantization buffers → >19 GB.
  Fix: `gather_to_disk()` (BT cell) materializes the E5 train slice as a disk-backed `.npy`, so the
  library-internal copy is the only RAM cost (fit peak ~9 GB). Also added `log_result()` →
  `SCRATCH/results.jsonl`: every completed fit in E1–E6 and the MLP appends its metrics immediately,
  so a later crash never loses finished results. Note the B0/B1 cache survives kernel crashes —
  restart recovery is cell 3 → B0/B1 (`[reuse]`) → BT → resume. Smoke-tested; E5 numbers identical
  to the in-RAM slice path.
- **2026-07-30** — **Uncapped 7-month protocol** (user decision: no train caps; identical training
  set for all models). Window moved to **Feb–Aug 2025** (1,254 files / 80.9 GB / **64.0M jobs**;
  failure rate 26.6%, 91.9% Payload / 8.1% Hardware, median wait 25.8 min, 87.5% matched);
  `MIN_QDATE=2025-02-01`, train Feb–Jun / test Jul–Aug (`TRAIN_YM`/`TEST_YM`). Removed
  `CB_LGBM_TRAIN_CAP` + `cap_train` entirely: E5's LightGBM/CatBoost now materialize the full
  training slice (same rows as XGBoost's streamed path). B0 gate re-budgeted for the materialized
  fit (`FIT_ROW` 60→161 B/row): est. working set ~16.3 GB at 64M rows vs ~17 GB usable in the 19 GB
  cgroup — passes, which is why 7 months is the uncapped ceiling (8 months was not). Window change
  alters the cache signature, so B0 rebuilds once. Smoke-tested end-to-end incl. cache reuse.
- **2026-07-27 (f)** — **Streamed XGBoost training** after the corrected gate showed the real
  constraint: the kernel cgroup allows only ~19 GB (machine has ~480 GB free) while per-fit float32
  train copies alone need ~12 GB at 71.1M rows. New BT cell: an `xgb.DataIter` streams row-batches
  from the memmaps into a compressed `QuantileDMatrix` (~1 B/cell), and `xgb.train` +
  batched `inplace_predict` replace the sklearn wrapper — **every XGBoost experiment (E1–E4, E6, and
  E5's XGB column) now trains on the full window with a ~2 GB fit working set**. LightGBM/CatBoost
  cannot stream construction, so E5 caps their training sample at `CB_LGBM_TRAIN_CAP` (20M rows,
  printed when applied; test set identical). Gate `FIT_ROW` 170→60 B/row; full-scale working set
  ~10 GB, comfortably inside the 19 GB budget. Smoke-tested end-to-end incl. multi-class and
  regression through the iterator and the cache-reuse path.
- **2026-07-27 (e)** — **Fixed the RAM-budget false alarm** on the server: B0's gate raised
  `MemoryError` with "budget ~0 GB" at 71.1M rows because the cgroup's `memory.current` charges the
  ~90 GB of parquet page cache left by the analysis cells against the limit — reclaimable memory the
  kernel evicts on demand. `_free_ram_gb()` now subtracts reclaimable file cache from cgroup usage
  (docker/k8s working-set convention, v1 + v2), the gate prints each candidate (machine vs cgroup),
  and a `RAM_BUDGET_GB` knob overrides detection outright. Server outputs also confirmed the 8-month
  scope: 1,395 files / 89.7 GB / **71.1M jobs**, failure rate 25.6%, 91.5% Payload / 8.5% Hardware,
  median wait 22.5 min; B0 working set ~18 GB at full scale.
- **2026-07-27 (d)** — **Dropped pre-window straggler jobs** (queued before 2025-01-01 UTC; submit
  times reached back to 2024-08-27 inside the 2025 files). Previously they were only excluded from
  the train/test split but still contaminated trailing-state history, feature standardization, and
  the idle-queue depth. Now filtered at scan time in both cell 3's `lf` (analysis + export) and B0's
  chunked scan (`MIN_QDATE` knob); the cache signature includes `MIN_QDATE`, so existing caches
  rebuild automatically. Local check: 634 straggler rows removed; "outside-window (dropped)" now 0.
- **2026-07-27 (c)** — **Disk-backed arrays + crash-recovery cache** after the chunked load succeeded
  but the server kernel still died downstream (post-load arrays + `Xmatch`/`Xsub` + per-fit train
  copies exceed a small kernel's RAM; JupyterHub cgroup limits are also invisible to `psutil`, so the
  (b) gate could pass and still OOM). B0/B1 now write all large arrays as **`.npy` memmaps** under
  `SCRATCH_DIR` (env `FIFE_SCRATCH`; `ON_DISK="auto"`), so resident RAM stays at the working-set
  scale (~18 GB at the full 8 months vs ~50 GB fully in-RAM). The RAM gate is **cgroup-aware**
  (min of machine-free and cgroup headroom) and still fails fast with a recommended `SUBSET_FRAC`.
  A **manifest** (files+`SUBSET_FRAC` signature) lets re-runs reload the cache in seconds — after any
  crash: restart kernel, cell 3 → B0 → B1 both print `[reuse]`, continue; cached B0 re-runs no longer
  need cell 3's label expressions. E2/E3 restructured to stack features **per split** (never a
  full-width copy); `codes`/`X_num` stored Fortran-order for fast column ops; `[mem]` peak-RSS lines
  after each stage localize any future failure. Smoke-tested: fresh build + second session hitting
  the cache reproduce identical numbers.
- **2026-07-27 (b)** — **OOM-hardened B0** after the first server attempt killed the kernel during the
  single whole-stream streaming collect. B0 now (i) estimates the whole-pipeline footprint up front
  (~550 B/row ≈ 40 GB at the full 8-month 73M jobs) against free RAM and raises a `MemoryError` with a
  recommended `SUBSET_FRAC` instead of dying silently; (ii) loads in **file-batch chunks**
  (`FILES_PER_CHUNK`, ~2–3 GB raw each), converting each chunk straight to compact numpy so polars
  peak memory stays chunk-sized (labels re-applied per chunk from cell 3's expressions, with a guard
  that demands a cell-3 re-run before re-running B0); (iii) trims steady state — `Xmatch`/`Xsub` are
  preallocated and filled in place, the base matrix `X` and trailing block are zero-copy views into
  `Xmatch`, and `X_num`/`node_k` are freed after B1. Re-smoke-tested end-to-end locally (incl.
  multi-chunk concat and the re-run guard). Recommended server procedure: restart kernel, run only
  cell 3 → B0 → B1 → experiments.
- **2026-07-27** — **Ported the baseline experiments to the GPU notebook** (`scripts/attribution-modeling.ipynb`,
  new cells B0–B1 + E5/E1/E2/E3/E4/E6 + MLP) so the corrected protocol can run on the full 8-month raw
  stream (Jan–Aug 2025) without the reduced-parquet round-trip. Self-contained: no `mlp/` import on the
  server. B0 materializes a compact modeling table in one polars streaming pass (categoricals hashed to
  int codes in-query — strings never held in RAM — then densified via `np.unique` for embedding
  cardinalities; ~200 B/row ≈ 15 GB at 73M jobs; `SUBSET_FRAC` drops whole ClusterId batches during the
  scan). B1 rebuilds `Xmatch`/`Xsub` exactly as in `baseline-experiments.ipynb`, using a **vectorized
  `trailing_rate`** (composite key×time searchsorted; validated bit-identical to `mlp/data.py` on
  randomized tests — the per-key Python loop would not scale to millions of node keys). Split is by
  submit **year-month** (`TRAIN_YM`/`TEST_YM`, default 202501–06 → 202507–08) because the 2025 files
  contain 2024 submit stragglers that a bare month index would misassign. GPU: XGB `device="cuda"`,
  CatBoost `task_type="GPU"`, LightGBM CPU unless a GPU build is present. New MLP arm: two
  family-scoped tabular nets (port of `mlp/model.py`, no sequence branch) — match-time net (17
  embeddings + Xmatch numerics → Failed+hw_fault heads), submission net (12 embeddings + Xsub numerics
  → wait head), BCE `pos_weight` mirroring `scale_pos_weight`, AMP. End-to-end smoke-tested locally at
  2% subset on the reduced parquet. Paper .tex updates deferred until the 8-month GPU run lands.
- **2026-07-24 (c)** — **Corrected 2025 baseline run** (`scripts/baseline-experiments.ipynb`, 25% subset,
  9.13M jobs, temporal train Jan+Feb → test Mar+Apr). Fixed the central bug in the (b) run: E5/E6 were
  benchmarking the bare `encode()` matrix (identity + requested/provisioned + match site), which omits
  the engineered features that carry the signal. Rebuilt with the paper's **disjoint admissible
  matrices**: `Xmatch` (base + cyclical(start) + leak-free trailing site/entry/node/campaign fail+hw
  health) for `Failed`/`hw_fault`, and a **submission-admissible** `Xsub` (submission cols only +
  cyclical(submit) + leak-free queue state) for `wait_time` — the match columns the old wait model read
  were themselves inadmissible at submit. Also fixed `idle_queue_depth`: counted over started jobs only,
  so never-started removals can't turn it into a monotonic time index (this alone moved wait $R^2(\log)$
  from ~0.22 to ~0.44). Corrected numbers: **Failed ROC 0.808–0.818** (CatBoost 0.818), **hw_fault ROC
  0.808–0.828** (CatBoost 0.828, up from 0.772 base / 0.68 in 2024), **wait $R^2(\log)$ 0.36–0.44**
  (CatBoost 0.44, median-AE ~21 min). E1 feature-matched split decomposition: Failed **random 0.970 →
  temporal 0.811, Δ=0.160**. E2 (leaks on top of the *honest* 0.811 model): +JobStatus +0.048, +LHR
  +0.045, month −0.012. E4 3-class: App F1 0.76 / Hardware 0.49 / **Submission 0.28** (macro 0.51) —
  App/Sub still inseparable. Wired paper Results (§4) + Discussion (§5) + methodology Feature-Evaluation
  to these numbers; updated intro Δ (0.970→0.811). Registered a `tachyon` Jupyter kernel for headless
  execution.
- **2026-07-24 (b)** — First **2025 baseline numbers** (`scripts/baseline-experiments.ipynb`, reduced Jan+Feb, 25% subset, temporal Jan→Feb). E5 cross-model: Failed ROC 0.78–0.81 (CatBoost best 0.805), hw_fault ROC 0.74–0.75 (XGB best 0.753, **up from 0.68 in 2024**), wait R²log ~0.5 / median-AE ~12 min. E1 split decomposition: Failed **random 0.938 → temporal 0.796, Δ=0.143** (reproduces the memorization gap; 2024 was 0.96→0.78). Library choice a wash (~0.02). Redesigned analysis D-cell (hanging count-bars). Wired `mlp/` + baselines to `local2025` (`data/2025/reduced_fife_batch_queues_2025.parquet`); `POMS_LAUNCHER` absent from the reduced export (minor).
- **2026-07-24** — Ran the in-depth analysis on 2025 Jan+Feb (full months, 17.4M jobs). Converted all analysis plots to **seaborn**; replaced raw time-series with heatmaps (day×hour failure, site×week hardware incidents), Lorenz concentration curves, failure-driver panels + correlation bar, and per-group fault composition (§12). Notebook renamed `train-gpu.ipynb` → `attribution-modeling.ipynb`.
- **2026-07-23 (b)** — Moved to **2025 raw data** (`/exp/nova/.../fifebatch-history-v3-*.parquet`,
  1,925 files, 120.9 GB, **106.1M jobs**). Full-population stats: **failure rate 24.4%**, fault split
  **91.2% Payload / 8.8% Hardware**, 88.5% matched. Raw schema differs: timestamps are microseconds
  (fixed `to_sec` to handle us/ns/Datetime), no `Group` column (derive it as the token before the
  first `.` in `AccountingGroup`, e.g. `group_uboone.prod.uboonepro` → `group_uboone`). Added §11
  experiment plan; added dataset-characterization cell (timeframe/users/sites/groups/batches).
- **2026-07-23** — Standardized **fault vs. failure** terminology (added a definitions block; a *failure* is the outcome, a *fault* is the attributed cause). Renamed the attribution concept "failure type" → **fault type** and the column `failure_type` → `fault_type` in go-forward code (`mlp/`, `attribution-modeling.ipynb`); reporting is Payload vs Hardware **fault**. `Failed` / "failure rate" keep their outcome meaning. Noted that the high raw failure rate (18–24%) is ~84% payload faults, ~16% Hardware — only ~3–4% of all jobs are Hardware faults.
- **2026-07-22** — Scaffolded the multi-modal neural workstream (`mlp/`): schema-agnostic raw-log loader (validated locally on the processed parquets), leak-free same-site sequence assembly, GRU+tabular fusion model, temporal-split trainer. Targets the raw server dumps (2022–2025) read-only. Analysis pending on the lab-server GPU box.
- **2026-07-21** — Collapsed failure attribution to Hardware-vs-Payload (App/Sub inseparable, 68% mixed campaigns). Dropped `Drain` (missingness leak) and scoped match tasks to non-null match attributes (fixed the earlier no-op). Trimmed windows to 1m/5m/1h. Added truthful wait metrics (median AE ~16 min). Honest results: Failed ROC 0.78, hw_fault ROC 0.68, wait R²(log) 0.50. Renamed notebook to `hold-failure-modeling.ipynb`; created this document.
- **2026-07-20** — Built the staged leak-free notebook (submission vs match families); validated trailing-state helper; added missingness audit. Established Failed ROC 0.80, Hardware recall 61% at match time vs node-failure 0% at submission (context-vs-leakage illustration).
- **2026-07-17** — Renamed failure classes to Application/Submission/Hardware ("what fixes it"); relabeled crash signals and runtime exceptions to Application. Feature-importance study; memorization diagnostics.
- **Earlier** — Split-protocol decomposition (random/campaign-aware/temporal); 3-GBM comparison; leakage identification (`JobStatus`, `JobPrio`, `JobCurrentWaitTime`, month).
