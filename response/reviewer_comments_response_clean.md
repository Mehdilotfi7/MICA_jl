# Response to Reviewers

**Manuscript:** MICA: Model-Informed Change-point Analysis  
**PLOS Computational Biology — Major Revision (PCOMPBIOL-D-26-00585)**

Dear Dr. Kaderali,

Thank you for the opportunity to revise our manuscript. We have addressed the substantive concerns raised by the reviewers and the journal, and we believe the revised manuscript now meets the publication criteria of PLOS Computational Biology.

The major changes include:

1. **Quantitative benchmarking** against 15 competitor change-point detection methods on nine synthetic toy datasets (ODE/SIR, continuous piecewise-linear, AR(1)) and on the Turing Change Point Dataset (TCPD; 42 objectives). Summary tables and mean-F1 figures are provided in the main text and Supplementary Material.
2. **Penalty and model-selection framework.** We clarified that MICA supports several penalty families. The principled information-criterion penalties BIC, MDL, and AIC count the total number of free parameters, including change-point locations, and are used for the TCPD benchmark, the synthetic toy benchmark, and the real-world applications. The legacy user-scalable $\kappa$-heuristic $\mathrm{pen}=\kappa p \log(n)$ is retained only for legacy sensitivity analyses (the original synthetic SIR experiments and the COVID-19/wind-turbine sensitivity checks). Exact formulas for all supported criteria are given in Supplementary Section~S5.
3. **Scaling and normalization sensitivity** for the COVID-19 multi-channel objective, showing that equal-weight log-transformed observations give the most stable and interpretable results.
4. **Uncertainty quantification and identifiability** via a profile-likelihood engine (BreakpointProfiles.jl) and structural-identifiability analysis of a rational simplification of the COVID-19 model.
5. **Greedy-segmentation refinement** study (TASK_A6) comparing the greedy forward-backward scan with an exhaustive BIC search on a toy ODE, demonstrating that a post-hoc refinement step recovers the global BIC optimum.
6. **Methodology figures (Figs 7, 9, 10).** Redesigned the three core methodological figures to clearly illustrate the segmentation module, the optimization module, and the Dynamic Chromosome Length Genetic Algorithm (DCLGA). The updated figures and their captions now explicitly show the segment pool, pop-and-sweep operations, forward/backward scans, and dynamic chromosome extension.
7. **Manuscript text and terminology:** softened claims about automatic inference of "which parameters change"; unified "Model-Informed" terminology throughout the manuscript and removed all remaining "BIC-based" wording in favor of "complexity penalty" or "information-criterion" language; expanded the Discussion limitations; added a discussion of critical slowing down and dynamical network biomarkers; added a physical-meaning paragraph for the wind-turbine parameters; added an over-segmentation caveat to the turbine results; added an Author Summary; converted Author Contributions to CRediT format; and updated figure legends to distinguish data, simulation, and change points.

All source code, data, and outputs are available at the repository referenced in the Data Availability statement.

---

## Editor comments

**Major revision required.** We have addressed each reviewer concern with new experiments, revised text, and additional supplementary material.

---

## Journal requirements

1. **CRediT author contributions.** Updated in the Declarations section of both the clean and marked manuscripts.
2. **Manuscript source files.** The revised `.tex` source files (`Manuscript_PLOS.tex`, `Manuscript_PLOS_diff.tex`, and `Supplementary_Material.tex`) are provided.
3. **Author Summary.** Added between the Abstract and Introduction, 182 words, written for a broad audience.
4. **Copyright/trademark symbols.** Removed all ®/TM symbols from the computer-specifications paragraph.
5. **Figure file formats.** New method figures are provided as `.pdf`, `.eps`, and `.tiff`; all main figures can be converted to PLOS-compliant `.tif` or `.eps` on request.
6. **Figure 13 copyright.** Figure 13 is a line plot of mean recall versus noise level generated directly from the benchmarking data using the GKS plotting library. It contains no third-party clip-art, icons, or copyrighted images; all graphical elements were produced by the authors.

---

## Reviewer #1

**1. Lack of quantitative comparison with alternative change point detection methods.**

We have added two quantitative benchmarking studies:

- **TCPD benchmark:** MICA is compared against 15 competitor methods on 42 TCPD objectives. Results are summarized in Fig 11 and Table 2 of the main manuscript and detailed in Supplementary Section S1.
- **Toy benchmark:** Nine synthetic datasets from three model families (ODE/SIR, continuous piecewise-linear regression, AR(1)) are used to compare MICA with the same competitor set. Results are shown in Fig 12 and Supplementary Section S3.

**2. Missing genetic-algorithm implementation details.**

The default GA settings are now reported in the Methods section and in Table `tab:ga-hyperparameters`. The default configuration is:

```julia
GA(populationSize=150, selection=uniformranking(20),
   crossover=MILX(0.01, 0.17, 0.5), mutationRate=0.3,
   crossoverRate=0.6, mutation=gaussian(0.0001))
```

A lighter variant is used for problems with fewer than 20 parameters, and the optimizer is backend-agnostic (Nelder-Mead, Metaheuristics, etc.).

**3. Parameter identifiability in the COVID-19 model.**

We performed structural identifiability analysis on a rational simplification of the baseline SEIRD model using `StructuralIdentifiability.jl`, and profile-likelihood analysis via `BreakpointProfiles.jl` to report confidence intervals for parameters and change-point locations. These results are discussed in Supplementary Section S7 and referenced in the main text.

**4. Synthetic results lack statistical uncertainty.**

We now report bootstrap confidence intervals for the toy-benchmark mean-F1 scores and profile-likelihood confidence intervals for the COVID-19 and toy-model parameters and change-point dates.

**5. "Model-agnostic" claim vs. ODE/difference-equation focus.**

We clarified that MICA is *simulation-agnostic*: it requires only a callable simulator. The examples use ODEs because they lack closed-form solutions and therefore represent a challenging case; the toy benchmark additionally uses linear regression and AR(1) models, demonstrating applicability beyond ODEs.

---

## Reviewer #2

**1. Downgrade claim that MICA infers "which parameters change".**

We agree. The revised Abstract and Introduction now state explicitly that the user pre-specifies the partition into global (non-segment-specific) and segment-specific parameters; MICA then estimates their values and the change-point locations. We removed any claim that MICA automatically selects which parameters change.

**2. Missing quantitative comparisons with standard/model-based CPD methods.**

Addressed by the TCPD and toy benchmarking studies described under Reviewer #1, point 1.

**3. Penalty coefficient κ and principled model-selection criteria.**

We revised the penalty implementation:

- The total number of free parameters is counted as  
  $p_{\text{total}} = n_{\text{global}} + n_{\text{segments}} \, n_{\text{segment-specific}} + n_{\text{CP}}$.
- MICA supports principled criteria `:bic` (κ=1), `:mdl` (κ=0.5), and `:aic` (κ=2).
- A user-tunable κ-heuristic is retained only as an optional sensitivity knob, not as a formal BIC approximation.
- On COVID-19, the main Figure 4 result remains the original eight-change-point segmentation obtained with the legacy empirical penalty ($\kappa=40$ on segment-specific parameters only) or with zero penalty. For comparison, the principled BIC and MDL criteria select 2 change points at indices 60 and 150 (approximately March 26 and June 24, 2020), and AIC selects 3 change points at indices 30, 60, and 150. A raw-scale BIC comparison figure is added in Section 3.2. The random change-point baseline in Supplementary Section~S6 shows that the BIC solution (loss 870.31) far outperforms the best random 8-change-point baseline (loss 2018.76), confirming that the selected breakpoints are not overfit artifacts.

**4. CP-to-policy alignment and data normalization.**

- The original eight-change-point fit remains the primary COVID-19 segmentation in the main manuscript (Figure 4), with the corresponding policy-event alignment retained as originally reported. The BIC/MDL/AIC results and the raw-scale BIC comparison figure are reported as principled alternatives, and Table~\ref{tab:covid_sensitivity} summarizes the sensitivity of detected change points to scaling and loss choices.
- The policy alignment is explicitly labeled as descriptive and post-hoc, not a formal statistical test.
- The objective uses equal per-channel weights on log-transformed observations, preventing large-magnitude channels (vaccination, death) from dominating the loss. Table~\ref{tab:covid_sensitivity} summarizes the sensitivity sweep over inverse-mean, inverse-standard-deviation, mean-normalized, sqrt, Box-Cox, relative, and absolute losses, confirming that scaling choices materially affect detected breakpoints.

**5. Wind-turbine over-segmentation.**

We agree. The turbine application was rerun with principled information-criterion penalties (BIC, MDL, and AIC). All three criteria select the same four change points (2021-01-01 23:20, 2021-01-04 11:20, 2021-01-08 23:40, and 2021-01-12 22:00), each aligned with a strong SCADA status event: startup, a data-communication-related thermal transition, low-wind external stops, and an external-stop/restart cycle. The original 15-change-point segmentation, obtained with a weaker empirical penalty, is retained in Supplementary Section S4 as a sensitivity analysis; most of its additional change points had weak or no SCADA correspondence and are now explicitly interpreted as over-segmentation artifacts. A κ-sensitivity ladder confirms that the 4-CP solution is stable for κ ≤ 100 and only collapses when the penalty is strongly over-regularized. The wind-turbine manuscript section, figure, and alignment table have been updated accordingly, and the original 15-CP solution is retained in Supplementary Section S4.

**6. Synthetic experiments are too narrow.**

The toy benchmark now covers three model families (ODE/SIR, continuous piecewise-linear, AR(1)) with varying noise levels, change-point configurations, and parameter settings. Full settings and results are in Supplementary Section S3.

**7. GA stability and random-seed sensitivity.**

The default GA hyperparameters and a lighter variant are reported. We ran MICA with multiple random seeds on representative datasets; detected change-point sets were stable across seeds, with small parameter variations consistent with the stochastic optimizer. Summary statistics are included in Supplementary Section S3.

**8. Parameter compensation and uncertainty intervals.**

Profile-likelihood confidence intervals for parameters and change-point locations are now reported for the COVID-19 and toy examples via `BreakpointProfiles.jl`. These intervals help assess whether parameter differences are supported by the data or are artifacts of compensation.

**9. Terminology and notation consistency.**

- Replaced all instances of "Model-based" with "Model-Informed" throughout the manuscript.
- Unified interval notation to half-open $[t_{i-1}, t_i)$ for segments.

**10. Critical slowing down and dynamical network biomarkers.**

We added a paragraph to the Discussion noting that model-free early-warning methods (critical slowing down, dynamical network biomarkers) are valuable when the mechanism is unknown, while MICA complements them by locating transitions precisely and attributing them to specific parameter changes when a mechanistic model is available.

---

## Reviewer #3

**Major Concern 1. Methodological transparency (GA hyperparameters, penalty sensitivity, greedy vs. exhaustive segmentation).**

- GA hyperparameters are reported in the Methods and in Table `tab:ga-hyperparameters`.
- Penalty sensitivity is addressed by the κ ladder and the BIC/MDL/AIC criteria (Reviewer #2, point 3).
- Greedy vs. exhaustive segmentation is quantified in TASK_A6: on a toy ODE with true CPs at 40 and 80, the greedy zero-penalty scan returns [39, 52, 83]; post-hoc BIC refinement collapses this to [40, 79], matching the exhaustive BIC optimum [40, 79].
- The three core methodological figures (segmentation module, optimization module, DCLGA) were redesigned and their captions expanded to make the pipeline visually explicit.

**Major Concern 2. Missing quantitative baselines.**

Addressed by the TCPD and toy benchmarking studies.

**Major Concern 3. Expand limitations discussion.**

The Discussion now contains an explicit limitations paragraph covering:
- heuristic nature and lack of finite-sample guarantees;
- penalty influence and model misspecification;
- conditional interpretation of detected change points;
- imperfect alignment with event logs;
- offline batch operation and scalability limits;
- availability of approximate uncertainty intervals via profile likelihood.

**Writing & Figures.**
- Long technical sentences were simplified where possible.
- Legends distinguishing Data, Simulation, and Change points were added/verified in the relevant figures (Figs 3, 4, 6 in the revised numbering; formerly Figs 7, 12, 14 in the original submission). Figure captions were updated to describe the legend entries explicitly.
- Equation numbering in Section 3.2 was standardized.

**Minor 1. Summary table comparing MICA with SOTA CPD methods.**

A qualitative comparison table is now Table 2 in the main manuscript.

**Minor 2. Link each COVID-19 CP to a specific policy intervention.**

The main COVID-19 analysis retains the original eight-change-point segmentation (Figure 4) and its associated policy-event alignment. The BIC/MDL/AIC results are reported as principled comparisons, and a random change-point baseline in Supplementary Section~S6 confirms that the information-criterion-selected change points are not overfit artifacts.

**Minor 3. Physical meaning of θ₁–θ₇ in the turbine model.**

We added a paragraph to Section 3.3 explaining that θ₁–θ₃ are coefficients of the wind-speed polynomial for electrical (copper) heat generation and are held global (NSP), while θ₄–θ₆ are coefficients of the wind-speed polynomial for thermal resistance (cooling efficiency) and θ₇ combines heat capacity with a baseline thermal-resistance term; these cooling-related parameters are allowed to vary across segments (SP).

---

## Reviewer #4

**Major Concern 1. Objective decrease is not statistical evidence.**

We agree. The revised Discussion explicitly states that MICA does not provide p-values, exact confidence sets, or calibrated hypothesis tests. Detected change points are conditional on the assumed model, loss, scaling, and penalty. For users who need uncertainty intervals, we provide profile-likelihood confidence intervals via `BreakpointProfiles.jl`.

**Major Concern 2. Greedy forward-backward path dependence.**

Addressed by the TASK_A6 refinement study (Reviewer #3, Major Concern 1).

**Major Concern 3. GA variability vs. true evidence.**

The revised text notes that the GA (or any user-selected metaheuristic) is stochastic and that close objective values between competing segmentations should be interpreted cautiously. Multi-seed stability checks are reported in Supplementary Section S3.

**Major Concern 4. "BIC-based" terminology and heuristic penalty.**

The distinction between principled information criteria (BIC, MDL, AIC) and the scaled $\kappa$-heuristic is now explicit in Sections 2 and 3.2, and the exact formulas for all supported penalty families are given in Supplementary Section~S5. We no longer describe arbitrary $\kappa$ values as BIC-based.

**Major Concern 5. Numerical examples not sufficiently specified.**

All benchmark settings (model equations, parameter values, noise levels, change-point locations, competitor grids, and objective functions) are specified in Supplementary Sections S1 and S3 and in the repository scripts.

**Major Concern 6. Expand simulation study to show boundary of applicability.**

The toy benchmark now includes three model families and multiple difficulty settings. The limitations paragraph explicitly discusses scalability and model-misspecification boundaries.

**Minor 1. Use "BIC-based" cautiously.**

We replaced the remaining "BIC-based" wording throughout the manuscript with "complexity penalty" or "information-criterion" language. The distinction between principled information criteria (BIC, MDL, AIC) and the scaled κ-heuristic is explicit in Sections 2 and 3.2.

**Minor 2. Unify "Model-Informed" vs. "Model-based" terminology.**

Addressed; only "Model-Informed" is used.

**Minor 3. Normalization/weighting of COVID-19 channels.**

Addressed by the equal-weight log-transformed objective and the scaling sensitivity sweep (Reviewer #2, point 4).

**Minor 4. Numerical tables of detected CPs and fitted parameters.**

Tables of detected change points and fitted parameters for the COVID-19 and toy examples are provided in Supplementary Sections S2 and S3.

**Minor 5. Acknowledge method is heuristic.**

Addressed by the expanded Discussion limitations paragraph.

---

We thank the reviewers and editor for their constructive comments, which substantially improved the manuscript.

Sincerely,

Mehdi Lotfi and Lars Kaderali
