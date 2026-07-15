# Methodology

This document explains the statistics behind each judgegate command, why
each method was chosen, and where the guarantees have limits.

## The problem judgegate solves

An LLM judge is a classifier, and like any classifier it should be
validated before its decisions matter. The common practice is worse than
no validation: a team eyeballs a handful of judge verdicts, computes a raw
percent agreement against a few human labels, and starts gating releases
on the judge's scores. Raw agreement is a misleading number. A judge that
answers "pass" every time scores 80% agreement on a dataset where humans
pass 80% of items, while carrying zero information.

judgegate replaces that with the standard toolkit for inter-rater
reliability, packaged as a gate with exit codes.

## Cohen's kappa

Kappa measures agreement beyond chance:

```
kappa = (po - pe) / (1 - pe)
```

where `po` is observed agreement and `pe` is the agreement two independent
raters with the same marginal label distributions would reach by chance.
The always-pass judge above scores kappa 0, not 80%. Kappa of 1 is perfect
agreement; 0 is chance; negative values indicate systematic disagreement.

For ordinal label sets (say `bad`, `ok`, `good`), weighted kappa charges
partial penalties for near misses: linear weights scale with the distance
between labels, quadratic weights with its square. judgegate implements
both; configure `gate.weighting` and declare the label order in
`labels.values`.

Degenerate edge: when the human labels cover a single class, chance
agreement is 1 and kappa is undefined. The gate treats such data as
unmeasurable and refuses to certify anything on it, whatever the judge
did, because agreement with a constant has no chance-corrected meaning.
Perfect observed agreement across two or more classes is handled
separately: the items bootstrap is blind at that boundary (every
resample of an all-diagonal matrix is all-diagonal), so the interval's
lower bound is derived from the binomial uncertainty of the agreement
rate itself, mapped through the kappa formula. Twenty flawless labels do
not certify a 0.9 bar, and now the interval says so.

## The confidence interval

Kappa from 100 labels is an estimate, not a fact, and the gate must not
treat it as one. judgegate reports a bias-corrected and accelerated (BCa)
bootstrap interval. Items are the resampling unit; because kappa depends
on the data only through the confusion matrix, resampling n items reduces
to drawing multinomial counts over the confusion cells, which lets the
whole bootstrap run vectorized. The jackknife behind the acceleration term
is exact and grouped: removing any item from the same cell yields the same
leave-one-out kappa, so one computation per occupied cell suffices.

The decision rule uses the interval, not the point estimate:

- interval entirely above `min_kappa`, minimum label count met, and no
  failed probe: TRUSTED
- interval entirely below `min_kappa`, or any failed probe: UNTRUSTED
- otherwise: INCONCLUSIVE

The large-sample standard error of kappa (Fleiss, Cohen, and Everitt,
1969) is also implemented, but it is used only inside power simulations
where running a bootstrap per replicate would be prohibitive. Reported
intervals always come from the bootstrap.

## The label budget

Human labels are the scarce resource in judge validation, and "label 200
and hope" is the folklore judgegate exists to replace. The power engine
answers two questions by simulation:

**How many labels do I need?** Given an assumed true kappa (typically the
pilot estimate), judgegate constructs a judge-human joint distribution
with exactly that kappa and the observed marginals, simulates label sets
of increasing size, and finds the smallest n where the gate certifies the
judge with the target power. The search screens candidates with a fast
large-sample approximation, then validates the final answer against the
real bootstrap decision procedure and enlarges it until the measured
certification rate meets the target, because the fast screen alone runs
a few points optimistic at small samples. The joint construction is a
mixture of perfect agreement and independence, which reproduces the
target kappa exactly while preserving both marginals.

**What can my current labels certify?** The inverse search: the smallest
true kappa that the current label count can reliably certify. If the
answer is 0.85 and your judge sits near 0.7, more labels are not optional.

A judge whose assumed kappa does not exceed the threshold returns "no
amount of labeling can certify this", which is the honest answer: power
analysis cannot rescue a judge that is not actually good enough.

The calibration suite verifies these predictions against the real decision
procedure: simulated judges of known kappa are pushed through the actual
bootstrap gate at the recommended label count, and the certification rate
must match the target power.

## Sequential labeling

Labels arrive one annotation session at a time, and stopping early is
real money. `judgegate sequential` applies a mixture sequential
probability ratio test to the per-item agreement indicators, anchored at
the agreement rate a threshold-kappa judge would achieve given the chance
agreement:

```
theta0 = pe + min_kappa * (1 - pe)
```

Because the indicators are Bernoulli and the null rate is `theta0`, the
information scale under the null is known: the test uses the fixed
variance `theta0 * (1 - theta0)` rather than the sample variance. This
keeps the verdict monotone in the evidence; a judge that agrees on every
single item is the strongest possible signal and certifies fastest,
instead of degenerating into a zero-variance corner case.

The running p-value is valid under optional stopping, so peeking after
every batch does not inflate the false positive rate the way repeated
fixed-n tests would. Honest caveats:

- The normal mixture over a Bernoulli mean is an approximation, and when
  `tau` is not pinned it is chosen from the early data. The test suite
  checks the realized false positive rate under optional stopping.
- Chance agreement `pe` is estimated from the full replay dataset. In a
  true prospective setting it would evolve with the data; the replay
  report states this assumption.
- The bar is interpreted as unweighted kappa; under a weighted gate the
  replay says so in its notes.
- The sequential verdict covers agreement only. Probes still need to run
  before a judge is promoted, and the report says so.

## The probe battery

High average agreement can coexist with systematic bias, and biased
judges are exploitable by whatever they gate. Each probe isolates one
failure mode, reports an effect size with a confidence interval, and
compares it against a configurable tolerance:

- **stability** reruns the judge on identical inputs. The effect is the
  mean per-rerun flip rate: across all reruns, the fraction of item
  verdicts that differed from the base run, with a Wilson interval. The
  per-rerun averaging keeps the number comparable when the runs setting
  changes. A judge that cannot repeat itself adds pure noise to every
  decision.
- **position** re-judges pairwise items with the candidates swapped. A
  content-driven verdict follows the content into its new slot; a verdict
  that stays with the slot is position bias, a well-documented failure of
  pairwise LLM judges.
- **verbosity** is fully offline: items are split at the median output
  length and the judge's deviation from the human verdict is compared
  across the halves in both directions (rating above the humans, and
  below them), as a two-proportion difference with a normal interval.
  The larger gap is reported, so the answer does not depend on which way
  the label list happens to be written.
- **format** strips markdown decoration (bold, headings, code fences)
  while preserving every word, then re-judges. Changed verdicts mean the
  judge is grading typography.

Probe decisions apply the same interval discipline as the kappa gate: a
probe FAILS only when its confidence interval proves the effect exceeds
the tolerance, and a point estimate above tolerance that is not
statistically confirmed surfaces as WARN with a note, never as a hard
failure. A failed probe forces UNTRUSTED regardless of kappa. Skipped
probes (offline runs, wrong judge type, insufficient data) are reported
as skipped, never silently dropped.

## Reproducibility

Set `gate.seed` and every resampling procedure becomes deterministic:
identical inputs produce identical reports. The judge response cache keys
on endpoint, model, parameters, and prompt, so verify plus the full probe
battery costs roughly one pass over the data.

## Known limitations

- Kappa inherits the quality of the human labels. If your annotators
  disagree with each other, measure that first; judgegate can do it by
  putting one annotator's labels in `human_label` and the other's in
  `judge_label`.
- The bootstrap assumes items are exchangeable. If your labels mix very
  different strata (easy smoke tests and hard adversarial cases), a
  per-stratum verification is more informative than one pooled verdict.
- Probes cover four failure modes, not all of them. A judge can pass every
  probe and still be wrong in ways your domain cares about.
- judgegate quantifies agreement and bias. It cannot decide whether your
  grading rubric itself measures what your users care about. Statistics
  after judgment, not instead of it.
