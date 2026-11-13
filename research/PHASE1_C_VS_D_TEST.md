# Phase 1 C versus D MAESTRO test result

## Status

The one-time MAESTRO test analysis completed on November 13, 2026 under the
approved and frozen plan in `research/PHASE1_TEST_ANALYSIS_PLAN.md`.

The final decision is:

```text
offset_improvement_below_worthwhile_threshold
```

System D produces a reproducible positive improvement over System C, but the
primary note+offset effect does not reach the approved minimum worthwhile effect
of `+0.0020` macro F1. The approved progression gate therefore does not pass.

## Protocol audit

All three test evaluations passed the frozen protocol audit:

- seeds: `20261108`, `20261109`, and `20261110`;
- split identity: `test` / `full_piece_paired_test`;
- 177 complete recordings per seed, with no piece limit;
- 177 unique recording names in identical order across all seeds;
- labels exactly `C` and `D`;
- all four expected metrics present for every recording;
- test metadata, evaluator, config, and all checkpoint SHA-256 values match;
- clean execution revision `e59e45a1dd389642827598d935a3ad867a5ce9d3`;
- evaluator SHA-256
  `0b699710f28f59826275919c4efcacebc4305d06eb37db379194b7e0788264f1`;
- common recording-order SHA-256
  `27cb6125576d77a8305bfab0be747819cfa0475e535543e7d861e13b8d5b27cf`.

The three source-result SHA-256 values are:

| Seed | Result SHA-256 |
|---|---|
| `20261108` | `69cdccc38f9fe3c88c4ecf8a95e6cb69d1932023a8c3c5bad2161c709955e11d` |
| `20261109` | `8902f230199a4094d38fb66fbe5df5a789ecfa155681670253cf40bb23e6bdd7` |
| `20261110` | `b2db43c585648f81d366b23e4abbecb0f5a1042284648e7da5e002a7415bcf63` |

## Aggregate results

Deltas are **D minus C**. The interval is the frozen 10,000-replicate paired
recording bootstrap with seed `20261112`. D-C is averaged across the three seeds
within each recording before the 177 recordings are resampled.

| Outcome | C precision | C recall | C F1 | D precision | D recall | D F1 | D-C F1 | Paired 95% interval | Improved / declined / tied recordings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Note | 0.997790 | 0.968941 | 0.982960 | 0.997807 | 0.969671 | 0.983349 | +0.000389 | [+0.000305, +0.000476] | 135 / 41 / 1 |
| Note+offset | 0.825173 | 0.801978 | 0.813248 | 0.826305 | 0.803653 | 0.814664 | +0.001416 | [+0.001012, +0.001824] | 128 / 49 / 0 |
| Note+velocity | 0.991619 | 0.963075 | 0.976947 | 0.991874 | 0.964025 | 0.977565 | +0.000618 | [+0.000496, +0.000748] | 141 / 36 / 0 |
| Note+velocity+offset | 0.820080 | 0.797142 | 0.808289 | 0.821397 | 0.798986 | 0.809881 | +0.001592 | [+0.001182, +0.002014] | 130 / 47 / 0 |

D improves both macro precision and macro recall on both offset-aware outcomes.
All four aggregate intervals are strictly above zero.

## Per-seed primary outcome

| Seed | C note+offset F1 | D note+offset F1 | D-C | Paired 95% interval |
|---|---:|---:|---:|---:|
| `20261108` | 0.814005 | 0.815360 | +0.001355 | [+0.000611, +0.002127] |
| `20261109` | 0.812345 | 0.813519 | +0.001174 | [+0.000323, +0.001985] |
| `20261110` | 0.813395 | 0.815113 | +0.001718 | [+0.000937, +0.002487] |

The note+offset effect is positive in every seed, and every seed-specific paired
interval excludes zero. Every seed's point estimate remains below `+0.0020`.

## Approved decision gate

| Criterion | Observed | Required | Result |
|---|---:|---:|---|
| Aggregate note+offset D-C | +0.001416 | at least +0.0020 | **Fail** |
| Note+offset interval lower bound | +0.001012 | greater than 0 | Pass |
| Note-only interval lower bound | +0.000305 | greater than -0.0010 | Pass |

The observed primary effect is approximately 71% of the approved minimum
worthwhile effect. It is statistically supported and consistent across seeds,
but it does not satisfy the practical-effect clause. The note-only safeguard
passes comfortably and shows a small improvement rather than a trade-off.

## Interpretation

The test result supports the scientific claim that allowing trained velocity,
timing, and boundary-presence attributes to influence semi-CRF decoding yields
a small positive transcription improvement over using the same trained
attributes only after interval selection.

It does **not** support the stronger practical claim required by the approved
Phase 1 progression rule. Therefore:

1. Phase 1 should be reported as a positive but sub-threshold result.
2. Progression to duration-aware offset modeling is not supported by the primary
   practical-effect gate.
3. The supportive note+velocity+offset improvement is also positive, but it
   cannot override the failed primary practical threshold.
4. No threshold, checkpoint, metric, or bootstrap setting should be changed in
   response to this result.
5. Any further mechanism or stratum analysis must remain explanatory and must
   not be used to relabel the confirmatory decision.

The test effect is smaller than the exploratory validation estimate
(`+0.001416` versus `+0.002465` for note+offset), illustrating why the untouched
test split and the practical-effect threshold were necessary.

## Reproducibility

Authoritative aggregate artifact:

```text
checkpoint/phase1-C-vs-D-full-test-aggregate.json
```

Artifact SHA-256:

```text
32eacfa5d58ddf57c25e85146767cc148a57d2a283e460fd9c46440e4ee80ed5
```

An independent rerun of `scripts/aggregate_phase1_c_vs_d.py` produced an exactly
equal JSON object for the protocol, per-seed results, aggregate summary,
decision, source hashes, and analysis provenance.

The generated JSON artifacts remain local and ignored by Git. This tracked
report records the compact final result needed for the thesis.