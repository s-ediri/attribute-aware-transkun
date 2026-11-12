# Phase 1 C versus D full-recording validation

## Status

Completed and audited on November 12, 2026.

This report covers the exploratory MAESTRO validation comparison between:

- **C:** structured attribute heads trained with score influence `0.0`;
- **D:** structured attribute heads trained with score influence `1.0`.

The three final seeds were `20261108`, `20261109`, and `20261110`. Each run
scored the same 137 complete validation recordings, in the same order, with no
piece limit. The comparison is paired and every reported delta is **D minus C**.

## Audit

All three result files passed the following checks:

- `complete == true`;
- `completed_pieces == piece_count == 137`;
- 137 unique recording names in an identical order across seeds;
- labels are exactly `C` and `D`;
- all four expected metric variants are present for every recording;
- metadata, evaluator, config, and checkpoint SHA-256 values match the current
  files;
- evaluator Git commit is `1d392b5c595b0fbcaad009400b5c6668e2f2dab4`;
- recorded Git state is clean.

The common recording-order SHA-256 is
`14a962eff0c9a6e02c2d6654edb3110cadc280e51a0e75db952e40a9cfd5a805`.

## Per-seed pre-specified outcomes

Intervals are the evaluator's 10,000-replicate paired piece bootstrap intervals.

| Seed | Outcome | C macro F1 | D macro F1 | D-C | Paired 95% interval | Improved / declined / tied pieces |
|---|---|---:|---:|---:|---:|---:|
| 20261108 | Note | 0.981538 | 0.982014 | +0.000476 | [+0.000296, +0.000662] | 93 / 38 / 6 |
| 20261108 | Note+offset | 0.769536 | 0.772642 | +0.003106 | [+0.002054, +0.004234] | 102 / 35 / 0 |
| 20261108 | Note+velocity+offset | 0.765215 | 0.768419 | +0.003204 | [+0.002126, +0.004366] | 101 / 36 / 0 |
| 20261109 | Note | 0.981642 | 0.981815 | +0.000173 | [+0.000004, +0.000339] | 84 / 44 / 9 |
| 20261109 | Note+offset | 0.770289 | 0.771620 | +0.001330 | [+0.000041, +0.002639] | 84 / 52 / 1 |
| 20261109 | Note+velocity+offset | 0.766021 | 0.767464 | +0.001443 | [+0.000175, +0.002745] | 83 / 54 / 0 |
| 20261110 | Note | 0.980825 | 0.981528 | +0.000703 | [+0.000529, +0.000886] | 103 / 31 / 3 |
| 20261110 | Note+offset | 0.770574 | 0.773532 | +0.002958 | [+0.001876, +0.004048] | 106 / 30 / 1 |
| 20261110 | Note+velocity+offset | 0.766450 | 0.769503 | +0.003053 | [+0.001977, +0.004137] | 102 / 34 / 1 |

The supportive note+velocity outcome was also positive in every seed:
`+0.000548`, `+0.000346`, and `+0.000751`, respectively. Each corresponding
paired interval excluded zero.

## Across-seed descriptive summary

The table below averages each macro result across the three seeds. Its interval
is an additional descriptive calculation: D-C was first averaged across seeds
within each of the 137 recordings, then those recordings were resampled 10,000
times using seed `20261112`. This preserves the recording as the independent
bootstrap unit and does not treat the three predictions for one recording as
independent observations.

| Outcome | C precision | C recall | C F1 | D precision | D recall | D F1 | Mean D-C F1 | Seed-averaged piece 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Note | 0.997686 | 0.965951 | 0.981335 | 0.997686 | 0.966815 | 0.981786 | +0.000451 | [+0.000334, +0.000567] |
| Note+offset | 0.782969 | 0.758066 | 0.770133 | 0.785114 | 0.760823 | 0.772598 | +0.002465 | [+0.001725, +0.003219] |
| Note+velocity | 0.992388 | 0.960936 | 0.976184 | 0.992493 | 0.961891 | 0.976733 | +0.000548 | [+0.000403, +0.000701] |
| Note+velocity+offset | 0.778608 | 0.753941 | 0.765895 | 0.780861 | 0.756795 | 0.768462 | +0.002567 | [+0.001829, +0.003322] |

D had higher F1 in all three seeds for every metric. For the primary
note+offset outcome, the seed effects ranged from `+0.001330` to `+0.003106`.
The note-only safeguard also favored D in every seed rather than indicating a
trade-off. The offset-aware gains reflect increases in both macro precision and
macro recall.

## Interpretation and frozen next decision

These results provide consistent validation-set evidence that allowing the
trained attributes to influence structured decoding improves full-recording
transcription relative to the matched post-hoc control. The largest gains are
on the two offset-aware outcomes, while note-only performance is preserved and
slightly improved.

The practical-effect and safeguard thresholds were subsequently approved on
November 12, 2026 and frozen in `research/PHASE1_TEST_ANALYSIS_PLAN.md`:

- minimum worthwhile note+offset D-C: `+0.0020` macro F1;
- note-only non-inferiority margin: `-0.0010` macro F1.

Applying that rule to these validation results gives
`progression_supported`: note+offset D-C is `+0.002465` with interval
`[+0.001725, +0.003219]`, and the note-only lower bound is `+0.000334`, above
the `-0.0010` safeguard margin.

This is a retrospective classification of exploratory validation evidence,
because the numerical thresholds were approved after validation was inspected.
It supports proceeding to the one-time test analysis but is not independent
confirmation. The MAESTRO test split remains untouched at this stage.

## Local result artifacts

- `checkpoint/phase1-C-vs-D-seed-20261108-full-validation.json`
- `checkpoint/phase1-C-vs-D-seed-20261109-full-validation.json`
- `checkpoint/phase1-C-vs-D-seed-20261110-full-validation.json`

These files are intentionally ignored by Git. They contain the complete
per-recording results and the authoritative per-seed bootstrap summaries.