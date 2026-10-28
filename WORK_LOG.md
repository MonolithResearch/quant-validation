# Bounded gate repair — 2026-10-28

## Scope and custody

Implementation work is confined to this separate clone of the public remote
`https://github.com/MonolithResearch/quant-validation.git`, starting at
`5f10186a23784b41b165b1c6bccf9e47e72ed448`. No private Git history, strategy data,
credentials, installs, subagents, commits, or pushes are part of this task.
Parent owns independent review and publication. MIT license remains intact.
Existing files receive timestamped, hash-verified sibling `.bak_*` copies before
rewrites; those copies are ignored. No repository AGENTS.md was present.

Read `C:/Users/there/AGENTS.md`; applied systematic-debugging, test-driven-development,
and verification-before-completion workflows to the supplied implementation brief.

## Plan and acceptance criteria

1. Reproduce the published defect and inspect the user-owned local diff read-only.
2. Add red regressions: missing evidence, domains, trial accounting, complete gate
   outcomes, monotonicity, and CLI error paths.
3. Repair the existing API without a broad redesign. `gate_pass` must be false
   whenever any required evidence is missing. Add explicit readiness/status and
   missing reasons; keep computable descriptive statistics. No invented PBO.
4. Align the CLI, truthful README, executable synthetic demo, and CI configuration.
5. Run the full suite and demo on installed Python 3.14, inspect the diff, verify
   backups/source hashes/license, and hand changes back uncommitted.

## Baseline and source review

- New clone: `py -3.14 -m pytest -q`: **27 passed**, exit 0.
- Reproduction on seed-11 normal synthetic returns (800 observations, mean .0016,
  std .008): published report claims `gate_pass=True` with no PBO or trade count,
  both for `n_trials=1` and `n_trials=500` without dispersion.
- Root causes: optional evidence is skipped in gate conditions; absent dispersion
  becomes zero; the trial-Sharpe branch replaces declared trial count with list size.
- Environment: CPython 3.14.3, NumPy 2.4.4, SciPy 1.17.1, pytest 9.0.3.
- Read-only source: `C:/Users/there/monolith_showcase/repos/quant-validation`.
  Three modified tracked files and one new CLI were inspected. Source SHA256:
  - `monolith_validation/core.py`: `1079DF06BA6E1CD1AB63ADDAC592ED97ED24D708DDE73E355C9FE5E194CA83B8`
  - `pyproject.toml`: `4EC1CE7C72AD17DE8483881B6AE41F8FBA72EE1927B10F7DDA4B45274473552B`
  - `tests/test_core.py`: `0E06102078CCD6A0C31774ACD14199DCF41337726754CADBD39F45B32797B5ED`
  - `monolith_validation/cli.py`: `C80AFC6013A539531FE10E1C709FFF8163F738C012209B16F132F715BE4E13C8`
- Attribution: retain the user's deflation-honesty guard concept and console entry
  point; adapt its regression to require PBO as well. Review CLI behavior before
  reuse: do not copy independent verdict logic, silent column fallback, ignored
  malformed key/value pairs, hard-coded thresholds, or non-standard JSON.
- Process shortfall: pytest was inadvertently run in the protected checkout
  (**28 passed**). No source was edited, but pytest/Python may have refreshed
  ignored cache/bytecode files. Do not modify that checkout to clean them up.
  All subsequent execution is confined to the new clone.

## Primary references

Review the authors' DSR and PBO papers, especially units, trial count/dispersion,
rank ties, and the distinction between a diagnostic and full gate evidence:

- https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf

## Validation and handoff

Implementation and local verification completed; independent parent review and
any publication remain with the parent. Changes are uncommitted and unpushed.

### Core red/green cycle

- Added 97 cases in `tests/test_gate_evidence.py` before implementation.
  Published baseline: 93 failed, 4 passed. Failures included actual false passes,
  missing readiness fields, count reduction, bool/float coercion, missing finite
  validation, ambiguous NumPy array truth tests, and CDF tail cancellation.
- `py -3.14 -m pytest -o addopts='' -q --tb=short` after the repair:
  **124 passed in 1.19s**, exit 0, no warnings (27 original + 97 added).
- DSR numerical anchor: authors' 2014 paper, pp. 9–10, N=100, T=1250,
  annual SR=2.5 with 250 periods/year, trial SR variance=1/500, skew=-3,
  kurtosis=10; DSR rounds to 0.9004. Original formula passes this anchor.
- Compatibility choice: keep existing names/signatures and boolean `gate_pass`.
  Add readiness/status, missing reasons, and applied thresholds. Missing DSR or
  expected maximum is now None rather than mislabelled PSR/zero. `to_dict()`
  serializes nonfinite diagnostic values as null. Invalid inputs now raise
  ValueError instead of being dropped/truncated. Lower-level single-trial/zero-
  dispersion expected maximum remains mathematically zero; the report refuses
  to certify zero-dispersion multi-trial evidence.
- Effective trial count is max(declared count, Sharpe sample size, matrix columns).
  No automatic correlation/effective-trials estimator or fabricated PBO is added.
- PBO paper fetch through browser search timed out; no claim of a fresh full-PBO
  formula audit is made. The existing CSCV ranking convention is retained.

### CLI and final numeric red/green cycles

- Before adding the reviewed CLI in this clone:
  `py -3.14 -m pytest tests/test_cli.py -o addopts='' -q --tb=no`:
  **28 failed, 2 passed in 57.45s**, exit 1. Final environment inspection found
  a pre-existing editable installation at
  `C:/Users/there/h2_validation_worktree/monolith_validation` (from installed
  distribution `direct_url.json`). Module resolution could find an existing CLI
  outside this clone while this clone had no CLI file. This red run is therefore
  not an isolated test of a fresh package installation and may also have refreshed
  ignored bytecode outside the intended clone. No source changes were made there.
- After adding the CLI:
  `py -3.14 -m pytest tests/test_cli.py -o addopts='' -q --tb=short`:
  **30 passed in 50.42s**, exit 0. Three further edge cases were then added.
- The adapted CLI keeps the author's command/option vocabulary and 0/1/2 exit
  convention, but delegates verdicts entirely to the API. Its parser uses
  CSV/text/inline inputs, strict errors and JSON; Parquet is deliberately omitted.
  Module provenance was explicitly verified to resolve package, core, and CLI
  under this clone. CLI subprocess tests now disable bytecode writes as well.
- Additional numeric review exposed four failures in 106 core evidence cases:
  two scale-invariance failures in CSCV (finite values at 1e-200 and 1e200),
  an overflowed deflation benchmark passing a relaxed zero DSR threshold, and an
  unhelpful exception for a malformed scenario mapping. These were reproduced
  before repair, then fixed with scaled column moments, an unusable-benchmark
  guard, and explicit mapping validation. Added threshold-boundary and benchmark-
  semantics checks as well. The CSCV ranking/tie policy was not redesigned.

### Final commands and results

Executed in this clone on CPython 3.14.3:

```text
py -3.14 -m pytest -o addopts='' -q -W error --tb=short
166 passed in 54.62s; exit 0 (warnings treated as errors)

py -3.14 -m examples.synthetic_demo
Returns only: INCONCLUSIVE
Declared search without dispersion: INCONCLUSIVE
Complete engineered fixture: PASS
Same fixture with too few trades: FAIL
All four asserted; exit 0

git diff --check
exit 0 (Git emitted its local LF/CRLF conversion advisory for README)
```

- Parsed `pyproject.toml` with `tomllib`, resolved the configured `gate-check`
  callable, and asserted package/core/CLI module paths are inside this clone.
  No installation was performed; an installed console wrapper for this change
  is therefore not locally verified. CI now exercises that wrapper after install.
- Rechecked all four protected source SHA256 values listed above: unchanged.
  Its Git status remains exactly three modified tracked files plus the new CLI.
  This confirms source preservation, not absence of the cache side effects
  already disclosed above.
- LICENSE disk Git object and baseline `HEAD:LICENSE` both equal
  `e049aff596f95ce975d210c7fb53f58caa0bcd67`.
- Backup copies were hash-compared immediately after creation. Every timestamped
  sibling backup is ignored by Git; none is intended for publication.
- New clone HEAD remains `5f10186a23784b41b165b1c6bccf9e47e72ed448`.
  No commits, pushes, installs, or spawned agents.

### Exact deliverable files

Modified:

- `monolith_validation/core.py`: validation, complete-evidence gate/status,
  truthful missing DSR/PBO, preserved declared trial count, stable numerics,
  JSON-safe serialization, and report contracts.
- `monolith_validation/__init__.py`: evidence-scoped package description.
- `pyproject.toml`: reviewed console entry point, accurate description, 3.14 classifier.
- `.github/workflows/ci.yml`: add 3.14 to configured matrix, synthetic demo and CLI smoke checks.
- `.gitignore`: ignore timestamped sibling backups.
- `README.md`: reproducible usage, explicit evidence/units/status contracts,
  limitations, and primary references; removed unsupported private/live claims.

Added:

- `monolith_validation/cli.py`: one API-backed CLI with strict parsing/errors/JSON.
- `tests/test_gate_evidence.py`: 106 evidence and numeric regression cases.
- `tests/test_cli.py`: 33 real subprocess integration cases.
- `examples/synthetic_demo.py`: deterministic four-outcome executable example.
- `WORK_LOG.md`: this provenance, validation, custody, and handoff record.

### Limits for parent review

- Only Python 3.14.3 ran the suite here. Python 3.10–3.14 CI is configured,
  not executed remotely in this task. Python 3.13 was not tested; no dependencies
  were installed. Packaging build and installed-wrapper execution remain unverified.
- The public API still exposes the existing low-level formulas; the complete
  evidence/readiness contract belongs to `litigious_report`. Report callers must
  handle None for unavailable DSR/expected maximum and strict ValueError inputs.
- Existing CSCV equal-block remainder truncation, deterministic tie handling,
  strictly-below-median convention, and combinatorial runtime are documented.
  No claim that the gate proves alpha, provenance, trial independence, or a
  complete search archive. No market/private data or new trading claims added.
- Parent owns fresh review, publication decision, commit, and push.


## Bounded independent-review follow-up - 2026-10-28

Scope: only core.py, cli.py, tests/test_gate_evidence.py, tests/test_cli.py,
README.md if semantics need clarification, and this log in this clone. Parent
retains staging/commit/push and wheel rebuild. No agents or installations.
Existing edits are preserved; each rewrite uses a dated sibling backup verified
byte-for-byte before writing. This follow-up does not access the original checkout.

Initial reproductions on the existing implementation:
- Fully masked returns, returns_matrix, and trial_sharpes each produced PASS
  against the complete 80-observation engineered fixture. np.asarray discards
  top-level masks and masks on nested rows; nested masked scalars may warn while
  becoming NaN, so validation must inspect masks before conversion.
- periods_per_year=10**309 raised OverflowError from math.sqrt rather than the
  documented ValueError. Integer validation accepted arbitrary-size values.
- CSV inspection confirms the first nonnumeric row is unconditionally inferred
  as a header; missing-marker rows can disappear. csv.Error is not handled by
  the CLI input-error boundary.

Plan: add actual API/subprocess regressions, record RED, implement bounded
validation/parser fixes, record GREEN, run the requested full warning-as-error
suite and synthetic demo, inspect only the owned diff, and record limitations.

### Follow-up RED evidence

Before production edits, all runs used Python 3.14, -B, pytest's cache provider
 disabled, empty addopts, and warnings as errors:

- tests/test_gate_evidence.py -k 'masked or unrepresentable': **18 failed,
  9 passed, 106 deselected in 1.87s**, exit 1. Failures were silent mask
  stripping, masked-scalar conversion warnings instead of ValueError, ppy
  OverflowError, and oversized count inputs being accepted.
- tests/test_cli.py -k 'first_row_missing or csv_reader_errors or huge_integer
  or unterminated or named_matrix': **23 failed, 8 passed, 33 deselected in
  46.95s**, exit 1. NA/null/N/A/None/<NA> became headers and yielded PASS;
  140,000-character fields emitted csv.Error tracebacks; huge ppy escaped as
  OverflowError; oversized trade/split/threshold counts were accepted. A CSV
  ending with an unterminated quoted numeric field also yielded PASS.
- tests/test_gate_evidence.py -k 'array_protocol': **3 failed, 133 deselected
  in 1.16s**, exit 1. np.asarray stripped masks returned by __array__ adapters.
- Existing NaN/blank-row rejection and valid unmasked/header controls passed;
  these are retained as compatibility and warning-free boundary checks.

Implemented for subsequent GREEN verification:
- Pre-coercion mask inspection over arrays/nested sequences; asanyarray preserves
  masks exposed by array adapters for inspection before numeric conversion.
- Finite-float range validation for integer inputs, retaining integer identity.
- Reject reserved missing/nonfinite first-row tokens before header inference;
  preserve named headers and numeric headerless observations.
- Strict CSV quoting plus csv.Error/OverflowError handling through the existing
  structured ERROR/exit-2 CLI boundary. No separate CLI gate computation.
- README records masks, numeric bounds, reserved headers and CSV parsing errors.


### Follow-up final validation and handoff

An additional adversarial check nested custom __array__ adapters inside a matrix:
**1 failed, 5 passed, 133 deselected in 1.18s**, exit 1, before the final mask
inspection refinement. The matrix could still discard child masks during NumPy
assembly. The validator now inspects array adapters encountered inside sequences
as well. Existing plain-array, nested-array, and unmasked behavior is preserved.

Targeted GREEN runs:
- API regressions, selector 'masked or unrepresentable or array_protocol':
  **33 passed, 106 deselected in 1.01s**, exit 0.
- CLI regressions, selector 'first_row_missing or csv_reader_errors or
  huge_integer or unterminated or named_matrix':
  **31 passed, 33 deselected in 44.26s**, exit 0.

Final full verification in this clone (CPython 3.14.3):

```text
py -3.14 -B -m pytest -p no:cacheprovider -o addopts='' -q -W error --tb=short
230 passed in 103.76s (0:01:43); exit 0

py -3.14 -B -m examples.synthetic_demo
Returns only: INCONCLUSIVE
Declared search without dispersion: INCONCLUSIVE
Complete engineered fixture: PASS
Same fixture with too few trades: FAIL
All four demo assertions passed; exit 0

git diff --check
exit 0
```

Exact follow-up changes, relative to the inherited implementation:
- monolith_validation/core.py: pre-conversion masked-evidence checks, including
  full/partial masks, nested sequences, masked scalars, object arrays and array
  adapters; finite-float bounds for integer inputs. +25/-3 lines.
- monolith_validation/cli.py: reserved missing/nonfinite tokens checked before
  header inference, strict CSV quoting, csv.Error/OverflowError converted to
  ERROR with exit 2 (strict JSON when requested). +7/-2 lines.
- tests/test_gate_evidence.py: 33 added API regression/control cases; 139 total.
- tests/test_cli.py: 31 added actual-subprocess cases; 64 total. Includes each of
  returns/matrix/trial-sample first-row markers, 140k-character fields, 10**309
  integer options, malformed quoting, and valid named-header complete PASS.
- README.md: documented masked input, integer bounds and reserved-header behavior.
- WORK_LOG.md: this reproduction, test-first sequence, final evidence and handoff.

Final source SHA256:
- core.py: 0c8a54deb1910f474eb0f0223e49bf3b2c37207df8b92f2c4f7f84bcce67b277
- cli.py: a500a4f70698104de49d191533afca7acc1db30219c6f3742e73c42f3c79f67d
- test_gate_evidence.py: 12dd67f67cbdd790a9edcdd52e7305dc5c4b2d61121242f2e3c2598a70e9c217
- test_cli.py: f1716c24570e2ac9e9399925f3fbe8876a1f0c17ae82ca6b17b5c56f7e4ae623

Custody and limitations:
- Every rewrite was preceded by a dated sibling .bak_ file, checked byte-for-byte
  against the original immediately before writing. Backups are ignored by Git.
- Only the six owned files were edited in this follow-up. The inherited edits in
  other files remain intact. No staging, commits, pushes, installs, agents,
  original-checkout access, live endpoints, or deployment operations were used.
- Package/core/CLI import paths were reasserted to resolve inside this clone.
  CLI subprocesses disable bytecode writes. Tests use temporary synthetic fixtures.
- Automatic header inference remains supported for nonreserved text labels;
  unknown vendor-specific missing-value markers cannot be distinguished from an
  intended header without an explicit schema. The reserved markers are documented.
- Masks already discarded by the caller before the API sees the input cannot be
  reconstructed; the API checks masks still present on supplied values/adapters.
- Only the installed Python 3.14 environment was exercised here. No wheel build,
  installation, remote CI, or public-release verification is claimed in this
  follow-up. Parent owns fresh independent review, wheel rebuild and publication.


## Parent publication verification

Independent parent execution of `py -3.14 -B -m pytest -p no:cacheprovider -o addopts='' -q -W error --tb=short` returned **230 passed in100.08s**. The four-outcome synthetic demo passed. A fresh wheel built without dependency installation via `pip wheel --no-build-isolation --no-deps`; SHA-256 `61048920ff596575646e84a9cf647754f2840d7db70517ccff69268b7744f7d4`,18,049 bytes. Direct wheel imports resolved from that archive, console-entry metadata resolved, and the CLI returned structured INCONCLUSIVE/exit2 for missing evidence. Installed-wrapper and Linux/Python-matrix verification belong to the subsequent GitHub workflow. Independent follow-up closure review is recorded in the external publication receipt.

Final independent follow-up review closed all three prior findings: 42 independent
checks passed (28 API and 14 actual CLI cases) under Python 3.14 with warnings as
errors. Masked evidence, first-row missing markers, long CSV fields and huge
annualization values reject through the intended contracts; valid controls pass.
Reviewed source hashes remained unchanged. Evidence: external
C:/Users/there/MONOLITH/FINAL_VALIDATION_REVIEW_20261028.md.
