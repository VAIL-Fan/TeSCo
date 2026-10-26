# Before this repository goes public

Four things block or complicate a public GitHub release. None of them is a code problem; all of
them need a decision rather than a patch, so they are recorded here instead of being acted on.

## 1. Citation metadata is missing, and the local paper copies are the anonymous versions

The paper — *Exploiting Textual Semantics for Robust Cross-View Object Correspondence* — has been
accepted and published at NeurIPS 2026, so there is no anonymity constraint on publishing this
repository.

The copies in this working tree are still the submission versions, though: `5271.pdf` has
"Anonymous Author(s)" on the title page and "Submitted to 40th Conference on Neural Information
Processing Systems (NeurIPS 2026). Do not distribute." in the footer, and `v9.tex` loads
`\usepackage{neurips_2026}` without the `final` or `nonanonymous` option. They therefore cannot
supply the author list, and no `CITATION.cff` or BibTeX entry has been written for that reason.

To add one, take the author list and venue details from the camera-ready version and drop them
into `CITATION.cff` plus a `## Citation` section in `README.md`. A public repository for a
published paper should carry both.

## 2. The weights exceed GitHub's free LFS quota, and the file count hurts clones

Measured on the working tree:

| | |
|---|---|
| tracked files (before the cleanup below) | **629,025**, of which `text_labels/HANDAL-X` is 618,880 |
| `.git/objects` — the actual history | **52 MB** |
| `.git/lfs` — the weight cache | **5.6 GB** |
| configured remote | `origin git@github.com:fb-reps/TeSCo.git` — already set |

`.gitattributes` already routes `*.pt` and `*.pth` through Git LFS, and it is working: the
checkpoints are committed as 134-byte pointers, not as blobs. **GitHub's 100 MB per-file limit is
therefore not the obstacle** — the weights never enter git as large objects.

The two real obstacles are:

* **LFS quota.** The free tier gives 1 GB of LFS storage and 1 GB of bandwidth per month, against
  ~3.4 GB of weights. A push exhausts it immediately. Either buy an LFS data pack, or host the
  weights off GitHub.
* **File count.** 629 k tracked files makes `git clone` slow enough to be a real barrier, even
  though the packed history is only 52 MB.

Both are addressed by keeping source in git and everything else out of it, which `.gitignore`
now does:

| path | why it is not tracked |
|---|---|
| `text_labels/` | 2.9 GB, 638,884 files; regenerable with `tools/precompute_dam_frame_captions.py` |
| `checkpoint/` | Meta's unmodified SAM 2.1 release — better fetched from its own source |
| `training/checkpoints_EgoExo/` | 787 MB each; host on a model hub and link from the README |
| `data/` | Ego-Exo4D and HANDAL-X carry their own licences and are not ours to redistribute |
| `outputs/`, `sam2_logs/`, `work_dirs/`, `*.log` | run intermediates |

That leaves **138 tracked files** — the code, configs and docs — which is what a reader actually
needs in order to reproduce.

Note that untracking with `git rm --cached` (which is what was done; no file was deleted from
disk) stops them being tracked **from the next commit onwards**. The earlier commits
`d0bbbb18` and `ecf86f7d` still reference them, so `.git` keeps its size and a `git push` would
still upload the LFS objects behind those commits. To get a genuinely small repository, start a
fresh history from the current tree, or rewrite with `git filter-repo`. Both discard the existing
two commits, so that is left for the author to decide.

## 3. There is no LICENSE

The repository has no licence file. Without one, "public on GitHub" grants no rights to use the
code at all, which is likely not the intent.

This is the author's decision, not something to pick on their behalf. The relevant constraint is
that `sam2/` is derived from Meta's SAM 2, released under **Apache 2.0**, so a compatible licence
is needed and the SAM 2 notice should be retained. `third_party/describe_anything` carries its own
terms and should be checked separately.

## 4. Datasets and ground truth are not redistributable here

`data/` holds only symlinks and the two GT JSONs. Ego-Exo4D requires accepting a licence from
[ego-exo4d-data.org](https://ego-exo4d-data.org), so neither the frames nor, most likely, the
ground-truth exports should be committed. `README.md` → "Datasets" already documents how to obtain
them; keep it that way and confirm the GT files' redistribution terms before including them in any
release archive.

---

## What is ready

The code, documentation and verification tooling are complete and self-consistent:

* `evaluation/unified_eval.py` — the Setting A scorer, validated against an input whose answer is
  known (GT fed back as its own prediction scores exactly 100.00 under skip, and exactly the
  covered fraction under full coverage).
* `tools/check_gt.py`, `tools/check_captions.py`, `tools/diff_checkpoints.py`,
  `tools/make_selftest_pred.py` — all four run and reproduce the figures quoted in the docs.
* `docs/METRIC_SETTINGS.md`, `docs/RESULTS.md`, `docs/REPRODUCE.md` — the two settings, Table 1's
  per-row provenance, and the reproduction path, including the two things this folder does **not**
  reproduce (the Ego→Exo direction, and 90 takes' `_exo` captions).

The honest limitations are stated in those files rather than omitted, which matters more for a
public release than a tidy results table: `REPRODUCE.md` §2 records that neither bundled checkpoint
produces the published Ego→Exo number, and `RESULTS.md` records that Table 1's O-MaMa row is on a
different metric setting from ours.
