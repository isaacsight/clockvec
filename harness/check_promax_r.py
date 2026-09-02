"""Check R's Promax implementations against the clockvec vector.

Optional adapter, kept out of the package and the test suite for the same
reason as check_mfcc_libraries: clockvec depends on numpy alone. Run it
when R with psych and GPArotation is on PATH.

    Rscript -e 'install.packages(c("psych","GPArotation"))'
    python harness/check_promax_r.py

Every implementation is driven through its PUBLIC API only. The three
checks are:

    stats::promax(L, m=4)                   -> expectation 0 (normalized)
    psych::Promax(L, m=4, normalize=FALSE)  -> expectation 1 (un-normalized)
    psych::Promax(L, m=4, normalize=TRUE)   -> expectation 0 (normalized)

On psych 2.6.5 the third check FAILS: the function returns the
un-normalized answer regardless of the argument, which is the
DeadNormalizeArgument flag. A passing third check is how a future psych
release proves the argument has been wired up.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from clockvec.promax import canonicalize
from clockvec.promax_vectors import build_vectors

R_SCRIPT = r"""
suppressMessages({library(psych); library(GPArotation)})
args <- commandArgs(trailingOnly = TRUE)
L <- as.matrix(read.csv(args[1], header = FALSE))
m <- as.numeric(args[2])
dump <- function(tag, z) {
  cat(tag, "\n")
  write.table(unclass(z), stdout(), sep = ",", row.names = FALSE, col.names = FALSE)
}
cat("VERSIONS psych", as.character(packageVersion("psych")),
    "GPArotation", as.character(packageVersion("GPArotation")), "\n")
dump("stats::promax", stats::promax(L, m = m)$loadings)
dump("psych::Promax normalize=FALSE", psych::Promax(L, m = m, normalize = FALSE)$loadings)
dump("psych::Promax normalize=TRUE", psych::Promax(L, m = m, normalize = TRUE)$loadings)
"""

# (label, expectation index, flag named on failure)
PLAN = [
    ("stats::promax", 0, None),
    ("psych::Promax normalize=FALSE", 1, None),
    ("psych::Promax normalize=TRUE", 0, "DeadNormalizeArgument"),
]


def run_r(L: np.ndarray, m: float) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "L.csv"
        np.savetxt(csv, L, delimiter=",")
        script = Path(td) / "run.R"
        script.write_text(R_SCRIPT)
        out = subprocess.run(
            ["Rscript", str(script), str(csv), str(m)],
            check=True, capture_output=True, text=True,
        ).stdout
    blocks: dict[str, list[list[float]]] = {}
    current = None
    for line in out.splitlines():
        if line.startswith("VERSIONS"):
            print(line)
            continue
        if "," not in line:
            current = line.strip()
            blocks[current] = []
        else:
            blocks[current].append([float(x) for x in line.split(",")])
    return {k: np.asarray(v) for k, v in blocks.items()}


def main() -> int:
    (v,) = build_vectors()
    d = v.declaration
    L = np.asarray(d.input_loadings).reshape(d.n_variables, d.n_factors)
    got = run_r(L, d.power_m)
    header = f"{'implementation':<32}{'expectation':<50}{'max|d|':>11}{'tol':>9}  verdict"
    print(header)
    print("-" * len(header))
    passes = fails = 0
    for label, idx, flag in PLAN:
        exp = v.expectations[idx]
        z = canonicalize(got[label]).ravel(order="C")
        diff = float(np.abs(z - np.asarray(exp.values)).max())
        ok = diff <= exp.tolerance
        passes += ok
        fails += not ok
        verdict = "PASS" if ok else f"FAIL  [{flag}]" if flag else "FAIL"
        print(f"{label:<32}{exp.implementation[:49]:<50}{diff:>11.3e}{exp.tolerance:>9.1e}  {verdict}")
    print("-" * len(header))
    print(f"{passes} passed, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
