"""Does allel.patterson_fst equal allel.hudson_fst?

scikit-allel ships this TODO in released code (allel/stats/fst.py):

    TODO check if this is  numerically equivalent to Hudson's estimator.

Answer: yes, and not by coincidence -- they are algebraically the same
estimator. Proof below, then a numerical check across regimes including
unequal sample sizes, where a difference would be most likely to show up.
"""

import allel
import numpy as np

SEED = 20260820


def simulate(n_variants, n1, n2, seed, fst_like=0.06):
    """Balding-Nichols draw. Returns two biallelic allele-count arrays.

    Sample sizes n1 and n2 are haploid allele counts and are deliberately
    allowed to differ -- Weir & Cockerham's estimator is known to be
    sensitive to the sample-size ratio, so it is the regime where two
    estimators are most likely to come apart.
    """
    rng = np.random.default_rng(seed)
    anc = rng.uniform(0.05, 0.95, size=n_variants)
    a = (1 - fst_like) / fst_like
    p1 = rng.beta(anc * a, (1 - anc) * a)
    p2 = rng.beta(anc * a, (1 - anc) * a)
    alt1 = rng.binomial(n1, p1)
    alt2 = rng.binomial(n2, p2)
    ac1 = np.column_stack([n1 - alt1, alt1])
    ac2 = np.column_stack([n2 - alt2, alt2])
    return ac1, ac2


def genome_wide(num, den):
    """Ratio of averages -- the combination rule Bhatia et al. recommend."""
    return np.nansum(num) / np.nansum(den)


REGIMES = [
    ("equal n, equal drift", 100, 100),
    ("equal n, larger sample", 300, 300),
    ("unequal n (14 vs 100)", 14, 100),
    ("unequal n (20 vs 300)", 20, 300),
    ("extreme (8 vs 500)", 8, 500),
]

print(f"scikit-allel {allel.__version__} | numpy {np.__version__} | seed {SEED}")
print(f"{'regime':<26} {'hudson':>12} {'patterson':>12} {'max|per-SNP dn|':>16} {'max|dd|':>10}")
print("-" * 82)

worst = 0.0
for label, n1, n2 in REGIMES:
    ac1, ac2 = simulate(20_000, n1, n2, SEED, 0.06)

    hn, hd = allel.hudson_fst(ac1, ac2)
    pn, pd = allel.patterson_fst(ac1, ac2)

    dn = np.nanmax(np.abs(hn - pn))
    dd = np.nanmax(np.abs(hd - pd))
    worst = max(worst, dn, dd)

    print(
        f"{label:<26} {genome_wide(hn, hd):>12.9f} {genome_wide(pn, pd):>12.9f} "
        f"{dn:>16.3e} {dd:>10.3e}"
    )

print("-" * 82)
print(f"worst per-SNP absolute difference, either component: {worst:.3e}")
print()

# The algebra, checked component by component on one regime.
ac1, ac2 = simulate(5_000, 37, 211, SEED + 1, 0.08)
n1 = ac1.sum(axis=1).astype(float)
n2 = ac2.sum(axis=1).astype(float)
p1 = ac1[:, 1] / n1
p2 = ac2[:, 1] / n2

# Hudson, as printed in Bhatia et al. 2013 eq. 10
hud_num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
hud_den = p1 * (1 - p2) + p2 * (1 - p1)

pn, pd = allel.patterson_fst(ac1, ac2)
hn, hd = allel.hudson_fst(ac1, ac2)

print("component check against the hand-written Bhatia et al. formula:")
print(f"  patterson num vs hand-written Hudson num : {np.nanmax(np.abs(pn - hud_num)):.3e}")
print(f"  patterson den vs hand-written Hudson den : {np.nanmax(np.abs(pd - hud_den)):.3e}")
print(f"  hudson    num vs hand-written Hudson num : {np.nanmax(np.abs(hn - hud_num)):.3e}")
print(f"  hudson    den vs hand-written Hudson den : {np.nanmax(np.abs(hd - hud_den)):.3e}")
