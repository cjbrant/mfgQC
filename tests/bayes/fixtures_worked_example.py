"""Fixture generator for spec test T2.5 (worked example, bayes-user-guide).

Regenerates every number printed in the user guide's worked example from the
normative algorithms in bayes-implementation-spec (engines A, C, D, G).
Run: python fixtures_worked_example.py  -> prints the fixture table and
asserts the pinned values, so drift in this script itself is detected.

Seeds (normative): data 20260703, Monte Carlo 7, reference 13, stream 17.
Sampling call order is normative: sigma2 via chisquare first, then mu.
"""
import numpy as np
from scipy import stats

LSL, USL, TARGET = 24.95, 25.05, 25.00
DRAWS, R, ALPHA = 100_000, 10_000, 0.005
DATA_SEED = 20260703


def worked_example_data() -> np.ndarray:
    """The normative T2.5 data: 12 subgroups of 5, raveled to 60 values."""
    return np.random.default_rng(DATA_SEED).normal(25.006, 0.011, size=(12, 5)).ravel()


def main() -> dict:
    fx = {}
    y = worked_example_data()
    n, ybar, s2 = y.size, y.mean(), y.var(ddof=1)
    s = float(np.sqrt(s2))
    fx["n"], fx["ybar"], fx["s"] = n, round(float(ybar), 4), round(s, 5)

    # (a) noninformative closed forms (BDA3 sec 3.2)
    nu = n - 1
    fx["mu95"] = [round(float(v), 4) for v in
                  stats.t.ppf([.025, .975], nu, loc=ybar, scale=np.sqrt(s2 / n))]
    sig2_dist = stats.invgamma(nu / 2, scale=nu * s2 / 2)
    fx["sd95"] = [round(float(np.sqrt(v)), 5) for v in sig2_dist.ppf([.025, .975])]
    fx["ppk_point"] = round(float(min(USL - ybar, ybar - LSL) / (3 * s)), 3)

    # (b) MC push-through (engine C), seed 7
    mc = np.random.default_rng(7)
    sig2 = nu * s2 / mc.chisquare(nu, DRAWS)
    mu = mc.normal(ybar, np.sqrt(sig2 / n))
    sd = np.sqrt(sig2)
    ppk = np.minimum(USL - mu, mu - LSL) / (3 * sd)
    p = float((ppk >= 1.33).mean())
    fx["p_ppk_133"] = round(p, 3)
    fx["p_mcse"] = round(float(np.sqrt(p * (1 - p) / DRAWS)), 3)
    fx["ppk_med_95"] = [round(float(v), 3) for v in
                        np.quantile(ppk, [.5, .025, .975])]
    ppm = 1e6 * (stats.norm.sf((USL - mu) / sd) + stats.norm.cdf((LSL - mu) / sd))
    fx["ppm_med_95"] = [round(float(v)) for v in np.quantile(ppm, [.5, .025, .975])]

    # (c) informative prior + compare (engines A, D)
    mu0, k0, s0, nu0 = 25.005, 20, 0.012, 20
    kn, nun = k0 + n, nu0 + n
    mun = (k0 * mu0 + n * ybar) / kn
    nunsn2 = nu0 * s0**2 + (n - 1) * s2 + k0 * n / kn * (ybar - mu0) ** 2
    fx["prior_weight"] = round(k0 / kn, 4)
    fx["mun"], fx["kn"], fx["nun"] = round(float(mun), 4), kn, nun
    fx["sn"] = round(float(np.sqrt(nunsn2 / nun)), 5)
    sig2i = nunsn2 / mc.chisquare(nun, DRAWS)
    mui = mc.normal(mun, np.sqrt(sig2i / kn))
    ppki = np.minimum(USL - mui, mui - LSL) / (3 * np.sqrt(sig2i))
    fx["p_ppk_133_inf"] = round(float((ppki >= 1.33).mean()), 3)

    yB = np.random.default_rng(11).normal(25.002, 0.009, 40)
    nB, ybB, s2B = yB.size, yB.mean(), yB.var(ddof=1)
    fx["ybarB"], fx["sB"] = round(float(ybB), 4), round(float(np.sqrt(s2B)), 5)
    sig2B = (nB - 1) * s2B / mc.chisquare(nB - 1, DRAWS)
    muB = mc.normal(ybB, np.sqrt(sig2B / nB))
    ppkB = np.minimum(USL - muB, muB - LSL) / (3 * np.sqrt(sig2B))
    fx["p_sdB_lt_sdA"] = round(float((np.sqrt(sig2B) < sd).mean()), 3)
    fx["p_ppkB_gt_ppkA"] = round(float((ppkB > ppk).mean()), 3)

    # (d) monitor (engine G): reference seed 13, stream seed 17
    ref = np.random.default_rng(13)
    m = 5
    sig2r = nu * s2 / ref.chisquare(nu, R)
    mur = ref.normal(ybar, np.sqrt(sig2r / n))
    reps = ref.normal(mur[:, None], np.sqrt(sig2r)[:, None], size=(R, m))
    Tm, Ts = reps.mean(1), reps.std(1, ddof=1)
    stream = np.random.default_rng(17)
    subs = [stream.normal(25.006, 0.011, m),
            stream.normal(25.006, 0.011, m),
            stream.normal(25.026, 0.011, m)]
    pv = []
    for sub in subs:
        row = []
        for Tobs, Trep in [(sub.mean(), Tm), (sub.std(ddof=1), Ts)]:
            phi = float((Trep >= Tobs).mean())
            row.append(round(min(2 * min(phi, 1 - phi), 1.0), 3))
        pv.append(row)
    fx["monitor_p"] = pv
    fx["flags"] = [any(x < ALPHA for x in row) for row in pv]
    return fx


PINNED = {
    "n": 60, "ybar": 25.0048, "s": 0.01291,
    "mu95": [25.0014, 25.0081], "sd95": [0.01095, 0.01575],
    "ppk_point": 1.168,
    "p_ppk_133": 0.074, "ppk_med_95": [1.161, 0.940, 1.393],
    "ppm_med_95": [268, 16, 2656],
    "prior_weight": 0.25, "mun": 25.0048, "kn": 80, "nun": 80, "sn": 0.01261,
    "p_ppk_133_inf": 0.085,
    "ybarB": 25.0013, "sB": 0.00771,
    "p_sdB_lt_sdA": 1.0, "p_ppkB_gt_ppkA": 1.0,
    "monitor_p": [[0.543, 0.779], [0.628, 0.059], [0.003, 0.106]],
    "flags": [False, False, True],
}


if __name__ == "__main__":
    fx = main()
    for k, v in fx.items():
        print(f"{k:>18}: {v}")
    for k, want in PINNED.items():
        got = fx[k]
        assert got == want, f"FIXTURE DRIFT {k}: got {got}, pinned {want}"
    print("\nall pinned values reproduced.")
