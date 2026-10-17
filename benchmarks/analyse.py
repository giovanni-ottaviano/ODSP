"""Compare the benchmark results with known values.

    python3 benchmarks/analyse.py [results-dir]

Prints every measured value next to the exact or literature value and a
PASS/FAIL verdict for each check, then a summary; exits with status 1 if any
check fails. Benchmarks whose results are missing are skipped. Python standard
library only.

Tolerances: exact results must agree within 3 standard errors; asymptotic
results also allow for the finite-size corrections expected at these sizes
(stated next to each check).
"""
import csv, glob, math, os, re, sys
from collections import defaultdict

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
results = []   # (section, description, ok)
section = ""


def read(name):
    with open(os.path.join(R, name)) as f:
        rows = [l for l in f if not l.startswith("#")]
    out = []
    for r in csv.DictReader(rows):
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def have(name):
    return os.path.exists(os.path.join(R, name))


def head(title):
    global section
    section = title.split(".")[0]
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def check(desc, ok):
    results.append((section, desc, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")


def check_near(desc, x, ref, err=0.0, sigmas=3.0, slack=0.0):
    """|x - ref| <= sigmas * err + slack."""
    zs = f", z = {(x - ref) / err:+.1f}" if err > 0 else ""
    check(f"{desc}: {x:.5g} vs {ref:.5g} (tol {sigmas * err + slack:.2g}{zs})", abs(x - ref) <= sigmas * err + slack)


def crossings(tables, xkey, ykey):
    """tables: {L: rows}; crossing of y_L(x) and y_2L(x) by linear interpolation."""
    Ls = sorted(tables)
    out = []
    for a, b in zip(Ls, Ls[1:]):
        xa = [r[xkey] for r in tables[a]]
        da = [r[ykey] - s[ykey] for r, s in zip(tables[a], tables[b])]
        for i in range(len(xa) - 1):
            if da[i] == 0 or da[i] * da[i + 1] < 0:
                out.append((a, b, xa[i] - da[i] * (xa[i + 1] - xa[i]) / (da[i + 1] - da[i])))
    return out


def dims_N(r):
    return math.prod(int(x) for x in str(r["dims"]).split("x"))


def mean_time(r):
    return r["mean_consensus_time"], r["sd_consensus_time"] / math.sqrt(r["replicas"])


# ------------------------------------------------------------------ performance
if have("timings.txt"):
    head("I. Single-thread speed (wall time includes setup; information only)")
    timings = {}
    for line in open(os.path.join(R, "timings.txt")):
        m = re.match(r"(\S+)\s+status (\d+)\s+([\d.]+) s", line)
        if m:
            timings[m.group(1)] = (int(m.group(2)), float(m.group(3)))   # the latest run of each job
    for name, (st, t) in timings.items():
        if name.startswith("perf_"):
            n, sweeps = (1e6, 20) if name.endswith("_1000") else (4096, 5000)
            print(f"  {name:28s} {n * sweeps / t / 1e6:7.1f} M updates/s   ({t:.2f} s)")
    failed = [k for k, (s, _) in timings.items() if s != 0]
    check(f"every job exited with status 0 (failed: {failed or 'none'})", not failed)


# --------------------------------------------------------- voter, complete graph
def exact_complete_voter(N):
    """Mean and sd of the consensus time (sweeps) from N/2 up: birth-death chain
    in updates, up and down each with p_n = n (N - n) / (N (N - 1))."""
    p = [k * (N - k) / (N * (N - 1)) for k in range(N + 1)]

    def solve(rhs):   # p_n (2 T(n) - T(n+1) - T(n-1)) = rhs(n), T(0) = T(N) = 0
        S = [0.0] * (N + 1)
        for n in range(1, N):
            S[n] = S[n - 1] + rhs[n] / p[n]
        S[N] = S[N - 1]
        D0 = sum(S[j] for j in range(N)) / N
        T = [0.0] * (N + 1)
        for j in range(N):
            T[j + 1] = T[j] + D0 - S[j]
        return T

    T1 = solve([1.0] * (N + 1))
    T2 = solve([2 * T1[n] - 1 for n in range(N + 1)])
    n0 = N // 2
    return T1[n0] / N, math.sqrt(T2[n0] - T1[n0] ** 2) / N


if have("voter_complete_sweep.csv"):
    head("A1. Voter, complete graph: consensus time vs the exact birth-death chain")
    for r in read("voter_complete_sweep.csv"):
        N = int(r["n"]); T, se = mean_time(r); ex, exsd = exact_complete_voter(N)
        check_near(f"N={N:4d} mean T (sweeps)", T, ex, se)
        check_near(f"N={N:4d} sd of T", r["sd_consensus_time"], exsd, 0.0, slack=0.05 * exsd)
        check_near(f"N={N:4d} P(+1)", r["exit_prob_up"], 0.5, math.sqrt(0.25 / r["replicas"]))


def voter_scaling(name, label, ref, scale_of, sname, slack, last_only=True):
    rows = read(name)
    for r in rows:
        N = dims_N(r) if "dims" in r else int(r["n"])
        T, se = mean_time(r); s = scale_of(N)
        print(f"    {label} N={N:6d}  T={T:10.1f} +- {se:7.1f}   T/{sname} = {T / s:.4f} +- {se / s:.4f}")
    for r in rows[-1:] if last_only else rows:
        N = dims_N(r) if "dims" in r else int(r["n"])
        T, se = mean_time(r); s = scale_of(N)
        check_near(f"{label} N={N} T/{sname}", T / s, ref, se / s, slack=slack * ref)


if have("voter_rr4_sweep.csv"):
    head("A2. Voter, random 4-regular graph: T/N -> ln2 (k-1)/(k-2) = 1.040 (Sood-Redner, pair approx.)")
    voter_scaling("voter_rr4_sweep.csv", "k=4", math.log(2) * 1.5, float, "N", slack=0.05)

if have("rfvoter_2d_sweep.csv"):
    head("A3. Voter, 2D torus: T/(N ln N) -> ln2/pi = 0.2206 (Cox 1989); log corrections, 15% allowed")
    NlnN = lambda N: N * math.log(N)
    voter_scaling("rfvoter_2d_sweep.csv", "rf-voter", math.log(2) / math.pi, NlnN, "NlnN", slack=0.15)
    if have("rfvoter_2d_L128_sweep.csv"):
        voter_scaling("rfvoter_2d_L128_sweep.csv", "rf-voter", math.log(2) / math.pi, NlnN, "NlnN", slack=0.15)
    if have("voter_2d_check_sweep.csv"):
        a = read("voter_2d_check_sweep.csv")[0]
        b = next(r for r in read("rfvoter_2d_sweep.csv") if dims_N(r) == 1024)
        (ta, ea), (tb, eb) = mean_time(a), mean_time(b)
        check_near("L=32 voter vs rf-voter mean T", ta, tb, math.hypot(ea, eb))

if have("rfvoter_3d_sweep.csv"):
    head("A4. Voter, 3D torus: T/N -> ln2/(1 - P_return) = 0.6931/0.65946 = 1.051 (Cox 1989); 10% allowed")
    voter_scaling("rfvoter_3d_sweep.csv", "rf-voter", math.log(2) / 0.659463, float, "N", slack=0.10)


# ------------------------------------------------------------ stationary scans
def scan(prefix, Ls):
    return {L: read(f"{prefix}{L}_sweep.csv") for L in Ls if have(f"{prefix}{L}_sweep.csv")}


def print_scan(tables, xkey, extra=()):
    for L, rows in sorted(tables.items()):
        print(f"    L={L}")
        for r in rows:
            s = (f"      {xkey}={r[xkey]:<9g} U={r['binder']:.4f}+-{r['binder_err']:.4f}  "
                 f"xi/L={r['xi_over_L']:.4f}+-{r['xi_over_L_err']:.4f}  |m|={r['abs_mean']:.4f}  tau={r['tau_abs']:.1f}")
            for k in extra:
                s += f"  {k}={r[k]:.5f}+-{r[k + '_err']:.5f}"
            print(s)


def check_crossings(tables, xkey, ref, tol):
    for key in ("binder", "xi_over_L"):
        for a, b, x in crossings(tables, xkey, key):
            check_near(f"crossing of {key} L={a}/{b}", x, ref, slack=tol)


def at(rows, key, x):
    return min(rows, key=lambda r: abs(r[key] - x))


TC = 2 / math.log(1 + math.sqrt(2))
ising = scan("ising_L", [16, 32, 64])
if ising:
    head(f"B. 2D Ising (Metropolis): T_c = {TC:.6f}, U* = 0.61069, xi/L* = 0.9050 (Salas-Sokal),"
         "\n   C(T_c) = 0.4945 ln L + const (Ferdinand-Fisher), e(T_c) -> -sqrt2 as 1/L")
    print_scan(ising, "temperature", extra=("energy", "specific_heat"))
    check_crossings(ising, "temperature", TC, 0.005)
    atc = {L: at(rows, "temperature", TC) for L, rows in ising.items()}
    for L in sorted(atc):   # long runs replace the scan values at T_c
        if have(f"ising_L{L}_Tc_sweep.csv"):
            atc[L] = read(f"ising_L{L}_Tc_sweep.csv")[0]
    if have("ising_L128_Tc_sweep.csv"):
        atc[128] = read("ising_L128_Tc_sweep.csv")[0]
    for L in sorted(atc):   # corrections ~ 1/L^2: 0.003 / 0.015 cover L = 16
        r = atc[L]
        check_near(f"L={L:3d} U(T_c)", r["binder"], 0.61069, r["binder_err"], slack=0.003)
        check_near(f"L={L:3d} xi/L(T_c)", r["xi_over_L"], 0.9050, r["xi_over_L_err"], slack=0.015)
    Ls = sorted(atc)
    for a, b in zip(Ls, Ls[1:]):
        if b == 2 * a:
            dC = atc[b]["specific_heat"] - atc[a]["specific_heat"]
            err = math.hypot(atc[a]["specific_heat_err"], atc[b]["specific_heat_err"])
            check_near(f"C(L={b}) - C(L={a})", dC, 0.4945 * math.log(2), err, slack=0.03)
    for a, b in zip(Ls, Ls[1:]):
        if b == 2 * a:
            ratio = (atc[b]["energy"] + math.sqrt(2)) / (atc[a]["energy"] + math.sqrt(2))
            check_near(f"(e+sqrt2)(L={b}) / (e+sqrt2)(L={a})", ratio, 0.5, slack=0.1)

TP = 1 / math.log(1 + math.sqrt(3))
potts = scan("potts3_L", [16, 32, 64])
if potts:
    head(f"C. 3-state Potts (heat bath): T_c = {TP:.6f}; e(T_c) -> -(1 + 1/sqrt3) as L^(-(1-alpha)/nu) = L^-0.8")
    print_scan(potts, "temperature", extra=("energy", "specific_heat"))
    check_crossings(potts, "temperature", TP, 0.003)
    de = {L: at(rows, "temperature", TP)["energy"] + 1 + 1 / math.sqrt(3) for L, rows in potts.items()}
    Ls = sorted(de)
    for a, b in zip(Ls, Ls[1:]):
        check_near(f"(e - e_c)(L={b}) / (e - e_c)(L={a})", de[b] / de[a], 2 ** -0.8, slack=0.06)

mv = scan("mv_L", [16, 32, 64])
if mv:
    head("D. Majority-vote model, 2D: q_c = 0.075 (Oliveira 1992), Ising class")
    print_scan(mv, "noise")
    check_crossings(mv, "noise", 0.075, 0.002)
    if 64 in mv:
        r = at(mv[64], "noise", 0.075)
        check_near("L=64 U(q_c) (Ising U* = 0.6107)", r["binder"], 0.61069, r["binder_err"], slack=0.01)


def mf_qvoter_m(q, p):
    f = lambda x: (1 - p) * ((1 - x) * x ** q - x * (1 - x) ** q) + p * (0.5 - x)
    lo, hi = 0.5 + 1e-9, 1.0
    if f(lo) <= 0:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(lo) * f(mid) <= 0 else (mid, hi)
    return 2 * lo - 1


for q, pc, deep in ((3, 1 / 3, 0.2), (4, 3 / 11, 0.15)):
    tabs = {N: read(f"qvoter{q}_N{N}_sweep.csv") for N in (500, 2000, 8000) if have(f"qvoter{q}_N{N}_sweep.csv")}
    if not tabs:
        continue
    head(f"E. q-voter with independence, q={q}, complete graph: p_c = (q-1)/(q-1+2^(q-1)) = {pc:.5f},"
         "\n   mean-field Binder U* = 0.2705")
    for N, rows in sorted(tabs.items()):
        print(f"    N={N}")
        for r in rows:
            print(f"      p={r['p']:<9g} |m|={r['abs_mean']:.4f}+-{r['abs_mean_err']:.4f} (MF {mf_qvoter_m(q, r['p']):.4f})"
                  f"  U={r['binder']:.4f}+-{r['binder_err']:.4f}  tau={r['tau_abs']:.1f}")
    near = {N: [r for r in rows if r["p"] > pc - 0.03] for N, rows in tabs.items()}
    for a, b, x in crossings(near, "p", "binder"):
        if a >= 2000:
            check_near(f"Binder crossing N={a}/{b}", x, pc, slack=0.003)
    if 8000 in tabs:
        r = at(tabs[8000], "p", pc)
        check_near("N=8000 U(p_c)", r["binder"], 0.2705, r["binder_err"], slack=0.01)
        r = at(tabs[8000], "p", deep)
        check_near(f"N=8000 |m| at p={deep} vs mean field", r["abs_mean"], mf_qvoter_m(q, deep), slack=0.003)


def cluster_stats(prefix, N=1000):
    out = []
    files = glob.glob(os.path.join(R, prefix + "_point*_clusters.csv"))
    for f in sorted(files, key=lambda s: int(re.search(r"point(\d+)", s).group(1))):
        by = defaultdict(list)
        for r in read(os.path.basename(f)):
            by[int(r["replica"])].append(r["size"])
        major = [sum(1 for s in v if s >= 0.1 * N) for v in by.values()]
        one = sum(1 for v in by.values() if max(v) == N)
        out.append((sum(major) / len(major), sum(len(v) for v in by.values()) / len(by), one / len(by)))
    return out


if have("deffuant_sweep.csv"):
    head("F. Deffuant (mu = 1/2), complete graph N=1000: floor(1/(2 eps)) major clusters (>= 10% of the nodes);"
         "\n   complete consensus, stragglers included, for eps > 1/2 (Fortunato 2004)")
    sweep = read("deffuant_sweep.csv")
    for r, (major, total, one) in zip(sweep, cluster_stats("deffuant")):
        eps = r["epsilon"]
        print(f"    eps={eps:<5g} major={major:5.2f}  all clusters={total:6.2f}  P(all in one)={one:.2f}")
        if 0.15 <= eps <= 0.3:
            check_near(f"eps={eps} major clusters", major, math.floor(1 / (2 * eps)), slack=0.3)
        if eps <= 0.4:
            check(f"eps={eps} P(complete consensus) = {one:.2f} < 0.5", one < 0.5)
        if eps >= 0.5:
            check(f"eps={eps} P(complete consensus) = {one:.2f} > 0.9", one > 0.9)

if have("hk_sweep.csv"):
    head("G. Hegselmann-Krause, complete graph N=1000: consensus for eps above ~0.2 (Fortunato 2005)")
    for r, (major, total, one) in zip(read("hk_sweep.csv"), cluster_stats("hk")):
        eps = r["epsilon"]
        print(f"    eps={eps:<5g} major={major:5.2f}  all clusters={total:6.2f}  P(consensus)={one:.2f}")
        if eps <= 0.15:
            check(f"eps={eps} P(consensus) = {one:.2f} < 0.1", one < 0.1)
        if eps >= 0.25:
            check(f"eps={eps} P(consensus) = {one:.2f} > 0.9", one > 0.9)


def columns(rows):
    return {k: [r[k] for r in rows] for k in rows[0]}


def slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pts); mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def window(ts, key, lo, hi):
    sel = [(t, v) for t, v in zip(ts["time"], ts[key]) if lo <= t <= hi]
    return [t for t, _ in sel], [v for _, v in sel]


def near_ratio(ts, key, tw, ratio):
    row = [(t, c) for t, c in zip(ts["time"], ts[key]) if not math.isnan(c) and t > 0]
    return min(row, key=lambda tc: abs(math.log(tc[0] / (ratio * tw))))


if have("coarsen_ising_timeseries.csv"):
    head("H1. Ising quench to T = 1.5, 256x256: L(t) ~ t^(1/2), excess broken bonds ~ t^(-1/2) (Allen-Cahn);"
         "\n   ageing C(t, t_w) = f(t/t_w)")
    ts = columns(read("coarsen_ising_timeseries.csv"))
    K = 1 / 1.5; k = 2 * math.sinh(2 * K) / math.cosh(2 * K) ** 2
    a, b = 1.0, math.sqrt(1 - k * k)
    for _ in range(40):
        a, b = (a + b) / 2, math.sqrt(a * b)
    u = -1 / math.tanh(2 * K) * (1 + 2 / math.pi * (2 * math.tanh(2 * K) ** 2 - 1) * math.pi / (2 * a))
    rho_eq = (1 + u / 2) / 2   # Onsager: equilibrium density of broken bonds at T = 1.5
    ts["excess_rho"] = [r - rho_eq for r in ts["rho_mean"]]
    sL = slope(*window(ts, "length_mean", 20, 2000))
    sR = slope(*window(ts, "excess_rho", 20, 2000))
    check_near("d ln L / d ln t, t = 20..2000 (effective, approaches 1/2 from below)", sL, 0.5, slack=0.08)
    check_near(f"d ln(rho - rho_eq) / d ln t (rho_eq = {rho_eq:.5f})", sR, -0.5, slack=0.08)
    prod = [L * e for t, L, e in zip(ts["time"], ts["length_mean"], ts["excess_rho"]) if 20 <= t <= 2000]
    check(f"L (rho - rho_eq) constant within 15%: {min(prod):.3f}..{max(prod):.3f}", max(prod) / min(prod) < 1.15)
    for ratio in (2, 5, 10, 20):
        cs = [near_ratio(ts, f"autocorrelation_tw{tw}_mean", tw, ratio) for tw in (10, 30, 100)]
        print(f"    C at t/t_w ~ {ratio:2d}: " + "  ".join(f"t_w={tw}: {c:.3f} (t/t_w={t / tw:.2f})"
                                                     for tw, (t, c) in zip((10, 30, 100), cs)))
    t1, c1 = near_ratio(ts, "autocorrelation_tw10_mean", 10, 10)
    t2, c2 = near_ratio(ts, "autocorrelation_tw100_mean", 100, 10)
    check_near("collapse: C(10 t_w, t_w) for t_w = 100 vs t_w = 10", c2, c1, slack=0.05)

if have("coarsen_voter_timeseries.csv"):
    head("H2. Voter, 256x256: rho(t) ~ pi / (2 ln t + ln 256) (Frachebourg-Krapivsky 1996)")
    ts = columns(read("coarsen_voter_timeseries.csv"))
    for t, rho, L in zip(ts["time"], ts["rho_mean"], ts["length_mean"]):
        if t in (10, 100, 1000, 10000):
            print(f"    t={t:7g}  rho={rho:.4f}  prediction={math.pi / (2 * math.log(t) + math.log(256)):.4f}  L_1/2={L:.2f}")
    t, rho = [(t, r) for t, r in zip(ts["time"], ts["rho_mean"]) if t <= 10000][-1]
    check_near(f"rho(t = {t:g})", rho, math.pi / (2 * math.log(t) + math.log(256)),
               slack=0.1 * math.pi / (2 * math.log(t) + math.log(256)))
    print(f"    apparent exponent of L_1/2, t = 100..10000: {slope(*window(ts, 'length_mean', 100, 10000)):.3f}"
          "  (G(r,t) ~ 1 - ln r^2 / ln t suggests t^(1/4) asymptotically)")


# --------------------------------------------------------------------- summary
print("\n" + "=" * 78)
fails = [(s, d) for s, d, ok in results if not ok]
print(f"{len(results) - len(fails)} of {len(results)} checks passed")
for s, d in fails:
    print(f"  FAIL [{s}] {d}")
sys.exit(1 if fails else 0)
