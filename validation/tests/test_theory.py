import ast
import math
from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from validation import theory
from validation.theory import (CITATIONS, ISING_TC, POTTS3_TC, exit_probability, ising_energy,
                               mean_absorption_time, noisy_voter_rates, nonlinear_voter_rates,
                               onsager_magnetization, qvoter_independence_rates, qvoter_mean_field_m,
                               qvoter_pc, stationary_distribution, voter_rates)

ROOT = Path(__file__).resolve().parents[2]


def _benchmark_function(name: str):
    """`name` from benchmarks/analyse.py, compiled alone: the script has no __main__ guard, so
    importing it would run the whole benchmark analysis."""
    path = ROOT / "benchmarks" / "analyse.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, f"{name} not found in {path}"
    namespace: dict = {"math": math, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_voter_exit_is_initial_share():
    P = exit_probability(*voter_rates(50))
    assert len(P) == 51
    for n in range(51):
        assert P[n] == approx(n / 50, abs=1e-12)


def test_voter_time_matches_benchmark_reference():
    exact_complete_voter = _benchmark_function("exact_complete_voter")
    for N in (20, 100):
        T = mean_absorption_time(*voter_rates(N))
        assert T[0] == 0 and T[N] == 0
        # the reference is the mean in sweeps from N // 2
        assert T[N // 2] / N == approx(exact_complete_voter(N)[0], rel=1e-10)


def test_nonlinear_alpha_one_is_voter():
    for N in (2, 10, 101):
        up, down = nonlinear_voter_rates(N, 1.0)
        vup, vdown = voter_rates(N)
        assert up == approx(vup, rel=1e-14, abs=0)
        assert down == approx(vdown, rel=1e-14, abs=0)


def test_qvoter_q1_p0_is_voter():
    for N in (2, 10, 101):
        up, down = qvoter_independence_rates(N, 1, 0.0)
        vup, vdown = voter_rates(N)
        assert up == approx(vup, rel=1e-14, abs=0)
        assert down == approx(vdown, rel=1e-14, abs=0)


def test_rates_on_small_complete_graph_by_enumeration():
    # N = 3, n = 1 up node: a down node (2 of 3) sees 1 up among its 2 neighbours.
    up, down = qvoter_independence_rates(3, 2, 0.2)
    assert up[1] == approx(2 / 3 * (0.8 * (1 / 2) ** 2 + 0.1))
    assert down[1] == approx(1 / 3 * (0.8 * 1.0 + 0.1))
    up, down = noisy_voter_rates(3, 0.2)
    assert up[1] == approx(2 / 3 * (0.8 * 1 / 2 + 0.1))
    assert down[1] == approx(1 / 3 * (0.8 * 1.0 + 0.1))
    assert up[3] == 0 and down[0] == 0
    up, down = nonlinear_voter_rates(3, 2.0)
    assert up[1] == approx(2 / 3 * (1 / 2) ** 2)
    assert down[1] == approx(1 / 3 * 1.0)


def test_noisy_stationary():
    N = 100
    for noise in (0.1, 0.001):
        pi = stationary_distribution(*noisy_voter_rates(N, noise))
        assert len(pi) == N + 1
        assert pi.sum() == approx(1, abs=1e-12)
        assert pi == approx(pi[::-1], rel=1e-10)
    d = np.diff(stationary_distribution(*noisy_voter_rates(N, 0.1)))      # noise N = 10: unimodal
    assert np.all(d[:N // 2] > 0) and np.all(d[N // 2:] < 0)
    d = np.diff(stationary_distribution(*noisy_voter_rates(N, 0.001)))    # noise N = 0.1: bimodal
    assert np.all(d[:N // 2] < 0) and np.all(d[N // 2:] > 0)


def test_stationary_large_N_is_finite():
    pi = stationary_distribution(*noisy_voter_rates(10000, 0.001))
    assert np.all(np.isfinite(pi)) and np.all(pi >= 0)
    assert pi.sum() == approx(1, abs=1e-12)


def test_chain_solvers_against_direct_linear_algebra():
    rng = np.random.default_rng(1)
    N = 12
    up = rng.uniform(0.05, 0.45, N + 1)
    down = rng.uniform(0.05, 0.45, N + 1)
    up[N] = 0.0
    down[0] = 0.0
    # absorbing chain: interior generator Q (sub-stochastic), (I - Q) T = 1, (I - Q) P = b
    n = np.arange(1, N)
    A = np.diag(up[n] + down[n]) - np.diag(up[n][:-1], 1) - np.diag(down[n][1:], -1)
    T = np.linalg.solve(A, np.ones(N - 1))
    b = np.zeros(N - 1)
    b[-1] = up[N - 1]
    P = np.linalg.solve(A, b)
    assert mean_absorption_time(up, down)[1:N] == approx(T, rel=1e-12)
    assert exit_probability(up, down)[1:N] == approx(P, rel=1e-12)
    # ergodic chain: pi is the null vector of the full generator
    up0 = up.copy()
    down0 = down.copy()
    up0[0] = 0.3
    down0[N] = 0.2
    G = np.diag(-(up0 + down0)) + np.diag(up0[:-1], 1) + np.diag(down0[1:], -1)
    w, v = np.linalg.eig(G.T)
    ref = np.real(v[:, np.argmin(np.abs(w))])
    ref /= ref.sum()
    assert stationary_distribution(up0, down0) == approx(ref, rel=1e-9)


def _qvoter_dxdt(x: float, q: int, p: float) -> float:
    # include/QVoterModel.hpp, independence
    return (1 - p) * ((1 - x) * x ** q - x * (1 - x) ** q) + p * (0.5 - x)


def test_qvoter_mean_field():
    assert qvoter_pc(2) == approx(1 / 3)
    assert qvoter_pc(4) == approx(3 / 11)
    assert qvoter_mean_field_m(3, qvoter_pc(3) + 0.01) == 0
    for q in (0, 1):                     # the voter case has no unique stationary m
        with pytest.raises(ValueError):
            qvoter_mean_field_m(q, 0.1)
    with pytest.raises(ValueError):
        qvoter_mean_field_m(1, 0.0)
    for q in (2, 3, 4, 5, 6, 8):
        for frac in (0.1, 0.5, 0.9, 0.999):
            p = frac * qvoter_pc(q)
            m = qvoter_mean_field_m(q, p)
            assert 0 < m <= 1
            assert abs(_qvoter_dxdt((1 + m) / 2, q, p)) < 1e-12
    # large-N limit of the finite-N chain: the stationary peak sits at the mean-field m
    N = 4000
    pi = stationary_distribution(*qvoter_independence_rates(N, 3, 0.1))
    assert abs(2 * np.argmax(pi) / N - 1) == approx(qvoter_mean_field_m(3, 0.1), abs=0.01)


def test_ising_exact():
    assert ISING_TC == approx(2 / math.log(1 + math.sqrt(2)))
    assert ising_energy(ISING_TC) == approx(-math.sqrt(2), abs=1e-9)
    assert ising_energy(0.05) == approx(-2, abs=1e-6)
    assert onsager_magnetization(0.5) == approx(1, abs=1e-6)
    assert onsager_magnetization(ISING_TC * 1.001) == 0
    for T in (ISING_TC * (1 - 1e-6), ISING_TC * (1 + 1e-6)):    # continuous through Tc
        assert ising_energy(T) == approx(-math.sqrt(2), abs=1e-4)
    assert ising_energy(1e3) == approx(-2 / 1e3, rel=1e-2)       # high T: u ~ -2 tanh(1/T)
    assert theory.ISING_EXPONENTS == {"nu": 1.0, "gamma/nu": 7 / 4, "beta/nu": 1 / 8}
    assert theory.ISING_BINDER[0] == approx(1 - 1.1679229 / 3, abs=1e-7)     # Salas & Sokal 2000
    assert theory.ISING_BINDER[1] == approx(4.7e-6 / 3, rel=0.05)


def test_ising_energy_matches_onsager_free_energy():
    # u = d(beta f)/d(beta) of Onsager's  -beta f = ln 2 + (1/8 pi^2) int int ln[cosh^2 2b -
    # sinh 2b (cos t1 + cos t2)]: an independent route to the closed form. The integrand is
    # periodic and analytic away from Tc, so a uniform grid converges exponentially.
    t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    c = np.cos(t)[:, None] + np.cos(t)[None, :]
    for T in (1.0, 1.8, 2.0, 2.6, 3.5, 10.0):
        b2 = 2 / T
        num = 4 * np.cosh(b2) * np.sinh(b2) - 2 * np.cosh(b2) * c
        den = np.cosh(b2) ** 2 - np.sinh(b2) * c
        u = -0.5 * np.mean(num / den)
        assert ising_energy(T) == approx(u, abs=1e-10)


def test_potts_constants():
    assert POTTS3_TC == approx(1 / math.log(1 + math.sqrt(3)))
    assert theory.POTTS3_EXPONENTS == approx({"nu": 5 / 6, "gamma/nu": 26 / 15, "beta/nu": 2 / 15})


def _referenced_citation_keys(path: Path) -> set[str]:
    """String keys a model module cites: `source="key"` keywords and `CITATIONS["key"]`."""
    keys: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.keyword) and node.arg == "source" and \
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            keys.add(node.value.value)
        if isinstance(node, ast.Subscript) and isinstance(node.value, (ast.Name, ast.Attribute)) and \
                getattr(node.value, "id", getattr(node.value, "attr", None)) == "CITATIONS" and \
                isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
    return keys


def _rho_2d_volterra(t_max: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """rho(t) = j(t)/4 from the Volterra equation j(t) = -P0'(t) - int_0^t j(u) P0'(t-u) du,
    P0 = (e^-t I0(t))^2, by the trapezoid rule: an independent route to voter_rho_2d."""
    from scipy.special import i0e, i1e
    t = np.arange(0.0, t_max + dt / 2, dt)
    K = 2 * i0e(t) * (i1e(t) - i0e(t))            # P0'(t)
    j = np.empty_like(t)
    j[0] = -K[0]
    for n in range(1, t.size):
        conv = 0.5 * j[0] * K[n] + np.dot(j[1:n], K[n - 1:0:-1])
        j[n] = (-K[n] - dt * conv) / (1 + 0.5 * dt * K[0])
    return t, j / 4


def test_voter_rho_2d():
    rho = theory.voter_rho_2d
    # independent solution of the same equations (trapezoid rule, Richardson-extrapolated)
    _, coarse = _rho_2d_volterra(5.0, 2e-3)
    _, fine = _rho_2d_volterra(5.0, 1e-3)
    for tt in (0.1, 0.5, 1.0, 2.0, 5.0):
        ref = (4 * fine[int(round(tt / 1e-3))] - coarse[int(round(tt / 2e-3))]) / 3
        assert rho(tt) == approx(ref, rel=1e-5)
    # small t: rho = 1/2 - t/4 + O(t^2) (from G(e) at t = 0)
    assert (rho(0.1) - 0.5) / 0.1 == approx(-0.25, abs=0.03)
    # vectorised; the inversion is stable against its number of terms
    ts = np.array([1.0, 10.0, 1e3, 1e6])
    assert rho(ts) == approx([rho(x) for x in ts], rel=1e-12)
    assert rho(ts, terms=16) == approx(rho(ts), rel=1e-4)
    # decreasing, and approaching the asymptote pi / (2 ln t + ln 256) (Frachebourg-Krapivsky)
    big = np.geomspace(10, 1e12, 12)
    vals = rho(big)
    assert np.all(np.diff(vals) < 0)
    fk = np.pi / (2 * np.log(big) + np.log(256))
    assert np.all(np.diff(np.abs(vals / fk - 1)) < 0) and abs(vals[-1] / fk[-1] - 1) < 0.03
    with pytest.raises(ValueError):
        rho(0.0)


def test_torus_eigentime():
    # 1D ring: sum_{k=1}^{L-1} 1 / (1 - cos(2 pi k / L)) = (L^2 - 1) / 6
    for L in (2, 3, 10, 101):
        assert theory.torus_eigentime(L, 1) == approx((L * L - 1) / 6, rel=1e-12)
    # 2x2 torus in 2D by hand: modes (0,1), (1,0) have lambda = 1, mode (1,1) has lambda = 2
    assert theory.torus_eigentime(2, 2) == approx(1 + 1 + 0.5)
    # 3D: M / N -> Watson's integral 1.516386 with a 1/L correction; 2D: M / N - ln(N) / pi -> const
    w = [theory.torus_eigentime(L, 3) / L ** 3 for L in (16, 32, 64)]
    assert w[0] < w[1] < w[2] < 1.516386
    assert (1.516386 - w[2]) / (1.516386 - w[1]) == approx(0.5, abs=0.03)
    c = [theory.torus_eigentime(L, 2) / L ** 2 - math.log(L * L) / math.pi for L in (64, 128, 256)]
    assert abs(c[2] - c[1]) < 0.25 * abs(c[1] - c[0]) + 1e-4
    with pytest.raises(ValueError):
        theory.torus_eigentime(1, 2)


def test_every_citation_key_is_used_or_documented():
    assert CITATIONS
    assert all(isinstance(k, str) and isinstance(v, str) and v for k, v in CITATIONS.items())
    for path in sorted((ROOT / "validation" / "models").glob("*.py")):
        missing = _referenced_citation_keys(path) - CITATIONS.keys()
        assert not missing, f"{path.name} cites unknown keys {sorted(missing)}"
