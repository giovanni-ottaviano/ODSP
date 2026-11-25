"""Exact and mean-field predictions the model checks compare with, each with its source.

Birth-death chains: the binary models on the complete graph of N nodes are chains on n, the
number of +1 nodes. `up[n]` / `down[n]` are the probabilities that ONE single update (the
driver's unit: N updates make a sweep) takes n to n + 1 / n - 1. The rates below are derived
from each model's `_single_update` in include/*.hpp exactly as implemented: a node is chosen
uniformly (probability 1/N each) and "a neighbour" is any of the other N - 1 nodes.

Ising and Potts: H = -J sum_<ij> s_i s_j and H = -J sum_<ij> delta(s_i, s_j) with J = 1 and
k_B = 1 on the square lattice, as in include/IsingModel.hpp and include/PottsModel.hpp.

Every prediction cites a key of `CITATIONS`; a Check's `source` is one of these keys.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, ellipkm1, logsumexp

CITATIONS: dict[str, str] = {
    "birth_death": (
        "S. Karlin and H. M. Taylor, A First Course in Stochastic Processes, 2nd ed. "
        "(Academic Press, New York, 1975) (birth-death chains: absorption probabilities "
        "and mean absorption times by first-step analysis; stationary law by detailed balance)"),
    "voter_complete": (
        "T. M. Liggett, Interacting Particle Systems (Springer, New York, 1985), ch. V; "
        "S. Redner, A Guide to First-Passage Processes (Cambridge University Press, Cambridge, "
        "2001): on the complete graph the voter model is a martingale birth-death chain, exit "
        "probability n/N, mean consensus time from the first-step recursion"),
    "voter_exit": (
        "T. M. Liggett, Interacting Particle Systems (Springer, New York, 1985), ch. V; "
        "P. L. Krapivsky, S. Redner and E. Ben-Naim, A Kinetic View of Statistical Physics "
        "(Cambridge University Press, Cambridge, 2010), ch. 8: with node update on a graph where "
        "every node has the same degree (lattices) the magnetization is a martingale, so "
        "P(+1 consensus) = initial fraction of +1 nodes"),
    "voter_lattice": (
        "J. T. Cox, \"Coalescing random walks and voter model consensus times on the torus in "
        "Z^d\", Ann. Probab. 17, 1333 (1989) (d >= 2: T ~ N ln N in d = 2, T ~ N in d >= 3; the "
        "consensus time is asymptotically proportional to the meeting time of two random walks); "
        "D. Aldous and J. A. Fill, Reversible Markov Chains and Random Walks on Graphs (2002, "
        "online monograph), ch. 3 (eigentime identity: mean hitting time from stationarity = "
        "sum over non-zero modes of 1/lambda_k, the exact finite-torus meeting time); "
        "P. L. Krapivsky, S. Redner and E. Ben-Naim, A Kinetic View of Statistical Physics "
        "(Cambridge University Press, Cambridge, 2010), ch. 8 (mean consensus time T ~ N^2 in "
        "d = 1, N ln N in d = 2, N in d > 2)"),
    "voter_heterogeneous": (
        "V. Sood and S. Redner, \"Voter model on heterogeneous graphs\", Phys. Rev. Lett. 94, "
        "178701 (2005); V. Sood, T. Antal and S. Redner, \"Voter models on heterogeneous "
        "networks\", Phys. Rev. E 77, 041121 (2008): node-update voter model (a random node "
        "copies a random neighbour), T ~ N mu1^2 / mu2; degree exponent 3 (Barabasi-Albert) "
        "has mu2 ~ ln N, so T ~ N / ln N"),
    "voter_interface_2d": (
        "L. Frachebourg and P. L. Krapivsky, \"Exact results for kinetics of catalytic "
        "reactions\", Phys. Rev. E 53, R3009 (1996), Eq. (23): density of disagreeing "
        "nearest-neighbour pairs on the square lattice rho(t) = pi / (2 ln t + ln 256) + ..., "
        "i.e. 1/rho = (2/pi) ln t + const (their time unit, tau = 4/D, gives each site the flip "
        "rate (fraction of disagreeing neighbours): one sweep of node updates); the exact rho(t) "
        "of voter_rho_2d solves the same closed pair-correlation equations"),
    "rejection_free": (
        "A. B. Bortz, M. H. Kalos and J. L. Lebowitz, \"A new algorithm for Monte Carlo "
        "simulation of Ising spin systems\", J. Comput. Phys. 17, 10 (1975) (n-fold way: "
        "skipping rejected attempts with geometric waiting times samples the same Markov chain, "
        "so every observable has the same distribution)"),
    "noisy_voter": (
        "A. Kirman, \"Ants, rationality, and recruitment\", Q. J. Econ. 108, 137 (1993); "
        "B. L. Granovsky and N. Madras, \"The noisy voter model\", Stoch. Proc. Appl. 55, 23 "
        "(1995): on the complete graph the number of +1 nodes is a birth-death chain whose "
        "stationary law follows from detailed balance (birth_death); bimodal for small noise N, "
        "unimodal for large"),
    "nonlinear_voter": (
        "F. Schweitzer and L. Behera, \"Nonlinear voter models: the transition from invasion to "
        "coexistence\", Eur. Phys. J. B 67, 301 (2009); C. Castellano, M. A. Munoz and "
        "R. Pastor-Satorras, \"Nonlinear q-voter model\", Phys. Rev. E 80, 041129 (2009) (flip "
        "with probability f^q, f the disagreeing fraction, q real: the nonlinear voter with "
        "alpha = q; mean field: coexistence for q < 1, ordering for q > 1); on the complete "
        "graph a birth-death chain, exit probability and mean consensus time by first-step "
        "analysis (birth_death)"),
    "qvoter": (
        "C. Castellano, M. A. Munoz and R. Pastor-Satorras, \"Nonlinear q-voter model\", "
        "Phys. Rev. E 80, 041129 (2009)"),
    "qvoter_independence": (
        "P. Nyczka, K. Sznajd-Weron and J. Cislo, \"Phase transitions in the q-voter model with "
        "two types of stochastic driving\", Phys. Rev. E 86, 011105 (2012) (mean-field "
        "p_c = (q-1)/(q-1+2^(q-1)), continuous for q <= 5); on the complete graph the number "
        "of +1 nodes is a birth-death chain, whose stationary law (detailed balance, "
        "birth_death) is exact at finite N"),
    "ising_tc": (
        "H. A. Kramers and G. H. Wannier, \"Statistics of the two-dimensional ferromagnet. "
        "Part I\", Phys. Rev. 60, 252 (1941); L. Onsager, \"Crystal statistics. I. A "
        "two-dimensional model with an order-disorder transition\", Phys. Rev. 65, 117 (1944)"),
    "ising_magnetization": (
        "L. Onsager, \"Crystal statistics. I. A two-dimensional model with an order-disorder "
        "transition\", Phys. Rev. 65, 117 (1944) (announced the result); C. N. Yang, \"The "
        "spontaneous magnetization of a two-dimensional Ising model\", Phys. Rev. 85, 808 (1952) "
        "(derivation)"),
    "ising_energy": (
        "L. Onsager, \"Crystal statistics. I. A two-dimensional model with an order-disorder "
        "transition\", Phys. Rev. 65, 117 (1944); R. J. Baxter, Exactly Solved Models in "
        "Statistical Mechanics (Academic Press, London, 1982), ch. 7"),
    "ising_exponents": (
        "L. Onsager, Phys. Rev. 65, 117 (1944) (nu = 1); C. N. Yang, Phys. Rev. 85, 808 (1952) "
        "(beta = 1/8); T. T. Wu, B. M. McCoy, C. A. Tracy and E. Barouch, \"Spin-spin "
        "correlation functions for the two-dimensional Ising model: exact theory in the scaling "
        "region\", Phys. Rev. B 13, 316 (1976) (gamma = 7/4); R. J. Baxter, Exactly Solved "
        "Models in Statistical Mechanics (Academic Press, London, 1982), ch. 7 (textbook "
        "account)"),
    "ising_binder": (
        "J. Salas and A. D. Sokal, \"Universal amplitude ratios in the critical two-dimensional "
        "Ising model on a torus\", J. Stat. Phys. 98, 551 (2000): <M^4>/<M^2>^2 = 1.1679229(47) "
        "at Tc on the periodic square lattice, L -> infinity (value and uncertainty used); "
        "secondary: G. Kamieniarz and H. W. J. Blote, \"Universal ratio of magnetization "
        "moments in two-dimensional Ising models\", J. Phys. A 26, 201 (1993), who quote the "
        "inverse ratio <M^2>^2/<M^4> = 0.856216(1)"),
    "potts_tc": (
        "R. B. Potts, \"Some generalized order-disorder transformations\", Proc. Camb. Phil. Soc. "
        "48, 106 (1952); R. J. Baxter, \"Potts model at the critical temperature\", J. Phys. C "
        "6, L445 (1973); F. Y. Wu, \"The Potts model\", Rev. Mod. Phys. 54, 235 (1982)"),
    "potts_exponents": (
        "M. P. M. den Nijs, \"A relation between the temperature exponents of the eight-vertex "
        "and q-state Potts model\", J. Phys. A 12, 1857 (1979) (nu = 5/6 for q = 3); "
        "B. Nienhuis, E. K. Riedel and M. Schick, \"Magnetic exponents of the two-dimensional "
        "q-state Potts model\", J. Phys. A 13, L189 (1980) (magnetic exponent: beta/nu = 2/15, "
        "gamma/nu = 2 - 2 beta/nu = 26/15 for q = 3); F. Y. Wu, \"The Potts model\", "
        "Rev. Mod. Phys. 54, 235 (1982) (q = 3: alpha = 1/3, beta = 1/9, gamma = 13/9, nu = 5/6)"),
    "mv_qc": (
        "U. Yu, \"Phase transition in the majority-vote model on the Archimedean lattices\", "
        "Phys. Rev. E 95, 012101 (2017), Table I and text: q_c = 0.07518(3) on the periodic "
        "square lattice from the Binder crossings of L <= 260 (value and uncertainty used), for "
        "the flip probability w = (1/2)[1 - (1 - 2q) s_i sgn(sum of the 4 nearest neighbours)] "
        "(sgn(0) = 0: a tie gives either opinion with probability 1/2), each site updated once "
        "per Monte Carlo step on average; earlier: M. J. de Oliveira, \"Isotropic majority-vote "
        "model on a square lattice\", J. Stat. Phys. 66, 273 (1992) (the model; q_c = "
        "0.075(1)); W. Kwak, J.-S. Yang, J.-I. Sohn and I.-M. Kim, Phys. Rev. E 75, 061110 "
        "(2007) (q_c = 0.075(1))"),
    "mv_exponents": (
        "the 2D Ising exponents (ising_exponents: nu = 1, gamma/nu = 7/4, beta/nu = 1/8) by "
        "universality: G. Grinstein, C. Jayaprakash and Y. He, \"Statistical mechanics of "
        "probabilistic cellular automata\", Phys. Rev. Lett. 55, 2527 (1985) (non-equilibrium "
        "up-down symmetric spin models are in the equilibrium Ising class); M. J. de Oliveira, "
        "J. Stat. Phys. 66, 273 (1992) (the square-lattice majority-vote model is in the 2D "
        "Ising class); U. Yu, Phys. Rev. E 95, 012101 (2017) (1/nu = 0.99(3), "
        "gamma/nu = 1.75(1), beta/nu = 0.123(3) at q_c = 0.07518(3))"),
    "mv_binder": (
        "U* of the 2D Ising class on the periodic square L x L lattice, J. Salas and A. D. Sokal, "
        "J. Stat. Phys. 98, 551 (2000) (ising_binder: value and uncertainty used); the "
        "cumulant at criticality is universal within a class for a given geometry and boundary "
        "conditions: V. Privman and M. E. Fisher, \"Universal critical amplitudes in "
        "finite-size scaling\", Phys. Rev. B 30, 322 (1984); K. Binder, \"Finite size scaling "
        "analysis of Ising model block distribution functions\", Z. Phys. B 43, 119 (1981); the "
        "majority-vote model is in the 2D Ising class: M. J. de Oliveira, J. Stat. Phys. 66, "
        "273 (1992); U. Yu, Phys. Rev. E 95, 012101 (2017)"),
    "majority_stripes": (
        "K. Barros, P. L. Krapivsky and S. Redner, \"Freezing into stripe states in "
        "two-dimensional ferromagnets and crossing probabilities in critical percolation\", "
        "Phys. Rev. E 80, 040101(R) (2009): after a quench to T = 0 the periodic square L x L "
        "system freezes into horizontal or vertical stripes with the probability P_01 + P_10 = "
        "0.3388... that critical continuum percolation on the square torus has a cluster winding "
        "(0, 1) or (1, 0); diagonal (1, +-1) windings reach the ground state on a much longer "
        "time scale; H. T. Pinson, \"Critical percolation on the torus\", J. Stat. Phys. 75, 1167 "
        "(1994) (the crossing probabilities, Eqs. (6)-(7) of BKR, evaluated in "
        "theory.torus_wrapping_probability); J. Olejarz, P. L. Krapivsky and S. Redner, \"Fate "
        "of 2D kinetic ferromagnets and critical percolation crossing probabilities\", Phys. Rev. "
        "Lett. 109, 195702 (2012) (Glauber dynamics with zero-field spins flipped with "
        "probability 1/2; Table I: stripes 0.3388, diagonal stripes 0.04196; diagonal stripes "
        "decay to the ground state on a time scale ~ L^3); V. Spirin, P. L. Krapivsky and "
        "S. Redner, Phys. Rev. E 63, 036118 (2001) and Phys. Rev. E 65, 016119 (2002) (the "
        "dynamics: zero-field spins flip with probability 1/2, one flip attempt per spin per unit "
        "time; stripe probability ~ 1/3). The start is BKR's quench from T = infinity: "
        "independent +-1 spins with P(+1) = 1/2"),
    "majority_coarsening": (
        "I. M. Lifshitz, Sov. Phys. JETP 15, 939 (1962); S. M. Allen and J. W. Cahn, \"A "
        "microscopic theory for antiphase boundary motion and its application to antiphase "
        "domain coarsening\", Acta Metall. 27, 1085 (1979); A. J. Bray, \"Theory of phase-"
        "ordering kinetics\", Adv. Phys. 43, 357 (1994): curvature-driven coarsening with a "
        "non-conserved order parameter, characteristic length l(t) ~ t^(1/2); the excess energy "
        "per bond (at T = 0 the density rho of unsatisfied bonds) scales as 1/l, so l = 1/rho; "
        "D. A. Huse, \"Corrections to late-stage behavior in spinodal decomposition: Lifshitz-"
        "Slyozov scaling and Monte Carlo simulations\", Phys. Rev. B 34, 7845 (1986) (the leading "
        "correction to a domain-growth law is an additive lattice-scale constant, l = A t^b + c); "
        "V. Spirin, P. L. Krapivsky and S. Redner, Phys. Rev. E 63, 036118 (2001) (zero-"
        "temperature Glauber dynamics in 2D: energy density decaying as t^(-1/2))"),
    "sznajd_complete": (
        "derived here (theory.sznajd_complete_exit) from include/SznajdModel.hpp as implemented: "
        "on the complete graph the first agreeing ordered pair converts every other node, so "
        "P(+1) = n(n-1) / [n(n-1) + (N-n)(N-n-1)] exactly, -> f^2 / (f^2 + (1-f)^2); the model: "
        "K. Sznajd-Weron and J. Sznajd, \"Opinion evolution in closed community\", Int. J. Mod. "
        "Phys. C 11, 1157 (2000) (outflow rule: an agreeing pair converts its neighbours)"),
    "sznajd_lattice": (
        "D. Stauffer, A. O. Sousa and S. Moss de Oliveira, \"Generalization to square lattice of "
        "Sznajd sociophysics model\", Int. J. Mod. Phys. C 11, 1239 (2000) (an agreeing pair of "
        "neighbours convinces its six neighbours; the probability of ending at +1 against the "
        "initial concentration sharpens with the lattice size into a step at 1/2); "
        "C. Castellano, S. Fortunato and V. Loreto, \"Statistical physics of social dynamics\", "
        "Rev. Mod. Phys. 81, 591 (2009) (review: the exit probability on the square lattice "
        "tends to a step at m = 0); E(1/2) = 1/2 by the +-1 symmetry of the rule and of a start "
        "with exactly half the nodes at +1"),
    "deffuant_rate_equation": (
        "E. Ben-Naim, P. L. Krapivsky and S. Redner, \"Bifurcations and patterns in compromise "
        "processes\", Physica D 183, 190 (2003) (arXiv:cond-mat/0212313): rate equation of the "
        "compromise (Deffuant) model with mu = 1/2, random pairs, |x1 - x2| < 1, opinions uniform "
        "on [-Delta, Delta] (Delta = 1/(2 eps) for opinions on [0, 1] with bound eps); one central "
        "cluster for Delta < 1.871, the central cluster splits into two major clusters at "
        "Delta = 1.871; bifurcation points (Delta_1, Delta_2, Delta_3) = (1.000, 1.871, 2.248), "
        "(3.289, 4.079, 4.455), (5.496, 6.259, 6.638), (7.676, 8.431, 8.810) (types: 1 minor pair "
        "born, 2 central cluster -> major pair, 3 central cluster born), period L ~ 2.155; major "
        "clusters of mass M > 1 (-> L), minor clusters of mass m < 1e-2 (-> 3e-4), integration "
        "mesh 0.0025 in Delta; complete consensus (no minor clusters) only for Delta < 1 "
        "(eps > 1/2); S. Fortunato, \"Universality of the threshold for complete consensus for "
        "the opinion dynamics of Deffuant et al.\", Int. J. Mod. Phys. C 15, 1301 (2004) "
        "(arXiv:cond-mat/0406054) (all agents in one cluster for eps > 1/2 on every graph "
        "studied); J. Lorenz, \"Continuous opinion dynamics under bounded confidence: a "
        "survey\", Int. J. Mod. Phys. C 18, 1819 (2007) (arXiv:0707.1762) (the consensus "
        "transition as the bifurcation of the central cluster into two major clusters, minor "
        "clusters neglected)"),
    "hk_threshold": (
        "S. Fortunato, \"On the consensus threshold for the opinion dynamics of "
        "Krause-Hegselmann\", Int. J. Mod. Phys. C 16, 259 (2005) (arXiv:cond-mat/0408648): "
        "complete graph, opinions uniform on [0, 1], agents updated one at a time in ordered "
        "sweeps (every agent once per sweep, the same order every sweep; random updating, he "
        "writes, would not change the final number of opinion clusters, though on special "
        "graphs it can change the probability of complete consensus), stop when no opinion "
        "changes by more than 1e-9 in a sweep, "
        "consensus = every "
        "agent with the same opinion (within 1e-9), 1000 samples per point at N = 5000 and "
        "10000: the step of the consensus probability for N -> infinity lies in [0.195, 0.202] "
        "(no comparison with synchronous updating); J. Lorenz, Int. J. "
        "Mod. Phys. C 18, 1819 (2007) (synchronous density-based model: consensus for eps >= "
        "0.19, with long convergence and consensus again at lower eps)"),
    "hk_clusters": (
        "V. D. Blondel, J. M. Hendrickx and J. N. Tsitsiklis, \"On Krause's multi-agent "
        "consensus model with state-dependent connectivity\", IEEE Trans. Autom. Control 54, "
        "2586 (2009) (arXiv:0807.2028): synchronous Hegselmann-Krause dynamics, bound 1, 5000 L "
        "agents on [0, L]: the number of clusters grows linearly with L with a coefficient "
        "slightly smaller than 1/2 (inter-cluster distance slightly larger than 2; about 2.2 on "
        "a semi-infinite interval) -- the \"2R conjecture\" (clusters ~ 1/(2 eps) on [0, 1]); "
        "C. Wang, Q. Li, W. E and B. Chazelle, \"Noisy Hegselmann-Krause systems: phase "
        "transition and the 2R-conjecture\", J. Stat. Phys. 166, 1209 (2017) "
        "(arXiv:1511.02975) (mean-field linear stability: 1/(2.29 eps) clusters)"),
}


# ---------------------------------------------------------------------------------------------
# Birth-death chains on n = 0..N
# ---------------------------------------------------------------------------------------------

def _rates(up: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    up = np.asarray(up, dtype=float)
    down = np.asarray(down, dtype=float)
    if up.ndim != 1 or up.shape != down.shape or len(up) < 2:
        raise ValueError("up and down must be 1-D arrays of the same length N + 1 >= 2")
    if not (np.all(np.isfinite(up)) and np.all(np.isfinite(down))):
        raise ValueError("rates must be finite")
    if np.any(up < 0) or np.any(down < 0) or np.any(up + down > 1 + 1e-12):
        raise ValueError("rates are probabilities per update: up, down >= 0 and up + down <= 1")
    return up, down, len(up) - 1


def _require_positive_interior(up: np.ndarray, down: np.ndarray, N: int) -> None:
    if N >= 2 and (np.any(up[1:N] <= 0) or np.any(down[1:N] <= 0)):
        raise ValueError("every interior state 0 < n < N needs up[n] > 0 and down[n] > 0")


def exit_probability(up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """P(absorbed at N | start at n) for n = 0..N, with 0 and N absorbing (up[0], down[N] and
    the other boundary rates are ignored).

    First-step analysis: P(n) = sum_{k<n} rho_k / sum_{k<N} rho_k, rho_k = prod_{j=1}^{k}
    down[j] / up[j] (Karlin & Taylor). rho is accumulated in log space and every term is
    positive, so there is neither overflow nor cancellation."""
    up, down, N = _rates(up, down)
    _require_positive_interior(up, down, N)
    log_rho = np.concatenate(([0.0], np.cumsum(np.log(down[1:N]) - np.log(up[1:N]))))
    w = np.exp(log_rho - log_rho.max())
    c = np.cumsum(w)
    return np.concatenate(([0.0], c / c[-1]))


def mean_absorption_time(up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Mean number of single updates until absorption at 0 or N, from n = 0..N.

    First-step analysis: T(0) = T(N) = 0 and, for 0 < n < N,
        (up[n] + down[n]) T(n) - up[n] T(n+1) - down[n] T(n-1) = 1
    (an update that leaves n unchanged costs one update too). The tridiagonal system is solved
    by elimination written without subtractions: the pivots are d_n = up[n] + down[n] e_{n-1}
    with e_n = 1 - up[n] / d_n = down[n] e_{n-1} / d_n, e_0 = 1, so every intermediate is a sum
    of positive terms and the result is accurate componentwise to O(N eps)."""
    up, down, N = _rates(up, down)
    _require_positive_interior(up, down, N)
    T = np.zeros(N + 1)
    if N < 2:
        return T
    d = np.empty(N)       # pivots, index 1..N-1
    y = np.empty(N)       # eliminated right-hand side
    e = 1.0
    for n in range(1, N):
        d[n] = up[n] + down[n] * e
        y[n] = 1.0 if n == 1 else 1.0 + down[n] * y[n - 1] / d[n - 1]
        e = down[n] * e / d[n]
    T[N - 1] = y[N - 1] / d[N - 1]
    for n in range(N - 2, 0, -1):
        T[n] = (y[n] + up[n] * T[n + 1]) / d[n]
    return T


def stationary_distribution(up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Stationary law of an ergodic chain (no absorbing states): up[n] > 0 for n < N and
    down[n] > 0 for n > 0. Detailed balance pi(n+1) / pi(n) = up[n] / down[n+1], summed in log
    space and normalised by logsumexp, so N = 10^4 and beyond neither overflows nor
    underflows the bulk."""
    up, down, N = _rates(up, down)
    if np.any(up[:N] <= 0) or np.any(down[1:] <= 0):
        raise ValueError("an ergodic chain needs up[n] > 0 for n < N and down[n] > 0 for n > 0")
    log_pi = np.concatenate(([0.0], np.cumsum(np.log(up[:N]) - np.log(down[1:]))))
    return np.exp(log_pi - logsumexp(log_pi))


# ---------------------------------------------------------------------------------------------
# Per-update rates on the complete graph (binary +/-1 opinions)
# ---------------------------------------------------------------------------------------------

def _states(N: int) -> np.ndarray:
    if N < 2:
        raise ValueError("the complete graph needs N >= 2 nodes")
    return np.arange(N + 1, dtype=float)


def voter_rates(N: int) -> tuple[np.ndarray, np.ndarray]:
    """VoterModel, Node update (include/VoterModel.hpp): a random node copies a random
    neighbour. n -> n+1 iff a -1 node is chosen ((N-n)/N) and its neighbour is +1 (n of its
    N-1 neighbours): up[n] = (N-n)/N * n/(N-1); symmetrically down[n] = n/N * (N-n)/(N-1)."""
    n = _states(N)
    return (N - n) / N * (n / (N - 1)), n / N * ((N - n) / (N - 1))


def noisy_voter_rates(N: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """NoisyVoterModel with the default opinions {+1, -1} (include/NoisyVoterModel.hpp): the
    chosen node draws a new opinion uniformly from {+1, -1} with probability `noise` (so it
    keeps its own half the time), else copies a random neighbour. For a -1 node (prob. (N-n)/N)
    the flip probability is (1-noise) n/(N-1) + noise/2:
        up[n]   = (N-n)/N * [(1-noise) n/(N-1)     + noise/2]
        down[n] =     n/N * [(1-noise) (N-n)/(N-1) + noise/2]"""
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    n = _states(N)
    up = (N - n) / N * ((1 - noise) * (n / (N - 1)) + noise / 2)
    down = n / N * ((1 - noise) * ((N - n) / (N - 1)) + noise / 2)
    return up, down


def nonlinear_voter_rates(N: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """NonlinearVoterModel (include/NonlinearVoterModel.hpp): the chosen node flips with
    probability f^alpha, f = disagreeing fraction of its N-1 neighbours. A -1 node sees
    f = n/(N-1), a +1 node f = (N-n)/(N-1):
        up[n] = (N-n)/N * (n/(N-1))^alpha,   down[n] = n/N * ((N-n)/(N-1))^alpha
    (alpha = 1 uses f itself, as the header does, so it equals voter_rates)."""
    if not alpha > 0:
        raise ValueError("alpha must be > 0")
    n = _states(N)
    f_up, f_down = n / (N - 1), (N - n) / (N - 1)
    if alpha != 1.0:
        f_up, f_down = f_up ** alpha, f_down ** alpha
    return (N - n) / N * f_up, n / N * f_down


def qvoter_independence_rates(N: int, q: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """QVoterModel with independence and the default panel with repetition
    (include/QVoterModel.hpp): with probability p the chosen node adopts +1 or -1 with
    probability 1/2 each; otherwise it draws q neighbours with repetition from its N-1 and
    adopts the panel's opinion only if unanimous. A -1 node meets an all-+1 panel with
    probability (n/(N-1))^q:
        up[n]   = (N-n)/N * [(1-p) (n/(N-1))^q     + p/2]
        down[n] =     n/N * [(1-p) ((N-n)/(N-1))^q + p/2]
    For N -> infinity, (up - down) with x = n/N gives the header's mean-field equation."""
    if q < 1 or int(q) != q:
        raise ValueError("q must be an integer >= 1")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    n = _states(N)
    q = int(q)
    up = (N - n) / N * ((1 - p) * (n / (N - 1)) ** q + p / 2)
    down = n / N * ((1 - p) * ((N - n) / (N - 1)) ** q + p / 2)
    return up, down


def qvoter_pc(q: int) -> float:
    """Mean-field critical independence p_c = (q-1) / (q-1 + 2^(q-1)) (Nyczka et al. 2012),
    where x = 1/2 of the header's dx/dt loses stability: d/dx dx/dt at 1/2 is
    (1-p) 2^(1-q) (q-1) - p. Continuous for q <= 5; for q > 5 it is the lower spinodal of a
    discontinuous transition."""
    if q < 1 or int(q) != q:
        raise ValueError("q must be an integer >= 1")
    return (q - 1) / (q - 1 + 2 ** (q - 1))


def qvoter_mean_field_m(q: int, p: float) -> float:
    """Stationary |m| = 2x - 1 of the q-voter with independence as N -> infinity, where x in
    (1/2, 1] solves the header's
        dx/dt = (1-p) [(1-x) x^q - x (1-x)^q] + p (1/2 - x) = 0,
    and 0 at and above p_c. Below p_c, dx/dt > 0 just above 1/2 and = -p/2 < 0 at x = 1, so a
    sign change is bracketed on a grid (geometric near 1/2) and refined by brentq; the largest
    root is taken, which is the stable ordered state. For q > 5 the ordered state also exists,
    metastable, in a window above p_c (hysteresis); this function returns 0 there, so do not
    compare q > 5 in that window with it.

    q >= 2 only (ValueError otherwise): for q = 1 (the voter / noisy voter case) there is no
    unique stationary m to compare with. At p = 0, dx/dt = 0 for every x; for p > 0 the
    finite-N stationary law is bimodal at small p N (noisy_voter_rates and
    stationary_distribution describe it exactly)."""
    if q < 2 or int(q) != q:
        raise ValueError("qvoter_mean_field_m needs an integer q >= 2: the voter case q = 1 has "
                         "no unique stationary m")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if p >= qvoter_pc(q):
        return 0.0
    if p == 0.0:          # consensus x = 1 is the stable fixed point for q >= 2
        return 1.0
    q = int(q)

    def dxdt(x: float | np.ndarray) -> float | np.ndarray:
        return (1 - p) * ((1 - x) * x ** q - x * (1 - x) ** q) + p * (0.5 - x)

    offsets = np.unique(np.concatenate((2.0 ** -np.arange(52, 1, -1), np.linspace(0, 0.5, 2001)[1:])))
    x = 0.5 + offsets
    positive = np.nonzero(dxdt(x) > 0)[0]
    if len(positive) == 0:      # root within 2^-52 of 1/2
        return 0.0
    i = positive[-1]
    root = brentq(dxdt, x[i], x[i + 1], xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    return 2.0 * root - 1.0


# ---------------------------------------------------------------------------------------------
# Voter model on the square lattice
# ---------------------------------------------------------------------------------------------

def torus_eigentime(L: int, d: int) -> float:
    """sum over the non-zero modes k of 1 / lambda_k, lambda_k = 1 - (1/d) sum_i cos(2 pi k_i / L),
    for the simple random walk on the periodic L^d lattice: by the eigentime identity (Aldous &
    Fill, ch. 3) the mean number of steps to hit a fixed site from a uniform start, i.e. the
    exact finite-size meeting time of the coalescing walks dual to the voter model (in units of
    the walk's steps). It grows as N G_d (G_3 = 1.516386, Watson) with a relative 1/L correction
    in 3D, as N (ln N / pi + const) in 2D and is (L^2 - 1)/6 on the ring. Cox (1989): in
    d >= 2 the mean consensus time is asymptotically proportional to it."""
    if L < 2 or d < 1:
        raise ValueError("torus_eigentime needs L >= 2 and d >= 1")
    c = np.cos(2.0 * np.pi * np.arange(L) / L) / d
    total = np.zeros(1)
    for _ in range(d):                      # sum_i cos(...) / d over the d-dimensional grid
        total = (total[:, None] + c[None, :]).ravel()
    lam = 1.0 - total[1:]                   # mode 0 (lambda = 0) excluded
    return float(np.sum(1.0 / lam))


def _stehfest_weights(terms: int) -> np.ndarray:
    h = terms // 2
    V = np.zeros(terms)
    for k in range(1, terms + 1):
        V[k - 1] = (-1) ** (k + h) * sum(
            j ** h * math.factorial(2 * j) / (math.factorial(h - j) * math.factorial(j)
                                              * math.factorial(j - 1) * math.factorial(k - j)
                                              * math.factorial(2 * j - k))
            for j in range((k + 1) // 2, min(k, h) + 1))
    return V


def voter_rho_2d(t, terms: int = 14):
    """Exact density of active links rho(t) (disagreeing nearest-neighbour pairs) of the voter
    model on the infinite square lattice from an uncorrelated start with m = 0, each site
    copying a random neighbour at rate 1 (one sweep of node updates per unit time).

    The pair correlation G(r, t) = <s_0 s_r> obeys the closed equations
    dG(r)/dt = 2 [(1/4) sum_e G(r+e) - G(r)] for r != 0, G(0, t) = 1, G(r, 0) = delta_r0
    (Frachebourg & Krapivsky 1996). Writing G as the free walk P(r, t) (hop rate 1/2 per
    direction, P(0, t) = (e^-t I0(t))^2) plus a source j(t) at the origin that keeps G(0) = 1,
    and using s P0^(s) - 1 = 2 (Pe^(s) - P0^(s)) for the neighbour e, gives
    rho = (1 - G(e))/2 = j/4, i.e. in Laplace space
        rho^(s) = (1 / (s P0^(s)) - 1) / 4,   P0^(s) = 2 K(k) / (pi (s + 2)),  k = 2/(s + 2),
    K the complete elliptic integral (lattice Green function). rho^(s) is inverted with the
    Gaver-Stehfest formula (`terms` even; with 14, rho agrees with a direct solution of the
    equations to ~3e-6 relative for t >= 0.1; at t << 0.1 the inversion loses accuracy to
    cancellation); 1 - k^2 is formed without cancellation for the small s of large t. rho -> pi / (2 ln t + ln 256),
    Frachebourg & Krapivsky's Eq. (23), only logarithmically slowly (3% off at t = 10^12).
    The driver's sweep of N random single updates and its exactly balanced start differ from
    this by O(1/N)."""
    t = np.asarray(t, dtype=float)
    if np.any(~(t > 0)) or np.any(~np.isfinite(t)):
        raise ValueError("t must be finite and > 0")
    if terms < 2 or terms % 2:
        raise ValueError("terms must be an even integer >= 2")
    V = _stehfest_weights(terms)
    a = math.log(2.0) / t[..., None]
    s = a * np.arange(1, terms + 1)
    p = 2.0 / (s + 2.0)
    one_minus_m = s * (s + 4.0) / (s + 2.0) ** 2           # 1 - p^2
    P0 = p / math.pi * ellipkm1(one_minus_m)
    rho = a[..., 0] * np.sum(V * (1.0 / (s * P0) - 1.0) / 4.0, axis=-1)
    return float(rho) if rho.ndim == 0 else rho


# ---------------------------------------------------------------------------------------------
# Square-lattice Ising (J = 1, k_B = 1)
# ---------------------------------------------------------------------------------------------

ISING_TC: float = 2.0 / math.log(1.0 + math.sqrt(2.0))      # sinh(2 / Tc) = 1
ISING_EXPONENTS: dict[str, float] = {"nu": 1.0, "gamma/nu": 7.0 / 4.0, "beta/nu": 1.0 / 8.0}
# U4 = 1 - <M^4> / (3 <M^2>^2) at Tc, periodic L x L, L -> infinity. Salas & Sokal (2000):
# <M^4>/<M^2>^2 = 1.1679229(47) -> U4 = 1 - 1.1679229/3 = 0.6106924, uncertainty 4.7e-6/3.
ISING_BINDER: tuple[float, float] = (0.6106924, 1.6e-6)


def onsager_magnetization(T: float) -> float:
    """Spontaneous magnetization per site, (1 - sinh(2/T)^-4)^(1/8) below Tc, 0 at and above
    (Onsager 1944, Yang 1952). sinh(x)^-1 = 2 e^-x / (1 - e^-2x) avoids overflow as T -> 0."""
    if not T > 0:
        raise ValueError("T must be > 0")
    if T >= ISING_TC:
        return 0.0
    E = math.exp(-2.0 / T)
    inv_sinh = 2.0 * E / (1.0 - E * E)
    return max(0.0, 1.0 - inv_sinh ** 4) ** 0.125


def ising_energy(T: float) -> float:
    """Exact energy per site of the infinite square lattice (Onsager 1944; Baxter 1982, ch. 7):
        u = -coth(2b) [1 + (2/pi) (2 tanh^2(2b) - 1) K(k)],   b = 1/T,
        k = 2 sinh(2b) / cosh^2(2b),
    with K the complete elliptic integral of the first kind of modulus k, i.e. scipy's
    ellipk(m) at PARAMETER m = k^2. At Tc, k = 1 and K diverges logarithmically while its
    prefactor vanishes linearly, so the term is 0 and u(Tc) = -sqrt 2; that limit is taken
    whenever m rounds to >= 1. Hyperbolic functions are written through e^(-2b) so T -> 0
    does not overflow (u -> -2)."""
    if not T > 0:
        raise ValueError("T must be > 0")
    E = math.exp(-2.0 / T)                      # e^(-2b)
    tanh = (1.0 - E * E) / (1.0 + E * E)        # tanh(2b)
    k = 4.0 * E * (1.0 - E * E) / (1.0 + E * E) ** 2
    m = k * k
    term = 0.0 if m >= 1.0 else (2.0 / math.pi) * (2.0 * tanh * tanh - 1.0) * float(ellipk(m))
    return -(1.0 + term) / tanh


# ---------------------------------------------------------------------------------------------
# Square-lattice 3-state Potts (J = 1, k_B = 1)
# ---------------------------------------------------------------------------------------------

POTTS3_TC: float = 1.0 / math.log(1.0 + math.sqrt(3.0))     # self-dual point, J / ln(1 + sqrt q)
POTTS3_EXPONENTS: dict[str, float] = {"nu": 5.0 / 6.0, "gamma/nu": 26.0 / 15.0, "beta/nu": 2.0 / 15.0}
