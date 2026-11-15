"""Final-state snapshots of lattices."""

from __future__ import annotations

import numpy as np

from ..errors import InputError
from ..io import Provenance, Table
from ..sizes import _parse_dims
from .style import finish, new_fig, PALETTE


def _warn(warnings, text):
    if warnings is not None:
        warnings.append(text)


def _panel(ax, grid2d, vmin, vmax, cmap, title):
    ax.imshow(grid2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=7)


def plot_states(t: Table, prov: Provenance, max_panels: int = 6,
                warnings: list[str] | None = None):
    """One panel per replica/chain column (at most max_panels): 2D lattices as images, 1D as a
    strip, 3D as the three mid-slices. Discrete opinions get a discrete colour map,
    continuous ones a continuous map. Networks are skipped."""
    cfg = prov.config
    if cfg.get("graph", "lattice") != "lattice" or "dims" not in cfg:
        _warn(warnings, f"{t.path.name}: snapshots are drawn for lattices only "
                        f"(graph = {cfg.get('graph', 'unknown')}); states figure skipped")
        return []
    dims = _parse_dims(cfg["dims"], str(t.path))
    if len(dims) > 3:
        _warn(warnings, f"{t.path.name}: snapshots are drawn for 1D, 2D and 3D lattices "
                        f"(dims = {cfg['dims']}); states figure skipped")
        return []
    cols = [n for n in t.names if n != t.names[0] and t[n].dtype != object]
    if not cols or len(t) != int(np.prod(dims)):
        _warn(warnings, f"{t.path.name}: {len(t)} nodes do not match dims {cfg['dims']}; "
                        "states figure skipped")
        return []
    if len(cols) > max_panels:
        _warn(warnings, f"{t.path.name}: {len(cols)} replicas/chains, first {max_panels} drawn "
                        f"({len(cols) - max_panels} skipped)")
        cols = cols[:max_panels]
    allv = np.concatenate([t[c] for c in cols])
    uniq = np.unique(allv[np.isfinite(allv)])
    discrete = 0 < uniq.size <= 8 and np.all(uniq == np.round(uniq))
    if discrete:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(PALETTE[: uniq.size] if uniq.size > 1 else PALETTE[:1])
        vmin, vmax = uniq.min() - 0.5, uniq.max() + 0.5
        if uniq.size > 1:
            vmin, vmax = uniq.min() - 0.5 * (uniq.max() - uniq.min()) / (uniq.size - 1), \
                uniq.max() + 0.5 * (uniq.max() - uniq.min()) / (uniq.size - 1)
    else:
        cmap, vmin, vmax = "viridis", float(np.nanmin(allv)), float(np.nanmax(allv))
    d = len(dims)
    if d == 3:
        fig, axes = new_fig(len(cols), 3, size=(2.2, 2.3))
    else:
        n = len(cols)
        nc = min(n, 3)
        nr = -(-n // nc)
        fig, axes = new_fig(nr, nc, size=(3.2, 2.6 if d == 2 else 1.2))
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
    for k, c in enumerate(cols):
        field = t[c].reshape(dims)
        if d == 1:
            _panel(axes.ravel()[k], field[None, :], vmin, vmax, cmap, c)
            axes.ravel()[k].set_aspect("auto")
        elif d == 2:
            _panel(axes.ravel()[k], field, vmin, vmax, cmap, c)
        else:
            for a in range(3):
                sl = np.take(field, dims[a] // 2, axis=a)
                _panel(axes[k, a], sl, vmin, vmax, cmap, f"{c}: slice x{a} = {dims[a] // 2}")
    if discrete and uniq.size <= 8:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=cmap(i), label=f"{v:g}") for i, v in enumerate(uniq)]
                   if uniq.size > 1 else [Patch(color=PALETTE[0], label=f"{uniq[0]:g}")],
                   loc="outside right upper", title="opinion", fontsize=7)
    else:
        sm = __import__("matplotlib.cm", fromlist=["cm"]).ScalarMappable(
            norm=__import__("matplotlib.colors", fromlist=["c"]).Normalize(vmin, vmax), cmap=cmap)
        fig.colorbar(sm, ax=axes.ravel().tolist(), label="opinion", shrink=0.8)
    return [("states", finish(fig, prov))]
