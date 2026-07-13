"""
ArraySpec instances for all 21 simulated microphone arrays from the paper's experiment.

Public interface:
    PAPER_ARRAYS_TRAIN  — 10 training arrays
    PAPER_ARRAYS_TEST   — 11 test arrays incl. ARIA
    PAPER_ARRAYS_ALL    — all 21 combined

Individual arrays are also importable by name (e.g. from datagenerator.paper_arrays import aria_rigid).

Usage in run_FT_JNF.py:
    from datagenerator.paper_arrays import PAPER_ARRAYS_TRAIN, PAPER_ARRAYS_TEST
    ARRAYS_TRAIN = PAPER_ARRAYS_TRAIN
    ARRAYS_TEST  = PAPER_ARRAYS_TEST

Run as a script to produce the paper's geometry figures:
    python datagenerator/paper_arrays.py
"""

import sys
import os

# When run as a script (`python datagenerator/paper_arrays.py`), Python adds the
# script's own directory to sys.path, not the project root.  Fix that here so
# the datagenerator package and its siblings are importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from shroom.geometry.sampling import sphereicalGrid

from datagenerator.generate_baseline_train_ds import ArraySpec
from datagenerator.helpers import FreeFieldArrayConfig, RigidSphereArrayConfig


# ── Private helpers ───────────────────────────────────────────────────────────

def _cart_to_sphere_grid(x, y, z):
    """Cartesian (x,y,z) → sphereicalGrid(az, co).  Safe for any finite radius."""
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    co = np.arccos(np.clip(z / r, -1.0, 1.0))
    return sphereicalGrid(az=az, co=co)


def _fibonacci_az_co(n):
    """Exact MATLAB fibonacci_sphere(N) formula → (az, co) arrays.

    Matches experiment_data_gen_3D.m lines 779–789 so that uniform_sphere_r01/r005
    reproduce the paper's arrays exactly (not the fibonacci_sphere_points() variant
    in helpers.py which uses a slightly different z-spacing).
    """
    indices = np.arange(n) + 0.5
    z = 1.0 - 2.0 * indices / n
    co = np.arccos(z)
    az = (2.0 * np.pi * indices / ((1.0 + np.sqrt(5.0)) / 2.0)) % (2.0 * np.pi)
    return az, co


def _semi_planar_pos(r):
    """Center mic + 3 middle-arc + 3 outer-arc mics in XY-plane (arrays 15 / 16)."""
    phi_mid = np.array([-np.pi / 4, 0.0, np.pi / 4])
    phi_out = np.array([-np.pi / 2, 0.0, np.pi / 2])
    x = np.concatenate([[0.0], 0.6 * r * np.cos(phi_mid), r * np.cos(phi_out)])
    y = np.concatenate([[0.0], 0.6 * r * np.sin(phi_mid), r * np.sin(phi_out)])
    return np.column_stack([x, y, np.zeros(7)])


def _sphere_mesh(r, n=50):
    """Unit-sphere (n+1)×(n+1) grid scaled by r — equivalent to MATLAB sphere(n)*r."""
    phi   = np.linspace(0, np.pi,     n + 1)
    theta = np.linspace(0, 2 * np.pi, n + 1)
    tg, pg = np.meshgrid(theta, phi)
    return r * np.sin(pg) * np.cos(tg), r * np.sin(pg) * np.sin(tg), r * np.cos(pg)


# ── Array 1: ULA along X-axis (free-field) ───────────────────────────────────
_x1 = np.linspace(-0.1, 0.1, 7)
_pos1 = np.column_stack([_x1, np.zeros(7), np.zeros(7)])

ula_x_axis = ArraySpec(
    name="ULA along X-axis",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=_pos1),
)

# ── Array 2: ULA tilted 20° around Y-axis (free-field) ───────────────────────
_th20 = np.radians(20)
_Ry20 = np.array([[np.cos(_th20), 0,  np.sin(_th20)],
                  [0,             1,  0            ],
                  [-np.sin(_th20), 0, np.cos(_th20)]])
_pos2 = (_Ry20 @ _pos1.T).T

ula_x_tilted_20 = ArraySpec(
    name="ULA along X-axis (tilt=20)",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=_pos2),
)

# ── Array 3: random sphere 1, ON+IN surface, r=0.1 (free-field) ──────────────
random_sphere1_on_in_r01 = ArraySpec(
    name="random sphere1 radius = 0.1",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=np.array([
        [ 0.0223, -0.0517,  0.0827],
        [ 0.0800, -0.0538,  0.0265],
        [ 0.0414,  0.0425, -0.0805],
        [-0.0121,  0.0666, -0.0636],
        [-0.0169, -0.0051,  0.0491],
        [ 0.0293, -0.0080,  0.0686],
        [ 0.0947, -0.0212, -0.0028],
    ])),
)

# ── Array 4: random sphere 2, ON+IN surface, r=0.1 (free-field) ──────────────
random_sphere2_on_in_r01 = ArraySpec(
    name="random sphere2 radius = 0.1",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=np.array([
        [ 0.0097, -0.0358, -0.0929],
        [ 0.0693, -0.0180,  0.0698],
        [-0.0277, -0.0412,  0.0868],
        [-0.0268, -0.0558,  0.0203],
        [ 0.0013, -0.0270, -0.0236],
        [-0.0018, -0.0418,  0.0189],
        [-0.0256,  0.0206, -0.0878],
    ])),
)

# ── Array 5: random sphere 1, ON surface, r=0.1 (rigid sphere) ───────────────
random_sphere1_on_r01 = ArraySpec(
    name="random sphere3 (rigid) radius = 0.1",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=_cart_to_sphere_grid(
            np.array([-0.0274, -0.0319,  0.0951,  0.0971, -0.0886, -0.0668,  0.0084]),
            np.array([-0.0759,  0.0711, -0.0308,  0.0213,  0.0359,  0.0615, -0.0856]),
            np.array([ 0.0590, -0.0626, -0.0020, -0.0109,  0.0293,  0.0419,  0.0509]),
        ),
        mic_radius=0.1,
    ),
)

# ── Array 6: random sphere 2, ON surface, r=0.1 (rigid sphere) ───────────────
random_sphere2_on_r01 = ArraySpec(
    name="random sphere4 (rigid) radius = 0.1",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=_cart_to_sphere_grid(
            np.array([-0.0154, -0.0421, -0.0468,  0.0451,  0.0639, -0.1000,  0.0888]),
            np.array([ 0.0935, -0.0891, -0.0690,  0.0737,  0.0593,  0.0010, -0.0230]),
            np.array([-0.0319,  0.0171, -0.0552,  0.0503, -0.0490,  0.0012,  0.0398]),
        ),
        mic_radius=0.1,
    ),
)

# ── Array 7: random sphere 1, ON surface, r=0.05 (rigid sphere) ──────────────
random_sphere1_on_r005 = ArraySpec(
    name="random sphere5 (rigid) radius = 0.05",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=_cart_to_sphere_grid(
            np.array([ 0.0337,  0.0376, -0.0410,  0.0165,  0.0282, -0.0019,  0.0234]),
            np.array([-0.0276, -0.0098, -0.0125,  0.0196,  0.0385,  0.0397, -0.0365]),
            np.array([-0.0246,  0.0314, -0.0256,  0.0429, -0.0150, -0.0303, -0.0249]),
        ),
        mic_radius=0.05,
    ),
)

# ── Array 8: random sphere 2, ON surface, r=0.05 (rigid sphere) ──────────────
random_sphere2_on_r005 = ArraySpec(
    name="random sphere6 (rigid) radius = 0.05",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=_cart_to_sphere_grid(
            np.array([-0.0337, -0.0423, -0.0257,  0.0236, -0.0426, -0.0252,  0.0196]),
            np.array([-0.0301,  0.0072,  0.0346, -0.0424, -0.0253, -0.0081, -0.0112]),
            np.array([-0.0214,  0.0257,  0.0254, -0.0120,  0.0068, -0.0424, -0.0446]),
        ),
        mic_radius=0.05,
    ),
)

# ── Arrays 9 & 10: uniform (Fibonacci) sphere, r=0.1 and r=0.05 (rigid sphere)
_az_fib, _co_fib = _fibonacci_az_co(7)
_fib_grid = sphereicalGrid(az=_az_fib, co=_co_fib)  # same angular layout for both

uniform_sphere_r01 = ArraySpec(
    name="uniform sphere (rigid) radius = 0.1",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(mics_grid=_fib_grid, mic_radius=0.1),
)

uniform_sphere_r005 = ArraySpec(
    name="uniform sphere (rigid) radius = 0.05",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(mics_grid=_fib_grid, mic_radius=0.05),
)

# ── Array 11: full circle in XY-plane, r=0.1 (rigid sphere) ──────────────────
full_circle_r01 = ArraySpec(
    name="full circle (rigid) radius = 0.1",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=sphereicalGrid(
            az=np.linspace(0, 2 * np.pi, 7, endpoint=False),
            co=np.full(7, np.pi / 2),
        ),
        mic_radius=0.1,
    ),
)

# ── Array 12: semi-circle in XY-plane, r=0.1 (rigid sphere) ──────────────────
semi_circle_r01 = ArraySpec(
    name="semi circle (rigid) radius = 0.1",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=sphereicalGrid(
            az=np.linspace(-np.pi / 2, np.pi / 2, 7),
            co=np.full(7, np.pi / 2),
        ),
        mic_radius=0.1,
    ),
)

# ── Array 13: uniform planar square, r=0.1 (free-field) ──────────────────────
# 7 of 9 points from a 3×3 grid; excludes the two side-midpoints (x=0, y=±0.1).
# Uses Fortran (column-major) ravel to match MATLAB's X(:) linearisation.
_vals13 = np.linspace(-0.1, 0.1, 3)
_X13, _Y13 = np.meshgrid(_vals13, _vals13)
_all13 = np.column_stack([_X13.ravel('F'), _Y13.ravel('F')])
_idx7 = np.array([0, 1, 2, 4, 6, 7, 8])  # 0-based equiv. of MATLAB 1-based [1,2,3,5,7,8,9]
_xy13 = _all13[_idx7]

planar_square_uniform = ArraySpec(
    name="planar",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(
        mic_positions=np.column_stack([_xy13, np.zeros(7)]),
    ),
)

# ── Array 14: uniform planar square, rotated 45°, r=0.1 (free-field) ─────────
_sc14 = 1.0 / np.sqrt(2)
_vals14 = np.linspace(-0.1 * _sc14, 0.1 * _sc14, 3)
_X14, _Y14 = np.meshgrid(_vals14, _vals14)
_all14 = np.column_stack([_X14.ravel('F'), _Y14.ravel('F')])
_xy14 = _all14[_idx7]
_R45 = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                 [np.sin(np.pi / 4),  np.cos(np.pi / 4)]])
_xy14_rot = (_R45 @ _xy14.T).T

planar_square_uniform_rot45 = ArraySpec(
    name="planar (rot=45deg)",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(
        mic_positions=np.column_stack([_xy14_rot, np.zeros(7)]),
    ),
)

# ── Array 15: uniform semi-planar, r=0.1 (free-field) ────────────────────────
semi_planar_r01 = ArraySpec(
    name="semi circle planar radius = 0.1",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=_semi_planar_pos(0.1)),
)

# ── Array 16: uniform semi-planar, r=0.05 (free-field) ───────────────────────
semi_planar_r005 = ArraySpec(
    name="semi circle planar radius = 0.05",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=_semi_planar_pos(0.05)),
)

# ── Array 17: random planar disc 1, r=0.1 (free-field) ───────────────────────
random_circle1_r01 = ArraySpec(
    name="random 2D array1 radius = 0.1",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=np.array([
        [-0.0585,  0.0193, 0.0],
        [-0.0798,  0.0575, 0.0],
        [-0.0567, -0.0487, 0.0],
        [-0.0133, -0.0421, 0.0],
        [-0.0885, -0.0284, 0.0],
        [ 0.0532,  0.0804, 0.0],
        [ 0.0145, -0.0605, 0.0],
    ])),
)

# ── Array 18: random planar disc 2, r=0.1 (free-field) ───────────────────────
random_circle2_r01 = ArraySpec(
    name="random 2D array2 radius = 0.1",
    array_type="free_field",
    free_field=FreeFieldArrayConfig(mic_positions=np.array([
        [-0.0562, -0.0450, 0.0],
        [ 0.0691, -0.0643, 0.0],
        [-0.0685,  0.0491, 0.0],
        [-0.0052,  0.0857, 0.0],
        [ 0.0262,  0.0775, 0.0],
        [ 0.0351,  0.0687, 0.0],
        [-0.0268,  0.0057, 0.0],
    ])),
)

# ── Array 19: front hemisphere 1, r=0.1 (rigid sphere) ───────────────────────
# 1 mic at (r,0,0) + 6 on a ring at x=r/1.5; all mics lie exactly on the sphere.
_r = 0.1
_Ryz19 = np.sqrt(_r**2 - (_r / 1.5)**2)
_phi19 = np.linspace(0, 2 * np.pi, 6, endpoint=False)  # MATLAB: linspace(0,2π,7); remove last → 6 pts
_x19 = np.concatenate([[_r],          np.full(6, _r / 1.5)])
_y19 = np.concatenate([[0.0],         _Ryz19 * np.cos(_phi19)])
_z19 = np.concatenate([[0.0],         _Ryz19 * np.sin(_phi19)])

front_hemisphere1 = ArraySpec(
    name="front hemisphere1 (rigid) radius = 0.1",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=_cart_to_sphere_grid(_x19, _y19, _z19),
        mic_radius=_r,
    ),
)

# ── Array 20: front hemisphere 2, r=0.1 (rigid sphere) ───────────────────────
_Ryz20 = np.sqrt(_r**2 - (_r / 4)**2)
_phi20 = np.linspace(0, 2 * np.pi, 6, endpoint=False)  # same: 6 ring mics
_x20 = np.concatenate([[_r],          np.full(6, _r / 4)])
_y20 = np.concatenate([[0.0],         _Ryz20 * np.cos(_phi20)])
_z20 = np.concatenate([[0.0],         _Ryz20 * np.sin(_phi20)])

front_hemisphere2 = ArraySpec(
    name="front hemisphere2 (rigid) radius = 0.1",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=_cart_to_sphere_grid(_x20, _y20, _z20),
        mic_radius=_r,
    ),
)

# ── Array 21: ARIA on rigid sphere (simulated) ────────────────────────────────
# Mic positions in cm → m.  Not at a uniform radius, so mic_radius = mean(r_i).
_aria_cm = np.array([
    [ 9.95, -4.76,  0.68],  # lower-lens right
    [10.59,  0.74,  5.07],  # nose bridge
    [ 9.95,  4.49,  0.76],  # lower-lens left
    [ 9.28,  6.41,  5.12],  # front left
    [ 9.93, -5.66,  5.22],  # front right
    [-0.42, -8.45,  3.35],  # rear right
    [-0.48,  7.75,  3.49],  # rear left
])
_aria_m = _aria_cm / 100.0
_aria_mean_r = float(np.mean(np.linalg.norm(_aria_m, axis=1)))

aria_rigid = ArraySpec(
    name="Aria on rigid sphere (simulated)",
    array_type="rigid_sphere",
    rigid_sphere=RigidSphereArrayConfig(
        mics_grid=_cart_to_sphere_grid(_aria_m[:, 0], _aria_m[:, 1], _aria_m[:, 2]),
        mic_radius=_aria_mean_r,
    ),
)


# ── Paper train / test lists  (Fig. 2 / Fig. 3 order from the paper) ─────────
# Order: 1D/2D free-field (arrays 1-4) → 3D free-field (array 5) → rigid sphere (6-10).
# ARIA is appended at the end of the test list and plotted as a standalone figure.

PAPER_ARRAYS_TRAIN = [
    ula_x_axis,                # paper Fig. 2  array  1 — ULA
    planar_square_uniform,     # paper Fig. 2  array  2 — rectangular planar
    semi_planar_r005,          # paper Fig. 2  array  3 — semi-circular planar, r=0.05
    random_circle1_r01,        # paper Fig. 2  array  4 — random planar disc
    random_sphere1_on_in_r01,  # paper Fig. 2  array  5 — random sphere (free-field, on+in)
    random_sphere1_on_r01,     # paper Fig. 2  array  6 — random sphere (rigid, r=0.1)
    random_sphere1_on_r005,    # paper Fig. 2  array  7 — random sphere (rigid, r=0.05)
    uniform_sphere_r01,        # paper Fig. 2  array  8 — uniform sphere (rigid, r=0.1)
    full_circle_r01,           # paper Fig. 2  array  9 — full circle (rigid, r=0.1)
    front_hemisphere1,         # paper Fig. 2  array 10 — front hemisphere (rigid)
]

PAPER_ARRAYS_TEST = [
    ula_x_tilted_20,             # paper Fig. 3  array  1 — ULA tilted 20°
    planar_square_uniform_rot45, # paper Fig. 3  array  2 — rectangular planar, rotated 45°
    semi_planar_r01,             # paper Fig. 3  array  3 — semi-circular planar, r=0.1
    random_circle2_r01,          # paper Fig. 3  array  4 — random planar disc
    random_sphere2_on_in_r01,    # paper Fig. 3  array  5 — random sphere (free-field, on+in)
    random_sphere2_on_r01,       # paper Fig. 3  array  6 — random sphere (rigid, r=0.1)
    random_sphere2_on_r005,      # paper Fig. 3  array  7 — random sphere (rigid, r=0.05)
    uniform_sphere_r005,         # paper Fig. 3  array  8 — uniform sphere (rigid, r=0.05)
    semi_circle_r01,             # paper Fig. 3  array  9 — semi-circle (rigid, r=0.1)
    front_hemisphere2,           # paper Fig. 3  array 10 — front hemisphere (rigid)
    aria_rigid,                  # paper Fig. 3  ARIA  — plotted as standalone figure
]

PAPER_ARRAYS_ALL = PAPER_ARRAYS_TRAIN + PAPER_ARRAYS_TEST


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_paper_arrays(save_dir=None):
    """Produce the paper's array-geometry figures (Fig. 2, Fig. 3) and a standalone ARIA plot.

    Three figures are created:
      Figure 1 — Training arrays  (10 subplots, paper order 1-10)
      Figure 2 — Test arrays      (10 subplots, paper order 1-10; ARIA separate)
      Figure 3 — ARIA on rigid sphere (standalone)

    Parameters
    ----------
    save_dir : str or None
        If given, each figure is saved as a PDF inside this directory.
    """
    import matplotlib.pyplot as plt

    # ── Per-array decoration table (name → (style_key, display_r)) ───────────
    # style_key drives the background geometry drawn behind the mic scatter.
    _DECOR = {
        "ula_x_axis":                 ("ula_h",           0.10),
        "ula_x_tilted_20":            ("ula_tilt",        0.10),
        "random_sphere1_on_in_r01":   ("sphere_ff",       0.10),
        "random_sphere2_on_in_r01":   ("sphere_ff",       0.10),
        "random_sphere1_on_r01":      ("sphere_rigid",    0.10),
        "random_sphere2_on_r01":      ("sphere_rigid",    0.10),
        "random_sphere1_on_r005":     ("sphere_rigid",    0.05),
        "random_sphere2_on_r005":     ("sphere_rigid",    0.05),
        "uniform_sphere_r01":         ("sphere_rigid",    0.10),
        "uniform_sphere_r005":        ("sphere_rigid",    0.05),
        "full_circle_r01":            ("circle_rigid",    0.10),
        "semi_circle_r01":            ("semicircle_rigid",0.10),
        "planar_square_uniform":      ("square",          0.10),
        "planar_square_uniform_rot45":("square_rot45",    0.10),
        "semi_planar_r01":            ("semi_planar",     0.10),
        "semi_planar_r005":           ("semi_planar",     0.05),
        "random_circle1_r01":         ("circle_ff",       0.10),
        "random_circle2_r01":         ("circle_ff",       0.10),
        "front_hemisphere1":          ("hemisphere",      0.10),
        "front_hemisphere2":          ("hemisphere",      0.10),
        "aria_rigid":                 ("sphere_rigid",    0.11),
    }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_xyz(arr):
        if arr.array_type == "free_field":
            p = arr.free_field.mic_positions
            return p[:, 0], p[:, 1], p[:, 2]
        r  = arr.rigid_sphere.mic_radius
        az = arr.rigid_sphere.mics_grid.az
        co = arr.rigid_sphere.mics_grid.co
        return (r * np.sin(co) * np.cos(az),
                r * np.sin(co) * np.sin(az),
                r * np.cos(co))

    def _draw_sphere_surf(ax, r, style='rigid'):
        xs, ys, zs = _sphere_mesh(r, n=50)
        if style == 'ff':
            # Free-field: warm orange wireframe — visually distinct from rigid (blue solid)
            ax.plot_wireframe(xs[::8, ::8], ys[::8, ::8], zs[::8, ::8],
                              color=(0.85, 0.45, 0.05), alpha=0.35, linewidth=0.8)
        else:
            ax.plot_surface(xs, ys, zs,
                            color=(0.7, 0.8, 1.0), alpha=0.1,
                            edgecolor=None, linewidth=0.2)

    def _draw_crosshairs(ax, r):
        kw = dict(color=(0.5, 0.5, 0.5), linestyle=':')
        r12 = r * 1.2
        ax.plot3D([-r12, r12], [0, 0],   [0, 0],   **kw)
        ax.plot3D([0, 0],   [-r12, r12], [0, 0],   **kw)
        ax.plot3D([0, 0],   [0, 0],   [-r12, r12], **kw)

    def _format_ax(ax, r, num):
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        # 3 ticks per axis keeps labels sparse and readable
        ticks = np.array([-r, 0.0, r])
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Y (m)', fontsize=7)
        ax.set_zlabel('Z (m)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True)
        # azim=-45 mirrors azim=45 across the X axis, putting +X on the right
        ax.view_init(elev=25, azim=-45)
        ax.set_title(f'({num})', fontsize=10, pad=2)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass

    def _draw_subplot(ax, arr, num):
        style, display_r = _DECOR.get(arr.name, ("sphere_rigid", 0.10))
        x, y, z = _get_xyz(arr)

        # ── Step 1: background geometry ────────────────────────────────────
        if style == "sphere_ff":
            _draw_sphere_surf(ax, 0.1, style='ff')
            _draw_crosshairs(ax, 0.1)

        elif style in ("sphere_rigid", "hemisphere"):
            _draw_sphere_surf(ax, display_r)
            _draw_crosshairs(ax, display_r)

        elif style == "circle_rigid":
            _draw_sphere_surf(ax, display_r)
            _draw_crosshairs(ax, display_r)
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot3D(display_r * np.cos(th), display_r * np.sin(th),
                      np.zeros(200), 'k--', linewidth=1.5)

        elif style == "semicircle_rigid":
            _draw_sphere_surf(ax, display_r)
            _draw_crosshairs(ax, display_r)
            th = np.linspace(-np.pi / 2, np.pi / 2, 200)
            ax.plot3D(display_r * np.cos(th), display_r * np.sin(th),
                      np.zeros(200), 'k--', linewidth=1.2)

        elif style == "ula_h":
            ax.plot3D([-0.1, 0.1], [0, 0], [0, 0], 'k--', linewidth=1.2)

        elif style == "ula_tilt":
            th = np.radians(20)
            Ry = np.array([[np.cos(th), 0, np.sin(th)],
                           [0, 1, 0],
                           [-np.sin(th), 0, np.cos(th)]])
            ends = Ry @ np.array([[-0.1, 0.1], [0, 0], [0, 0]])
            ax.plot3D(ends[0], ends[1], ends[2], 'k--', linewidth=1.5)

        elif style == "square":
            sq_x = 0.1 * np.array([-1, 1, 1, -1, -1])
            sq_y = 0.1 * np.array([-1, -1, 1, 1, -1])
            ax.plot3D(sq_x, sq_y, np.zeros(5), 'k--', linewidth=1.2)
            for v in np.linspace(-0.1, 0.1, 3):
                ax.plot3D([-0.1, 0.1], [v, v], [0, 0], ':', color=(0.7, 0.7, 0.7))
                ax.plot3D([v, v], [-0.1, 0.1], [0, 0], ':', color=(0.7, 0.7, 0.7))

        elif style == "square_rot45":
            sc = 1.0 / np.sqrt(2)
            R45 = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                            [np.sin(np.pi / 4),  np.cos(np.pi / 4)]])
            sq = 0.1 * sc * np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]])
            sqr = R45 @ sq
            ax.plot3D(sqr[0], sqr[1], np.zeros(5), 'k--', linewidth=1.2)
            for v in np.linspace(-0.1 * sc, 0.1 * sc, 3):
                l1 = R45 @ np.array([[-0.1 * sc, 0.1 * sc], [v, v]])
                ax.plot3D(l1[0], l1[1], [0, 0], ':', color=(0.7, 0.7, 0.7))
                l2 = R45 @ np.array([[v, v], [-0.1 * sc, 0.1 * sc]])
                ax.plot3D(l2[0], l2[1], [0, 0], ':', color=(0.7, 0.7, 0.7))

        elif style == "semi_planar":
            th = np.linspace(-np.pi / 2, np.pi / 2, 200)
            ax.plot3D(display_r * np.cos(th), display_r * np.sin(th),
                      np.zeros(200), 'k--', linewidth=1.2)
            ax.plot3D([0, 0], [-display_r, display_r], [0, 0], 'k--', linewidth=1.2)

        elif style == "circle_ff":
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot3D(0.1 * np.cos(th), 0.1 * np.sin(th),
                      np.zeros(200), 'k--', linewidth=1.5)

        # ── Step 2: scatter ────────────────────────────────────────────────
        edge = 'k' if style in ("sphere_ff", "hemisphere") else 'none'
        ax.scatter3D(x, y, z, s=80,
                     color=[0, 0.447, 0.741], edgecolors=edge, depthshade=True)

        # ── Step 3: format ─────────────────────────────────────────────────
        _format_ax(ax, display_r, num)

    # ── Figure 1: Training arrays ─────────────────────────────────────────────
    n = len(PAPER_ARRAYS_TRAIN)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig1 = plt.figure(figsize=(3 * cols, 3 * rows), facecolor='white')
    for i, arr in enumerate(PAPER_ARRAYS_TRAIN, start=1):
        ax = fig1.add_subplot(rows, cols, i, projection='3d')
        _draw_subplot(ax, arr, i)
    fig1.suptitle('Training Arrays', fontsize=12, y=1.01)
    fig1.tight_layout()

    # ── Figure 2: Test arrays (ARIA plotted separately) ───────────────────────
    test_no_aria = PAPER_ARRAYS_TEST[:-1]
    n2 = len(test_no_aria)
    cols2 = int(np.ceil(np.sqrt(n2)))
    rows2 = int(np.ceil(n2 / cols2))

    fig2 = plt.figure(figsize=(3 * cols2, 3 * rows2), facecolor='white')
    for i, arr in enumerate(test_no_aria, start=1):
        ax = fig2.add_subplot(rows2, cols2, i, projection='3d')
        _draw_subplot(ax, arr, i)
    fig2.suptitle('Test Arrays', fontsize=12, y=1.01)
    fig2.tight_layout()

    # ── Figure 3: ARIA standalone ─────────────────────────────────────────────
    fig3 = plt.figure(figsize=(8, 6), facecolor='white')
    ax3 = fig3.add_subplot(111, projection='3d')

    R_display = 0.11  # mean ARIA sphere radius for visual context
    xs_a, ys_a, zs_a = _sphere_mesh(R_display, n=40)
    ax3.plot_surface(xs_a, ys_a, zs_a,
                     color=[0, 0.5, 1], alpha=0.1, edgecolor='none')

    xa, ya, za = _aria_m[:, 0], _aria_m[:, 1], _aria_m[:, 2]
    ax3.scatter3D(xa, ya, za, s=120, color='red', depthshade=True)

    for i1, i2 in [(3, 1), (1, 4), (3, 2), (4, 0), (2, 0), (3, 6), (4, 5)]:
        ax3.plot3D([xa[i1], xa[i2]], [ya[i1], ya[i2]], [za[i1], za[i2]],
                   'k-', linewidth=1.5)

    for i, label in enumerate(
        ['lower R', 'bridge', 'lower L', 'front L', 'front R', 'rear R', 'rear L']
    ):
        ax3.text(xa[i], ya[i], za[i], f'  {label}', fontsize=10, color='k')

    _format_ax(ax3, R_display, 'ARIA')
    ax3.set_title('Aria Glasses Microphone Positions on a 0.11 m Sphere', fontsize=10, pad=2)

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, 'arrays_train.pdf'), bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, 'arrays_test.pdf'),  bbox_inches='tight')
        fig3.savefig(os.path.join(save_dir, 'aria.pdf'),          bbox_inches='tight')
        print(f"Figures saved to {save_dir}/")

    return fig1, fig2, fig3


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot_paper_arrays()
    plt.show()
