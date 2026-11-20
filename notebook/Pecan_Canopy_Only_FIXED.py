# ============================================================
# PECAN CANOPY + EVEN / ODD THINNING DIAGONALS — ANY ORCHARD IMAGE
# Canopy only (no grass/weeds, no soil, no shadow), one crown per planting
# position, EVEN / ODD diagonals, canopy per diagonal, thinning START.
#
# SAME RESULTS AS THE DASHBOARD: with ENGINE = "dashboard" (default) this
# notebook runs the Emmanuel Dashboard's own analysis engine (js/analyze.js,
# embedded at the end, run with Node.js). The same image + DSM give exactly
# the same crowns, canopy (to the m2) and diagonals here and on
# https://emmanueldashboard.github.io. Kaggle without Node.js: switch
# Internet on (notebook settings) so it can be installed (nodejs-bin).
# ENGINE = "python" runs the older full-resolution Python analysis (steps
# 1-10 below), whose numbers differ from the dashboard by about 1-2 %.
#
# INPUT: attach one or more drone orthomosaics (GeoTIFF, RGB) and, if you
# have them, the matching elevation rasters (DSM/DEM). Every orthomosaic is
# analysed and gets its own results ZIP; each one uses the elevation raster
# that covers it. Any file name, any projected or lat/long CRS, 8- or 16-bit,
# clipped to the field or not: the image is converted as needed.
#
# WHY A GREENNESS MASK FAILS
#   The old mask was "ExG = 2G-R-B >= 8" + Gaussian support. Every green
#   pixel passes that test: inter-row grass, weeds and the field-edge lawn
#   became "crowns", while self-shaded crown halves (dark, low ExG) were
#   dropped. Colour indices separate vegetation from soil; they cannot
#   separate tree canopy from ground vegetation.
#
# WHAT THIS VERSION DOES (RGB only; optional DSM/height upgrade)
#   1. Per-pixel colour + texture features at native resolution.
#      Texture = local std of log-luminance at 0.25/0.5/1.0 m: pecan crowns
#      are rough (leaf clusters + self-shadow); grass and soil are smooth.
#   2. Training pixels are harvested automatically from the image itself with
#      adaptive (histogram-valley) thresholds:
#         canopy      = interior of large textured green blobs, own pixel green
#         ground veg  = green but smooth, >= 1 m from textured blobs
#         soil/road   = bright, not green
#         shadow      = dark and colour-neutral (skylight-lit)
#   3. A gradient-boosted classifier learns P(canopy) from those pixels.
#   4. P(canopy) >= 0.5, light morphology, enclosed gaps < 4 m2 filled.
#   5. DSM (auto-detected): terrain is modelled from RGB ground
#      pixels; canopy height = DSM - terrain. Height decides WHAT is a
#      tree (patch 90th pct >= 1 m; nothing > 0.5 m beyond height-
#      confirmed canopy). RGB decides WHERE the crown edge is, because
#      photogrammetric DSMs smear crown edges.
#   6. Clear cast shadow is trimmed from crown edges: dark AND not
#      leaf-green AND at ground level (DSM). Shaded leaves stay canopy.
#   7. Crowns are split by marker-controlled watershed: treetops from
#      the canopy height model (or distance maxima if RGB only).
#
# EVEN / ODD DIAGONALS (measured from the trees, not fixed lines)
#   8. The planting grid (in-row spacing, row spacing, row direction) is
#      measured from the treetops. Diagonal thinning removes "alternate
#      trees in each row and column", a checkerboard on that grid, so each
#      crown's diagonal comes from its grid position.
#   9. Along every row neighbouring crowns always alternate (1 blue,
#      2 red, 3 blue ...), also across a missing tree.
#  10. Each diagonal is a smooth curve through the treetops on it, so it
#      crosses the top of every crown on it, also where rows bend.
#
# Raw detections are preserved separately. START + arrow: first diagonal of the family with the smaller canopy (D01 or D02).
# ============================================================

from pathlib import Path
import sys
import importlib.util
import subprocess

# ==================== SETTINGS ====================

# ---------- Engine ----------
# "dashboard": the analysis engine of the Emmanuel Dashboard website
#   (https://emmanueldashboard.github.io, js/analyze.js, run here with
#   Node.js). The results are IDENTICAL to analysing the same image and DSM
#   in the dashboard (same crowns, same canopy to the m2, same diagonals).
# "python": the full-resolution Python analysis below (finer crown edges;
#   its numbers differ from the dashboard by about 1-2 %).
ENGINE = "dashboard"
# Detail of the dashboard engine, as in the dashboard's Analyse window:
# "fast", "standard" (the dashboard's default) or "fine".
DASHBOARD_DETAIL = "standard"

# Where the drone images are (all sub-folders are searched). On Kaggle:
# the attached datasets. Elsewhere: the current folder.
DATASET_DIR = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()

# The orthomosaic to analyse. A file name alone is searched for in
# DATASET_DIR and all its sub-folders (here: the Hahira field, clipped to
# the orchard). None = analyse EVERY orthomosaic found in DATASET_DIR (one
# results ZIP each). Or a full path, or a list of names / paths.
TIFF_PATH = "Pecan_Inside_Red_Boundary.tif"
OUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "pecan_results"

# ---------- Canopy-only detection (RGB) ----------
# Texture scales (Gaussian sigma, metres) for local std of log-luminance.
TEXTURE_SCALES_M = (0.25, 0.50, 1.00)

# Crown cores used as canopy training pixels: textured green blobs of at
# least MIN_CORE_BLOB_M2, eroded by CORE_EROSION_M.
MIN_CORE_BLOB_M2 = 4.0
CORE_EROSION_M = 0.60

# Ground-vegetation training pixels must be this far from textured blobs.
GROUND_VEG_MIN_DISTANCE_M = 1.0

TRAIN_PIXELS_PER_CLASS = 60000
RANDOM_SEED = 0

# Decision boundary on P(canopy). 0.50 is the unbiased choice. Raising it
# moves every crown edge inward (see canopy_probability_sensitivity.csv).
BOUNDARY_PROBABILITY = 0.50

PROBABILITY_SMOOTHING_M = 0.20
CLOSING_RADIUS_M = 0.25
OPENING_RADIUS_M = 0.20
MAX_FILLED_GAP_M2 = 4.0
MIN_CANOPY_PATCH_M2 = 2.0

# ---------- Elevation: DSM (strongly recommended) ----------
# "auto": for each orthomosaic, the single-band elevation raster in
#         DATASET_DIR that covers it (file name contains dem/dsm/elev/
#         height/chm/surface; finest resolution wins). A raster that shows
#         no tree heights (e.g. a terrain-only DTM) is ignored.
# None:   RGB only.   Or give an explicit path to the GeoTIFF.
ELEVATION_PATH = "auto"
ELEVATION_IS_HEIGHT = "auto"   # True = raster is already height above ground
MIN_TREE_HEIGHT_M = 1.0        # a canopy patch must reach this (90th pct.)
# Photogrammetric DSMs smear crown edges, so the crown EDGE comes from RGB:
# RGB canopy is kept up to this distance beyond height-confirmed canopy.
RGB_EDGE_TOLERANCE_M = 0.5
# Terrain model = DSM on pixels the RGB classifier calls ground.
GROUND_PROBABILITY_MAX = 0.2
GROUND_CLEARANCE_M = 0.6
# Crown splitting with height: treetops = local maxima of the smoothed CHM.
TREETOP_SMOOTHING_M = 0.5
TREETOP_SEPARATION_M = 2.5     # keep below half the in-row tree spacing

# ---------- Cast shadow (trimmed from the crown edges) ----------
# Shadow that the trees cast on the ground is dark, like the shaded side of
# a crown, but it lies at ground level and is less green. Only clear
# shadow is trimmed: a canopy pixel is removed when it is darker than
# SHADOW_LUMA ("auto" = the field's 30th brightness percentile) AND at
# ground level in the DSM (below SHADOW_MAX_HEIGHT_M) AND less green than
# halfway between ground shadow and shaded leaves. Without a DSM: darker
# than the field's typical ground shadow AND less green. Shaded leaves
# (high and green) stay canopy. This changes canopy area by about 2 %.
REMOVE_CAST_SHADOW = True
SHADOW_LUMA = "auto"
SHADOW_MAX_HEIGHT_M = 0.5
SHADOW_SMOOTHING_M = 0.25
# Gaps between leaves fully inside a crown, smaller than this, stay crown.
SHADOW_HOLE_FILL_M2 = 1.0

# ---------- Crown splitting ----------
SEED_SEPARATION_M = 3.0
SEED_HEIGHT_M = 0.35

MAX_CANDIDATES = 200000
# Larger images are resampled to this many pixels (about 0.06 m per pixel
# for 20 ha); finer detail does not improve crown outlines.
MAX_NATIVE_PIXELS = 60000000

# ---------- EVEN / ODD diagonals (measured from the treetops) ----------
# Planting design (ft): (between trees in the row, between rows). The grid
# steps are searched near these lengths, then fitted to the trees. If the
# trees do not match (another design), or with None, the spacing is
# measured from the trees alone.
DESIGN_SPACING_FT = (20.0, 40.0)
# Row direction and diagonal angle are measured from the trees.
# This only picks WHICH of the two diagonal directions to draw: the one
# closest to this image angle (90-180 = upper-right to lower-left).
DIAGONAL_DIRECTION_HINT_DEG = 114.0
# A treetop joins the grid if it lies within this share of a grid step of
# where its neighbours place it (it may move up to 0.4 before it is OFF_GRID).
GRID_TOLERANCE_FRACTION = 0.30
# A separately planted block (shifted grid) needs this many trees.
MIN_BLOCK_TREES = 25
# One crown per planting position: canopy covering a position with no
# detected treetop becomes that position's crown, so touching crowns along
# a row alternate EVEN / ODD (flag INFERRED_FROM_SPACING).
SPLIT_COVERED_POSITIONS = True
# True: along every row, crowns next to each other always have opposite
# colours (1 blue, 2 red, 3 blue ...), also across a missing tree: the
# crowns on one side of the gap move to the neighbouring diagonal (flag
# MOVED_ACROSS_GAP; per row the side with fewer crowns moves). False: the
# colour comes from the grid position only, so the two neighbours of a
# missing tree share a colour.
ALTERNATE_ALONG_ROWS = True
# True: young (replanted) pecan trees are counted. Their crowns are too small
# for MIN_CROWN_AREA_M2 and too low for the DSM height rule, so a small,
# round, sunlit crown as green as the pecan leaves, standing on an EMPTY
# planting position of a row on bare ground, is counted as a young tree (flag
# YOUNG_TREE). Grass, weeds and shadow are not on a planting position, or
# not round, sunlit and standing out from bare ground, and are not counted.
# (ENGINE = "dashboard", as in the dashboard.)
YOUNG_TREES = True

# Combined EVEN + ODD limit (safety cap; reported if it binds).
MAX_RETAINED_CROWNS = 100000

# Provisional minimum region area.
# This is not a validated tree-size threshold.
MIN_CROWN_AREA_M2 = 2.0

MAX_ANNOTATED_CROWNS = MAX_RETAINED_CROWNS

MAP_DPI = 300
DISPLAY_MAPS = True

# Where thinning starts. "auto": the family (EVEN or ODD) with the SMALLER
# total canopy is removed and thinning starts on its first diagonal (D01
# when ODD is smaller, D02 when EVEN is smaller), as in the dashboard.
# Or a diagonal number, e.g. 2 for D02 (EVEN) or 1 for D01 (ODD).
START_DIAGONAL = "auto"

MAKE_ZIP = True
INSTALL_MISSING = True


# ==================== DEPENDENCIES ====================

packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "rasterio": "rasterio",
    "shapely": "shapely",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "pyproj": "pyproj",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
}

missing = [
    package
    for module, package in packages.items()
    if importlib.util.find_spec(module) is None
]

if missing:
    if not INSTALL_MISSING:
        raise ImportError(
            "Missing packages: " + ", ".join(missing)
        )

    print("Installing:", ", ".join(missing), flush=True)

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        *missing,
    ])

import json
import math
import time
import shutil
import traceback

from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import cv2

from rasterio.crs import CRS as RasterCRS
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import (
    breadth_first_order,
    connected_components,
    minimum_spanning_tree,
)
from scipy.spatial import cKDTree

from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from sklearn.ensemble import HistGradientBoostingClassifier

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from shapely.ops import transform as geom_transform
from shapely.prepared import prep

from pyproj import CRS, Transformer

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import patheffects as pe

START_TIME = time.time()


# ==================== HELPERS ====================

def log(message):
    elapsed = (time.time() - START_TIME) / 60
    print(f"[{elapsed:6.1f} min] {message}", flush=True)


# Areas are measured in m2 and reported in square feet (1 m2 = 10.764 ft2).
FT2_PER_M2 = 1 / 0.3048 ** 2


def save_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parts(geometry, kind):
    if geometry.is_empty:
        return []

    if geometry.geom_type == kind:
        return [geometry]

    return [
        part
        for child in getattr(geometry, "geoms", [])
        for part in parts(child, kind)
    ]


def show_file(path):
    try:
        from IPython.display import display, Image
        display(Image(filename=str(path), width=1200))
    except ImportError:
        print(path)


def is_output(path):
    """Files written by this notebook (never analysed as inputs)."""
    try:
        return Path(OUTPUT_ROOT).resolve() in Path(path).resolve().parents
    except OSError:
        return False


def raster_kind(path):
    """'image' (RGB orthomosaic), 'elevation' (one band) or None."""
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            if src.count >= 3 and src.dtypes[0] in ("uint8", "uint16"):
                return "image"
            if src.count == 1:
                return "elevation"
    except Exception:
        return None
    return None


def find_images():
    """Orthomosaics to analyse: TIFF_PATH (one path or a list), otherwise
    every RGB GeoTIFF in DATASET_DIR and its sub-folders."""
    if TIFF_PATH is not None:
        given = TIFF_PATH if isinstance(TIFF_PATH, (list, tuple)) else [TIFF_PATH]
        paths = []
        for path in map(Path, given):
            if not path.is_file() and path.name == str(path):  # name only
                found = sorted(
                    candidate for candidate in Path(DATASET_DIR).rglob(path.name)
                    if candidate.is_file() and not is_output(candidate)
                )
                if found:
                    path = found[0]
            if not path.is_file():
                attached = sorted(
                    candidate.name for candidate in Path(DATASET_DIR).rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.lower() in (".tif", ".tiff")
                    and not is_output(candidate)
                )
                raise FileNotFoundError(
                    f"{path} not found in {DATASET_DIR}. GeoTIFFs attached: "
                    + (", ".join(attached) or "none")
                    + ". Set TIFF_PATH to one of them (or None for all)."
                )
            paths.append(path)
    else:
        paths = [
            path for path in sorted(Path(DATASET_DIR).rglob("*"))
            if path.is_file()
            and path.suffix.lower() in (".tif", ".tiff")
            and not is_output(path)
            and raster_kind(path) == "image"
        ]
        if not paths:
            raise RuntimeError(
                f"No RGB orthomosaic (GeoTIFF with 3 or more bands) found in "
                f"{DATASET_DIR}. Attach one, or set TIFF_PATH."
            )

    if (
        ELEVATION_PATH is not None
        and str(ELEVATION_PATH).lower() != "auto"
        and not Path(ELEVATION_PATH).is_file()
    ):
        raise FileNotFoundError(ELEVATION_PATH)

    return paths


WEB_MERCATOR = {3857, 3785, 900913, 102100, 102113}


def utm_crs_for(src):
    """WGS 84 / UTM zone of the image centre."""
    left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    lon, lat = (left + right) / 2, (bottom + top) / 2
    zone = int((lon + 180) // 6) % 60 + 1
    return RasterCRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def prepare_image(path, work):
    """The orthomosaic as the analysis needs it: 8-bit RGB with a mask, in
    metres (UTM if it was lat/long, Web Mercator or feet), square pixels, at
    most MAX_NATIVE_PIXELS. Returns (path to use, list of notes, whether the
    file had its own field mask: alpha band or nodata)."""
    notes = []

    with rasterio.open(path) as src:
        has_mask = any(
            flag in (MaskFlags.per_dataset, MaskFlags.alpha, MaskFlags.nodata)
            for flag in src.mask_flag_enums[0]
        )
        crs = CRS(src.crs)
        in_metres = crs.is_projected and all(
            abs(axis.unit_conversion_factor - 1) < 1e-8 for axis in crs.axis_info[:2]
        )
        target_crs = src.crs
        if not in_metres or crs.to_epsg() in WEB_MERCATOR:
            target_crs = utm_crs_for(src)
            notes.append(f"reprojected from {crs.to_string()} to {target_crs.to_string()} (true metres)")

        interp = list(src.colorinterp)
        rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        bands = [interp.index(c) + 1 for c in rgb] if all(c in interp for c in rgb) else [1, 2, 3]
        eight_bit = all(src.dtypes[b - 1] == "uint8" for b in bands)
        t = src.transform
        px, py = math.hypot(t.a, t.d), math.hypot(t.b, t.e)
        square = (
            abs(px - py) <= 0.01 * max(px, py)
            and abs(t.a * t.b + t.d * t.e) <= 0.01 * px * py
        )

        if (
            target_crs == src.crs and eight_bit and square
            and bands == [1, 2, 3] and tuple(interp[:3]) == rgb
            and src.width * src.height <= MAX_NATIVE_PIXELS
        ):
            return Path(path), notes, has_mask  # already fine: used as it is

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        resolution = abs(transform.a)
        if width * height > MAX_NATIVE_PIXELS:
            resolution *= math.sqrt(width * height / MAX_NATIVE_PIXELS) * 1.01
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds,
                resolution=resolution,
            )
            notes.append(f"resampled to {resolution:.3f} m per pixel (MAX_NATIVE_PIXELS)")
        if not eight_bit:
            notes.append(f"converted from {src.dtypes[bands[0] - 1]} to 8-bit")

        with WarpedVRT(
            src, crs=target_crs, transform=transform, width=width, height=height,
            resampling=Resampling.average if resolution > 1.2 * px else Resampling.bilinear,
            add_alpha=True,
        ) as vrt:
            data = vrt.read(bands)
            mask = vrt.dataset_mask() > 0

    if not eight_bit:
        stretched = np.zeros(data.shape, dtype="uint8")
        for k in range(data.shape[0]):
            values = data[k][mask]
            low, high = (np.percentile(values[:: max(1, values.size // 2_000_000)], [1, 99]) if values.size else (0, 1))
            stretched[k] = np.clip((data[k] - low) / max(high - low, 1e-9) * 255, 0, 255)
        data = stretched

    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{Path(path).stem}_prepared.tif"
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 4, "crs": target_crs,
        "transform": transform, "width": width, "height": height,
        "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data.astype("uint8"), [1, 2, 3])
        dst.write(np.where(mask, 255, 0).astype("uint8"), 4)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]

    notes.append(f"working copy {width} x {height} pixels")
    return out, notes, has_mask


def field_without_mask(src):
    """No mask in the original file: the field is the image minus uniform
    black or white areas connected to the image edge (fill around a field
    that was clipped without transparency)."""
    rgb = src.read([1, 2, 3])
    empty = np.zeros(rgb.shape[1:], dtype=bool)

    for flat in ((rgb <= 3).all(axis=0), (rgb >= 252).all(axis=0)):
        labels, count = ndi.label(flat)
        if count:
            edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            empty |= np.isin(labels, edge[edge > 0])

    return ~empty


def validate(src):
    if src.count < 3 or any(
        dtype != "uint8" for dtype in src.dtypes[:3]
    ):
        raise ValueError("An 8-bit RGB GeoTIFF is required.")

    if tuple(src.colorinterp[:3]) != (
        ColorInterp.red,
        ColorInterp.green,
        ColorInterp.blue,
    ):
        raise ValueError("Bands 1, 2 and 3 must be RGB.")

    if src.crs is None or not CRS(src.crs).is_projected:
        raise ValueError("Use a projected CRS in metres.")

    if any(
        abs(axis.unit_conversion_factor - 1) > 1e-8
        for axis in CRS(src.crs).axis_info[:2]
    ):
        raise ValueError("CRS units must be metres.")

    transform = src.transform
    px = math.hypot(transform.a, transform.d)
    py = math.hypot(transform.b, transform.e)

    if (
        abs(px - py) > 0.01 * max(px, py)
        or abs(
            transform.a * transform.b
            + transform.d * transform.e
        ) > 0.01 * px * py
    ):
        raise ValueError("Use square orthogonal pixels.")

    if src.width * src.height > MAX_NATIVE_PIXELS:
        raise MemoryError(
            "Raster exceeds MAX_NATIVE_PIXELS."
        )

    if not 0.05 <= BOUNDARY_PROBABILITY <= 0.95:
        raise ValueError(
            "BOUNDARY_PROBABILITY must be between 0.05 and 0.95."
        )


# ==================== CANOPY-ONLY MASK ====================

CLASS_NAMES = {
    0: "pecan_canopy",
    1: "ground_vegetation",
    2: "soil_road_dry_grass",
    3: "shadow",
}

FEATURE_NAMES = [
    "r", "g", "b", "Y", "exg", "Lstar", "astar", "bstar",
    "t025", "t050", "t100", "ll_m1", "ll_m3", "reldark",
    "exg_s05", "exg_s15", "bstar_s05", "astar_s05",
]


def gaussian(image, sigma_px):
    if sigma_px <= 0:
        return image

    size = int(2 * math.ceil(3 * sigma_px) + 1)

    return cv2.GaussianBlur(
        image,
        (size, size),
        sigma_px,
        borderType=cv2.BORDER_REFLECT,
    )


def masked_gaussian(image, weight, sigma_px, weight_blur=None):
    """Normalized convolution: average only inside the field."""
    if weight_blur is None:
        weight_blur = gaussian(weight, sigma_px)

    return gaussian(image * weight, sigma_px) / np.maximum(
        weight_blur, 1e-3
    )


def disk(radius_px):
    radius = max(1, int(round(radius_px)))

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )


def drop_small(mask, min_pixels):
    components, count = ndi.label(mask)

    if count == 0:
        return mask

    sizes = np.bincount(components.ravel())
    keep = sizes >= min_pixels
    keep[0] = False

    return keep[components]


def histogram_valley(values, low, high, bins=60):
    """Lowest point of the smoothed histogram between its two main modes."""
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    smooth = np.convolve(counts, np.ones(5) / 5, mode="same")
    centres = 0.5 * (edges[1:] + edges[:-1])

    peaks = [
        i for i in range(1, bins - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
    ]

    if len(peaks) < 2:
        return None

    first, second = sorted(
        sorted(peaks, key=lambda i: smooth[i])[-2:]
    )

    if second - first < 3:
        return None

    return float(
        centres[first + int(np.argmin(smooth[first:second + 1]))]
    )


def canopy_features(rgb, valid, pixel_m):
    """Colour and multi-scale texture features (float32, native grid)."""
    weight = valid.astype("float32")

    red, green, blue = [
        rgb[..., i].astype("float32") for i in range(3)
    ]

    total = red + green + blue + 1.0
    r, g, b = red / total, green / total, blue / total
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    log_luma = np.log1p(luma).astype("float32")

    del red, green, blue, total

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype("float32")

    features = {
        "r": r.astype("float32"),
        "g": g.astype("float32"),
        "b": b.astype("float32"),
        "Y": luma.astype("float32"),
        "exg": (2 * g - r - b).astype("float32"),
        "Lstar": lab[..., 0] * (100.0 / 255.0),
        "astar": lab[..., 1] - 128.0,
        "bstar": lab[..., 2] - 128.0,
    }

    del lab, r, g, b, luma

    for name, sigma_m in zip(
        ("t025", "t050", "t100"),
        TEXTURE_SCALES_M,
    ):
        sigma = sigma_m / pixel_m
        weight_blur = gaussian(weight, sigma)
        mean = masked_gaussian(log_luma, weight, sigma, weight_blur)
        mean_sq = masked_gaussian(
            log_luma * log_luma, weight, sigma, weight_blur
        )
        features[name] = np.sqrt(
            np.maximum(mean_sq - mean * mean, 0)
        ).astype("float32")

        if name == "t100":
            features["ll_m1"] = mean.astype("float32")

    features["ll_m3"] = masked_gaussian(
        log_luma, weight, 3.0 / pixel_m
    ).astype("float32")

    features["reldark"] = (
        features["ll_m3"] - log_luma
    ).astype("float32")

    for name, source, sigma_m in (
        ("exg_s05", "exg", 0.5),
        ("exg_s15", "exg", 1.5),
        ("bstar_s05", "bstar", 0.5),
        ("astar_s05", "astar", 0.5),
    ):
        features[name] = masked_gaussian(
            features[source], weight, sigma_m / pixel_m
        ).astype("float32")

    return features


def harvest_training_pixels(features, valid, pixel_m):
    """High-confidence class samples taken from the image itself."""
    pixel_area = pixel_m * pixel_m

    texture = features["t050"]
    texture_wide = features["t100"]
    luma = features["Y"]
    own_exg = gaussian(features["exg"], 1.0)
    own_astar = gaussian(features["astar"], 1.0)
    exg_local = features["exg_s05"]

    green_loose = valid & (exg_local > 0.05)
    texture_sample = texture[green_loose][::7]

    if texture_sample.size < 1000:
        raise RuntimeError(
            "Too few green pixels to learn pecan canopy."
        )

    texture_otsu = float(threshold_otsu(texture_sample))
    texture_valley = histogram_valley(texture_sample, 0.0, 0.8)

    if texture_valley is None or not (
        0.05 < texture_valley < texture_otsu
    ):
        texture_valley = 0.5 * texture_otsu

    field_luma = luma[valid]
    dark_luma = float(np.percentile(field_luma, 30))
    bright_luma = float(np.percentile(field_luma, 50))

    dark = valid & (luma < dark_luma)
    dark_exg_valley = histogram_valley(own_exg[dark][::5], -0.1, 0.4)

    if dark_exg_valley is None or not (
        0.03 < dark_exg_valley < 0.25
    ):
        dark_exg_valley = 0.12

    textured_green = (
        green_loose & (texture > texture_valley)
    ).astype("uint8")

    textured_green = cv2.morphologyEx(
        textured_green, cv2.MORPH_OPEN, disk(0.3 / pixel_m)
    ).astype(bool)

    textured_green = drop_small(
        textured_green,
        int(MIN_CORE_BLOB_M2 / pixel_area),
    )

    core = cv2.erode(
        textured_green.astype("uint8"),
        disk(CORE_EROSION_M / pixel_m),
    ).astype(bool)

    distance_to_texture = ndi.distance_transform_edt(
        ~textured_green
    ) * pixel_m

    own_green = (
        (own_exg > dark_exg_valley + 0.01)
        & (own_astar < -5.0)
    )

    own_neutral = (
        (own_exg < dark_exg_valley - 0.04)
        & (own_astar > -4.5)
    )

    seeds = np.full(luma.shape, -1, dtype="int8")

    # 1 ground vegetation: green, smooth, away from any textured blob.
    seeds[
        valid
        & (own_exg > 0.08)
        & (exg_local > 0.06)
        & (texture < 0.85 * texture_valley)
        & (texture_wide < texture_otsu)
        & (luma > dark_luma)
        & (distance_to_texture > GROUND_VEG_MIN_DISTANCE_M)
    ] = 1

    # 2 soil, road, dry grass: bright and not green (also next to crowns).
    seeds[
        valid & (own_exg < 0.03) & (luma > bright_luma) & ~core
    ] = 2

    # 3 shadow: dark and colour-neutral (also next to crowns).
    seeds[
        valid & (luma < dark_luma) & own_neutral & ~core
    ] = 3

    # 0 canopy: crown core AND the pixel itself is green and textured.
    seeds[
        core & valid & own_green & (texture > texture_valley)
    ] = 0

    thresholds = {
        "texture_otsu": texture_otsu,
        "texture_valley": float(texture_valley),
        "dark_luma_p30": dark_luma,
        "bright_luma_p50": bright_luma,
        "dark_exg_valley": float(dark_exg_valley),
    }

    counts = {
        CLASS_NAMES[k]: int(np.count_nonzero(seeds == k))
        for k in CLASS_NAMES
    }

    return seeds, thresholds, counts


def train_canopy_classifier(features, seeds):
    rng = np.random.default_rng(RANDOM_SEED)
    rows, labels = [], []

    for class_id in CLASS_NAMES:
        index = np.flatnonzero(seeds.ravel() == class_id)

        if index.size == 0:
            continue

        pick = rng.choice(
            index,
            size=min(TRAIN_PIXELS_PER_CLASS, index.size),
            replace=False,
        )

        rows.append(np.stack(
            [features[name].ravel()[pick] for name in FEATURE_NAMES],
            axis=1,
        ))
        labels.append(np.full(pick.size, class_id))

    if not any((label == 0).any() for label in labels):
        raise RuntimeError(
            "No pecan-canopy training pixels were found. "
            "Check that the TIFF is the RGB orchard ortho."
        )

    if len(labels) < 2:
        raise RuntimeError("Need at least two classes to train.")

    x = np.concatenate(rows)
    y = np.concatenate(labels)

    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_SEED,
    )

    classifier.fit(x, y)

    return classifier, float(classifier.score(x, y))


def predict_canopy_probability(classifier, features, valid, seeds_info):
    """P(canopy) for every field pixel; obvious bright soil is skipped."""
    canopy_column = int(
        np.flatnonzero(classifier.classes_ == 0)[0]
    )

    # Same rule as the soil training class: bright and not green.
    obvious_soil = (
        (gaussian(features["exg"], 1.0) < 0.03)
        & (features["Y"] > seeds_info["bright_luma_p50"])
        & (features["t050"] < seeds_info["texture_valley"])
    )

    todo = np.flatnonzero((valid & ~obvious_soil).ravel())
    probability = np.zeros(valid.size, dtype="float32")

    chunk = 2_000_000

    for start in range(0, todo.size, chunk):
        index = todo[start:start + chunk]

        block = np.stack(
            [features[name].ravel()[index] for name in FEATURE_NAMES],
            axis=1,
        )

        probability[index] = classifier.predict_proba(
            block
        )[:, canopy_column]

    return probability.reshape(valid.shape), int(todo.size)


def regularize_canopy(probability, valid, pixel_m):
    pixel_area = pixel_m * pixel_m

    smooth = gaussian(
        probability, PROBABILITY_SMOOTHING_M / pixel_m
    )

    sensitivity = [
        {
            "boundary_probability": t,
            "canopy_area_m2_before_morphology": float(
                np.count_nonzero((smooth >= t) & valid) * pixel_area
            ),
            "used": bool(abs(t - BOUNDARY_PROBABILITY) < 1e-9),
        }
        for t in sorted(set([
            0.40, 0.50, 0.60, 0.70, float(BOUNDARY_PROBABILITY)
        ]))
    ]

    mask = ((smooth >= BOUNDARY_PROBABILITY) & valid).astype("uint8")

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, disk(CLOSING_RADIUS_M / pixel_m)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, disk(OPENING_RADIUS_M / pixel_m)
    ).astype(bool)

    holes = ndi.binary_fill_holes(mask) & ~mask
    holes &= ~drop_small(holes, int(MAX_FILLED_GAP_M2 / pixel_area))
    mask |= holes

    mask = drop_small(
        mask, int(MIN_CANOPY_PATCH_M2 / pixel_area)
    ) & valid

    return mask, sensitivity


def resolve_elevation(src):
    """Path of the DSM/CHM to use, or None for RGB only."""
    if ELEVATION_PATH is None:
        return None

    if str(ELEVATION_PATH).lower() != "auto":
        return Path(ELEVATION_PATH)

    ortho = Path(src.name).resolve()
    keys = ("dem", "dsm", "elev", "height", "chm", "surface")
    found = set()

    for root in (Path("/kaggle/input"), Path(DATASET_DIR)):
        if not root.exists():
            continue

        for path in root.rglob("*"):
            name = path.name.lower()
            if (
                path.is_file()
                and path.suffix.lower() in (".tif", ".tiff")
                and any(key in name for key in keys)
                and path.resolve() != ortho
                and not is_output(path)
            ):
                found.add(path.resolve())

    box = src.bounds
    field_box_area = (box.right - box.left) * (box.top - box.bottom)
    usable = []

    for path in sorted(found):
        try:
            with rasterio.open(path) as elevation:
                if elevation.crs is None or elevation.count != 1:
                    continue

                left, bottom, right, top = transform_bounds(
                    elevation.crs, src.crs, *elevation.bounds
                )
                overlap = (
                    max(0.0, min(right, box.right) - max(left, box.left))
                    * max(0.0, min(top, box.top) - max(bottom, box.bottom))
                )

                # Must cover this field (rejects DEMs of other farms).
                if overlap < 0.5 * field_box_area:
                    continue

                density = elevation.width * elevation.height / max(
                    (right - left) * (top - bottom), 1e-9
                )
                usable.append((density, str(path)))
        except Exception:
            continue

    if not usable:
        return None

    # Finest resolution first.
    return Path(sorted(usable, reverse=True)[0][1])


def rgb_guided_terrain(surface, known, probability, rgb_mask, pixel_m):
    """Ground elevation under the trees.

    The DSM is sampled only where the RGB classifier sees ground
    (P(canopy) low, away from crowns); a robust 1 m median is then
    inpainted under the crowns by multi-scale normalized convolution.
    """
    import warnings

    clearance = ndi.distance_transform_edt(~rgb_mask) * pixel_m
    ground = (
        known
        & (probability < GROUND_PROBABILITY_MAX)
        & (clearance >= GROUND_CLEARANCE_M)
    )
    del clearance

    if np.count_nonzero(ground) < 1000:
        raise RuntimeError("Too few ground pixels to model the terrain.")

    height, width = surface.shape
    step = max(1, int(round(1.0 / pixel_m)))
    rows = math.ceil(height / step) * step
    cols = math.ceil(width / step) * step

    cells = np.pad(
        np.where(ground, surface, np.nan),
        ((0, rows - height), (0, cols - width)),
        constant_values=np.nan,
    ).reshape(
        rows // step, step, cols // step, step
    ).transpose(0, 2, 1, 3).reshape(
        rows // step, cols // step, step * step
    )

    usable = np.isfinite(cells).sum(axis=2) >= 0.25 * step * step
    level = np.zeros(usable.shape, dtype="float32")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        level[usable] = np.nanmedian(cells[usable], axis=1)

    del cells

    weight = usable.astype("float32")
    terrain = np.full(usable.shape, np.nan, dtype="float32")

    for sigma_m in (1.5, 3.0, 6.0, 12.0, 24.0, 48.0):
        sigma = sigma_m / (step * pixel_m)
        support = gaussian(weight, sigma)
        estimate = gaussian(level * weight, sigma) / np.maximum(
            support, 1e-6
        )
        todo = ~np.isfinite(terrain) & (support > 0.15)
        terrain[todo] = estimate[todo]

    terrain[~np.isfinite(terrain)] = float(np.median(level[usable]))
    terrain = gaussian(terrain, 1.0 / (step * pixel_m))

    terrain = cv2.resize(
        terrain, (cols, rows), interpolation=cv2.INTER_LINEAR
    )[:height, :width]

    return terrain, float(ground.mean())


def label_percentiles(labels, count, values, percentiles):
    """Per-label percentiles of finite values. Returns {q: array[count+1]}."""
    finite = np.isfinite(values) & (labels > 0)
    flat_labels = labels[finite]
    flat_values = values[finite]
    order = np.argsort(flat_labels, kind="stable")
    flat_labels = flat_labels[order]
    flat_values = flat_values[order]
    bounds = np.searchsorted(flat_labels, np.arange(count + 2))

    result = {q: np.full(count + 1, np.nan) for q in percentiles}

    for label_id in range(1, count + 1):
        chunk = flat_values[bounds[label_id]:bounds[label_id + 1]]
        if chunk.size:
            for q, value in zip(
                percentiles, np.percentile(chunk, percentiles)
            ):
                result[q][label_id] = value

    return result


def elevation_refine(src, rgb_mask, probability, valid, pixel_m):
    """Use the DSM where it is reliable; keep RGB where it is reliable.

    * Height decides WHAT is a tree: canopy must be within
      RGB_EDGE_TOLERANCE_M of height-confirmed canopy (CHM >=
      MIN_TREE_HEIGHT_M), and every patch must reach MIN_TREE_HEIGHT_M.
    * RGB decides WHERE the crown edge is (DSMs smear crown edges).

    Returns (mask, info, chm or None, removed_by_height or None).
    """
    path = resolve_elevation(src)

    if path is None:
        return rgb_mask, {
            "used": False,
            "reason": "No DSM/CHM found (or ELEVATION_PATH = None).",
        }, None, None

    log(f"Elevation raster: {path}")

    surface = np.full(rgb_mask.shape, np.nan, dtype="float32")

    with rasterio.open(path) as elevation:
        reproject(
            source=rasterio.band(elevation, 1),
            destination=surface,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    known = np.isfinite(surface) & valid
    coverage = float(known.sum() / max(valid.sum(), 1))

    if coverage < 0.5:
        log(
            f"WARNING: {path.name} covers only {100 * coverage:.0f}% of this "
            "field. It is ignored: RGB only."
        )
        return rgb_mask, {
            "used": False,
            "path": str(path),
            "reason": f"covers only {100 * coverage:.0f}% of the field",
        }, None, None

    is_height = ELEVATION_IS_HEIGHT

    if str(is_height).lower() == "auto":
        is_height = bool(
            -1.0 <= float(np.percentile(surface[known], 5)) <= 1.0
        )

    if is_height:
        chm = surface
        ground_share = None
    else:
        log("Building the terrain model from RGB ground pixels.")
        terrain, ground_share = rgb_guided_terrain(
            surface, known, probability, rgb_mask, pixel_m
        )
        chm = surface - terrain
        del terrain

    del surface
    chm[~known] = np.nan

    # A raster with no tree heights under the canopy (a terrain-only DTM,
    # or one that is not aligned with the image) would delete every crown.
    under = chm[rgb_mask & np.isfinite(chm)]
    if under.size and float(np.percentile(under, 90)) < MIN_TREE_HEIGHT_M:
        log(
            f"WARNING: {path.name} shows no tree heights under the canopy "
            "(terrain-only DTM, or not aligned with the image?). It is "
            "ignored: RGB only."
        )
        return rgb_mask, {
            "used": False,
            "path": str(path),
            "reason": "no tree heights under the RGB canopy",
        }, None, None

    pixel_area = pixel_m * pixel_m

    # 1. RGB canopy near height-confirmed canopy (NaN compares False).
    confirmed = rgb_mask & (chm >= MIN_TREE_HEIGHT_M)
    near = cv2.dilate(
        confirmed.astype("uint8"),
        disk(RGB_EDGE_TOLERANCE_M / pixel_m),
    ).astype(bool)
    del confirmed

    mask = rgb_mask & (near | ~known)
    del near

    holes = ndi.binary_fill_holes(mask) & ~mask
    holes &= ~drop_small(holes, int(MAX_FILLED_GAP_M2 / pixel_area))
    mask |= holes

    mask = cv2.morphologyEx(
        mask.astype("uint8"),
        cv2.MORPH_OPEN,
        disk(OPENING_RADIUS_M / pixel_m),
    ).astype(bool)

    after_edge_rule = float(mask.sum() * pixel_area)

    # 2. Every canopy patch must be a tree (90th-percentile height).
    components, count = ndi.label(mask)
    p90 = label_percentiles(components, count, chm, [90])[90]
    keep = ~(p90 < MIN_TREE_HEIGHT_M)          # unknown height: keep
    keep[0] = False
    low_patches = int(count - keep[1:].sum())
    mask = keep[components]
    del components

    mask = drop_small(
        mask, int(MIN_CANOPY_PATCH_M2 / pixel_area)
    ) & valid

    removed = rgb_mask & ~mask

    info = {
        "used": True,
        "path": str(path),
        "field_coverage": coverage,
        "is_height_raster": bool(is_height),
        "terrain_model": (
            None if is_height else
            "DSM median on RGB-ground pixels (1 m cells), inpainted"
        ),
        "ground_pixel_share": ground_share,
        "min_tree_height_m": MIN_TREE_HEIGHT_M,
        "rgb_edge_tolerance_m": RGB_EDGE_TOLERANCE_M,
        "rgb_canopy_m2": float(rgb_mask.sum() * pixel_area),
        "after_edge_rule_m2": after_edge_rule,
        "low_patches_removed": low_patches,
        "final_canopy_m2": float(mask.sum() * pixel_area),
        "removed_by_height_m2": float(removed.sum() * pixel_area),
    }

    return mask, info, chm, removed


def remove_cast_shadow(rgb, mask, valid, chm, pixel_m, dark_luma):
    """Trim clear cast shadow (shadow on the ground) from the canopy.

    Cast shadow and the shaded side of a crown are both dark. They differ in
    height (shadow lies on the ground) and colour (shadow on soil or dry
    grass is less green than shaded leaves). A dark canopy pixel is removed
    only when it is BOTH at ground level in the DSM AND less green than
    halfway between the field's ground shadow and its dark canopy. Without
    a DSM it must be darker than the field's typical ground shadow and less
    green. This is deliberately gentle: only thin shadow slivers at crown
    edges go.

    Returns (mask, removed, info).
    """
    sigma = SHADOW_SMOOTHING_M / pixel_m
    red, green, blue = [rgb[..., i].astype("float32") for i in range(3)]
    luma = gaussian(0.299 * red + 0.587 * green + 0.114 * blue, sigma)
    exg = gaussian((2 * green - red - blue) / (red + green + blue + 1.0), sigma)
    del red, green, blue

    cut_luma = (
        float(dark_luma) if str(SHADOW_LUMA).lower() == "auto"
        else float(SHADOW_LUMA)
    )
    dark = luma < cut_luma

    ground_shadow = valid & ~mask & dark
    dark_canopy = mask & dark

    enough = (
        np.count_nonzero(ground_shadow) > 1000
        and np.count_nonzero(dark_canopy) > 1000
    )
    cut_exg = (
        0.5 * (
            float(np.median(exg[ground_shadow])) + float(np.median(exg[dark_canopy]))
        )
        if enough else 0.12
    )
    shadow = dark_canopy & (exg < cut_exg)
    del exg

    very_dark = None

    if chm is not None:
        with np.errstate(invalid="ignore"):
            shadow &= chm < SHADOW_MAX_HEIGHT_M  # unknown height (NaN): kept
    else:
        # RGB only: no height, so only pixels darker than the field's
        # typical ground shadow count.
        very_dark = (
            float(np.median(luma[ground_shadow])) if enough else 0.6 * cut_luma
        )
        shadow &= luma < very_dark
    del dark_canopy, dark, ground_shadow, luma

    pixel_area = pixel_m * pixel_m
    cleaned = cv2.morphologyEx(
        (mask & ~shadow).astype("uint8"),
        cv2.MORPH_OPEN,
        disk(OPENING_RADIUS_M / pixel_m),
    ).astype(bool)
    holes = ndi.binary_fill_holes(cleaned) & ~cleaned  # small leaf gaps
    holes &= ~drop_small(holes, int(SHADOW_HOLE_FILL_M2 / pixel_area))
    cleaned |= holes & mask
    del holes
    cleaned = drop_small(cleaned, int(MIN_CANOPY_PATCH_M2 / pixel_area)) & valid
    removed = mask & ~cleaned

    info = {
        "used": True,
        "dark_luma_cut": cut_luma,
        "very_dark_luma_cut_rgb_only": very_dark,
        "greenness_cut_exg": cut_exg,
        "max_height_m": SHADOW_MAX_HEIGHT_M if chm is not None else None,
        "canopy_before_m2": float(mask.sum() * pixel_area),
        "shadow_removed_m2": float(removed.sum() * pixel_area),
        "canopy_after_m2": float(cleaned.sum() * pixel_area),
        "rule": (
            "dark AND less green than the cut AND at ground level in the DSM "
            "(RGB only: darker than typical ground shadow); shaded leaves are "
            "high and green and stay canopy"
        ),
    }

    return cleaned, removed, info


def save_review_tiles(rgb, valid, mask, pixel_m, out, removed=None, shadow=None):
    """Four zoomed tiles: RGB with the canopy outline (visual check).

    Orange = RGB canopy removed by the height check (when a DSM is used).
    Purple = cast shadow taken out of the canopy.
    """
    height, width = mask.shape
    size = int(round(45.0 / pixel_m))
    step = max(size, 1)
    best = []

    for row in range(0, height - size, step):
        for col in range(0, width - size, step):
            window = (slice(row, row + size), slice(col, col + size))
            if valid[window].mean() > 0.95:
                best.append((mask[window].mean(), row, col))

    if not best:
        return None

    best.sort()
    picks = [best[int(q * (len(best) - 1))] for q in (0.2, 0.45, 0.7, 0.95)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout="constrained")

    for ax, (fraction, row, col) in zip(axes.ravel(), picks):
        tile = rgb[row:row + size, col:col + size].copy()

        if removed is not None:
            gone = removed[row:row + size, col:col + size]
            tile[gone] = (
                0.45 * tile[gone] + 0.55 * np.array([255, 150, 0])
            ).astype("uint8")

        if shadow is not None:
            gone = shadow[row:row + size, col:col + size]
            tile[gone] = (
                0.45 * tile[gone] + 0.55 * np.array([200, 0, 255])
            ).astype("uint8")

        outline = mask[row:row + size, col:col + size].astype("uint8")
        contours, _ = cv2.findContours(
            outline, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(tile, contours, -1, (255, 40, 40), 2)
        ax.imshow(tile)
        ax.set_title(
            f"rows {row}-{row + size}, cols {col}-{col + size} | "
            f"canopy {100 * fraction:.0f}%",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(
        "Pecan canopy (red outline), 45 m detail\n"
        + "; ".join(
            text for text, show in (
                ("orange = removed by height check", removed is not None),
                ("purple = shadow removed", shadow is not None),
            ) if show
        ),
        fontsize=14,
        weight="bold",
    )

    path = out / "canopy_review_tiles.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

    return path


def write_band(path, array, src, dtype, nodata):
    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    profile.pop("photometric", None)

    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array.astype(dtype), 1)


# ==================== DETECT RAW CANDIDATES ====================

def detect(src, valid_native, out):
    pixel_m = math.hypot(src.transform.a, src.transform.d)
    pixel_area = pixel_m * pixel_m
    valid = valid_native

    log(
        f"Reading native RGB at {pixel_m:.3f} m/pixel "
        f"({src.width} x {src.height})."
    )

    rgb = np.ascontiguousarray(
        src.read([1, 2, 3]).transpose(1, 2, 0)
    )

    log("Computing colour + texture features.")
    features = canopy_features(rgb, valid, pixel_m)

    log("Harvesting training pixels from the image (adaptive thresholds).")
    seeds, thresholds, seed_counts = harvest_training_pixels(
        features, valid, pixel_m
    )

    for name, count in seed_counts.items():
        log(f"  {name}: {count * pixel_area:,.0f} m2 of training pixels")

    log(
        f"  texture valley {thresholds['texture_valley']:.3f} | "
        f"shadow/shaded-leaf greenness valley "
        f"{thresholds['dark_exg_valley']:.3f}"
    )

    write_band(
        out / "canopy_training_pixels.tif",
        np.where(valid, (seeds + 1).astype("uint8"), np.uint8(255)),
        src, "uint8", 255,
    )

    log("Training the canopy classifier.")
    classifier, train_accuracy = train_canopy_classifier(features, seeds)
    del seeds

    log("Predicting P(canopy) for every field pixel.")
    probability, predicted_pixels = predict_canopy_probability(
        classifier, features, valid, thresholds
    )
    del features

    write_band(
        out / "canopy_probability.tif",
        np.where(valid, np.round(probability * 100), 255),
        src, "uint8", 255,
    )

    mask, sensitivity = regularize_canopy(probability, valid, pixel_m)

    pd.DataFrame(sensitivity).to_csv(
        out / "canopy_probability_sensitivity.csv",
        index=False,
    )

    log(f"RGB canopy (grass, soil and shadow excluded): "
        f"{mask.sum() * pixel_area:,.1f} m2")

    mask, elevation_info, chm, removed = elevation_refine(
        src, mask, probability, valid, pixel_m
    )
    del probability

    if chm is not None:
        log(
            f"Height check: removed {elevation_info['removed_by_height_m2']:,.0f} m2 "
            f"that is not tree ({elevation_info['low_patches_removed']} "
            f"low patches + ground beyond the crown edge)."
        )

        write_band(
            out / "canopy_height_model_cm.tif",
            np.where(
                valid & np.isfinite(chm),
                np.clip(np.round(np.nan_to_num(chm) * 100), -32000, 32000),
                -32768,
            ),
            src, "int16", -32768,
        )
    else:
        log("No elevation raster: RGB-only canopy and crown splitting.")

    shadow_removed = None
    shadow_info = {"used": False}

    if REMOVE_CAST_SHADOW:
        mask, shadow_removed, shadow_info = remove_cast_shadow(
            rgb, mask, valid, chm, pixel_m, thresholds["dark_luma_p30"]
        )
        log(
            f"Cast shadow trimmed from the crown edges: "
            f"{shadow_info['shadow_removed_m2']:,.0f} m2 (dark AND not "
            f"leaf-green AND at ground level)."
        )

    canopy_m2 = float(mask.sum() * pixel_area)

    log(f"Pecan canopy (final): {canopy_m2:,.1f} m2")

    write_band(
        out / "canopy_mask.tif",
        np.where(valid, mask.astype("uint8"), np.uint8(255)),
        src, "uint8", 255,
    )

    review = save_review_tiles(
        rgb, valid, mask, pixel_m, out, removed, shadow_removed
    )
    del removed, shadow_removed

    if DISPLAY_MAPS and review is not None:
        show_file(review)

    del rgb

    # ---------------- crown splitting (watershed) ----------------
    log("Splitting canopy into crowns (marker-controlled watershed).")

    components, component_count = ndi.label(mask)

    if chm is not None:
        # Treetops = local maxima of the smoothed canopy height model.
        score = gaussian(
            np.where(mask, np.nan_to_num(chm, nan=0.0), 0.0)
            .astype("float32"),
            TREETOP_SMOOTHING_M / pixel_m,
        )
        separation_m = TREETOP_SEPARATION_M
        peak_threshold = MIN_TREE_HEIGHT_M
    else:
        # RGB only: crown centres = maxima of distance to canopy edge.
        distance = ndi.distance_transform_edt(
            mask,
            sampling=pixel_m,
        ).astype("float32")
        score = gaussian(distance, 0.35 / pixel_m)
        separation_m = SEED_SEPARATION_M
        peak_threshold = SEED_HEIGHT_M
        del distance

    peaks = peak_local_max(
        score,
        min_distance=max(1, round(separation_m / pixel_m)),
        threshold_abs=peak_threshold,
        labels=mask.astype("uint8"),
        exclude_border=False,
    )

    markers = np.zeros(mask.shape, dtype="int32")
    markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)

    unseeded = np.setdiff1d(
        np.arange(1, component_count + 1),
        np.unique(components[markers > 0]),
    )

    if len(unseeded):
        positions = ndi.maximum_position(
            score,
            labels=components,
            index=unseeded,
        )

        for marker_id, (row, col) in enumerate(
            positions,
            start=len(peaks) + 1,
        ):
            markers[row, col] = marker_id

    total_candidates = int(markers.max())

    # Treetop of every crown = its watershed seed (row, col, seed score).
    # With the DSM this is the highest point of the crown; RGB only, it is
    # the crown centre. The EVEN/ODD diagonals pass through these points.
    seeds_rc = np.concatenate([
        peaks.reshape(-1, 2),
        np.asarray(positions if len(unseeded) else [], dtype=float).reshape(-1, 2),
    ]).astype(int)
    tops = np.column_stack([
        seeds_rc,
        score[seeds_rc[:, 0], seeds_rc[:, 1]] if len(seeds_rc) else [],
    ]).astype(float).reshape(-1, 3)

    log(
        f"Seeds: {len(peaks):,} peak-based + "
        f"{len(unseeded):,} small separate canopy patches."
    )

    if total_candidates > MAX_CANDIDATES:
        raise RuntimeError(
            f"{total_candidates:,} raw candidates exceed "
            "the safety limit. Review detection settings."
        )

    if total_candidates:
        labels = watershed(-score, markers, mask=mask).astype("int32")
    else:
        labels = markers

    assert np.array_equal(labels > 0, mask)

    del components, score, markers

    native_labels = labels
    native_labels[~valid] = 0

    # Per-crown tree height (same table later joins the inventory).
    if chm is not None and total_candidates:
        stats = label_percentiles(
            native_labels, total_candidates, chm, [50, 90, 99]
        )
        heights = pd.DataFrame({
            "label_id": np.arange(1, total_candidates + 1),
            "height_median_m": stats[50][1:],
            "height_p90_m": stats[90][1:],
            "height_top_m": stats[99][1:],
        })
        sums = np.bincount(
            native_labels.ravel(),
            weights=np.nan_to_num(chm).ravel(),
            minlength=total_candidates + 1,
        )
        counts = np.bincount(
            (native_labels * np.isfinite(chm)).ravel(),
            minlength=total_candidates + 1,
        )
        heights["height_mean_m"] = (
            sums[1:] / np.maximum(counts[1:], 1)
        )
        heights.to_csv(out / "crown_heights.csv", index=False)
        del stats, sums, counts

    del chm

    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype="int32",
        nodata=0,
        compress="deflate",
    )
    profile.pop("photometric", None)

    with rasterio.open(
        out / "candidate_labels.tif",
        "w",
        **profile,
    ) as destination:
        destination.write(native_labels, 1)

    diagnostics = {
        "method": (
            "RGB colour + multi-scale texture, self-trained "
            "gradient-boosted pixel classifier (canopy vs ground "
            "vegetation vs soil vs shadow), native resolution."
        ),
        "pixel_m": pixel_m,
        "canopy_area_m2": canopy_m2,
        "boundary_probability": BOUNDARY_PROBABILITY,
        "adaptive_thresholds": thresholds,
        "training_pixels_m2": {
            k: v * pixel_area for k, v in seed_counts.items()
        },
        "training_accuracy_on_seed_pixels": train_accuracy,
        "classifier_iterations": int(classifier.n_iter_),
        "pixels_classified": predicted_pixels,
        "elevation": elevation_info,
        "cast_shadow": shadow_info,
        "crown_markers": (
            f"canopy-height treetops (smoothing {TREETOP_SMOOTHING_M} m, "
            f"min spacing {TREETOP_SEPARATION_M} m)"
            if elevation_info.get("used")
            else f"distance-to-canopy-edge maxima (min spacing "
                 f"{SEED_SEPARATION_M} m)"
        ),
        "peak_seed_count": len(peaks),
        "patch_seed_count": len(unseeded),
        "note": (
            "Training pixels are high-confidence automatic labels, so "
            "training accuracy is not an accuracy estimate. Inspect "
            "canopy_review_tiles.png and canopy_mask.tif."
        ),
    }

    save_json(
        out / "segmentation_diagnostics.json",
        diagnostics,
    )

    return native_labels, len(peaks), diagnostics, tops


# ==================== GEOMETRY ====================

def create_geometry(src, labels, valid):
    log(
        "Converting coherent candidate regions "
        "to crown polygons."
    )

    groups = {}

    for geometry, label_id in shapes(
        labels,
        mask=labels > 0,
        transform=src.transform,
    ):
        groups.setdefault(
            int(label_id), []
        ).append(shape(geometry))

    crowns = [
        (label_id, unary_union(polygons))
        for label_id, polygons in sorted(groups.items())
    ]

    field = unary_union([
        shape(geometry)
        for geometry, value in shapes(
            valid.astype("uint8"),
            mask=valid,
            transform=src.transform,
        )
        if value == 1
    ])

    return crowns, field


# ==================== PLANTING GRID + EVEN/ODD DIAGONALS ====================
#
# Pecan orchards are planted on a grid, and diagonal thinning removes
# "alternate trees in each row and column" (UGA Extension B1314): a
# checkerboard on that grid. So each crown's diagonal is read from its grid
# position (not from a fixed straight line), and each diagonal is drawn as
# a smooth curve through the treetops on it.

UNPLACED = np.iinfo(np.int64).min


def measure_planting_grid(points):
    """Two grid steps (2x2, metres; columns = in-row step, row-to-row step).

    Peaks of the neighbour-displacement histogram give the two planting
    directions; a least-squares fit over all neighbour pairs refines them.
    """
    tree = cKDTree(points)
    nearest = float(np.median(tree.query(points, k=2)[0][:, 1]))
    radius = 3.2 * nearest
    pairs = tree.query_pairs(radius, output_type="ndarray")

    if len(points) < 20 or len(pairs) < 50:
        raise RuntimeError("Too few crowns to measure the planting grid.")

    shifts = points[pairs[:, 1]] - points[pairs[:, 0]]
    shifts = np.concatenate([shifts, -shifts])
    width = nearest / 24
    edges = np.arange(-radius, radius + width, width)
    counts = ndi.gaussian_filter(
        np.histogram2d(shifts[:, 0], shifts[:, 1], bins=[edges, edges])[0],
        1.2,
    )
    window = max(3, int(round(nearest / 3 / width)) | 1)
    peaks = np.argwhere(
        (counts == ndi.maximum_filter(counts, window))
        & (counts > 0.15 * counts.max())
    )
    centres = (edges[:-1] + edges[1:]) / 2
    found = [
        (np.array([centres[i], centres[j]]), counts[i, j])
        for i, j in peaks
        if math.hypot(centres[i], centres[j]) > 0.4 * nearest
    ]

    if not found:
        raise RuntimeError("No regular planting grid found in the treetops.")

    strongest = max(strength for _, strength in found)
    strong = [(v, s) for v, s in found if s >= 0.35 * strongest]

    def crossing(u, v):
        return abs(u[0] * v[1] - u[1] * v[0]) / (math.hypot(*u) * math.hypot(*v))

    a = b = None

    if DESIGN_SPACING_FT:
        # Known design (20 x 40 ft): strongest peaks within 10 % of those
        # lengths, the row-to-row step nearly square (> 75 deg) to the row.
        in_row_m, between_m = (0.3048 * value for value in DESIGN_SPACING_FT)
        near = [
            (v, s) for v, s in strong
            if abs(math.hypot(*v) - in_row_m) < 0.10 * in_row_m
        ]
        if near:
            a = max(near, key=lambda item: item[1])[0]
            near = [
                (v, s) for v, s in strong
                if abs(math.hypot(*v) - between_m) < 0.10 * between_m
                and crossing(a, v) > math.sin(math.radians(75))
            ]
            b = max(near, key=lambda item: item[1])[0] if near else None

    if a is None or b is None:
        ordered = sorted((v for v, _ in strong), key=lambda v: math.hypot(*v))
        a = ordered[0]
        b = next(
            (v for v in ordered[1:] if crossing(a, v) > math.sin(math.radians(20))),
            None,
        )

        if b is None:
            raise RuntimeError("Only one planting direction found in the treetops.")

    for _ in range(20):  # shortest (reduced) pair of grid steps
        if math.hypot(*b) < math.hypot(*a):
            a, b = b, a
        multiple = round(float(a @ b) / float(a @ a))
        if multiple == 0:
            break
        b = b - multiple * a

    basis = np.column_stack([a, b])
    pairs = tree.query_pairs(
        1.3 * max(math.hypot(*a), math.hypot(*b)), output_type="ndarray"
    )
    shifts = points[pairs[:, 1]] - points[pairs[:, 0]]

    for _ in range(4):
        steps = np.round(shifts @ np.linalg.inv(basis).T)
        error = np.hypot(*(shifts - steps @ basis.T).T)
        good = (
            (error < 0.25 * math.hypot(*basis[:, 0]))
            & (np.abs(steps).max(axis=1) <= 2)
            & (np.abs(steps).sum(axis=1) > 0)
        )
        basis = np.linalg.lstsq(steps[good], shifts[good], rcond=None)[0].T

    return basis, nearest


def orient_grid(basis, transform):
    """In-row step points down the image, row step points right."""
    linear = np.array([[transform.a, transform.b], [transform.d, transform.e]])
    pixel = np.linalg.solve(linear, basis)
    basis = basis.copy()

    if pixel[1, 0] < 0:
        basis[:, 0] *= -1
    if pixel[0, 1] < 0:
        basis[:, 1] *= -1

    return basis


def grid_positions(points, basis):
    """Planting block and integer grid position (tree, row) of each treetop.

    Neighbour links that match one grid step (within GRID_TOLERANCE_FRACTION
    of a step in each grid direction) are joined into a minimum spanning
    tree, best-fitting links first, and positions are passed along it, so
    gently curving rows are followed. A group separated by a gap joins a
    block only when nearby trees agree on its position; a large group that
    agrees with no block (planted separately, shifted grid) starts its own
    block. Off-grid treetops get block -1.
    """
    count = len(points)
    inverse = np.linalg.inv(basis)
    in_row = math.hypot(*basis[:, 0])
    link = 1.25 * max(in_row, math.hypot(*basis[:, 1]))
    tree = cKDTree(points)

    def fits(shift):
        units = shift @ inverse.T
        steps = np.round(units)
        return steps, np.abs(units - steps).max(axis=-1)

    pairs = tree.query_pairs(link, output_type="ndarray")
    steps, error = fits(points[pairs[:, 1]] - points[pairs[:, 0]])
    good = (
        (error < GRID_TOLERANCE_FRACTION)
        & (np.abs(steps).max(axis=1) <= 2)
        & (np.abs(steps).sum(axis=1) > 0)
    )
    graph = coo_matrix(
        (
            error[good] + 0.01 * np.abs(steps[good]).sum(axis=1) + 1e-6,
            (pairs[good, 0], pairs[good, 1]),
        ),
        shape=(count, count),
    ).tocsr()
    spanning = minimum_spanning_tree(graph)
    spanning = (spanning + spanning.T).tocsr()
    group_count, group = connected_components(spanning, directed=False)

    local = np.zeros((count, 2), dtype=np.int64)

    for g in range(group_count):
        nodes = np.flatnonzero(group == g)
        order, parent = breadth_first_order(
            spanning, nodes[0], directed=False, return_predecessors=True
        )
        for node in order[1:]:
            local[node] = local[parent[node]] + np.round(
                inverse @ (points[node] - points[parent[node]])
            ).astype(np.int64)

    sizes = np.bincount(group)
    positions = np.full((count, 2), UNPLACED, dtype=np.int64)
    block = np.full(count, -1)
    pending = [int(g) for g in np.argsort(-sizes, kind="stable")]
    blocks = 0

    while pending:
        changed = True

        while changed and blocks:
            changed = False
            placed = np.flatnonzero(block >= 0)
            placed_tree = cKDTree(points[placed])

            for g in list(pending):
                nodes = np.flatnonzero(group == g)
                votes = {}

                for u in nodes:
                    near = placed[
                        placed_tree.query_ball_point(points[u], 2.5 * link)
                    ]
                    if not len(near):
                        continue
                    shift = points[u] - points[near]
                    step, err = fits(shift)
                    for v, s, e, d in zip(near, step, err, np.hypot(*shift.T)):
                        if e < GRID_TOLERANCE_FRACTION:
                            key = (
                                int(block[v]),
                                *(positions[v] + s.astype(np.int64) - local[u]).tolist(),
                            )
                            votes[key] = votes.get(key, 0.0) + 1.0 / (1.0 + d / in_row)

                if not votes:
                    continue

                key = max(votes, key=votes.get)
                same_block = sum(w for k, w in votes.items() if k[0] == key[0])

                if votes[key] >= 0.6 * same_block:
                    positions[nodes] = local[nodes] + np.array(key[1:])
                    block[nodes] = key[0]
                    pending.remove(g)
                    changed = True

        if not pending or (blocks and sizes[pending[0]] < MIN_BLOCK_TREES):
            break

        nodes = np.flatnonzero(group == pending.pop(0))  # new planting block
        positions[nodes] = local[nodes]
        block[nodes] = blocks
        blocks += 1

    # Each treetop vs where the placed neighbours of its block put it.
    offset = np.full((count, 2), np.nan)

    for t in np.flatnonzero(block >= 0):
        near = [
            v for v in tree.query_ball_point(points[t], link)
            if v != t and block[v] == block[t]
        ]
        if near:
            predicted = np.median(
                [points[v] + basis @ (positions[t] - positions[v]) for v in near],
                axis=0,
            )
            offset[t] = inverse @ (points[t] - predicted)

    fit_m = np.hypot(*(offset @ basis.T).T)
    off = (block >= 0) & (np.nan_to_num(np.abs(offset)).max(axis=1) > 0.4)
    block[off] = -1
    positions[block < 0] = UNPLACED

    both = good & (block[pairs[:, 0]] >= 0) & (block[pairs[:, 0]] == block[pairs[:, 1]])
    agree = (
        positions[pairs[both, 1]] - positions[pairs[both, 0]]
        == steps[both].astype(np.int64)
    ).all(axis=1)

    info = {
        "planting_blocks": int(blocks),
        "trees_per_block": [int((block == b).sum()) for b in range(blocks)],
        "off_grid": int((block < 0).sum()),
        "grid_fit_median_m": float(np.nanmedian(fit_m)),
        "grid_fit_p90_m": float(np.nanpercentile(fit_m, 90)),
        "neighbour_links_consistent": float(agree.mean()) if len(agree) else 1.0,
    }

    return block, positions, fit_m, info


def curve_through(along, offset, half, direction, normal, merge=0.5):
    """Wavy line through the crowns of one diagonal: crowns closer than
    `merge` metres along the diagonal (two crowns of one tree) share one
    point at their mean; a shape-preserving cubic (PCHIP) joins the points
    and the line runs on half a tree step beyond the end crowns."""
    order = np.argsort(along, kind="stable")
    along, offset = along[order], offset[order]
    groups = np.split(
        np.arange(len(along)),
        np.flatnonzero(np.diff(along) > max(merge, 0.5)) + 1,
    )
    along = np.array([along[g].mean() for g in groups])
    offset = np.array([offset[g].mean() for g in groups])

    if len(along) == 1:
        s = np.array([along[0] - half, along[0] + half])
        c = np.array([offset[0], offset[0]])
    else:
        inner = np.union1d(np.arange(along[0], along[-1], 1.0), along)
        s = np.r_[along[0] - half, inner, along[-1] + half]
        c = np.r_[offset[0], PchipInterpolator(along, offset)(inner), offset[-1]]

    return LineString(s[:, None] * direction + c[:, None] * normal)


def ease(t):
    """0 -> 1 with very flat ends (7th-order smooth step): the line stays
    square to the row through each crown and makes its turn in the alley."""
    t = np.clip(t, 0.0, 1.0)
    return t ** 4 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))


def crown_extent(outline_uv, u):
    """Across-row extent (v) of a crown on the across-row line at u."""
    _, low, _, high = outline_uv.bounds
    cut = LineString([(u, low - 1.0), (u, high + 1.0)]).intersection(outline_uv)

    if cut.is_empty:
        return low, high

    return cut.bounds[1], cut.bounds[3]


def wavy_diagonal(
    members, top_u, top_v, extents, row_spacing, slope, crossing,
    separate=(), run_on=1.5,
):
    """One diagonal as u(v) in grid coordinates (u along rows, v across).

    One connected S-wave from its first crown to its last crown. It passes
    every crown of the diagonal at its treetop, square to the row, and makes
    its one-tree shift along the row in the open alley. In a row where the
    diagonal has no tree (missing tree) it crosses the row at the best gap
    between the other trees (`crossing`). Crowns of one tree share one
    point; crowns in `separate` get their own point. The line ends `run_on`
    metres beyond the outer edge of its end crowns: no line over open
    ground outside the orchard.
    """
    groups = []

    def separate_group(group):
        return any(t in separate for t in group["members"])

    for t in sorted(members, key=lambda t: top_v[t]):
        low, high = extents[t]
        last = groups[-1] if groups else None

        if (
            last is not None
            and t not in separate
            and not separate_group(last)
            and low <= last["hi"]
            and abs(top_v[t] - last["v"]) < 0.3 * row_spacing
        ):
            last["members"].append(t)
            last["u"] = float(np.mean(top_u[last["members"]]))
            last["v"] = float(np.mean(top_v[last["members"]]))
            last["lo"] = min(last["lo"], low)
            last["hi"] = max(last["hi"], high)
        else:
            groups.append({
                "members": [t], "u": float(top_u[t]), "v": float(top_v[t]),
                "lo": low, "hi": high,
            })

    knots = [groups[0]]

    for group in groups[1:]:
        previous = knots[-1]
        missing = int(round((group["v"] - previous["v"]) / row_spacing)) - 1

        for q in range(1, missing + 1):  # missing trees inside the diagonal
            f = q / (missing + 1)
            v = previous["v"] + f * (group["v"] - previous["v"])
            u = previous["u"] + f * (group["u"] - previous["u"])
            knots.append({"u": crossing(u, v), "v": v})

        knots.append(group)

    # Two points of the diagonal inside one row (no shared point): the line
    # crosses that row slanting from one to the other, never a sideways jump.
    gap = 0.45 * row_spacing
    clusters = [[knots[0]]]

    for knot in knots[1:]:
        if knot["v"] - clusters[-1][-1]["v"] < gap:
            clusters[-1].append(knot)
        else:
            clusters.append([knot])

    knots = []

    for cluster in clusters:
        if len(cluster) > 1:
            middle = float(np.mean([knot["v"] for knot in cluster]))
            cluster.sort(key=lambda knot: knot["u"] * (1.0 if slope >= 0 else -1.0))
            spread = np.linspace(middle - gap / 2, middle + gap / 2, len(cluster))
            cluster = [{**knot, "v": float(v)} for knot, v in zip(cluster, spread)]
        knots.extend(cluster)

    # Straight run-on (square to the row) just past the end crowns.
    begin = min(min(group["lo"] for group in groups), knots[0]["v"]) - run_on
    end = max(max(group["hi"] for group in groups), knots[-1]["v"]) + run_on
    u_knots = np.array(
        [knots[0]["u"]] + [knot["u"] for knot in knots] + [knots[-1]["u"]]
    )
    v_knots = np.array(
        [begin] + [knot["v"] for knot in knots] + [end]
    )

    for q in range(1, len(v_knots)):  # strictly increasing across the rows
        v_knots[q] = max(v_knots[q], v_knots[q - 1] + 0.05)

    u_parts, v_parts = [], []

    for q in range(len(v_knots) - 1):
        v0, v1 = v_knots[q], v_knots[q + 1]
        v = np.linspace(v0, v1, max(4, int((v1 - v0) / 0.25) + 1))
        if q > 0:
            v = v[1:]
        v_parts.append(v)
        u_parts.append(
            u_knots[q] + (u_knots[q + 1] - u_knots[q]) * ease((v - v0) / (v1 - v0))
        )

    return np.concatenate(u_parts), np.concatenate(v_parts)


def choose_diagonal(transform, basis):
    """The diagonal direction closest to DIAGONAL_DIRECTION_HINT_DEG.

    Returns (vector of one diagonal step, grid step (1, +-1), other vector).
    """
    linear = np.array([[transform.a, transform.b], [transform.d, transform.e]])

    def angle_gap(vector):
        col, row = np.linalg.solve(linear, vector)
        gap = abs(math.degrees(math.atan2(row, col)) % 180 - DIAGONAL_DIRECTION_HINT_DEG) % 180
        return min(gap, 180 - gap)

    a, b = basis[:, 0], basis[:, 1]
    options = [(a + b, (1, 1)), (a - b, (1, -1))]
    options.sort(key=lambda option: angle_gap(option[0]))
    (vector, step), (other, _) = options

    return vector, step, other


def diagonal_guides(
    src, field, points, scores, block, positions, basis, use, outlines,
    knot_points=None, number_positions=None,
):
    """`positions` decide each crown's diagonal; `number_positions` (the
    planting grid before alternate_along_rows) fix the numbering, so D01 and
    D02 stay at the same corner."""
    transform = src.transform
    linear = np.array([[transform.a, transform.b], [transform.d, transform.e]])

    def image_angle(vector):
        col, row = np.linalg.solve(linear, vector)
        return math.degrees(math.atan2(row, col)) % 180

    def bearing(vector):
        return math.degrees(math.atan2(vector[0], vector[1])) % 180

    a, b = basis[:, 0], basis[:, 1]
    vector, step, other = choose_diagonal(transform, basis)

    step_m = float(np.hypot(*vector))
    pitch = abs(float(np.linalg.det(basis))) / step_m
    direction = vector / step_m
    normal = np.array([-direction[1], direction[0]])

    if normal @ np.array([transform.a, transform.d]) < 0:
        normal = -normal

    # Diagonal index: constant along one in-row step + one row step.
    index = np.where(
        block >= 0,
        positions[:, 0] * step[1] - positions[:, 1] * step[0],
        UNPLACED,
    )
    base_positions = positions if number_positions is None else number_positions
    base_index = np.where(
        block >= 0,
        base_positions[:, 0] * step[1] - base_positions[:, 1] * step[0],
        UNPLACED,
    )
    along = points @ direction
    offset = points @ normal
    guides = []

    for part in range(int(block.max()) + 1):
        keys = np.unique(index[block == part])
        base_keys = np.unique(base_index[block == part])
        trend = (
            np.polyfit(
                base_keys,
                [
                    offset[(block == part) & (base_index == k)].mean()
                    for k in base_keys
                ],
                1,
            )[0]
            if len(base_keys) >= 2 else 1.0
        )
        sign = 1 if trend > 0 else -1
        first = base_keys.max() if sign > 0 else base_keys.min()
        numbered = len(guides)
        low_key = int(min(keys.min(), base_keys.min()))
        high_key = int(max(keys.max(), base_keys.max()))

        # D01 = outermost diagonal on the side the normal points to (as before).
        for k in range(low_key, high_key + 1):
            number = numbered + int((first - k) * sign) + 1
            members = np.flatnonzero((block == part) & (index == k))
            knots = {}

            for t in members:  # one treetop per grid site: the highest
                site = int(positions[t, 0])
                if site not in knots or scores[t] > scores[knots[site]]:
                    knots[site] = t

            chosen = np.array(list(knots.values()), dtype=int)
            geometry = (
                curve_through(
                    along[chosen], offset[chosen], step_m / 2,
                    direction, normal,
                ).intersection(field)
                if len(chosen) else GeometryCollection()
            )
            guides.append({
                "diagonal": number,
                "family": "ODD" if number % 2 else "EVEN",
                "block": part + 1,
                "index": int(k),
                "geometry": geometry,
                "tree_sites": len(knots),
                "mean_offset": (
                    float(offset[list(knots.values())].mean())
                    if knots else float("nan")
                ),
            })

    guides.sort(key=lambda guide: guide["diagonal"])
    by_index = {(guide["block"] - 1, guide["index"]): guide for guide in guides}
    drawn = [guide for guide in guides if not guide["geometry"].is_empty]
    assigned = []

    for t in range(len(points)):
        point = Point(points[t])
        own = by_index.get((int(block[t]), int(index[t])))

        if own is not None and not own["geometry"].is_empty:
            guide = own
        else:  # off-grid crown: nearest drawn diagonal
            near = [
                g for g in drawn
                if abs(g["mean_offset"] - offset[t]) < 3 * pitch
            ] or drawn
            guide = min(near, key=lambda g: g["geometry"].distance(point))

        assigned.append((guide["diagonal"], guide["family"]))

    # Final wavy lines (wavy_diagonal): one connected S-wave per diagonal,
    # from its first crown to its last crown, through every crown of the
    # diagonal, crossing rows without its own tree at the best gap.
    diagonal_of = np.array([entry[0] for entry in assigned])
    row_hat = a / math.hypot(*a)
    across_hat = np.array([-row_hat[1], row_hat[0]])

    if across_hat @ b < 0:
        across_hat = -across_hat

    in_row = math.hypot(*a)
    row_spacing = abs(float(np.linalg.det(basis))) / in_row
    slope = float(vector @ row_hat) / float(vector @ across_hat)
    knots_xy = points if knot_points is None else knot_points
    top_u, top_v = knots_xy @ row_hat, knots_xy @ across_hat
    to_grid = [row_hat[0], row_hat[1], across_hat[0], across_hat[1], 0.0, 0.0]
    to_map = np.array([row_hat, across_hat])
    outlines_uv = {
        t: affine_transform(outlines[t], to_grid) for t in np.flatnonzero(use)
    }
    extents = {t: crown_extent(outlines_uv[t], top_u[t]) for t in outlines_uv}
    boxes_uv = np.array([
        outlines_uv[t].bounds if t in outlines_uv else (np.inf, np.inf, -np.inf, -np.inf)
        for t in range(len(points))
    ])

    def crossing_for(number):
        """Where to cross a row with no tree of this diagonal: the spot near
        the planting position with the least canopy of other trees."""
        half, reach = 0.35 * row_spacing, 0.7 * in_row

        def crossing(u_guess, v):
            near = np.flatnonzero(
                use & (diagonal_of != number)
                & (boxes_uv[:, 0] <= u_guess + reach)
                & (boxes_uv[:, 2] >= u_guess - reach)
                & (boxes_uv[:, 1] <= v + half) & (boxes_uv[:, 3] >= v - half)
            )
            if not len(near):
                return u_guess
            blocker = unary_union([outlines_uv[t] for t in near])
            best_key, best_u = None, u_guess
            for u in u_guess + np.arange(-reach, reach + 1e-9, 0.15):
                inside = LineString([(u, v - half), (u, v + half)]).intersection(blocker).length
                key = (round(inside, 1), abs(u - u_guess))
                if best_key is None or key < best_key:
                    best_key, best_u = key, float(u)
            return best_u

        return crossing

    def field_pieces(u, v, members, run_on=1.5):
        """The line inside the field; each piece runs from its first to its
        last crown of this diagonal (no line over open ground)."""
        kept_pieces = []
        for piece in parts(
            LineString(np.column_stack([u, v]) @ to_map).intersection(field),
            "LineString",
        ):
            xy = np.asarray(piece.coords)
            piece_v = xy @ across_hat
            own = [t for t in members if piece_v.min() <= top_v[t] <= piece_v.max()]
            if not own:
                continue
            low = min(extents[t][0] for t in own) - run_on
            high = max(extents[t][1] for t in own) + run_on
            xy = xy[(piece_v >= low) & (piece_v <= high)]
            if len(xy) >= 2 and LineString(xy).length >= 1.0:
                kept_pieces.append(LineString(xy))
        return kept_pieces

    # Lines through the crowns on planting positions.
    for guide in guides:
        members = np.flatnonzero(
            use & (block >= 0) & (diagonal_of == guide["diagonal"])
        )
        guide["geometry"] = GeometryCollection()

        if not len(members):
            continue

        crossing = crossing_for(guide["diagonal"])
        separate = set()

        for attempt in range(2):
            u, v = wavy_diagonal(
                members, top_u, top_v, extents, row_spacing, slope,
                crossing, separate,
            )
            inside = field_pieces(u, v, members)
            guide["geometry"] = (
                MultiLineString(inside) if len(inside) > 1
                else inside[0] if inside else GeometryCollection()
            )
            # A crown of this diagonal missed: give it its own point.
            separate = {
                t for t in members
                if not guide["geometry"].intersects(outlines[t])
            }
            if not separate:
                break

    # A crown off the planting grid takes the diagonal whose line runs
    # through it (longest stretch), else the nearest line: the lines never
    # bend to reach it.
    drawn = [guide for guide in guides if not guide["geometry"].is_empty]

    for t in np.flatnonzero(use & (block < 0)) if drawn else []:
        through = [
            (guide["geometry"].intersection(outlines[t]).length, guide)
            for guide in drawn
            if guide["geometry"].intersects(outlines[t])
        ]
        guide = (
            max(through, key=lambda item: item[0])[1] if through
            else min(drawn, key=lambda g: g["geometry"].distance(Point(points[t])))
        )
        assigned[t] = (guide["diagonal"], guide["family"])

    by_number = {guide["diagonal"]: guide for guide in guides}
    assigned = [
        (
            number, family,
            by_number[number]["geometry"].distance(Point(points[t]))
            if not by_number[number]["geometry"].is_empty else float("nan"),
        )
        for t, (number, family) in enumerate(assigned)
    ]

    settings = {
        "method": (
            "Planting grid (design 20 x 40 ft) measured from crown treetops; "
            "each crown's diagonal comes from its grid position; each "
            "diagonal is one connected S-wave from its first to its last "
            "crown through every crown on it (cut where it leaves the "
            "field); one crown per planting position (20 ft spacing); "
            "neighbouring crowns along a row alternate EVEN / ODD, also "
            "across a missing tree (ALTERNATE_ALONG_ROWS)."
        ),
        "in_row_spacing_m": float(math.hypot(*a)),
        "row_spacing_m": float(abs(np.linalg.det(basis)) / math.hypot(*a)),
        "grid_angle_deg": float(math.degrees(math.acos(min(
            1.0, abs(a @ b) / math.hypot(*a) / math.hypot(*b)
        )))),
        "row_bearing_deg": bearing(a),
        "diagonal_image_angle_deg": image_angle(vector),
        "diagonal_bearing_deg": bearing(vector),
        "other_diagonal_image_angle_deg": image_angle(other),
        "diagonal_spacing_m": pitch,
        "tree_step_along_diagonal_m": step_m,
        "diagonal_grid_step": [int(step[0]), int(step[1])],
        "direction_hint_deg": DIAGONAL_DIRECTION_HINT_DEG,
        "diagonal_count": len(guides),
    }

    return guides, assigned, along, settings, index


def empty_grid_sites(points, block, positions, basis, max_missing=3):
    """Grid sites inside a row (trees on both sides) with no treetop."""
    sites = {}

    for t in np.flatnonzero(block >= 0):
        sites.setdefault(
            (int(block[t]), int(positions[t, 0]), int(positions[t, 1])), t
        )

    empty = set()

    for (part, i, j) in sites:
        for gap in range(1, max_missing + 1):
            if (part, i + gap + 1, j) in sites and all(
                (part, i + q, j) not in sites for q in range(1, gap + 1)
            ):
                empty.update((part, i + q, j) for q in range(1, gap + 1))

    found = []

    for (part, i, j) in sorted(empty):
        near = [
            points[sites[(part, i + di, j + dj)]] + basis @ np.array([-di, -dj])
            for di in range(-2, 3) for dj in (-1, 0, 1)
            if (part, i + di, j + dj) in sites
        ]
        found.append(((part, i, j), np.median(near, axis=0)))

    return found


def fill_positions(
    src, labels, label_ids, points, seeds, block, positions, fit, eligible,
    outlines, basis,
):
    """One crown per planting position along every row (20 ft spacing).

    A planting position inside a row with no treetop gets its crown: an
    off-grid crown sitting on it is snapped to it; otherwise the canopy
    covering it (pixels of its two row neighbours that are closer to it
    than to their own treetop) becomes its crown. Bare positions stay empty
    (missing tree). `labels` is updated in place.
    """
    transform = src.transform
    inverse = ~transform
    pixel_m = math.hypot(transform.a, transform.d)
    in_row = math.hypot(*basis[:, 0])
    radius = int(math.ceil(in_row / pixel_m))
    min_area = 0.2 * float(np.median([outlines[t].area for t in np.flatnonzero(eligible)]))
    site_crowns = {}

    for t in np.flatnonzero(eligible & (block >= 0)):
        site_crowns.setdefault(
            (int(block[t]), int(positions[t, 0]), int(positions[t, 1])), []
        ).append(t)

    off_grid = np.flatnonzero(eligible & (block < 0))
    off_tree = cKDTree(points[off_grid]) if len(off_grid) else None
    off_label = {int(label_ids[t]): t for t in off_grid}
    crossing_points = {}
    snapped, new, changed = 0, [], set()
    next_label = int(labels.max()) + 1

    for site, spot in empty_grid_sites(points, block, positions, basis):
        if site in site_crowns:
            continue

        if off_tree is not None:
            distance, k = off_tree.query(spot)
            t = off_grid[k]
            if distance < 0.5 * in_row and block[t] < 0:
                block[t], positions[t], fit[t] = site[0], site[1:], distance
                site_crowns[site] = [t]
                snapped += 1
                continue

        # An off-grid crown covering the position is that position's crown
        # (its diagonal line then crosses it at the position).
        sc, sr = inverse * tuple(spot)
        row_px, col_px = int(math.floor(sr)), int(math.floor(sc))
        t = (
            off_label.get(int(labels[row_px, col_px]))
            if 0 <= row_px < labels.shape[0] and 0 <= col_px < labels.shape[1]
            else None
        )
        if t is not None and block[t] < 0:
            block[t], positions[t] = site[0], site[1:]
            fit[t] = float(math.hypot(*(points[t] - spot)))
            site_crowns[site] = [t]
            crossing_points[t] = np.asarray(spot, dtype=float)
            snapped += 1
            continue

        part, i, j = site
        neighbours = (
            site_crowns.get((part, i - 1, j), []) + site_crowns.get((part, i + 1, j), [])
        )
        if not neighbours:
            continue

        r0, r1 = max(0, int(sr) - radius), min(labels.shape[0], int(sr) + radius + 1)
        c0, c1 = max(0, int(sc) - radius), min(labels.shape[1], int(sc) + radius + 1)
        window = labels[r0:r1, c0:c1]  # view: edits go into labels
        spot_r, spot_c = int(sr) - r0, int(sc) - c0

        if not (
            0 <= spot_r < window.shape[0] and 0 <= spot_c < window.shape[1]
        ) or window[spot_r, spot_c] not in label_ids[neighbours]:
            continue  # bare planting position: missing tree

        rows, cols = np.mgrid[r0:r1, c0:c1] + 0.5
        to_spot = (cols - sc) ** 2 + (rows - sr) ** 2
        take = np.zeros(window.shape, dtype=bool)

        # Each neighbour keeps the canopy closer to its own position: its
        # planting position if it was placed there by covering it (its
        # treetop may be far away), otherwise its treetop.
        reference = {
            t: crossing_points.get(t, points[t]) for t in neighbours
        }

        for t in neighbours:
            tc, tr = inverse * tuple(reference[t])
            take |= (window == label_ids[t]) & (to_spot < (cols - tc) ** 2 + (rows - tr) ** 2)

        pieces_found, _ = ndi.label(take)
        take = pieces_found == pieces_found[spot_r, spot_c]  # one piece, on the spot

        if not take[spot_r, spot_c] or take.sum() * pixel_m ** 2 < min_area:
            continue

        window[take] = next_label

        # A neighbour must stay in one piece: canopy of it cut off on the far
        # side of the new crown joins the new crown (never red-blue-red for
        # one tree along the row).
        R0, R1 = max(0, int(sr) - 2 * radius), min(labels.shape[0], int(sr) + 2 * radius + 1)
        C0, C1 = max(0, int(sc) - 2 * radius), min(labels.shape[1], int(sc) + 2 * radius + 1)
        wide = labels[R0:R1, C0:C1]  # view
        grown = cv2.dilate((wide == next_label).astype("uint8"), disk(1)).astype(bool)

        for t in neighbours:
            pieces_left, count = ndi.label(wide == label_ids[t])
            if count < 2:
                continue
            ref_c, ref_r = inverse * tuple(reference[t])
            ref_r, ref_c = int(ref_r) - R0, int(ref_c) - C0
            inside = 0 <= ref_r < wide.shape[0] and 0 <= ref_c < wide.shape[1]
            main = pieces_left[ref_r, ref_c] if inside else 0
            if not main:
                main = int(np.argmax(np.bincount(pieces_left.ravel())[1:])) + 1
            for piece in range(1, count + 1):
                if piece != main and (grown & (pieces_left == piece)).any():
                    wide[pieces_left == piece] = next_label

        new.append((site, spot, next_label, (R0, R1, C0, C1), (sr, sc)))
        changed.update(neighbours)
        site_crowns[site] = [-1]
        next_label += 1

    def geometry_of(label, box):
        r0, r1, c0, c1 = box
        sub = labels[r0:r1, c0:c1]
        polygons = [
            shape(geometry) for geometry, _ in shapes(
                sub, mask=sub == label,
                transform=transform * Affine.translation(c0, r0),
            )
        ]
        return unary_union(polygons)

    for t in changed:  # neighbours that gave canopy to a planting position
        corners = np.array([
            inverse * corner for corner in [
                outlines[t].bounds[:2], outlines[t].bounds[2:],
                (outlines[t].bounds[0], outlines[t].bounds[3]),
                (outlines[t].bounds[2], outlines[t].bounds[1]),
            ]
        ])
        box = (
            max(0, int(corners[:, 1].min()) - 2), min(labels.shape[0], int(corners[:, 1].max()) + 3),
            max(0, int(corners[:, 0].min()) - 2), min(labels.shape[1], int(corners[:, 0].max()) + 3),
        )
        outlines[t] = geometry_of(int(label_ids[t]), box)

    inferred = np.zeros(len(points) + len(new), dtype=bool)
    inferred[len(points):] = True

    if new:
        label_ids = np.r_[label_ids, [entry[2] for entry in new]]
        points = np.vstack([points, [entry[1] for entry in new]])
        seeds = np.vstack([seeds, [[entry[4][0], entry[4][1], 0.0] for entry in new]])
        block = np.r_[block, [entry[0][0] for entry in new]]
        positions = np.vstack([positions, [entry[0][1:] for entry in new]]).astype(np.int64)
        fit = np.r_[fit, np.zeros(len(new))]
        outlines = outlines + [geometry_of(entry[2], entry[3]) for entry in new]
        eligible = np.r_[eligible, [outlines[-len(new) + q].area >= MIN_CROWN_AREA_M2 for q in range(len(new))]]

    return (
        label_ids, points, seeds, block, positions, fit, eligible, outlines,
        inferred, snapped, crossing_points,
    )


def alternate_along_rows(block, positions, use, step):
    """Crowns next to each other in a row always get opposite colours.

    On the grid the colour comes from the planting position, so where a
    position in a row has no crown (missing tree, or only shadow) the two
    crowns on either side of it share a colour. Each row is cut into runs of
    crowns on consecutive positions. Across a gap that would give the same
    colour, one of the two runs moves one position toward the gap: it then
    takes the neighbouring diagonal and the other colour. Runs only move
    into a gap, never past their neighbours or beyond the ends of the row.

    Each row has two ways to do this (move one set of runs or the other).
    Where a moved run in one row lines up with a moved run in the next row,
    the diagonals between them stay smooth; elsewhere they make an extra
    one-tree step. The choice for all rows together that gives the fewest
    extra steps (then the fewest moved crowns) is found exactly by dynamic
    programming over the rows of each planting block.

    Returns (positions used for the diagonals, moved flag per crown,
    number of extra steps in the diagonals).
    """
    shifted = positions.copy()
    moved = np.zeros(len(positions), dtype=bool)
    extra_steps = 0
    rows = {}

    for t in np.flatnonzero(use & (block >= 0)):
        rows.setdefault((int(block[t]), int(positions[t, 1])), []).append(t)

    def row_options(members, row):
        """The ways to make this row alternate: [(shift per crown, diagonal
        index -> shift, crowns moved)]."""
        members = np.array(sorted(members, key=lambda t: positions[t, 0]))
        runs = np.split(
            members, np.flatnonzero(np.diff(positions[members, 0]) > 1) + 1
        )
        gaps = [
            int(positions[runs[r + 1][0], 0] - positions[runs[r][-1], 0])
            for r in range(len(runs) - 1)
        ]
        # An even step (odd number of empty positions) keeps the colour:
        # the run after it must take the other colour.
        flip = [0]
        for gap in gaps:
            flip.append(flip[-1] ^ int(gap % 2 == 0))
        flip = np.array(flip)
        patterns = [flip] if len(runs) < 2 else [flip, 1 - flip]
        options = []

        for pattern in patterns:
            shift = {}
            for r, run in enumerate(runs):
                if not pattern[r]:
                    s = 0
                else:
                    into_left = r > 0 and not pattern[r - 1] and gaps[r - 1] % 2 == 0
                    into_right = (
                        r < len(runs) - 1 and not pattern[r + 1] and gaps[r] % 2 == 0
                    )
                    s = -1 if into_left else 1 if into_right else (-1 if r > 0 else 1)
                for t in run:
                    shift[int(t)] = s
            diagonal = {
                int((positions[t, 0] + s) * step[1] - row * step[0]): s
                for t, s in shift.items()
            }
            options.append((shift, diagonal, sum(1 for s in shift.values() if s)))

        return options

    for part in sorted({part for part, _ in rows}):
        row_ids = sorted(row for p, row in rows if p == part)
        choices = [row_options(rows[(part, row)], row) for row in row_ids]

        def steps_between(earlier, later):
            return sum(
                1 for k, s in later[1].items()
                if k in earlier[1] and earlier[1][k] != s
            )

        # Viterbi: cost = extra diagonal steps + 0.01 per moved crown.
        cost = [[0.01 * option[2] for option in choices[0]]]
        back = [[0] * len(choices[0])]

        for n in range(1, len(row_ids)):
            adjacent = row_ids[n] - row_ids[n - 1] == 1
            here, previous = [], []
            for option in choices[n]:
                best = min(
                    range(len(choices[n - 1])),
                    key=lambda p: cost[n - 1][p] + (
                        steps_between(choices[n - 1][p], option) if adjacent else 0
                    ),
                )
                here.append(
                    cost[n - 1][best] + 0.01 * option[2]
                    + (steps_between(choices[n - 1][best], option) if adjacent else 0)
                )
                previous.append(best)
            cost.append(here)
            back.append(previous)

        pick = int(np.argmin(cost[-1]))
        picked = [0] * len(row_ids)
        for n in range(len(row_ids) - 1, -1, -1):
            picked[n] = pick
            pick = back[n][pick]

        for n, row in enumerate(row_ids):
            shift = choices[n][picked[n]][0]
            for t, s in shift.items():
                if s:
                    shifted[t, 0] += s
                    moved[t] = True
            if n and row_ids[n] - row_ids[n - 1] == 1:
                extra_steps += steps_between(
                    choices[n - 1][picked[n - 1]], choices[n][picked[n]]
                )

    return shifted, moved, extra_steps


def planting_diagonals(src, field, crowns, tops, labels):
    log("Measuring the planting grid from the treetops.")

    label_ids = np.array([label_id for label_id, _ in crowns])
    areas = np.array([geometry.area for _, geometry in crowns])
    seeds = tops[label_ids - 1]
    xs, ys = src.transform * (seeds[:, 1] + 0.5, seeds[:, 0] + 0.5)
    points = np.column_stack([xs, ys]).astype(float)

    eligible = areas >= MIN_CROWN_AREA_M2
    basis, nearest = measure_planting_grid(points[eligible])
    basis = orient_grid(basis, src.transform)

    block = np.full(len(points), -1)
    positions = np.full((len(points), 2), UNPLACED, dtype=np.int64)
    fit = np.full(len(points), np.nan)
    (
        block[eligible], positions[eligible], fit[eligible], grid_info,
    ) = grid_positions(points[eligible], basis)

    # Every planting position along the rows gets its crown (20 ft spacing):
    # touching crowns in a row then alternate EVEN / ODD.
    outlines = [geometry for _, geometry in crowns]
    inferred = np.zeros(len(points), dtype=bool)
    snapped = 0
    crossing_points = {}

    if SPLIT_COVERED_POSITIONS:
        (
            label_ids, points, seeds, block, positions, fit, eligible,
            outlines, inferred, snapped, crossing_points,
        ) = fill_positions(
            src, labels, label_ids, points, seeds, block, positions, fit,
            eligible, outlines, basis,
        )

    # Where each diagonal line crosses a crown: its treetop, or the planting
    # position for a crown whose treetop lies far from its position.
    knot_points = points.copy()

    for t, spot in crossing_points.items():
        knot_points[t] = spot

    updated_crowns = list(zip(label_ids.tolist(), list(outlines)))

    # One planting position = one crown: crowns on the same grid site are
    # pieces of one tree (split by the watershed) and are merged.
    pieces = np.ones(len(points), dtype=int)
    merged_into = {}
    site_members = {}

    for t in np.flatnonzero(eligible & (block >= 0)):
        site_members.setdefault(
            (int(block[t]), int(positions[t, 0]), int(positions[t, 1])), []
        ).append(t)

    for members in site_members.values():
        if len(members) > 1:
            keep = max(members, key=lambda t: seeds[t, 2])
            outlines[keep] = unary_union([outlines[t] for t in members])
            pieces[keep] = len(members)
            for t in members:
                if t != keep:
                    merged_into[t] = keep

    kept = eligible.copy()
    kept[list(merged_into)] = False
    split_merged = len(merged_into)

    # An off-grid crown touching a tree of the grid within one tree spacing
    # is a piece of that tree (one planting position = one crown): it joins
    # the touching tree with the nearest treetop.
    in_row = math.hypot(*basis[:, 0])
    anchors = np.flatnonzero(kept & (block >= 0))
    pieces_joined = 0

    if len(anchors):
        anchor_tree = cKDTree(points[anchors])

        for t in np.flatnonzero(kept & (block < 0)):
            touching = [
                anchors[k]
                for k in anchor_tree.query_ball_point(points[t], in_row)
                if outlines[anchors[k]].distance(outlines[t]) < 0.3
            ]
            if not touching:
                continue
            keep = min(touching, key=lambda s: math.hypot(*(points[s] - points[t])))
            outlines[keep] = unary_union([outlines[keep], outlines[t]])
            pieces[keep] += 1
            merged_into[t] = keep
            kept[t] = False
            pieces_joined += 1

    # Along every row, neighbouring crowns alternate EVEN / ODD, also across
    # a missing tree (the crowns on one side move to the next diagonal).
    if ALTERNATE_ALONG_ROWS:
        diagonal_positions, moved, extra_steps = alternate_along_rows(
            block, positions, kept, choose_diagonal(src.transform, basis)[1]
        )
        for t, keep in merged_into.items():  # pieces follow their tree
            if block[t] >= 0 and block[keep] >= 0:
                diagonal_positions[t] = diagonal_positions[keep]
    else:
        diagonal_positions, moved = positions, np.zeros(len(points), dtype=bool)
        extra_steps = 0

    guides, assigned, along, settings, index = diagonal_guides(
        src, field, points, seeds[:, 2], block, diagonal_positions, basis, kept,
        outlines, knot_points, number_positions=positions,
    )
    # A crown off the planting grid (no row position) takes the colour
    # opposite to the crowns next to it along the row (nearest crown on each
    # side within 1.5 tree spacings), on the nearest line of that colour.
    off_grid_recoloured = 0

    if ALTERNATE_ALONG_ROWS:
        row_hat = basis[:, 0] / math.hypot(*basis[:, 0])
        across_hat = np.array([-row_hat[1], row_hat[0]])
        row_spacing = abs(float(np.linalg.det(basis))) / math.hypot(*basis[:, 0])
        where = knot_points
        on_u, on_v = where @ row_hat, where @ across_hat
        placed = np.flatnonzero(kept)
        placed_tree = cKDTree(where[placed])
        drawn_guides = [g for g in guides if not g["geometry"].is_empty]

        for t in np.flatnonzero(kept & (block < 0)):
            sides = {}
            for k in placed_tree.query_ball_point(where[t], 1.5 * in_row):
                s = placed[k]
                du, dv = on_u[s] - on_u[t], on_v[s] - on_v[t]
                if s == t or abs(dv) >= 0.4 * row_spacing or du == 0:
                    continue
                side = du > 0
                if side not in sides or abs(du) < abs(on_u[sides[side]] - on_u[t]):
                    sides[side] = s
            colours = {assigned[s][1] for s in sides.values()}
            if len(colours) != 1 or assigned[t][1] not in colours:
                continue
            wanted = "EVEN" if assigned[t][1] == "ODD" else "ODD"
            options = [g for g in drawn_guides if g["family"] == wanted]
            if not options:
                continue
            point = Point(points[t])
            guide = min(options, key=lambda g: g["geometry"].distance(point))
            assigned[t] = (
                guide["diagonal"], guide["family"],
                guide["geometry"].distance(point),
            )
            off_grid_recoloured += 1

    settings.update(grid_info)
    settings["off_grid_crowns_recoloured_to_alternate"] = off_grid_recoloured
    settings["alternate_along_rows"] = bool(ALTERNATE_ALONG_ROWS)
    settings["crowns_moved_across_gap"] = int((moved & kept).sum())
    settings["diagonal_extra_steps_from_moves"] = int(extra_steps)
    settings["nearest_treetop_spacing_m"] = nearest
    settings["split_crowns_merged"] = split_merged
    settings["off_grid_pieces_joined_to_their_tree"] = pieces_joined
    settings["positions_given_crown_from_spacing"] = int(inferred.sum())
    settings["off_grid_crowns_snapped_to_position"] = int(snapped)
    step = settings["diagonal_grid_step"]

    # Check (the planting rule): crowns next to each other in a row (no other
    # crown between them, touching or not) must have opposite colours.
    family_of = np.array([entry[1] for entry in assigned])
    rows_of = {}
    for t in np.flatnonzero(kept & (block >= 0)):
        rows_of.setdefault((int(block[t]), int(positions[t, 1])), []).append(t)
    row_pairs = same_colour = touching_pairs = touching_same = 0
    for members in rows_of.values():
        members.sort(key=lambda t: positions[t, 0])
        for t, s in zip(members, members[1:]):
            same = bool(family_of[t] == family_of[s])
            row_pairs += 1
            same_colour += same
            if outlines[t].distance(outlines[s]) < 0.3:
                touching_pairs += 1
                touching_same += same
    settings["row_neighbours"] = row_pairs
    settings["row_neighbours_same_colour"] = int(same_colour)

    # The same check as the eye sees it on the map, for every crown (on the
    # grid or not): crowns grouped into rows by position across the rows,
    # neighbours along the row within 1.5 tree spacings.
    map_row_hat = basis[:, 0] / math.hypot(*basis[:, 0])
    map_across = np.array([-map_row_hat[1], map_row_hat[0]])
    map_spacing = abs(float(np.linalg.det(basis))) / math.hypot(*basis[:, 0])
    shown_crowns = np.flatnonzero(kept)
    map_u = knot_points[shown_crowns] @ map_row_hat
    map_v = knot_points[shown_crowns] @ map_across
    order = np.argsort(map_v, kind="stable")
    map_rows = np.zeros(len(shown_crowns), dtype=int)
    map_rows[order] = np.cumsum(np.r_[0, np.diff(map_v[order]) > 0.4 * map_spacing])
    map_pairs = map_same = 0
    for r in np.unique(map_rows):
        members = np.flatnonzero(map_rows == r)
        members = members[np.argsort(map_u[members], kind="stable")]
        for a_, b_ in zip(members, members[1:]):
            if map_u[b_] - map_u[a_] <= 1.5 * in_row:
                map_pairs += 1
                map_same += family_of[shown_crowns[a_]] == family_of[shown_crowns[b_]]
    settings["map_row_neighbours"] = int(map_pairs)
    settings["map_row_neighbours_same_colour"] = int(map_same)
    settings["touching_row_neighbours"] = touching_pairs
    settings["touching_row_neighbours_same_colour"] = int(touching_same)

    # Checks: each crown is crossed by its own line; does another line reach
    # into it (passing along the edge between two crowns is not counted)?
    lines = {guide["diagonal"]: guide["geometry"] for guide in guides}
    numbers = np.array([entry[0] for entry in assigned])
    crossed = np.array([
        bool(kept[t]) and lines[numbers[t]].intersects(outlines[t])
        for t in range(len(points))
    ])
    boxes = np.array([geometry.bounds for geometry in outlines])
    touched = np.zeros(len(points), dtype=bool)
    entered = np.zeros(len(points), dtype=bool)
    cores = {}

    for guide in guides:
        guide["bridge"] = GeometryCollection()
        guide["solid"] = guide["geometry"]
        if guide["geometry"].is_empty:
            continue
        left, bottom, right, top = guide["geometry"].bounds
        test = prep(guide["geometry"])
        crossed_other = []
        for t in np.flatnonzero(
            kept & (numbers != guide["diagonal"])
            & (boxes[:, 0] <= right) & (boxes[:, 2] >= left)
            & (boxes[:, 1] <= top) & (boxes[:, 3] >= bottom)
        ):
            if test.intersects(outlines[t]):
                touched[t] = True
                if t not in cores:
                    cores[t] = outlines[t].buffer(-0.5)
                if not cores[t].is_empty and test.intersects(cores[t]):
                    entered[t] = True
                    crossed_other.append(outlines[t])
        # Where the line passes canopy of another tree (its own tree is
        # missing there) it is drawn dotted: connected, but clearly not
        # running through a crown of its own.
        if crossed_other:
            other_canopy = unary_union(crossed_other)
            guide["bridge"] = guide["geometry"].intersection(other_canopy)
            guide["solid"] = guide["geometry"].difference(other_canopy)

    settings["crowns_crossed_by_their_diagonal"] = float(crossed[kept].mean())
    settings["crowns_entered_by_another_diagonal"] = int(entered[kept].sum())
    settings["crowns_touched_at_edge_by_another_diagonal"] = int(
        (touched & ~entered)[kept].sum()
    )

    # Canopy 1, 2, 3 ... along each diagonal, from its upper (START) end.
    tree_on_diagonal = np.full(len(points), -1)

    for number in np.unique(numbers[kept]):
        members = np.flatnonzero(kept & (numbers == number))
        members = members[np.argsort(seeds[members, 0], kind="stable")]
        tree_on_diagonal[members] = np.arange(1, len(members) + 1)

    # 1-based grid row / tree numbers per planting block.
    row_no = np.full(len(points), -1)
    tree_no = np.full(len(points), -1)

    for part in range(grid_info["planting_blocks"]):
        inside = block == part
        tree_no[inside] = positions[inside, 0] - positions[inside, 0].min() + 1
        row_no[inside] = positions[inside, 1] - positions[inside, 1].min() + 1

    merged = {
        int(label_ids[t]): int(label_ids[keep]) for t, keep in merged_into.items()
    }

    # Grid sites inside rows with no treetop: missing tree, or a gap that a
    # neighbour's canopy has grown into.
    to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    by_index = {(guide["block"] - 1, guide["index"]): guide for guide in guides}
    first_i = {
        part: positions[block == part, 0].min()
        for part in range(grid_info["planting_blocks"])
    }
    first_j = {
        part: positions[block == part, 1].min()
        for part in range(grid_info["planting_blocks"])
    }
    empty_rows = []

    for (part, i, j), point in empty_grid_sites(points, block, positions, basis):
        guide = by_index.get((part, i * step[1] - j * step[0]))
        col, row = ~src.transform * tuple(point)
        col, row = int(math.floor(col)), int(math.floor(row))
        label = (
            int(labels[row, col])
            if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]
            else 0
        )
        label = merged.get(label, label)
        lon, lat = to_wgs84.transform(*point)
        empty_rows.append({
            "planting_block": part + 1,
            "grid_row": int(j - first_j[part] + 1),
            "grid_tree": int(i - first_i[part] + 1),
            "diagonal": guide["diagonal"] if guide else None,
            "family": guide["family"] if guide else None,
            "x": float(point[0]),
            "y": float(point[1]),
            "longitude": lon,
            "latitude": lat,
            "canopy_here": f"C{label:05d}" if label else "none",
        })

    flags = []

    for t in range(len(label_ids)):
        crown_flags = []
        if block[t] < 0:
            crown_flags.append("OFF_GRID")
        if pieces[t] > 1:
            crown_flags.append("MERGED_SPLIT_CROWN")
        if inferred[t]:
            crown_flags.append("INFERRED_FROM_SPACING")
        if moved[t]:
            crown_flags.append("MOVED_ACROSS_GAP")
        if kept[t] and not crossed[t]:
            crown_flags.append("LINE_MISSES_CROWN")
        flags.append(";".join(crown_flags))

    table = pd.DataFrame({
        "label_id": label_ids,
        "diagonal": [entry[0] for entry in assigned],
        "family": [entry[1] for entry in assigned],
        "tree_on_diagonal": pd.array(tree_on_diagonal, dtype="Int64"),
        "guide_distance_m": [entry[2] for entry in assigned],
        "along_m": along - along.mean(),
        "top_x": points[:, 0],
        "top_y": points[:, 1],
        "planting_block": pd.array(
            np.where(block >= 0, block + 1, 0), dtype="Int64"
        ),
        "grid_row": pd.array(row_no, dtype="Int64"),
        "grid_tree": pd.array(tree_no, dtype="Int64"),
        "grid_fit_m": fit,
        "crown_pieces": pieces,
        "grid_flags": flags,
    })

    for column in ["planting_block", "grid_row", "grid_tree"]:
        table.loc[block < 0, column] = pd.NA

    table.loc[tree_on_diagonal < 1, "tree_on_diagonal"] = pd.NA
    table = table[
        [t not in merged_into for t in range(len(points))]
    ].reset_index(drop=True)

    log(
        f"Planting grid: {settings['in_row_spacing_m']:.2f} m in the row x "
        f"{settings['row_spacing_m']:.2f} m between rows (rows at bearing "
        f"{settings['row_bearing_deg']:.1f} deg); "
        f"{grid_info['planting_blocks']} planting block(s); "
        f"{int((block >= 0).sum()):,} of {len(points):,} crowns on the grid; "
        f"{split_merged} split crowns merged into their tree; "
        f"{pieces_joined} off-grid pieces joined the tree they touch."
    )
    log(
        f"Diagonals: {len(guides)} wavy lines at image angle "
        f"{settings['diagonal_image_angle_deg']:.1f} deg, "
        f"{settings['diagonal_spacing_m']:.3f} m apart; "
        f"{100 * settings['crowns_crossed_by_their_diagonal']:.1f} % of crowns "
        f"crossed by their own line; "
        f"{len(empty_rows):,} grid sites inside rows have no treetop."
    )
    log(
        f"Planting rule: {int(inferred.sum())} positions got their crown from "
        f"the 20 ft spacing; {settings['crowns_moved_across_gap']} crowns moved "
        f"to the next diagonal across a gap ({extra_steps} extra steps in the "
        f"diagonals); neighbours in a row with the same colour: "
        f"{same_colour} of {row_pairs} (on the map, all crowns: "
        f"{settings['map_row_neighbours_same_colour']} of "
        f"{settings['map_row_neighbours']}); "
        f"{off_grid_recoloured} off-grid crowns recoloured to alternate."
    )

    return guides, table, settings, pd.DataFrame(empty_rows), merged, updated_crowns


def inventory(crowns, peak_count, grid_table, src):
    rows = []

    typical_area = float(np.median([
        geometry.area
        for _, geometry in crowns
        if geometry.area >= MIN_CROWN_AREA_M2
    ] or [0.0]))

    for label_id, geometry in crowns:
        centroid = geometry.centroid

        col, row = (
            ~src.transform
        ) * (centroid.x, centroid.y)

        status = (
            "PEAK_BASED_CANDIDATE"
            if label_id <= peak_count
            else "SEPARATE_PATCH_CANDIDATE"
        )

        flags = []

        if geometry.area < 0.7:
            flags.append("SMALL")

        if geometry.area > max(182.536, 4 * typical_area):
            flags.append("LARGE_OR_MERGED")

        # Two touching trees in a row with no visible neck between them.
        if typical_area and geometry.area >= 2 * typical_area:
            box = np.asarray(
                geometry.minimum_rotated_rectangle.exterior.coords
            )
            sides = np.hypot(*np.diff(box[:3], axis=0).T)
            if sides.min() > 0 and sides.max() / sides.min() >= 1.8:
                flags.append("POSSIBLE_MERGED_TREES")

        if geometry.geom_type == "MultiPolygon":
            flags.append("MULTIPART")

        rows.append({
            "crown_id": f"C{label_id:05d}",
            "label_id": label_id,
            "status": status,
            "area_m2": geometry.area,
            "area_ft2": geometry.area / 0.3048**2,
            "x": centroid.x,
            "y": centroid.y,
            "pixel_col": col,
            "pixel_row": row,
            "shape_flags": ";".join(flags),
        })

    df = pd.DataFrame(rows).merge(grid_table, on="label_id", how="left")
    df["review_flags"] = [
        ";".join(filter(None, [shape_flags, grid_flags]))
        for shape_flags, grid_flags in zip(
            df.pop("shape_flags"), df.pop("grid_flags")
        )
    ]
    df.loc[
        df["review_flags"].str.contains("INFERRED_FROM_SPACING"), "status"
    ] = "SPACING_INFERRED_CANDIDATE"

    columns = [
        "crown_id", "label_id", "status",
        "area_m2", "area_ft2", "x", "y",
        "pixel_col", "pixel_row", "diagonal",
        "family", "tree_on_diagonal", "guide_distance_m", "along_m",
        "top_x", "top_y", "planting_block",
        "grid_row", "grid_tree", "grid_fit_m",
        "crown_pieces", "review_flags",
    ]

    return df[columns]


# ==================== REPORTS ====================

def save_diagonal_chart(df, diagonal_summary, out):
    """Canopy area of every diagonal, with the EVEN / ODD totals at the end."""
    table = diagonal_summary.sort_values("diagonal")
    table = table[table["candidate_count"] > 0]

    if table.empty:
        return None

    totals = (
        df.assign(area_ft2=df["area_m2"] * FT2_PER_M2)
        .groupby("family")
        .agg(crowns=("crown_id", "size"), area=("area_ft2", "sum"))
        .reindex(["EVEN", "ODD"], fill_value=0)
    )
    area_ft2 = table["area_m2"] * FT2_PER_M2

    fig, (ax, end) = plt.subplots(
        1, 2,
        figsize=(17, 6.6),
        layout="constrained",
        gridspec_kw={"width_ratios": [7.5, 1.0]},
    )

    bars = ax.bar(
        table["diagonal"],
        area_ft2,
        color=[FAMILY_COLOURS[family] for family in table["family"]],
        width=0.78,
    )
    ax.bar_label(
        bars,
        labels=[f"{value:,.0f}" for value in area_ft2],
        rotation=90,
        padding=2,
        fontsize=5.5,
    )
    ax.set_xticks(table["diagonal"])
    ax.set_xticklabels(
        [f"D{number:02d}" for number in table["diagonal"]],
        rotation=90,
        fontsize=5.5,
    )
    ax.set(
        xlim=(table["diagonal"].min() - 0.8, table["diagonal"].max() + 0.8),
        ylim=(0, max(float(area_ft2.max()), 1.0) * 1.16),
        ylabel="Canopy area (ft²)",
        title="Each diagonal",
    )
    ax.legend(
        handles=[
            Patch(facecolor=FAMILY_COLOURS[family], label=family)
            for family in ("EVEN", "ODD")
        ],
        loc="upper left",
        fontsize=10,
    )

    end_bars = end.bar(
        ["EVEN", "ODD"],
        totals["area"],
        color=[FAMILY_COLOURS["EVEN"], FAMILY_COLOURS["ODD"]],
        width=0.62,
    )
    end.bar_label(
        end_bars,
        labels=[
            f"{area:,.0f} ft²\n{int(count):,} crowns"
            for area, count in zip(totals["area"], totals["crowns"])
        ],
        padding=4,
        fontsize=9,
    )
    end.set(
        ylim=(0, max(float(totals["area"].max()), 1.0) * 1.28),
        ylabel="Canopy area (ft²)",
        title=f"Total\n{totals['area'].sum():,.0f} ft²",
    )

    for axis in (ax, end):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)

    fig.suptitle("Canopy area per diagonal", fontsize=16, weight="bold")

    path = out / "canopy_area_per_diagonal.png"
    fig.savefig(path, dpi=MAP_DPI, facecolor="white")
    plt.close(fig)

    return path


def make_even_odd_report(df, diagonal_summary, out):
    table = (
        df.groupby("family")
        .agg(
            candidate_count=("crown_id", "size"),
            area_m2=("area_m2", "sum"),
        )
        .reindex(["EVEN", "ODD"], fill_value=0)
        .reset_index()
    )
    table.insert(2, "area_ft2", table["area_m2"] * FT2_PER_M2)
    if "area_ft2" not in diagonal_summary.columns and "area_m2" in diagonal_summary.columns:
        diagonal_summary = diagonal_summary.copy()
        diagonal_summary.insert(diagonal_summary.columns.get_loc("area_m2"), "area_ft2", diagonal_summary["area_m2"] * FT2_PER_M2)

    assert table.candidate_count.sum() <= MAX_RETAINED_CROWNS

    table.to_csv(
        out / "odd_even_summary.csv",
        index=False,
    )

    total = float(table.area_m2.sum())

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5.3),
        layout="constrained",
    )

    for ax, column, title, unit in zip(
        axes,
        ["candidate_count", "area_ft2"],
        ["Trees (crowns)", "Total canopy area"],
        ["Number of trees", "Area (ft²)"],
    ):
        values = table[column].to_numpy()

        bars = ax.bar(
            table.family,
            values,
            color=["#168bd2", "#e45756"],
            width=0.52,
        )

        ax.bar_label(
            bars,
            labels=[
                f"{value:,.0f}"
                if column == "candidate_count"
                else f"{value:,.0f} ft²\n{100 * value / max(values.sum(), 1e-9):.1f} %"
                for value in values
            ],
            padding=6,
            fontsize=12,
        )

        ax.set(
            title=title,
            ylabel=unit,
            ylim=(
                0,
                max(float(values.max()), 1) * 1.18,
            ),
        )

        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"EVEN vs ODD in total (all: {table.area_ft2.sum():,.0f} ft², "
        f"{int(table.candidate_count.sum()):,} trees)",
        fontsize=17,
        weight="bold",
    )

    chart = out / "bar_chart_even_odd_totals.png"

    fig.savefig(
        chart,
        dpi=MAP_DPI,
        facecolor="white",
    )
    plt.close(fig)

    per_diagonal = save_diagonal_chart(df, diagonal_summary, out)

    with pd.ExcelWriter(
        out / "EVEN_ODD_Totals.xlsx",
        engine="openpyxl",
    ) as writer:
        table.to_excel(
            writer,
            sheet_name="Even_Odd_Totals",
            index=False,
        )

        diagonal_summary.to_excel(
            writer,
            sheet_name="Diagonal_Summary",
            index=False,
        )

        # Each diagonal above, then the total of each family at the end.
        sheet = writer.sheets["Diagonal_Summary"]
        columns = list(diagonal_summary.columns)
        sheet.append([])

        for name, family, count, area in [
            *[
                (f"{row.family} total", row.family, row.candidate_count, row.area_m2)
                for row in table.itertuples()
            ],
            ("ALL total", "EVEN + ODD", table.candidate_count.sum(), table.area_m2.sum()),
        ]:
            values = [None] * len(columns)
            values[columns.index("diagonal")] = name
            values[columns.index("family")] = family
            values[columns.index("candidate_count")] = int(count)
            values[columns.index("area_m2")] = float(area)
            if "area_ft2" in columns:
                values[columns.index("area_ft2")] = float(area) * FT2_PER_M2
            sheet.append(values)

        df.drop(
            columns=["status", "review_flags"],
            errors="ignore",
        ).to_excel(
            writer,
            sheet_name="Crown_Inventory",
            index=False,
        )

        from openpyxl.drawing.image import Image as XLImage

        image = XLImage(str(chart))
        image.width = 840
        image.height = 371

        writer.sheets["Even_Odd_Totals"].add_image(
            image, "A6"
        )

        if per_diagonal is not None:
            image = XLImage(str(per_diagonal))
            image.width = 1150
            image.height = 447
            sheet.add_image(image, "J2")

        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = (
                f"A1:{worksheet.cell(1, worksheet.max_column).column_letter}"
                f"{len(diagonal_summary) + 1}"
                if worksheet is sheet else worksheet.dimensions
            )

            for cells in worksheet.columns:
                worksheet.column_dimensions[
                    cells[0].column_letter
                ].width = min(
                    46,
                    max(
                        16,
                        len(str(cells[0].value)) + 3,
                    ),
                )

    print(table.to_string(index=False), flush=True)

    log(
        f"Combined EVEN + ODD: "
        f"{int(table.candidate_count.sum()):,}"
    )

    log(f"Retained pecan canopy: {total * FT2_PER_M2:,.0f} ft² ({total:,.2f} m²)")

    if DISPLAY_MAPS:
        show_file(chart)
        if per_diagonal is not None:
            show_file(per_diagonal)

    return table


def export_geojson(path, geometries, properties, crs):
    transformer = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True,
    )

    def project(geometry):
        try:  # shapely >= 2
            import shapely
            return shapely.transform(
                geometry,
                lambda xy: np.column_stack(
                    transformer.transform(xy[:, 0], xy[:, 1])
                ),
            )
        except AttributeError:  # shapely 1.x
            return geom_transform(transformer.transform, geometry)

    features = [
        {
            "type": "Feature",
            "geometry": mapping(
                project(geometry)
            ),
            "properties": attributes,
        }
        for geometry, attributes in zip(
            geometries,
            properties,
        )
        if not geometry.is_empty
    ]

    save_json(
        path,
        {
            "type": "FeatureCollection",
            "features": features,
        },
    )


# ==================== MAPS + START ARROW ====================

FAMILY_COLOURS = {
    "EVEN": "#1e9be9",
    "ODD": "#ef4b3f",
}

# Darker shade for the lines drawn over the coloured crowns.
LINE_COLOURS = {
    "EVEN": "#0a4f9e",
    "ODD": "#b01e16",
}


def to_pixels(src, coords):
    inverse = ~src.transform
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)

    return np.column_stack([
        inverse.a * coords[:, 0] + inverse.b * coords[:, 1] + inverse.c,
        inverse.d * coords[:, 0] + inverse.e * coords[:, 1] + inverse.f,
    ])


def guide_parts(geometry):
    return [
        line for line in parts(geometry, "LineString")
        if line.length > 0
    ]


def label_point(src, guide):
    """Upper end of the longest part of a diagonal (pixel col, row)."""
    lines = guide_parts(guide["geometry"])
    if not lines:
        return None
    points = to_pixels(src, max(lines, key=lambda line: line.length).coords)
    return points[np.argmin(points[:, 1])]


def emptiest_corner(valid):
    """Map corner with the least field in it: (left?, top?)."""
    step = max(1, int(max(valid.shape) / 800))
    small = valid[::step, ::step]
    h, w = small.shape
    bh, bw = max(1, int(h * 0.17)), max(1, int(w * 0.30))
    corners = {
        (False, False): small[h - bh:, w - bw:],
        (True, False): small[h - bh:, :bw],
        (False, True): small[:bh, w - bw:],
        (True, True): small[:bh, :bw],
    }
    return min(corners, key=lambda corner: float(corners[corner].mean()))


def legend_below(ax, entries, totals=None):
    """Legend [(handle, label)] and canopy totals under the map, never
    over the field."""
    if entries:
        ax.legend(
            handles=[handle for handle, _ in entries],
            labels=[label for _, label in entries],
            loc="upper right" if totals else "upper center",
            bbox_to_anchor=(0.49 if totals else 0.5, -0.006),
            fontsize=10,
            framealpha=1.0,
            edgecolor="#bbbbbb",
            handlelength=2.6,
        )
    if totals:
        ax.text(
            0.51 if entries else 0.5, -0.006, totals,
            transform=ax.transAxes,
            ha="left" if entries else "center",
            va="top",
            fontsize=10,
            linespacing=1.55,
            family="DejaVu Sans Mono",
            bbox={"boxstyle": "round,pad=0.5", "fc": "white",
                  "ec": "#bbbbbb", "lw": 0.8},
        )


def line_end(points):
    """Lower (far) end of a line in pixels and its direction of travel."""
    if points[0, 1] > points[-1, 1]:
        points = points[::-1]
    back = points[max(0, len(points) - 12)]
    direction = points[-1] - back
    norm = np.hypot(*direction)
    return points[-1], (direction / norm if norm else direction)


def label_line_end(
    ax, src, points, text, colour, size, placed, px_per_pt, axis=None,
    gap_m=1.2, owner=None,
):
    """Text at the far (lower) end of a line, along the diagonal direction
    (`axis`, pixels), reading outward; moved further out while it (or its
    leader) would overlap a label, leader or arrow head already placed.
    `placed`: list of (owner, shape in pixels)."""
    if points[0, 1] > points[-1, 1]:
        points = points[::-1]
    pixel_m = math.hypot(src.transform.a, src.transform.d)
    travelled = np.r_[0.0, np.cumsum(
        np.hypot(*np.diff(points, axis=0).T)
    )] * pixel_m
    back = points[max(0, int(np.searchsorted(
        travelled, travelled[-1] - 13.5, side="right"
    )) - 1)]
    direction = points[-1] - back
    if axis is not None:  # all labels parallel: the diagonal direction
        axis = np.asarray(axis, dtype=float)
        direction = axis if direction @ axis >= 0 else -axis
    if not np.hypot(*direction):
        return
    direction = direction / np.hypot(*direction)
    normal = np.array([-direction[1], direction[0]])
    length = (0.62 * len(text) + 0.6) * size * px_per_pt
    half = 0.45 * size * px_per_pt

    reach = (gap_m + 45.0) / pixel_m + length
    tip = Point(points[-1])
    others = [
        shape_ for who, shape_ in placed
        if (who != owner or owner is None) and shape_.distance(tip) < reach
    ]
    best = None

    # Along the line first, then beside it; the first free spot wins, else
    # the spot with the least overlap.
    for offset in (0, 1, -1, 2, -2):
        for shift_m in np.arange(gap_m, gap_m + 40.0, 2.5):
            start = (
                points[-1] + direction * (shift_m / pixel_m)
                + normal * (offset * 2.4 * half)
            )
            box = Polygon([
                start + normal * half, start - normal * half,
                start - normal * half + direction * length,
                start + normal * half + direction * length,
            ])
            moved = shift_m > gap_m or offset != 0
            leader = LineString([points[-1], start]) if moved else None
            clash = sum(
                box.intersection(other).area
                + (1e9 if leader is not None and leader.intersects(other) else 0.0)
                for other in others
                if box.intersects(other)
                or (leader is not None and leader.intersects(other))
            )
            if best is None or clash < best[0]:
                best = (clash, start, box, leader, moved)
            if clash == 0:
                break
        if best[0] == 0:
            break

    _, start, box, leader, moved = best
    placed.append((owner, box))

    if moved:  # thin leader from the arrow tip to the moved label
        placed.append((owner, leader.buffer(0.3 / pixel_m)))
        ax.plot(
            [points[-1][0], start[0]], [points[-1][1], start[1]],
            color=colour, lw=0.4, alpha=0.8, zorder=8,
        )

    angle = math.degrees(math.atan2(-direction[1], direction[0]))
    ha = "left"
    if abs(angle) > 90:  # keep the text upright, still reading outward
        angle -= math.copysign(180.0, angle)
        ha = "right"
    ax.text(
        *start, text, rotation=angle, rotation_mode="anchor",
        ha=ha, va="center", fontsize=size, color=colour, weight="bold",
        zorder=9, clip_on=True,
        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
    ).set_in_layout(False)


def add_scale_and_north(ax, src, valid=None):
    """Scale bar and north arrow together, in the emptiest map corner."""
    left, top = emptiest_corner(valid) if valid is not None else (False, False)
    x0 = 0.03 if left else 0.72
    y0 = 0.12 if top else 0.955
    metres_per_pixel = math.hypot(src.transform.a, src.transform.d)
    maximum = src.width * metres_per_pixel * 0.18
    scale = 10 ** math.floor(math.log10(maximum))
    scale *= max(value for value in [1, 2, 5] if value <= maximum / scale)
    x, y = src.width * x0, src.height * y0
    stroke = [pe.withStroke(linewidth=2.5, foreground="black")]

    ax.plot(
        [x, x + scale / metres_per_pixel], [y, y],
        color="black", lw=5, solid_capstyle="butt",
        path_effects=[pe.Stroke(linewidth=8, foreground="white"), pe.Normal()],
    )
    ax.text(
        x + scale / metres_per_pixel / 2, y - src.height * 0.02,
        f"{scale:g} m", ha="center", va="bottom", color="black",
        fontsize=11, weight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    linear = np.array([[src.transform.a, src.transform.b],
                       [src.transform.d, src.transform.e]])
    north = np.linalg.solve(linear, [0.0, 1.0])
    north = north / np.hypot(*north) * src.height * 0.09
    base = np.array([src.width * (0.26 if left else 0.965), y])
    ax.annotate(
        "", xy=tuple(base + north), xytext=tuple(base),
        arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 2.4,
                    "mutation_scale": 18},
    ).arrow_patch.set_path_effects(
        [pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()]
    )
    ax.text(
        *(base + north * 1.22), "N", ha="center", va="center",
        color="white", fontsize=13, weight="bold", path_effects=stroke,
    )


def arrow_at_end(ax, src, points, colour, size=8, lw=1.0, back_m=6.0):
    """Arrow head at the lower (far) end of a line, along the line."""
    if points[0, 1] > points[-1, 1]:
        points = points[::-1]
    travelled = np.r_[0.0, np.cumsum(
        np.hypot(*np.diff(points, axis=0).T)
    )] * math.hypot(src.transform.a, src.transform.d)
    tail = points[max(0, int(np.searchsorted(
        travelled, travelled[-1] - back_m, side="right"
    )) - 1)]
    ax.annotate(
        "", xy=tuple(points[-1]), xytext=tuple(tail),
        arrowprops={"arrowstyle": "-|>", "color": colour, "lw": lw,
                    "mutation_scale": size, "shrinkA": 0, "shrinkB": 0},
        zorder=6,
    )


def draw_start_arrow(ax, src, pieces, colour, solid=None, bridge=()):
    """The START diagonal (its own colour), arrow head at its far end;
    dotted where it passes canopy of another tree."""
    outline = [pe.Stroke(linewidth=5.4, foreground="white"), pe.Normal()]

    for points in (pieces if solid is None else solid):
        ax.plot(
            points[:, 0], points[:, 1],
            color=colour, lw=3.0, solid_capstyle="round", zorder=19,
            path_effects=outline,
        )

    for points in bridge:
        ax.plot(
            points[:, 0], points[:, 1],
            color=colour, lw=3.0, linestyle=(0, (1.2, 1.2)), zorder=19,
        )

    last = pieces[-1]
    travelled = np.r_[0.0, np.cumsum(
        np.hypot(*np.diff(last, axis=0).T)
    )] * math.hypot(src.transform.a, src.transform.d)
    tail = last[max(0, int(np.searchsorted(
        travelled, travelled[-1] - 8.0, side="right"
    )) - 1)]

    ax.annotate(
        "", xy=tuple(last[-1]), xytext=tuple(tail),
        arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 3.0,
                    "mutation_scale": 30, "shrinkA": 0, "shrinkB": 0},
        zorder=20,
    ).arrow_patch.set_path_effects(
        [pe.Stroke(linewidth=5.0, foreground="white"), pe.Normal()]
    )


def save_diagonal_tiles(src, rgb, crowns, df, guides, out):
    """Zoomed view (60 m tiles): each wavy diagonal crosses every pecan
    crown on it; crown outlines in the colour of their diagonal."""
    pixel_m = math.hypot(src.transform.a, src.transform.d)
    size = int(round(60.0 / pixel_m))

    if size >= min(src.width, src.height):
        return None

    tops = to_pixels(src, df[["top_x", "top_y"]].to_numpy())
    on_grid = df["planting_block"].notna().to_numpy()
    centres = []

    for band in np.array_split(np.argsort(tops[:, 0]), 4):
        band = band[on_grid[band]]
        if len(band):
            pick = band[np.argsort(tops[band, 1])[len(band) // 2]]
            centres.append(tops[pick])

    if not centres:
        return None

    families = dict(zip(df.label_id, df.family))
    rings = {"EVEN": [], "ODD": []}
    fills = {"EVEN": [], "ODD": []}

    for label_id, geometry in crowns:
        if label_id in families:
            for polygon in parts(geometry, "Polygon"):
                ring = to_pixels(src, polygon.exterior.coords)
                fills[families[label_id]].append(ring)
                rings[families[label_id]].append(ring)
                for hole in polygon.interiors:
                    rings[families[label_id]].append(to_pixels(src, hole.coords))

    lines = [
        (
            guide,
            to_pixels(src, line.coords),
            [to_pixels(src, part.coords) for part in guide_parts(guide.get("solid", line))],
            [to_pixels(src, part.coords) for part in guide_parts(guide.get("bridge", GeometryCollection()))],
        )
        for guide in guides
        for line in guide_parts(guide["geometry"])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 13.6), layout="constrained")

    for ax, (cx, cy) in zip(axes.ravel(), centres):
        c0 = int(np.clip(cx - size / 2, 0, src.width - size))
        r0 = int(np.clip(cy - size / 2, 0, src.height - size))
        box = (c0 - 60, c0 + size + 60, r0 - 60, r0 + size + 60)

        def visible(points):
            return (
                (points[:, 0] > box[0]) & (points[:, 0] < box[1])
                & (points[:, 1] > box[2]) & (points[:, 1] < box[3])
            )

        ax.imshow(
            rgb[r0:r0 + size, c0:c0 + size],
            extent=(c0, c0 + size, r0 + size, r0),
            interpolation="nearest",
        )

        # Each crown tinted in its own colour, white borders between crowns.
        for family in ("EVEN", "ODD"):
            ax.add_collection(PolyCollection(
                [ring for ring in fills[family] if visible(ring).any()],
                facecolors=FAMILY_COLOURS[family], edgecolors="none", alpha=0.38,
            ))
            ax.add_collection(LineCollection(
                [ring for ring in rings[family] if visible(ring).any()],
                colors="white", linewidths=0.9, alpha=0.95,
            ))

        for guide, points, solid, bridge in lines:
            if not visible(points).any():
                continue
            ax.plot(
                points[:, 0], points[:, 1],
                color=LINE_COLOURS[guide["family"]], lw=2.4,
                path_effects=[pe.Stroke(linewidth=3.8, foreground="white"), pe.Normal()],
            )
            end = points[np.argmax(points[:, 1])]
            if c0 < end[0] < c0 + size and r0 < end[1] < r0 + size:
                arrow_at_end(
                    ax, src, points, LINE_COLOURS[guide["family"]],
                    size=18, lw=2.4, back_m=3.0,
                )

        ax.set(xlim=(c0, c0 + size), ylim=(r0 + size, r0), aspect="equal")
        ax.set_title(f"({'abcd'[list(axes.ravel()).index(ax)]}) 60 m x 60 m", fontsize=11)
        ax.axis("off")

    axes.ravel()[0].legend(
        handles=[
            (
                Patch(facecolor=FAMILY_COLOURS[f], alpha=0.38, edgecolor="white"),
                Line2D([0], [0], color=LINE_COLOURS[f], lw=2.4),
            )
            for f in ("EVEN", "ODD")
        ],
        labels=["EVEN diagonal and crowns", "ODD diagonal and crowns"],
        loc="lower left", fontsize=10, framealpha=0.92,
    )
    fig.suptitle(
        "EVEN and ODD diagonals (60 m detail)",
        fontsize=15, weight="bold",
    )

    path = out / "diagonal_check_tiles.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

    return path


def save_web_preview(src, rgb, valid, out, max_side=4096):
    """Downsampled orthomosaic (JPEG) + its bounds, for the web dashboard
    (Emmanuel Dashboard for Pecan Thinning: open the results ZIP there)."""
    if src.transform.b or src.transform.d:  # rotated raster: no simple overlay
        return None

    step = max(1, math.ceil(max(src.width, src.height) / max_side))
    preview = rgb[::step, ::step]

    if preview.dtype != np.uint8:  # e.g. 16-bit: stretch to 0-255
        low, high = np.percentile(preview[valid[::step, ::step]], [1, 99])
        preview = np.clip((preview - low) / max(high - low, 1e-9) * 255, 0, 255)

    preview = preview.astype(np.uint8)
    preview[~valid[::step, ::step]] = 255
    path = out / "web_preview.jpg"
    cv2.imwrite(
        str(path),
        np.ascontiguousarray(preview[:, :, ::-1]),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    left, bottom, right, top = src.bounds
    save_json(
        out / "web_preview.json",
        {
            "crs": str(src.crs),
            "bounds": [left, bottom, right, top],
            "width": int(preview.shape[1]),
            "height": int(preview.shape[0]),
        },
    )
    return path


def draw_maps(
    src, rgb, valid, crowns, df,
    guides, settings, out,
):
    step = max(
        1,
        math.ceil(max(valid.shape) / 4000),
    )

    preview = rgb[::step, ::step].copy()
    preview[~valid[::step, ::step]] = 255

    colors = FAMILY_COLOURS

    # Thinning removes the family with the smaller total canopy and starts on
    # its first drawn diagonal (D01 for ODD, D02 for EVEN), as the dashboard.
    drawn_numbers = sorted(g["diagonal"] for g in guides if guide_parts(g["geometry"]))
    if str(START_DIAGONAL).lower() == "auto":
        area_by_family = df.groupby("family")["area_m2"].sum()
        smaller = (
            "ODD" if float(area_by_family.get("ODD", 0.0)) <= float(area_by_family.get("EVEN", 0.0))
            else "EVEN"
        )
        start_number = next(
            (d for d in drawn_numbers if ("ODD" if d % 2 else "EVEN") == smaller),
            drawn_numbers[0] if drawn_numbers else None,
        )
    else:
        start_number = int(START_DIAGONAL)

    requested = next(
        (guide for guide in guides if guide["diagonal"] == start_number),
        None,
    )

    if requested is None:
        raise ValueError(
            f"START_DIAGONAL = {START_DIAGONAL}: no such diagonal. "
            'Use START_DIAGONAL = "auto".'
        )

    start_family = requested["family"]
    segments = guide_parts(requested["geometry"])

    if not segments:
        raise ValueError(
            f"D{start_number:02d} has no line in the field. "
            'Use START_DIAGONAL = "auto" or another diagonal.'
        )

    # START at the upper end; the arrow follows the wavy diagonal down.
    oriented = []

    for line in segments:
        coords = np.asarray(line.coords)
        pixels = to_pixels(src, coords)
        if pixels[0, 1] > pixels[-1, 1]:
            coords, pixels = coords[::-1], pixels[::-1]
        oriented.append((pixels, coords))

    oriented.sort(key=lambda item: item[0][0, 1])
    start_pieces = [pixels for pixels, _ in oriented]
    line_length = float(sum(line.length for line in segments))

    col, row = map(float, start_pieces[0][0])
    x, y = oriented[0][1][0]

    lon, lat = Transformer.from_crs(
        src.crs,
        "EPSG:4326",
        always_xy=True,
    ).transform(x, y)

    start_name = f"D{start_number:02d}"

    start = {
        "diagonal": start_number,
        "family": start_family,
        "pixel_col": col,
        "pixel_row": row,
        "longitude": lon,
        "latitude": lat,
        "arrow_length_m": line_length,
        "rule": (
            "family with the smaller total canopy, its first diagonal"
            if str(START_DIAGONAL).lower() == "auto" else "START_DIAGONAL setting"
        ),
        "note": (
            f"Thinning starts at the upper end of {start_name} ({start_family}); the "
            "arrow follows its wavy line through every crown on it."
        ),
    }

    export_geojson(
        out / "start_arrow_wgs84.geojson",
        [MultiLineString([LineString(coords) for _, coords in oriented])],
        [start],
        src.crs,
    )

    save_json(
        out / "suggested_start.json",
        start,
    )

    log(
        f"START: {start_name}, {start_family} ({start['rule']}) | "
        f"arrow along the wavy line: {line_length:.2f} m"
    )

    # Every retained candidate is annotated.
    assert len(df) <= MAX_RETAINED_CROWNS

    df.to_csv(
        out / "annotated_crowns.csv",
        index=False,
    )

    save_json(
        out / "annotation_summary.json",
        {
            "retained_candidates": len(df),
            "annotated_candidates": len(df),
            "selection": (
                "All retained canopy-only crown candidates."
            ),
            "note": (
                "Map, chart, inventory and area totals use "
                "the same retained candidates."
            ),
        },
    )

    families = dict(
        zip(df.label_id, df.family)
    )

    outlines = {
        "EVEN": [],
        "ODD": [],
    }
    fills = {
        "EVEN": [],
        "ODD": [],
    }

    for label_id, geometry in crowns:
        for polygon in parts(
            geometry,
            "Polygon",
        ):
            fills[families[label_id]].append(
                to_pixels(src, polygon.exterior.coords)
            )
            for ring in [
                polygon.exterior,
                *polygon.interiors,
            ]:
                outlines[families[label_id]].append(
                    to_pixels(src, ring.coords)
                )

    def pixel_parts(geometry):
        return [to_pixels(src, line.coords) for line in guide_parts(geometry)]

    guide_pixels = [
        (guide, pixel_parts(guide["geometry"])) for guide in guides
    ]

    # Canopy totals (square feet): each diagonal (written at the end of its
    # line) and each family (under the map).
    area_of = (df.groupby("diagonal")["area_m2"].sum() * FT2_PER_M2).to_dict()
    crowns_of = df.groupby("diagonal").size().to_dict()
    family_table = (
        df.assign(area_ft2=df["area_m2"] * FT2_PER_M2)
        .groupby("family")
        .agg(
            crowns=("crown_id", "size"),
            area=("area_ft2", "sum"),
            diagonals=("diagonal", "nunique"),
        )
        .reindex(["EVEN", "ODD"], fill_value=0)
    )

    def totals_text(shown):
        lines = ["Total canopy"]
        for family in shown:
            row = family_table.loc[family]
            lines.append(
                f"{family:<4} {row['area']:>10,.0f} ft² · "
                f"{int(row['crowns']):>5,} crowns · "
                f"{int(row['diagonals']):>3} diagonals"
            )
        if len(shown) > 1:
            lines.append(
                f"{'All':<4} {family_table['area'].sum():>10,.0f} ft² · "
                f"{int(family_table['crowns'].sum()):>5,} crowns · "
                f"{int(family_table['diagonals'].sum()):>3} diagonals"
            )
        return "\n".join(lines)

    # Area labels sized to fit between neighbouring diagonals.
    pixel_m = math.hypot(src.transform.a, src.transform.d)
    points_per_m = 13 * 72 / (src.width * pixel_m)
    px_per_pt = 1.1 / (points_per_m * pixel_m)
    label_size = float(np.clip(
        0.72 * settings.get("diagonal_spacing_m", 5.4) * points_per_m, 3.0, 8.0
    ))
    theta = math.radians(settings.get("diagonal_image_angle_deg", 135.0))
    label_axis = np.array([math.cos(theta), math.sin(theta)])
    margin = 40.0 / pixel_m  # white margin: labels outside the field fit

    modes = [
        ("overview", "Pecan field"),
        ("both", "EVEN and ODD diagonals"),
        ("odd", "ODD diagonals"),
        ("even", "EVEN diagonals"),
        ("crowns", "EVEN and ODD crowns"),
        ("final_thinning_start", f"Thinning start: {start_name} ({start_family})"),
    ]

    for mode, title in modes:
        # Room for the title above and the legend + totals under the map.
        fig, ax = plt.subplots(
            figsize=(
                13,
                13 * src.height / src.width + (1.3 if mode == "overview" else 2.9),
            ),
            layout="constrained",
        )

        ax.imshow(
            preview,
            extent=(0, src.width, src.height, 0),
            interpolation="nearest",
        )

        shown = (
            ["EVEN", "ODD"] if mode in ("both", "crowns")
            else [mode.upper()] if mode in ("even", "odd")
            else []
        )

        # Crown segmentation, each crown in the colour of its diagonal.
        # Crowns map: solid fill, white borders (1 blue, 2 red ... along
        # every row). Diagonal maps: light fill + coloured outline, so the
        # line is seen running through each of its crowns.
        for family in shown:
            ax.add_collection(
                PolyCollection(
                    fills[family],
                    facecolors=colors[family],
                    edgecolors="none",
                    alpha=0.75 if mode == "crowns" else 0.40,
                )
            )
            ax.add_collection(
                LineCollection(
                    outlines[family],
                    colors="white",
                    linewidths=0.35,
                    alpha=0.9,
                )
            )

        if mode in ("both", "odd", "even"):
            # Arrow heads first, so no area label is written over one.
            placed = []

            for guide, full in guide_pixels:
                if guide["family"] in shown and full:
                    tip, heading = line_end(max(full, key=lambda p: p[:, 1].max()))
                    placed.append((
                        guide["diagonal"],
                        LineString([tip, tip - heading * (3.5 / pixel_m)]).buffer(1.0 / pixel_m),
                    ))

            for guide, full in guide_pixels:
                if guide["family"] not in shown or not full:
                    continue

                colour = LINE_COLOURS[guide["family"]]

                for points in full:
                    ax.plot(
                        points[:, 0],
                        points[:, 1],
                        color=colour,
                        lw=0.9,
                        solid_capstyle="round",
                        path_effects=[
                            pe.Stroke(linewidth=1.9, foreground="white"),
                            pe.Normal(),
                        ],
                    )

                # Arrow and total canopy of the diagonal at its far end.
                end_piece = max(full, key=lambda p: p[:, 1].max())
                arrow_at_end(ax, src, end_piece, colour, size=8, lw=0.9)
                label_line_end(
                    ax, src, end_piece,
                    f"{area_of.get(guide['diagonal'], 0.0):,.0f} ft²",
                    colour, label_size, placed, px_per_pt, label_axis,
                    owner=guide["diagonal"],
                )

        entries = []
        totals = totals_text(shown) if shown else None

        if mode == "crowns":
            entries = [
                (
                    Patch(facecolor=colors[family], alpha=0.75, edgecolor="white"),
                    f"{family} crowns",
                )
                for family in shown
            ]
        elif mode in ("both", "odd", "even"):
            entries = [
                (
                    (
                        Patch(facecolor=colors[family], alpha=0.40,
                              edgecolor="white"),
                        Line2D([0], [0], color=LINE_COLOURS[family], lw=2.0,
                               marker=">", markersize=6),
                    ),
                    f"{family} diagonals and crowns",
                )
                for family in shown
            ]
            totals += "\n(ft² at the end of each line = canopy of that diagonal)"

        # Final thinning map: ONLY the START diagonal, with its arrow.
        if mode == "final_thinning_start":
            draw_start_arrow(ax, src, start_pieces, colors[start_family])

            ax.annotate(
                f"START\n{start_name} ({start_family})",
                xy=tuple(start_pieces[0][0]),
                xytext=(-16, 10),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=13,
                weight="bold",
                zorder=21,
                bbox={
                    "boxstyle": "round,pad=.35",
                    "fc": "white",
                    "ec": "#263747",
                    "lw": 1.2,
                },
                arrowprops={"arrowstyle": "-", "color": "#263747", "lw": 1.2},
            )

            entries.append(
                (
                    Line2D([0], [0], color=colors[start_family], lw=3.0,
                           marker=">", markersize=10),
                    f"{start_name} ({start_family}): start, arrow shows direction",
                )
            )
            start_count = int(crowns_of.get(start_number, 0))
            totals = (
                f"{start_name} canopy\n"
                f"{area_of.get(start_number, 0.0):,.0f} ft² · "
                f"{start_count} crown{'' if start_count == 1 else 's'}"
            )

        legend_below(ax, entries, totals)

        ax.set(
            xlim=(-margin, src.width + margin),
            ylim=(src.height + margin, -margin),
            aspect="equal",
        )

        ax.axis("off")

        ax.set_title(
            title,
            fontsize=17,
            weight="bold",
            pad=10,
        )

        add_scale_and_north(ax, src, valid)

        path = out / f"{mode}_map.png"

        fig.savefig(
            path,
            dpi=MAP_DPI,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.15,
        )

        plt.close(fig)

        if DISPLAY_MAPS:
            show_file(path)

    return start


# ==================== DASHBOARD ENGINE (identical to the website) ====================
#
# The image is analysed by the dashboard's own engine (js/analyze.js,
# embedded at the end of this file) with Node.js. The image and DSM are read
# here at full resolution and handed over unchanged; the engine averages
# them onto its analysis grid exactly as the website does. The maps,
# tables, Excel file, GeoJSON and ZIP are then made from its result.

DASHBOARD_NODE_RUNNER = r"""
const fs = require("fs");
const path = require("path");
globalThis.window = globalThis;
require(process.argv[2]);
const P = window.PecanAnalysis;
const meta = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
function load(file, type) {
  const buf = fs.readFileSync(file);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  if (type === "uint16") return new Uint16Array(ab);
  if (type === "float32") return new Float32Array(ab);
  return new Uint8Array(ab);
}
function save(name, arr) {
  fs.writeFileSync(path.join(meta.out, name), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
}
(async () => {
  const bands = meta.bands.map((b) => load(b.file, b.dtype));
  const { px, py } = P.groundPixelSize({ kind: meta.kind, rx: meta.rx, ry: meta.ry, miny: meta.miny, maxy: meta.maxy, unitFactor: meta.unitFactor });
  const fraction = meta.hasAlpha ? P.countField(bands[3]) / (meta.width * meta.height) : 1;
  const { W, H, pixel } = P.gridSize(meta.width, meta.height, px, py, fraction, meta.budget);
  console.log("  analysis grid " + W + " x " + H + " at " + (pixel * 100).toFixed(1) + " cm");
  const grid = P.createGrid(meta.width, meta.height, W, H, { hasAlpha: meta.hasAlpha, nodata: meta.nodata, bits: meta.bits });
  grid.addRows(0, meta.height, bands);
  const rgba = grid.finish();
  bands.length = 0;
  let surface = null, dsmProblem = "";
  if (meta.dsm) {
    const data = load(meta.dsm.file, "float32");
    const sampled = P.sampleSurface(
      { data, width: meta.dsm.width, height: meta.dsm.height, left: meta.dsm.left, top: meta.dsm.top, resX: meta.dsm.resX, resY: meta.dsm.resY, nodata: meta.dsm.nodata },
      meta.img, meta.width, meta.height, W, H);
    if (sampled.inside < 0.2 * W * H) dsmProblem = "it does not cover this image (another field?)";
    else surface = sampled.surface;
  }
  // full-resolution windows for the young-tree look (8-bit images, as the website)
  let chip = null;
  if (meta.bits <= 8) {
    const fds = meta.bands.map((b) => fs.openSync(b.file, "r"));
    chip = {
      width: meta.width, height: meta.height, hasAlpha: meta.hasAlpha, nodata: meta.nodata,
      read: async (x0, y0, x1, y1) => fds.map((fd) => {
        const ww = x1 - x0, out = new Uint8Array(ww * (y1 - y0));
        for (let y = y0; y < y1; y++) fs.readSync(fd, out, (y - y0) * ww, ww, y * meta.width + x0);
        return out;
      }),
    };
  }
  const result = await P.analyze({ rgba, width: W, height: H, pixel, surface },
    { designFt: meta.designFt, directionHintDeg: meta.hint, youngTrees: meta.youngTrees !== false, returnRasters: true, chip },
    (text) => console.log("  " + text));
  save("mask.bin", result.mask);
  save("labels.bin", result.labels);
  if (result.heights) save("heights.bin", result.heights);
  fs.writeFileSync(path.join(meta.out, "result.json"), JSON.stringify({
    crowns: result.crowns, diagonals: result.diagonals, settings: result.settings,
    segmentation: result.segmentation, field_area_m2: result.field_area_m2,
    canopy_area_m2: result.canopy_area_m2, W, H, pixel, px, py, fraction,
    dsm_problem: dsmProblem, version: P.VERSION,
  }));
})().catch((error) => { console.error((error && error.stack) || error); process.exit(1); });
"""


def dashboard_engine_source():
    import base64
    import zlib
    return zlib.decompress(base64.b64decode("".join(DASHBOARD_ENGINE_B64))).decode("utf-8")


def node_command():
    """How to run Node.js: `node` if installed, else the nodejs-bin package
    (pip; needs Internet on in the Kaggle notebook settings)."""
    import os
    override = os.environ.get("PECAN_NODE")
    if override:
        return [override]
    node = shutil.which("node")
    if node:
        return [node]
    if importlib.util.find_spec("nodejs") is None and INSTALL_MISSING:
        print("Installing Node.js (nodejs-bin) ...", flush=True)
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "nodejs-bin"])
    if importlib.util.find_spec("nodejs") is not None:
        return [sys.executable, "-m", "nodejs"]
    raise RuntimeError(
        "Node.js is needed for ENGINE = 'dashboard' and could not be installed. "
        "In Kaggle: Settings (right panel) > Internet: On, then run again. "
        "Or set ENGINE = 'python'."
    )


def georef_kind(src):
    """Coordinate kind and unit factor, as the dashboard reads them."""
    crs = CRS(src.crs)
    if crs.is_geographic:
        return "geographic", 1.0
    if crs.to_epsg() in WEB_MERCATOR:
        return "webmercator", 1.0
    factor = float(crs.axis_info[0].unit_conversion_factor or 1.0)
    if abs(factor - 0.3048) < 1e-9:
        return "projected", 0.3048
    if abs(factor - 1200 / 3937) < 1e-9:
        return "projected", 1200 / 3937
    return "projected", 1.0


def has_alpha_band(path, src):
    """Alpha band as the dashboard decides it (TIFF ExtraSamples)."""
    if src.count < 4:
        return False
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            tag = tf.pages[0].tags.get("ExtraSamples")
            extra = [] if tag is None else list(np.atleast_1d(tag.value))
        return 1 in extra or 2 in extra or (not extra and src.count == 4)
    except Exception:
        return src.colorinterp[3] == ColorInterp.alpha or src.count == 4


def run_dashboard(tif):
    """Analyse one orthomosaic with the dashboard engine; returns its folder."""
    tif = Path(tif)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tif.stem)[:60]
    out = OUTPUT_ROOT / f"Pecan_Canopy_Results_{name}_{timestamp}"
    out.mkdir()
    work = OUTPUT_ROOT / "_engine" / out.name
    work.mkdir(parents=True, exist_ok=True)

    log(f"Image: {tif}")
    log("Engine: dashboard (same analysis as https://emmanueldashboard.github.io).")
    log(f"Output: {out}")

    with rasterio.open(tif) as src:
        if src.count < 3:
            raise ValueError(f"{tif.name} has {src.count} band(s): an RGB orthomosaic is needed.")
        kind, unit_factor = georef_kind(src)
        t = src.transform
        alpha = has_alpha_band(tif, src)
        dtype = "uint16" if src.dtypes[0] in ("uint16", "int16") else "uint8"
        bands = []
        for k, b in enumerate([1, 2, 3] + ([4] if alpha else [])):
            data = src.read(b)
            file = work / f"band{k}.bin"
            data.astype("<u2" if dtype == "uint16" else "uint8").tofile(file)
            bands.append({"file": str(file), "dtype": dtype})
            del data
        top, left = t.f, t.c
        bottom = top + src.height * t.e
        meta = {
            "width": src.width, "height": src.height, "bands": bands,
            "hasAlpha": bool(alpha), "nodata": None if src.nodata is None else float(src.nodata),
            "bits": 16 if dtype == "uint16" else 8,
            "kind": kind, "rx": abs(t.a), "ry": abs(t.e), "miny": min(top, bottom), "maxy": max(top, bottom),
            "unitFactor": unit_factor,
            "budget": {"fast": 3e6, "standard": 5e6, "fine": 9e6}[DASHBOARD_DETAIL],
            "designFt": list(DESIGN_SPACING_FT) if DESIGN_SPACING_FT else None,
            "hint": DIAGONAL_DIRECTION_HINT_DEG,
            "youngTrees": bool(YOUNG_TREES),
            "img": {"left": left, "top": top, "resX": t.a, "resY": t.e},
            "dsm": None, "out": str(work),
        }

        # Elevation (DSM): read at full resolution, handed over unchanged.
        dsm_path = resolve_elevation(src)
        dsm_note = "no elevation file"
        if dsm_path is not None:
            log(f"Elevation raster: {dsm_path}")
            with rasterio.open(dsm_path) as dem:
                if CRS(dem.crs) == CRS(src.crs):
                    values = dem.read(1).astype("float32")
                    dt = dem.transform
                    nodata = dem.nodata
                else:  # other CRS: onto this image's CRS first
                    dt, w2, h2 = calculate_default_transform(dem.crs, src.crs, dem.width, dem.height, *dem.bounds)
                    values = np.full((h2, w2), np.nan, dtype="float32")
                    reproject(rasterio.band(dem, 1), values, dst_transform=dt, dst_crs=src.crs,
                              dst_nodata=np.nan, resampling=Resampling.nearest)
                    nodata = None
                file = work / "dsm.bin"
                values.astype("<f4").tofile(file)
                meta["dsm"] = {
                    "file": str(file), "width": int(values.shape[1]), "height": int(values.shape[0]),
                    "left": dt.c, "top": dt.f, "resX": dt.a, "resY": dt.e,
                    "nodata": None if nodata is None else float(nodata),
                }
                del values

        engine = work / "analyze.js"
        engine.write_text(dashboard_engine_source(), encoding="utf-8")
        runner = work / "run_engine.js"
        runner.write_text(DASHBOARD_NODE_RUNNER, encoding="utf-8")
        (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        log("Running the dashboard engine (Node.js) ...")
        command = node_command() + ["--max-old-space-size=8192", str(runner), str(engine), str(work / "meta.json")]
        if command[:3] == [sys.executable, "-m", "nodejs"]:
            command = command[:3] + command[4:]  # nodejs-bin: no V8 flags
        process = subprocess.run(command, capture_output=True, text=True)
        for line in (process.stdout or "").splitlines():
            log(line.strip())
        if process.returncode:
            raise RuntimeError("Dashboard engine failed:\n" + (process.stderr or process.stdout)[-3000:])
        result = json.loads((work / "result.json").read_text(encoding="utf-8"))

        W, H, pixel = result["W"], result["H"], result["pixel"]
        seg = result["segmentation"]
        elevation = seg.get("elevation", {})
        if dsm_path is not None:
            elevation["path"] = str(dsm_path)
            if result.get("dsm_problem"):
                elevation = {"used": False, "path": str(dsm_path), "reason": result["dsm_problem"]}
                seg["elevation"] = elevation
        if dsm_path is not None and not elevation.get("used"):
            log(f"WARNING: {Path(dsm_path).name} not used: {elevation.get('reason')}.")
        log(
            f"Analysis grid {W} x {H} at {pixel * 100:.1f} cm; elevation (DSM) "
            f"{'used' if elevation.get('used') else 'not used'}; cast shadow trimmed "
            f"{seg.get('cast_shadow', {}).get('shadow_removed_m2', 0):,.0f} m2."
        )

        # Local metres of the engine (origin lower-left) -> image CRS.
        sx = (src.width * t.a) / (W * pixel)
        sy = (src.height * abs(t.e)) / (H * pixel)

        def to_crs(points):
            return [(left + x * sx, bottom + y * sy) for x, y in points]

        crowns, records = [], []
        for c in result["crowns"]:
            polygons = []
            for poly in c["polys"]:
                rings = [to_crs(ring) for ring in poly if len(ring) >= 4]
                if rings:
                    polygons.append(Polygon(rings[0], rings[1:]))
            geometry = unary_union([p if p.is_valid else p.buffer(0) for p in polygons])
            label_id = int(c["id"][1:])
            crowns.append((label_id, geometry))
            top_x, top_y = to_crs([c["top"]])[0]
            family = "ODD" if abs(int(c["diagonal"])) % 2 else "EVEN"
            centroid = geometry.centroid
            records.append({
                "crown_id": c["id"], "label_id": label_id, "status": "DASHBOARD_ENGINE",
                "area_m2": float(c["area"]), "area_ft2": float(c["area"]) / 0.3048 ** 2,
                "x": centroid.x, "y": centroid.y, "diagonal": int(c["diagonal"]), "family": family,
                "top_x": top_x, "top_y": top_y,
                "planting_block": 1 if c.get("grid_row") is not None else None,
                "grid_row": c.get("grid_row"), "grid_tree": c.get("grid_tree"),
                "review_flags": c.get("flags", ""),
                "height_median_m": c.get("height_median_m"), "height_p90_m": c.get("height_p90_m"),
                "height_top_m": c.get("height_top_m"), "height_mean_m": c.get("height_mean_m"),
            })

        guides = []
        for d in result["diagonals"]:
            lines = [LineString(to_crs(piece)) for piece in d["pieces"] if len(piece) > 1]
            geometry = (MultiLineString(lines) if len(lines) > 1 else lines[0] if lines else GeometryCollection())
            guides.append({
                "diagonal": int(d["number"]), "family": "ODD" if int(d["number"]) % 2 else "EVEN",
                "block": 1, "index": None, "geometry": geometry, "tree_sites": 0,
            })
        guides.sort(key=lambda g: g["diagonal"])
        by_number = {g["diagonal"]: g for g in guides}

        df = pd.DataFrame(records)
        for column in ["planting_block", "grid_row", "grid_tree"]:
            df[column] = pd.array(df[column], dtype="Int64")
        df["guide_distance_m"] = [
            by_number[r.diagonal]["geometry"].distance(Point(r.top_x, r.top_y))
            if r.diagonal in by_number and not by_number[r.diagonal]["geometry"].is_empty else np.nan
            for r in df.itertuples()
        ]
        order = df.sort_values(["diagonal", "top_y"], ascending=[True, False]).groupby("diagonal").cumcount() + 1
        df["tree_on_diagonal"] = pd.array(order.reindex(df.index), dtype="Int64")
        for g in guides:
            g["tree_sites"] = int((df["diagonal"] == g["diagonal"]).sum())
        df = df.sort_values("label_id").reset_index(drop=True)

        # Rasters at the engine's resolution.
        grid_transform = Affine(src.width * t.a / W, 0.0, left, 0.0, src.height * t.e / H, top)
        profile = {"driver": "GTiff", "width": W, "height": H, "count": 1, "crs": src.crs,
                   "transform": grid_transform, "compress": "deflate"}
        mask = np.fromfile(work / "mask.bin", dtype="uint8").reshape(H, W)
        labels = np.fromfile(work / "labels.bin", dtype="<i4").reshape(H, W)
        keep = np.isin(labels, df["label_id"].to_numpy())
        with rasterio.open(out / "canopy_mask.tif", "w", dtype="uint8", nodata=255, **profile) as dst:
            dst.write(mask, 1)
        with rasterio.open(out / "candidate_labels.tif", "w", dtype="int32", nodata=0, **profile) as dst:
            dst.write(np.where(keep, labels, 0).astype("int32"), 1)
        if (work / "heights.bin").is_file():
            heights = np.fromfile(work / "heights.bin", dtype="<f4").reshape(H, W)
            with rasterio.open(out / "canopy_height_model_cm.tif", "w", dtype="int16", nodata=-32768, **profile) as dst:
                dst.write(np.where(np.isfinite(heights), np.clip(np.round(heights * 100), -32000, 32000), -32768).astype("int16"), 1)
            del heights
        del mask, labels, keep

        settings = dict(result["settings"])
        settings["engine"] = f"dashboard js/analyze.js {result.get('version')}"
        settings["analysis_pixel_m"] = pixel
        settings["analysis_grid"] = [W, H]

        df.to_csv(out / "all_candidates.csv", index=False)
        df.to_csv(out / "raw_candidates.csv", index=False)
        grouped = (
            df.groupby(["diagonal", "family"])
            .agg(candidate_count=("crown_id", "size"), area_m2=("area_m2", "sum"))
            .reset_index()
        )
        base = pd.DataFrame([
            {"diagonal": g["diagonal"], "family": g["family"], "planting_block": g["block"],
             "grid_sites_with_tree": g["tree_sites"], "line_length_m": float(g["geometry"].length)}
            for g in guides
        ])
        grouped = base.merge(grouped, how="left", on=["diagonal", "family"])
        for column in ["candidate_count", "area_m2"]:
            grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(0)
        grouped["candidate_count"] = grouped["candidate_count"].astype(int)
        grouped.insert(grouped.columns.get_loc("area_m2"), "area_ft2", grouped["area_m2"] * FT2_PER_M2)
        grouped.to_csv(out / "diagonal_summary.csv", index=False)
        family_summary = (
            df.groupby("family").agg(candidate_count=("crown_id", "size"), area_m2=("area_m2", "sum")).reset_index()
        )
        family_summary.insert(2, "area_ft2", family_summary["area_m2"] * FT2_PER_M2)
        family_summary.to_csv(out / "odd_even_summary.csv", index=False)

        young = int(df["review_flags"].fillna("").str.contains("YOUNG_TREE").sum())
        log(f"Young trees counted on empty planting positions (flag YOUNG_TREE): {young}")
        log(f"Pecan canopy (final): {result['canopy_area_m2'] * FT2_PER_M2:,.0f} ft2 ({result['canopy_area_m2']:,.1f} m2)")
        make_even_odd_report(df, grouped, out)

        export_geojson(out / "crowns_wgs84.geojson", [g for _, g in crowns],
                       json.loads(df.to_json(orient="records")), src.crs)
        export_geojson(out / "guides_wgs84.geojson", [g["geometry"] for g in guides],
                       base.to_dict("records"), src.crs)

        log(
            f"Planting rule: neighbours in a row with the same colour: "
            f"{settings.get('row_neighbours_same_colour')} of {settings.get('row_neighbours')}; "
            f"{settings.get('crowns_moved_across_gap')} crowns moved across a gap."
        )
        print("\nODD / EVEN\n" + family_summary.to_string(index=False), flush=True)

        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb / max(float(np.percentile(rgb, 99)), 1.0) * 255, 0, 255).astype(np.uint8)
        valid = src.dataset_mask() > 0
        log("Rendering maps, crown annotations and the START arrow.")
        save_web_preview(src, rgb, valid, out)
        start = draw_maps(src, rgb, valid, crowns, df, guides, settings, out)
        tiles = save_diagonal_tiles(src, rgb, crowns, df, guides, out)
        if DISPLAY_MAPS and tiles is not None:
            show_file(tiles)

        save_json(out / "summary.json", {
            "input": str(tif),
            "engine": settings["engine"],
            "image_preparation": [f"analysed on a {W} x {H} grid at {pixel * 100:.1f} cm (dashboard engine, Detail = {DASHBOARD_DETAIL})"],
            "crs": str(src.crs),
            "candidate_count": len(df),
            "young_tree_count": young,
            "candidate_area_m2": float(df["area_m2"].sum()),
            "candidate_area_ft2": float(df["area_m2"].sum() * FT2_PER_M2),
            "field_area_m2": float(result["field_area_m2"]),
            "field_area_acres": float(result["field_area_m2"]) / 4046.8564224,
            "canopy_area_m2": float(result["canopy_area_m2"]),
            "canopy_area_ft2": float(result["canopy_area_m2"] * FT2_PER_M2),
            "diagonal_settings": settings,
            "segmentation": seg,
            "start": start,
            "checks": "Same engine as the Emmanuel Dashboard: identical results for the same image and DSM.",
        })

    (out / "READ_ME.txt").write_text(
        "PECAN CANOPY + EVEN / ODD THINNING DIAGONALS\n"
        f"Image: {tif.name}\n"
        "Engine: the Emmanuel Dashboard's analysis (js/analyze.js), run with\n"
        "Node.js: results are identical to the dashboard for the same image and\n"
        f"DSM (Detail = {DASHBOARD_DETAIL}; analysis grid {W} x {H} at {pixel * 100:.1f} cm).\n"
        "Canopy: colour + texture classifier trained on this image; with the\n"
        "DSM only canopy at least 1 m (3 ft) tall counts as a tree; clear cast\n"
        "shadow is trimmed from the crown edges. One crown per planting position.\n"
        "Young (replanted) trees on empty planting positions are counted too\n"
        f"(flag YOUNG_TREE): {young} in this field.\n"
        "Along every row neighbouring crowns alternate EVEN / ODD (1 blue,\n"
        "2 red ...), also across a missing tree (flag MOVED_ACROSS_GAP).\n"
        "Areas: area_ft2 columns in square feet (area_m2 also given).\n"
        "Open this ZIP in the dashboard to check, correct and export the\n"
        "thinning plan (trees to remove with GPS positions).\n",
        encoding="utf-8",
    )
    shutil.rmtree(work, ignore_errors=True)

    if MAKE_ZIP:
        log("Creating result ZIP.")
        zip_path = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        log(f"ZIP: {zip_path}")

    log(f"FINISHED {tif.name}.")
    return out


# ==================== MAIN WORKFLOW ====================

def run(tif):
    """Analyse one orthomosaic; returns its results folder."""
    if (
        not isinstance(MAX_RETAINED_CROWNS, int)
        or not 1 <= MAX_RETAINED_CROWNS <= 1000000
    ):
        raise ValueError(
            "MAX_RETAINED_CROWNS must be between 1 and 1,000,000."
        )

    tif = Path(tif)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tif.stem)[:60]
    out = OUTPUT_ROOT / f"Pecan_Canopy_Results_{name}_{timestamp}"
    out.mkdir()

    log(f"Image: {tif}")
    log(f"Elevation setting: {ELEVATION_PATH}")
    log(f"Output: {out}")

    working, preparation, has_mask = prepare_image(tif, OUTPUT_ROOT / "_prepared")

    for note in preparation:
        log(f"Image {note}.")

    with rasterio.open(working) as src:
        validate(src)

        valid = src.dataset_mask() > 0

        if not has_mask:
            valid &= field_without_mask(src)
            preparation.append("no field mask: black / white borders removed")
            log(
                "No field mask in the image: the whole image is analysed "
                f"({100 * valid.mean():.0f}% after removing empty borders). "
                "Clipping the orthomosaic to the orchard gives cleaner results."
            )

        if not valid.any():
            raise ValueError("The image has no valid pixels.")

        log(
            f"Image: {src.width} x {src.height}; "
            f"CRS: {src.crs}."
        )

        labels, peak_count, diagnostics, tops = detect(
            src, valid, out
        )

        pixel_area = abs(
            src.transform.a * src.transform.e
            - src.transform.b * src.transform.d
        )

        label_counts = np.bincount(labels.ravel())

        crowns, field = create_geometry(
            src, labels, valid
        )

        for label_id, geometry in crowns:
            assert math.isclose(
                geometry.area,
                label_counts[label_id] * pixel_area,
                rel_tol=1e-7,
                abs_tol=1e-5,
            )

        del label_counts

        # EVEN / ODD diagonals from the measured planting grid.
        (
            guides, grid_table, settings, empty_sites, merged, crowns,
        ) = planting_diagonals(src, field, crowns, tops, labels)

        # Planting positions that got their crown from the spacing changed
        # the label raster: save it again so every output agrees.
        profile = src.profile.copy()
        profile.update(count=1, dtype="int32", nodata=0, compress="deflate")
        profile.pop("photometric", None)

        with rasterio.open(out / "candidate_labels.tif", "w", **profile) as destination:
            destination.write(labels, 1)

        # Tree height (DSM) of every final crown: also the crowns made for
        # planting positions from the spacing, with split-tree pieces joined
        # to their tree. Replaces the first crown_heights.csv.
        chm_path = out / "canopy_height_model_cm.tif"

        if chm_path.is_file():
            with rasterio.open(chm_path) as height_src:
                raw_height = height_src.read(1)
            chm = np.where(
                raw_height == -32768, np.nan, raw_height / 100.0
            ).astype("float32")
            del raw_height

            remap = np.arange(
                int(max([labels.max(), *merged.keys(), *merged.values()])) + 1,
                dtype="int32",
            )
            for source, target in merged.items():
                remap[source] = target
            final_labels = remap[labels]
            count = int(final_labels.max())

            stats = label_percentiles(final_labels, count, chm, [50, 90, 99])
            finite = np.isfinite(chm)
            sums = np.bincount(
                final_labels[finite], weights=chm[finite], minlength=count + 1
            )
            counts = np.bincount(final_labels[finite], minlength=count + 1)
            present = np.flatnonzero(np.bincount(final_labels.ravel(), minlength=count + 1))
            present = present[present > 0]

            pd.DataFrame({
                "label_id": present,
                "height_median_m": stats[50][present],
                "height_p90_m": stats[90][present],
                "height_top_m": stats[99][present],
                "height_mean_m": sums[present] / np.maximum(counts[present], 1),
            }).to_csv(out / "crown_heights.csv", index=False)

            del chm, finite, final_labels, remap, stats, sums, counts

        del labels

        # One planting position = one crown: merge the pieces of split trees.
        grouped = {}

        for label_id, geometry in crowns:
            grouped.setdefault(merged.get(label_id, label_id), []).append(geometry)

        crowns = [
            (label_id, geoms[0] if len(geoms) == 1 else unary_union(geoms))
            for label_id, geoms in sorted(grouped.items())
        ]

        raw_df = inventory(
            crowns,
            peak_count,
            grid_table,
            src,
        )

        heights_path = out / "crown_heights.csv"

        if heights_path.is_file():
            raw_df = raw_df.merge(
                pd.read_csv(heights_path),
                how="left",
                on="label_id",
            )

            # Far taller than the orchard: possibly a non-pecan tree
            # (e.g. woodland at the field edge). Review, not removed.
            q25, q75 = raw_df["height_top_m"].quantile([0.25, 0.75])
            tall = raw_df["height_top_m"] > q75 + 3 * (q75 - q25)
            raw_df.loc[tall, "review_flags"] = [
                ";".join(filter(None, [str(flags or ""), "TALLER_THAN_ORCHARD"]))
                for flags in raw_df.loc[tall, "review_flags"]
            ]

        # ----------------------------------------------------
        # Preserve raw detections.
        # ----------------------------------------------------

        raw_df.to_csv(
            out / "raw_candidates.csv",
            index=False,
        )

        export_geojson(
            out / "raw_crowns_wgs84.geojson",
            [geometry for _, geometry in crowns],
            json.loads(
                raw_df.to_json(orient="records")
            ),
            src.crs,
        )

        # ----------------------------------------------------
        # Filter small regions, then cap the COMBINED total.
        # ----------------------------------------------------

        eligible = raw_df[
            raw_df["area_m2"] >= MIN_CROWN_AREA_M2
        ]

        selected = (
            eligible.sort_values(
                ["area_m2", "label_id"],
                ascending=[False, True],
            )
            .head(MAX_RETAINED_CROWNS)
        )

        accepted_ids = set(
            selected["label_id"].astype(int)
        )

        df = (
            selected.sort_values("label_id")
            .reset_index(drop=True)
        )

        if df.empty:
            raise RuntimeError(
                "No crowns remain after filtering. "
                "Review MIN_CROWN_AREA_M2."
            )

        excluded = raw_df[
            ~raw_df["label_id"].isin(accepted_ids)
        ].copy()

        excluded["exclusion_reason"] = np.where(
            excluded["area_m2"] < MIN_CROWN_AREA_M2,
            "BELOW_MINIMUM_AREA",
            "COUNT_CAP_AREA_RANK",
        )

        excluded.to_csv(
            out / "excluded_candidates.csv",
            index=False,
        )

        crowns = [
            (label_id, geometry)
            for label_id, geometry in crowns
            if label_id in accepted_ids
        ]

        cap_exclusions = int(
            (
                excluded["exclusion_reason"]
                == "COUNT_CAP_AREA_RANK"
            ).sum()
        )

        selection = {
            "raw_candidate_count": len(raw_df),
            "retained_candidate_count": len(df),
            "excluded_candidate_count": len(excluded),
            "minimum_crown_area_m2": MIN_CROWN_AREA_M2,
            "count_cap": MAX_RETAINED_CROWNS,
            "below_minimum_count": int(
                (raw_df["area_m2"] < MIN_CROWN_AREA_M2).sum()
            ),
            "count_cap_exclusions": cap_exclusions,
            "note": (
                "Canopy-only crown candidates. "
                "The cap is not evidence of true tree count "
                "or detection accuracy."
            ),
        }

        save_json(
            out / "candidate_selection.json",
            selection,
        )

        log(
            f"Raw candidates: {len(raw_df):,} | "
            f"retained: {len(df):,} | "
            f"excluded: {len(excluded):,}."
        )

        if cap_exclusions:
            log(
                f"WARNING: the {MAX_RETAINED_CROWNS:,} cap removed "
                f"{cap_exclusions:,} canopy crowns. Review before use."
            )

        # ----------------------------------------------------
        # Filter the label TIFF so every output agrees.
        # ----------------------------------------------------

        raw_label_path = out / "raw_candidate_labels.tif"

        (out / "candidate_labels.tif").rename(
            raw_label_path
        )

        size = int(max([raw_df["label_id"].max(), *merged.keys()])) + 1
        remap = np.arange(size, dtype="int32")  # split-tree pieces -> tree

        for source, target in merged.items():
            remap[source] = target

        lookup = np.zeros(size, dtype=bool)

        lookup[list(accepted_ids)] = True
        retained_pixel_count = 0

        with rasterio.open(raw_label_path) as raw_src:
            with rasterio.open(
                out / "candidate_labels.tif",
                "w",
                **raw_src.profile,
            ) as destination:
                for _, window in raw_src.block_windows(1):
                    tile = remap[raw_src.read(
                        1,
                        window=window,
                    )]

                    tile[~lookup[tile]] = 0

                    retained_pixel_count += int(
                        np.count_nonzero(tile)
                    )

                    destination.write(
                        tile,
                        1,
                        window=window,
                    )

        assert len(df) <= MAX_RETAINED_CROWNS

        df.to_csv(
            out / "all_candidates.csv",
            index=False,
        )

        grouped = (
            df.groupby(["diagonal", "family"])
            .agg(
                candidate_count=("crown_id", "size"),
                area_m2=("area_m2", "sum"),
            )
            .reset_index()
        )

        base = pd.DataFrame([
            {
                "diagonal": guide["diagonal"],
                "family": guide["family"],
                "planting_block": guide["block"],
                "grid_sites_with_tree": guide["tree_sites"],
                "line_length_m": float(guide["geometry"].length),
            }
            for guide in guides
        ])

        grouped = base.merge(
            grouped,
            how="left",
            on=["diagonal", "family"],
        )

        for column in ["candidate_count", "area_m2"]:
            grouped[column] = pd.to_numeric(
                grouped[column],
                errors="coerce",
            ).fillna(0)

        grouped["candidate_count"] = (
            grouped["candidate_count"].astype(int)
        )

        grouped.to_csv(
            out / "diagonal_summary.csv",
            index=False,
        )

        family_summary = (
            df.groupby("family")
            .agg(
                candidate_count=("crown_id", "size"),
                area_m2=("area_m2", "sum"),
            )
            .reset_index()
        )

        family_summary.to_csv(
            out / "odd_even_summary.csv",
            index=False,
        )

        make_even_odd_report(
            df, grouped, out
        )

        assert grouped["candidate_count"].sum() == len(df)

        assert family_summary["candidate_count"].sum() <= MAX_RETAINED_CROWNS

        assert math.isclose(
            df["area_m2"].sum(),
            retained_pixel_count * pixel_area,
            rel_tol=1e-7,
            abs_tol=1e-5,
        )

        # Grid sites inside rows with no treetop (missing tree, or a gap
        # a neighbour's canopy grew into). The diagonals pass over them.
        empty_sites.to_csv(
            out / "empty_grid_sites.csv",
            index=False,
        )

        export_geojson(
            out / "crowns_wgs84.geojson",
            [geometry for _, geometry in crowns],
            json.loads(
                df.to_json(orient="records")
            ),
            src.crs,
        )

        export_geojson(
            out / "guides_wgs84.geojson",
            [guide["geometry"] for guide in guides],
            base.to_dict("records"),
            src.crs,
        )

        log(
            "NUMERIC RESULTS READY — "
            "canopy-only crown candidates, not confirmed trees."
        )

        print(
            "\nODD / EVEN\n"
            + family_summary.to_string(index=False),
            flush=True,
        )

        print(
            "\nPLANTING GRID + DIAGONALS\n"
            + json.dumps(settings, indent=2),
            flush=True,
        )

        print(
            "\nP(canopy) BOUNDARY SENSITIVITY\n"
            + pd.read_csv(
                out / "canopy_probability_sensitivity.csv"
            ).to_string(index=False),
            flush=True,
        )

        rgb = src.read(
            [1, 2, 3]
        ).transpose(1, 2, 0)

        log(
            "Rendering diagonal maps, retained crown "
            "annotations and START arrow."
        )

        save_web_preview(src, rgb, valid, out)

        start = draw_maps(
            src, rgb, valid, crowns, df,
            guides, settings, out,
        )

        tiles = save_diagonal_tiles(
            src, rgb, crowns, df, guides, out,
        )

        if DISPLAY_MAPS and tiles is not None:
            show_file(tiles)

        save_json(
            out / "summary.json",
            {
                "input": str(tif),
                "image_preparation": preparation or ["used as it is"],
                "crs": str(src.crs),
                "candidate_count": len(df),
                "candidate_area_m2": float(
                    df["area_m2"].sum()
                ),
                "field_area_m2": (
                    int(valid.sum()) * pixel_area
                ),
                "diagonal_settings": settings,
                "empty_grid_sites": len(empty_sites),
                "segmentation": diagnostics,
                "candidate_selection": selection,
                "start": start,
                "checks": (
                    "Retained polygon areas match retained label pixels. "
                    "Each retained candidate is assigned once. "
                    "Combined EVEN + ODD does not exceed MAX_RETAINED_CROWNS. "
                    "On-grid treetops lie on their own diagonal line."
                ),
                "missed_tree_count": (
                    "Unknown. Inspect against imagery; "
                    "no zero-miss guarantee."
                ),
            },
        )

    readme = (
        "PECAN CANOPY + EVEN / ODD THINNING DIAGONALS\n"
        f"Image: {tif.name}\n"
        f"Image preparation: {'; '.join(preparation) or 'used as it is'}.\n"
        "Canopy = RGB colour + texture classifier trained on pixels\n"
        "harvested automatically from this image (canopy, ground\n"
        "vegetation, soil/road, shadow). Grass, weeds, soil and cast\n"
        "shadows are excluded; self-shaded crown sides are kept.\n"
        "With a DSM: canopy height = DSM - terrain (terrain from\n"
        "RGB ground pixels). Patches below 1 m and anything more than\n"
        "0.5 m beyond height-confirmed canopy are removed; crown edges\n"
        "come from RGB. Crowns are split at canopy-height treetops.\n"
        "Clear cast shadow is trimmed from the crown edges: a pixel goes\n"
        "only when it is dark AND less green than leaves AND at ground\n"
        "level in the DSM (below 0.5 m); shaded leaves stay canopy.\n"
        "canopy_mask.tif: 1 = pecan canopy, 0 = not canopy, 255 = outside.\n"
        "canopy_height_model_cm.tif: height above ground in centimetres.\n"
        "crown_heights.csv: median / p90 / top / mean height per crown.\n"
        "canopy_probability.tif: RGB P(canopy) in percent.\n"
        "canopy_training_pixels.tif: 1 canopy, 2 ground vegetation,\n"
        "  3 soil/road, 4 shadow, 0 unlabelled (automatic labels).\n"
        "canopy_review_tiles.png: zoomed visual check (orange = removed\n"
        "  by the height check, purple = shadow removed).\n"
        "canopy_probability_sensitivity.csv: RGB area vs boundary threshold.\n"
        "\n"
        "EVEN / ODD DIAGONALS\n"
        "The planting grid (in-row spacing, row spacing, row direction) is\n"
        "measured from the treetops (crown centres if RGB only). Each crown's\n"
        "diagonal comes from its grid position, so EVEN and ODD alternate\n"
        "tree by tree along every row and across rows (diagonal thinning =\n"
        "alternate trees in each row and column). Planting design 20 x 40 ft.\n"
        "Each diagonal is a smooth S-wave from its first to its last crown\n"
        "(nothing over open ground; cut where it leaves the field). It crosses\n"
        "each of its crowns at the treetop, square to the row, and makes its\n"
        "one-tree shift in the alley between rows. In a row where its own\n"
        "tree is missing it crosses at the best gap between the other trees.\n"
        "Maps: each crown tinted in the colour of its diagonal (white borders\n"
        "between crowns); the arrow and the total canopy (m2) of each\n"
        "diagonal are at the end of its line; the EVEN / ODD totals are\n"
        "under the map.\n"
        "canopy_area_per_diagonal.png: canopy area of every diagonal, with\n"
        "  the EVEN and ODD totals at the end (EVEN_ODD_Totals.xlsx, sheet\n"
        "  Diagonal_Summary: the same, totals in the last rows).\n"
        "One crown per planting position: canopy that covers a position with\n"
        "no detected treetop becomes that position's crown (flag\n"
        "INFERRED_FROM_SPACING), so touching crowns in a row alternate.\n"
        "Along each planted row neighbouring crowns always alternate\n"
        "EVEN / ODD (1 blue, 2 red, 3 blue ...), also across a missing\n"
        "tree: the crowns on one side of the gap (the side with fewer\n"
        "crowns) move to the neighbouring diagonal (flag MOVED_ACROSS_GAP).\n"
        "Check in summary.json: row_neighbours_same_colour (0 = no two\n"
        "neighbours in a row with the same colour). Two crowns on one planting\n"
        "position (one tree split in two) are merged into one crown; an\n"
        "off-grid crown piece touching a tree within one tree spacing joins\n"
        "that tree.\n"
        "tree_on_diagonal: canopy 1, 2, 3 ... along each diagonal from its\n"
        "  upper (START) end.\n"
        "guides_wgs84.geojson: the wavy diagonals (D01 = outermost on the\n"
        "  east/south-east side; numbering continues block by block).\n"
        "final_thinning_start_map.png: the START diagonal only (first line of\n"
        "  its arrow (direction of travel).\n"
        "diagonal_check_tiles.png: 60 m zoom; each crown tinted in the\n"
        "  colour of its diagonal, white borders between crowns.\n"
        "Inventory columns: top_x/top_y = treetop; guide_distance_m = treetop\n"
        "  to its diagonal (0 = on the line); planting_block, grid_row,\n"
        "  grid_tree = position in the planting grid; grid_fit_m = treetop\n"
        "  offset from its grid site.\n"
        "Flags: OFF_GRID = crown not on the planting grid (assigned to the\n"
        "  nearest diagonal); MERGED_SPLIT_CROWN = pieces of one tree merged\n"
        "  (crown_pieces = how many); LINE_MISSES_CROWN = its own line does\n"
        "  not cross it; OTHER_LINE_ENTERS = a line of another diagonal runs\n"
        "  more than 0.5 m inside it.\n"
        "empty_grid_sites.csv: grid sites inside rows with no treetop\n"
        "  (canopy_here = neighbouring crown grown over it, or none =\n"
        "  bare, likely a missing tree).\n"
        "web_preview.jpg + web_preview.json: small copy of the orthomosaic and\n"
        "  its bounds, so the web dashboard can show it under the crowns\n"
        "  (open this results ZIP in Emmanuel Dashboard for Pecan Thinning).\n"
        "Combined retained EVEN + ODD candidates do not exceed MAX_RETAINED_CROWNS.\n"
        "raw_candidates.csv preserves every raw detection.\n"
        "excluded_candidates.csv records every exclusion and reason.\n"
        "START: first diagonal of the family (EVEN or ODD) with the smaller\n"
        "Crown boundaries and diagonals still deserve a visual check.\n"
    )

    (out / "READ_ME.txt").write_text(
        readme,
        encoding="utf-8",
    )

    if MAKE_ZIP:
        log("Creating result ZIP.")

        zip_path = shutil.make_archive(
            str(out),
            "zip",
            root_dir=out.parent,
            base_dir=out.name,
        )

        log(f"ZIP: {zip_path}")

    log(f"FINISHED {tif.name}.")

    return out


def run_all():
    """Analyse every orthomosaic (one results folder + ZIP each)."""
    images = find_images()
    log(
        f"{len(images)} orthomosaic(s) to analyse: "
        + ", ".join(path.name for path in images)
    )

    analyse = run_dashboard if ENGINE == "dashboard" else run
    log(f"Engine: {ENGINE}.")

    if len(images) == 1:
        try:
            return [analyse(images[0])]
        finally:  # working copies are not results
            shutil.rmtree(OUTPUT_ROOT / "_prepared", ignore_errors=True)
            shutil.rmtree(OUTPUT_ROOT / "_engine", ignore_errors=True)

    done, rows = [], []

    for number, tif in enumerate(images, 1):
        log(f"========== {number} of {len(images)}: {tif.name} ==========")
        try:
            out = analyse(tif)
            done.append(out)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            family = pd.read_csv(out / "odd_even_summary.csv").set_index("family")
            rows.append({
                "image": tif.name,
                "status": "done",
                "crowns": summary["candidate_count"],
                "EVEN_canopy_m2": round(float(family["area_m2"].get("EVEN", 0.0)), 1),
                "ODD_canopy_m2": round(float(family["area_m2"].get("ODD", 0.0)), 1),
                "elevation": "DSM" if summary["segmentation"]["elevation"].get("used") else "RGB only",
                "results": out.name + (".zip" if MAKE_ZIP else ""),
            })
        except Exception as error:  # keep going with the other images
            traceback.print_exc()
            log(f"FAILED {tif.name}: {error}")
            rows.append({"image": tif.name, "status": f"failed: {error}"})
        finally:  # working copies are not results
            shutil.rmtree(OUTPUT_ROOT / "_prepared", ignore_errors=True)
            shutil.rmtree(OUTPUT_ROOT / "_engine", ignore_errors=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_ROOT / "all_images_summary.csv", index=False)
    print("\nALL IMAGES\n" + table.to_string(index=False), flush=True)

    return done


# ==================== EMBEDDED DASHBOARD ENGINE ====================
# js/analyze.js of the Emmanuel Dashboard (zlib + base64), unchanged.
# ---- BEGIN EMBEDDED DASHBOARD ENGINE ----
DASHBOARD_ENGINE_B64 = (
    "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"
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
