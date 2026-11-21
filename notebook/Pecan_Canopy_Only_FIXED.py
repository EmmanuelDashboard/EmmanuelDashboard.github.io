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
    "OtcMeoNvDvv9w0HvK9xSrfsYwej6fwHv2MOd"
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
