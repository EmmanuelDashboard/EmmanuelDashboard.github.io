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
# Diagonals, as in the dashboard's Analyse window: "auto" (oblique 45 deg
# where the rows run east-west, else the true diagonal), "true" (next tree
# in the next row, about 63 deg to the rows) or "oblique" (two trees along
# in the next row, about 45 deg; the same trees are removed in every row).
DIAGONAL_STYLE = "auto"

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
    { designFt: meta.designFt, directionHintDeg: meta.hint, youngTrees: meta.youngTrees !== false, returnRasters: true, chip, diagonalStyle: meta.diagonalStyle || "auto" },
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
            "diagonalStyle": globals().get("DIAGONAL_STYLE", "auto"),
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
    "eNrcvf2S28aSL/i/nwJy7MwFJDZFtj4sd6vlkG3Jdoxsa2TP+Ogy+p4DkmA3ukmCDYDdpDW6sW+xb7B/b9xHmH2TfZLNX2bW"
    "FwCy27InYmNPHKtBoCqrKqsqKzMrPx7e/yy6H71aLNLlOptH36bV+bhIy2k0K8robTZJl9Gv5/lymS/PjqJ8kZ5lUbpM59sq"
    "r6J8GdXnWTQui5sqK/sECLB+SRdZtMjq82IapRWXWDEc+q9YbaNlUWfjorg8QuEoGvajSTEv1mX0IFqs53V+UE3SeRbV2aZe"
    "l1k0y1L8rY6l+GE/qss0R3+iVb7J5lV0npbXWVVn1OeyWHCD0tG8rrL5LIql4V50Vhbr5VTg0P+us7OsTuu8WPaiqsjn0cOo"
    "LNIp/ThPp8VNQgOlEUSTeVpV+SzPSsGJgku0Q4/60U1en1PhKJtn1wwvmuU0gvjbX35Mjsywz7P87LyOTiJ6Gx3Q8EoM4zgq"
    "lvOt7ZKWTetonqVVHQ0jGk86n0eE7ZQGnhHUs5L6w327ybJpFZ0VSS+a0BxQB6ZnWWWBhdjQ7j4mdBPskpoi+DJSApwvFj7+"
    "PHDUYLas55kZ75O+fK2iajXP62hM3Y0WaXmZlQeTYlmXxXxOoG5SGmF1Tk8xul0Xq8qBJxT0bDcJp9LchNqhiRZ0ulkEgkzj"
    "T/vRap4ua8z+WZlPaZ2lFa0Or+umtWMpsCqqnKekmEXZdVZ6uEajwL8Z7opm2EK39eJUP9MULAEhmqxrXvo3henWF/1oS0vr"
    "LIrLjCFk04Q7Uh0RcqoFzWAv4sVHi2u9BNoEJu2PMyq21I1i+yYbhuaJFjZVqWm20SnqDr1+9ePbX9939JT+P05px8gydwge"
    "Z5NikWEBSaPxbJ6eRe9//refvvv7r+9evaKFPqfduj47px2DYVJtGuZhtPjP/0V7gtbgsYVFnVwKHqNssaq3AX6B/seDaBMd"
    "DqJZLfgncBUttwlWArZPakFVKa0f4CwFbudY0ARDxoyxmMmk4Zf5CmuchoVdjgUR3VBH/F65lSDrIq2pSaE+06zKz5a8Y/Cz"
    "An2SPtE+XdIOmqWl2zTZDY19jlVRrYoa1GU+u0m3hMX6BjNlBkwbQ6fajIRmimibhcS7MzkmtJdrRo8sEV4We6DyTqd5cmtB"
    "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"
    "pbr1j/pfHfZ6h/2jB7ilRnebwK78/wInM+Gj"
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
