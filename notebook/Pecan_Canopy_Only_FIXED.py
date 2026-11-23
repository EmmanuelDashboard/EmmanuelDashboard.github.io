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
        # A real first line: at least 2 trees, or 1 tree of normal size (the
        # corner tree of a block), not a lone small crown at the field edge.
        per_diag = df.groupby("diagonal")["area_m2"].agg(["count", "max"])
        median_area = float(df["area_m2"].median()) if len(df) else 0.0
        def real_line(d):
            if d not in per_diag.index:
                return False
            row = per_diag.loc[d]
            return int(row["count"]) >= 2 or float(row["max"]) >= 0.5 * median_area
        same_family = [d for d in drawn_numbers if ("ODD" if d % 2 else "EVEN") == smaller]
        start_number = next(
            (d for d in same_family if real_line(d)),
            same_family[0] if same_family else (drawn_numbers[0] if drawn_numbers else None),
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
    "eNrcvf2S28aSL/i/nwJy3JkLSGyKbH1Y7lbLIduS7RjZRyN7xkeX0dcHJMFudJMEGwC7SWu0sW+xb7B/b9xHmH2TfZLNX2bW"
    "FwCy27InYmNPHKtBoCqrKqsqKzMrPx7e/yy6H71aLNLlOptH36bV+bhIy2k0K8robTZJl9Ev5/lymS/PjqJ8kZ5lUbpM59sq"
    "r6J8GdXnWTQui5sqK/sECLB+ThdZtMjq82IapRWXWDEc+q9YbaNlUWfjorg8QuEoGvajSTEv1mX0IFqs53V+UE3SeRbV2aZe"
    "l1k0y1L8rY6l+GE/qss0R3+iVb7J5lV0npbXWVVn1OeyWHCD0tG8rrL5LIql4V50Vhbr5VTg0P+us7OsTuu8WPaiqsjn0cOo"
    "LNIp/ThPp8VNQgOlEUSTeVpV+SzPSsGJgku0Q4/60U1en1PhKJtn1wwvmuU0gvjbn39Mjsywz7P87LyOTiJ6Gx3Q8EoM4zgq"
    "lvOt7ZKWTetonqVVHQ0jGk86n0eE7ZQGnhHUs5L6w327ybJpFZ0VSS+a0BxQB6ZnWWWBhdjQ7j4mdBPskpoi+DJSApwvFj7+"
    "PHDUYLas55kZ75O+fK2iajXP62hM3Y0WaXmZlQeTYlmXxXxOoG5SGmF1Tk8xul0Xq8qBJxT0bDcJp9LchNqhiRZ0ulkEgkzj"
    "T/vRap4ua8z+WZlPaZ2lFa0Or+umtWMpsCqqnKekmEXZdVZ6uEajwL8Z7opm2EK39eJUP9MULAEhmqxrXvo3henWF/1oS0vr"
    "LIrLjCFk04Q7Uh0RcqoFzWAv4sVHi2u9BNoEJu2PMyq21I1i+yYbhuaJFjZVqWm20SnqDr1+9ePbX9539JT+P05px8gydwge"
    "Z5NikWEBSaPxbJ6eRe//9m8/fffbL+9evaKFPqfduj47px2DYVJtGuZhtPjP/0V7gtbgsYVFnVwKHqNssaq3AX6B/seDaBMd"
    "DqJZLfgncBUttwlWArZPakFVKa0f4CwFbudY0ARDxoyxmMmk4Zf5CmuchoVdjgUR3VBH/F65lSDrIq2pSaE+06zKz5a8Y/Cz"
    "An2SPtE+XdIOmqWl2zTZDY19jlVRrYoa1GU+u0m3hMX6BjNlBkwbQ6fajIRmimibhcS7MzkmtJdrRo8sEV4We6DyTqd5cmtB"
    "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"
    "XVqljlkRuuo4JSLmWhxbTI9EBSAbSW+bvuKcVQ0jYae8uLuSMXCl1WqfvNA+N4ErFT2igeV4HeiJL8h+LgvJn7aduVj3I5Mq"
    "CybdHgX5tvzAbRK/f6F+zy5RB5YE/2DBi36YmpN1TuVY6aJr3Ms616QhAjHx9jeDMoIImxLYXcSfetGXA7a+7Xt2dp86N8tC"
    "jpZz3jOVr+eWs8awQQdMFEAletBLwFuOg6NCtc1EIPEm76OPmnE2k+y36Jr1oqpMeFrEcrjuNZJS0L+zvFzssz3ZO/umukyW"
    "h36dwBcnzM8FGya4JZzm8xSzYwCZ/YE8qg+DXYIuONbvD3aUEy02usgBDTQ/r7f0gr5Kg9BE/IzotN8jEUdML003PdL9mPlW"
    "nLtmkUllVaL4tUBi3OCMBuUVa350OazSenIu/u/sXo7FI/qcLwf1+YFbq7qiGNPRoh/kZDaJ2yZ85H006fpcV5rhPtfZTusi"
    "gQHzgj+R7nDfxtT2R7awXt3q99tzVdgeBsECb94Ck1nVwbPMhYHg8GtcuSFjc8qAOVbNPdO3U0MvEDvJ0Qv7WWhG16pXIq8w"
    "E69noU3cnVZxgFABabHWtYCRLIkX8I61e9hau7fSasfR2V6FMVxntUk9cystMkRVEYU9ehRxT0HfDfqEvnqXG6CzIjczYfzN"
    "sE1HloEiqlz9Jhvkt5IY+aw8suxOz1MLLKj89Lfx1pRdHLLTPdPTAxkK5BEgCBYeN7+tZOp+06pH3nQauB97zl7dyB3fcMCJ"
    "ScpOe+mUGKy6zBcgv9az1BMPkIh4WeMgSCsNmSFg7N3mUTRNy8vo5U/f0mqvcJ8K1zT2paeWEFkCn1LLMcmlog+KBJNjVgtj"
    "obCc0mOYnku+YPe/EyexXbFa2OrRMYJG3sh88TO/jsuzcWpEF5Vb+LD22B9p6f0nyzDvu9mWXvRqHz9zF5HF6R8xjBGCC+AO"
    "47vgBYe/7UVfN18eWt3Xe90X/cMvv6SP70Q2fwav7O/4eThEpa9N+VdSnh0zv4Oijv77mh35UBVVvvbomy/kcGYpUae8904Z"
    "d4biusgWedVZpGnCvZy+2pw5VotWtsdp+eLaPe9Q3VbCRcvMBqwGH2x7QN4JSrYs1mfnatnMPXRJAR4NOOiDbSj4EnSFiaKC"
    "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"
    "i6i860xRSMG5whvCSlMzJ02ZwnJ6FIHsdFcWusFA233V4KZ3UQZllRFvNA7wJ6NWraZGYZtzv70hHqiWiwXkSu9cjemucHl4"
    "kk5Vp3Ym6N2nMuKEBt63IfX3eOxbaCCXbNPARneuurtz1dWdRtVVd9VVu2qbwv5pyhlYCau9mGGkvGxC3cYTleRGkbAAcvV1"
    "5F++wVghY+ro54m2/MI4mxSLTMRu4StiGIEYUqrciY0hHL3691c/RUTcv/026YfZB60Ox4sNbMMCt1iSP8NkuNWJLQJoTGJC"
    "PgPKoECDhQgN3inPtxOEOMS6+Be5NQmulct0mrMAJy7kWT6P5ejpinO+yJcvxQQWlh33fVnmU9Rl3vY0n+0Cgi2cRyFYUGpx"
    "IXzUgzwwcgwX0uLORnkvujiVkM4ErMUJnzHPPjzmB1hJ8lNbF8B8HNP9GGcWSjMdMmzmRadYwvEpeVGfNDwIbQeupPkrNE5Q"
    "6cnG0A9avArbMlAb/oLcVf62++Qs9WqY4fP10Y4R3bm3bYke3PgVksAYsG230YoIExark/sfUDXkNKEX3McmwUp6niKgXbhB"
    "xk47mmw4jKALaMMlnKlw6AhzctAqOTxtXe3anA8Te29RuhsMHDk3+pIez7tdOJW3rx1DU5XKg1WTjlHMJWLwlQR+IWpda0Ty"
    "Dtfcm6VcXMmu6+ncP+CInbr/5PCcM7S0bo3snsBoH2JiMJ6W2VEQ577tOjxopPOuoCoQ2oNorkE633NxH67gMqxF2qOatCBO"
    "fIiTAOKNQpzsgwj1QcvjMWCpS+Gp6WHYxVVPhK2mh2HLV8mbOTfFLhtx29l3zmKLJK5DTlP+aWduf4w7yZArsykQvtI9gSPd"
    "bpauqisMxPeb0puiFd8NY2a2jSuidtcR3OIg0n3lbotXW/N2eGrvjKUsUhi2iuKlKSlKdc2gYrC2M7ih0RcZ4VNvZdjedjAY"
    "YjP6n5+bc60bq36gN+ZoUGqZTZAlQKM5YFRtQpPVnrIdDSZsn7z03ga7POHIOxJjeNfu12vJms8Er1DS3febc2RajBlsKwpy"
    "U2MppVYFm12ssRDWuDbcGY7vmk9fKihuFmummCLerX25b+3JfSh00+NXN6dyxF1bptyM6zqRJC2Z3uFfc/Zk/gmx6ZoYTe0r"
    "1sN1sj/OF9fjSEW3T/SvJuPFtz//qEaTTowH70iLEhFjK8N2lnkl/tb6MQTHvqsSvCSa5isboTuvbUgne4eDPHXF6tg30mwv"
    "wgoNgsrmJpCIrc/ZmRoAx0gEfG6D0hr3sZWJNGVTbFiLZVrdMad8XS8vJYVIguyd6nKGoTRGGHQBg8oldhWzfrRD0tCc1FaM"
    "cb+GzB/ZLF3Pa4STFx5+6jDgB7wSY3/6xIlls/m8iWnjnsRm1BqTyuS/WqWl6sHc2A1ueta22wKjInMFdFOs51NkMp0jwnRj"
    "ZWlCLrEEg8EOxIJ8OctKklN+gkXqL8VKqLCXZ8TnC89XYvaza4utWcdGS0a2yvkqtIxYO8uI8xV8mgBu3UXXz7Gd6TMb+pwL"
    "MV95BlDn3FMHqyMyxs1SynDRF6EdY1eocaYOtbIbHUEn2QSjGUb/NgzoybmWiBmFnoVhNH1vgpa1Jsa+H1kysOvstKenuYgq"
    "Tj2+UE69XlRslTnUEsNGiW6YlYambxkAssDlCekFjsPzDbdDD9sk6Qz5yREQuiKyh+FhNTQseBLuAT232ZLQQs67czoHGxCb"
    "LrETNfwtGVIiRnR+6a2U1n6Hpbujh24s5d/aJzk4+NVzMMvzILIMoibIGnW8063hNXnnYNEi6gIa8uJ3PYG5JuJGoEAScLYe"
    "QWDqfWRoPOj0Z61gsEQ1OfU6E/p4Lz0qZjUdAufpNWIYLonCtqFVJFTPXYNipsAnRxWdE5CMDSjgIzrPM+tTnHvJYV3nYTFu"
    "bbotsT4OfALEBiKEnFdtYOLxgtDnWVY3DhyENdcDsX08JZ3g7nBgeUJEGagphp7VbNceUe1hRzCdBlvvSxHnWL64vxRGvymT"
    "8CKnz8yfoIEuOSCAtzHwNgE8lUh4i8nnfRvz8tDXyXdvJqWMl4ceaQzZqMtD7nds97Ny23bHMkt+Ajzf50F+MDgUD9mWIefu"
    "7SaX/PbI6JiDjx3hjbg4m496CitcuXQHCP7YYrnZu9ioxVkTZnj8B2Esl+4DxjtcCFSDNXzpLXROb875BPQ2h+WCI7OZmFfk"
    "ZK1gcoqllw1TgcFF30/8BnlA1JK4Cqoa7widYNRKRGAdz9dZr8muTHlA9lop5cgCmq8g6bfQ9K5TGschaeXndzslcq9YW4b+"
    "plMqDyB/s1My3wHZm6tpsSzK/XyFFHwDJzEu/KAZxMe79rTB5DrCI7PlPLGNVbQjzxIrMhnCzhhJIC7vjM7gXafO4BujM/hm"
    "uIMGeJG5d1IAb/9XxLJAVfCGM73LKEVYHCS7OR8eaZuHYiZCLLEGuxgbksMOWWIdnLagavOstabFwK20aLUvox7uEFI7BNXD"
    "tqRKz8zZrjtN2M0l++9Ze6C75NpvOgXbb1pSLfEv2FVS9sa9fY6NJGU7bjjDCXRysE7ltVDyN5Kyy5tKkYd97F5b5FrkdInG"
    "3QS7+VtX9oghcrgzenEb9Za90nnNH5zhDe0ruGjeqKehDrbcoYP1yu/QxBpfZXadMTjiQPZGW1JO+Opw0BFhD9WSrgwfJ4Jf"
    "jY1oy1o7QA6sH72wP/CRXUm5G5ddqGAW0Zpe+PkiGZVNhXl4Y73uRZenEiGVR9heVhI0HAtIBrVf6Nml7vmvVfggYNGyveLz"
    "aWKwI/61w32rbz9aBA33FA1oy0Hed+L7bQgnobZyq6I6ikZ8G4NbpR4r/3o88UdOyGwkmbouiDWBzn73hX+H5zsinfFFaJUc"
    "kQwCH4V0Aptw6JB8d3hx6WVNJEfktpbJKnCUa0QFhNf0ivMVwfdXfIbhrv7qx7e/vO+wsSI0csrbnoHG5tFb2z8owuzFamp4"
    "ldk8PYve/+3ffvrut1/evXqVsLd9DmnC6IQMOFaBsa20NYY2Xvru+tXP0es8kjm8M/GYk3PrmAwv28Yr6tCv6fzS/oavaaMI"
    "XgVlFvk0+I2k6cELHv/PdSmRIOyNC2saxeG+y8fWn9ViZ4SATjfbHwI77vBbMZmsaQVMd1t1Ew7eIc1BF6djPaSrP2Egfa/D"
    "wKOL2piusgo3vMMOgg/OqtD2bec1bk8C0ur4mMT715X2A87HCxT+Si2swXdKcsc86TluteTUuHnCSUapqY407R6pkTArIAxJ"
    "59AyOBw0M7DBb/BH6rSN0o52+bq9Z6yswwrj9eQyq6uuyWvlJyHMdVjWto/bWo9a7oy76m2XGrpSjctl7Za1aDS/2zaNkf3W"
    "ZdfYDCyUbc7YdU3dij786eBIJhwdx8X/jo6id/Tf19BLxe+oxHf039dquvIxxPx8vUj9vgz6hxwp1OuHXJI9+yJ4K+4A+DIc"
    "Pm5+ceGaOIcCyDiEz/Vynpt4OkRvrrOqEUQnlpAmV2viDIhZbiR70yqdW3h3mFbPzVfGioHSwdzyzVf4MnM8Q1SysU5BJWU8"
    "J6a4Ct5f+UYi8qkXHT7hhI79bleNTwqohboanHeh8XXBYPiRlvx4AUHQpXuN0AKN4AEdTpporTPO1SIKfAlDCMsOt8yFum1W"
    "68XffVd9E9XY+/7+lu9vdn//lJBk852OnrvtPyMMcjRnO1mMCHZ4D9Ttmt+81zeezyG/f6Pv7VLsJhDMY+AGr5LIGtUudyC9"
    "823OTrpR+yV7EDwaeEfAY4lKEBg2uSbZBsmtfA3VcIBmJTeNbuQ4Va5MmZlJWpnYHsJTwXuMFTW5l+LRFICmR+NR4arLZ7ga"
    "+x4Xgt/IqmxsmcVuX4Oh2oUu1CbUABGjUJk9z41x0H/KcU2D989PLCZx/aWz99CU6iAi3VuqLl5JWDiEZfDCrrQYlFQ4tl1r"
    "HxFKJFTCjiJixm7uS5pnrCTg2bkzWV7Y9TkExZx5FcYvIAl1D+9F3N/PoUnAf9FeZf7Mm+zdpmG74yS0cnEfirHIgd3rZg2E"
    "ymWz78OvYQ+nCIzOM80FzBP0eI2SsmokGprOPFdxz2ywrIUaeJAZYsZlzloUfQHOZS6cC2Ys+O1EQS0MVmae7DgQOeVJUCTv"
    "TlJuDORFmM49YXq30bwre3N3Q3rPlB6rwF8nFyJw8+owa5GRc7FDWWgLgdW9CGRc2UdC/D/r1E1enKpc3yjUDAjhM0ecIWuV"
    "mQxRbJ2gfrbqTVH+qMmmiDhxESW//dZO+85sT3ZN6aEmx41ppHJrRLNux2uheq1wLSKXbqCmxcbB33OayaHk0d4Od2aLz527"
    "yofbd6Demz6nlhJpbWPevaDWEmlR32FtbKkc92Zr3r2g3iTSo62/uDdN3T29OIhKtqM/Fhj+163/ddOl1KeXD6JS2ett14XC"
    "1ivRQP8Yzi8baP82AzFbHIMIb/FmOwjV+r7VW+MkHCNy7/g8uQ31MDaLm5EQthycbQydoYmggO40NFKG9Ev703ye1llMP6jH"
    "N+g1rbPW6BCMN7w4oDd9MD9dHrSNJC06po48LUriMJhLlsjlaZ/Jac7pCmIE/B2bkT/QkfPAL2ngY44dsWlFhAnYa89x8AMP"
    "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"
    "miIbK+jLvrGXlx8BqDwDwrrsIHu3Ur9RScRvn5kUO/no9g9w9KGQBWSXf7tdWOr0aVP1R1Onah9/Hj1pPbIkWuMN5utiLpmF"
    "Ujer1A2WonttQ6WCgwrowWpQghH8Df7u19vDyEDRZVLAV4HzLzuXmaF7DmYR9zJOv2Y8zNzYEe1pxpzsvD52pJwuOW0bN+l6"
    "nJmIiELyP+07rjqONziC+dQZWKzVQQHVrwvZyCeKOUgSAm1cIB1ye1UU81Y6fpfC87i1ywt1CygQxmxARW7JxrP8+//Ytv7n"
    "//P/4mz4+//QA2UvQ8LbhNJlnI3g7u+FYmyXm3Q5nZPr5LbM/kp9KD0d7B5CizHCPTbYgM720JfQ1TdPmLCQRIsuVkRZrmWW"
    "xlz+psZZaaLePkvmWaHmDltr7bQiKrd7DTimw222wNrCFsnP2W77Z7bbYmb+Zbr9JzPdVqihasSNqQsazbnmYuJQ3xNtxD31"
    "uSezAsfe6xBLLE7vzs3ujXCZYgfpNFUsLUVaBgjmf//v/ZYL6GP8ZjU2GukPFPdI1+tIIBejNlWkL1VYq6IZLutRUwZ7WzqO"
    "qcbHYj2q2rsIqhpm5J2HqmZbYdg/QKxRKYFYY+pSpSjNSYCypsYpMGu01hWgtQq6VEXQJrAGzf9c+4FiGaiXJrqtfkC7o5a9"
    "FwWlK5GlVVrVvL0NzWjSFHDFoVuAlFJiDh9YAgrAVxcPKK8jJZCgLVw0Nh3SwvGfOj0NWnjLtILcNIm89xdz/2yU7NOg58L1"
    "wIqN1nNseNbC4Uv3MItNe6lq1mggZIPgJ1QK7/FJQUFoZFx1YfOc3C87yuLCt7ioSyQHyZc6ZZ+bzaVtnCzU/K/ZUwpauo7E"
    "46eX2ENkgfTurwmLF36qC85cKxM0v5We6TvwSQuo1WyGIUVgO21NlWAxz8iyqxbFSRrIyWhKzsOnsQHZvMyoA8Z2FsP4pXAz"
    "37hciU8j0IJb53ZdSH5ExhBQPa7m/CODMsUsRzPzOtn9XkmAKYt8YZjp/yYcVu0i2Iy/aqK8KoE32pPCwXpux7Chk/NGiMCG"
    "THhrLMEzG8ap5/sNLQ053Tx6nFST+R6yd69jR4HaxfooYNDhfkzLG2Esj6I8hI+EdHn7o6L9CrB0s0gnfY6LKQRatGcApxVX"
    "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"
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
