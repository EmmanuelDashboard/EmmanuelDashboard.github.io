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
    "eNrcvetyG9eyJvjfT1FyzDmNkkAIICVZJkU5ZFuyFS3bOrLP8VYj2HsXgCJZJIACqwokYB11zFvMG8zviX6EnjeZJ5n8MnPd"
    "6gJSsnfExOzYFgtVa+W658p7Prz/RXQ/erlYJMt1Oo++T8rzSZ4Us+g0L6K36TRZRr+dZ8tltjw7jLJFcpZGyTKZb8usjLJl"
    "VJ2n0aTIb8q0GBAgwPo1WaTRIq3O81mUlFxixXDov3y1jZZ5lU7y/PIQhaNoNIim+TxfF9GDaLGeV9leOU3maVSlm2pdpNFp"
    "muBveSTF9wdRVSQZ+hOtsk06L6PzpLhOyyqlPhf5ghuUjmZVmc5Po5403I/Oiny9nAkc+t91epZWSZXly35U5tk8ehgVeTKj"
    "H+fJLL+JaaA0gmg6T8oyO83SQuZEwcXaoYNBdJNV51Q4SufpNcOLTjMaQe/7X3+KD82wz9Ps7LyKjiN6G+3R8AoM4yjKl/Ot"
    "7ZKWTaponiZlFY0iGk8yn0c02wkNPCWoZwX1h/t2k6azMjrL4340pTWgDszO0tICC2dDu/uIpptgF9QUwZeREuBssfDnzwNH"
    "DabLap6a8T4eyNcyKlfzrIom1N1okRSXabE3zZdVkc/nBOomoRGW5/TUQ7erfFU68DQFfdtNmlNpbkrt0ELLdLpVxASZxp8M"
    "otU8WVZY/bMim9E+S0raHV7XTWtHUmCVlxkvSX4apddp4c01GsX8m+GuaIUtdFuvl+hnWoIlIETTdcVb/yY33fpqEG1pa51F"
    "vSJlCOks5o6UhzQ55YJWsB/x5qPNtV5i2gQmnY8zKrbUg2L7JgeG1ok2NlWpaLXRKeoOvX7509vf3rf0lP4/SejEyDZ3EzxJ"
    "p/kixQaSRnun8+Qsev/Lv//8w99/e/fyJW30OZ3W9dk5nRgMk2rTMPejxf/6n3QmaA8eWVjUyaXMY5QuVtU2mF9M/6NhtIn2"
    "h9FpJfNP4ErablPsBByfxIIqE9o/mLMEczvHhiYYMmaMxSwmDb/IVtjjNCyccmyI6IY64vfK7QTZF0lFTQr2maVldrbkE4Of"
    "JfCT9InO6ZJO0GlSuEOT3tDY59gV5SqvgF3mpzfJlmaxusFKmQHTwdClNiOhlSLcZiHx6YyPaNqLNU+PbBHeFjug8kmndXJ7"
    "QVtBFeymVTLlvXB6ykOi8mZiqjw/4hXqY3poJ/hbKi+ys4wwt8GNtPOya2q5l6+qcjA95ylmZG27UlYZoR5ZZ9pYFtScsDc1"
    "l/AxOF1TGTq2+XyNWtjwMlDZbDRui82GgwPeUzjgFhgh6QgFZ4RCdL0FKU1o2WZA4lFPEB1GK8hvntG9ME8qc/6eDqKX//Hy"
    "Z9qsv3z/fTTLkrOcRlriVN0k11sqv+S9UPAmtwiOFgzrvF7SDXbpbScUZ2xgINEenZd5lFAtav3mPKtSTEhS0pSlcjT1Lpxl"
    "RTrFPLh5P9UPAqpvkAhAoWKa0D7MqUjRjy7TdGXWVq8Cqe6goaC5MHmRAazi1eadc4oZXCXVOW082sVnelj9DbXETTQhAJlZ"
    "Jao4p92IP4S1l4S6o94omszXaZ/QAA7VYDCgsXqzYKER8s9kJFg8EAHR6+VqXR1G73749oW5o4G5ovJqLfgp46P+D/72D9AK"
    "QPu9ZV4Q4l9jHyZyTdPWzDBpdN7/QYf7NJmm/zjkC9RdtYop+FQz5N7Pyc90zS5z2lJVEqNHD7/o9eLo+Hn04Yso+nJdCk6Z"
    "Vl8efUEvHj6M9v70/yLamLjxCN7peslbIJrkm7R8lRc/JOuy7BESWiS0/NS3g5h7QncQHTM6RbOU9thx9BMt26C8Kqpeb7RP"
    "w+ca5m9Mu3tJNNIImx6bNK2iG1vrlM5k0WNA+j0jwoe+/wst4PHxcTSMqfTe3pHX7M2aalORB9G+/3phYPI10tYVIl+W9ExV"
    "+Z+96BE9LM2PA/xAb3t7j2yBOPabKLM/aMGPo/GJvMXF0MOAMno5PKI/zyK6mrMHD2IpO1ity/MeXi+ibwDykHqvIIuU6MOl"
    "lMObj1/U1uBtgtkvpv1oVlaEGvvROd3G9C+hxD+IYqEp0+XAnBXRM56tDyg9KNMKVQmNSzNHBN6Ng3YsZmuEwWKWCm99AKvZ"
    "gjfWrYx1S4Oiw7rFWE0ZAz+nIlvM4ZH9gJrJdMp17UsL85Le7xVH9JcGgb+AiuIPjiMaxjinDvLaLrJl74YWZtTX38mmNyQE"
    "FMcnLWA30tUNdfWG/oRdjTBRDHlzQuXQ2n2emSOvyC192NA7mbwTehGWkp5t6H3hd+7jF/7fj4QR6FibXsn0yTwt05vo1TxP"
    "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"
    "P/B9Plj1x0iRo2qpAK52nQpC5r6O0FeLyJv9bSbQOmUSWIWFMggT3M8Ru81WvcvazNUzpb07pspvFK0EiSQALnTecgR8cGn3"
    "o68eY54sw9q6JkmpAlR68NnyLwJdryPMTUwdHGsbQwVmM+Yrram5nbLqKEp8QNwS0srAVkojbUp4G5tFpqeXH0L1WDCe0tZd"
    "cWCL7E3Xc5RITYKDMq1qNItK9x+HHku1WSQOnoOvfbW7GE12ixVUaUQF3A1joyTBkVjwz+3HRoTwMPK6W0dBBhjv3n3FedSr"
    "Pb9XBe9NeAk+42KdhiA/p8J+YYoPA97LLvEymsLrU5ZXSvYR7y2rQlDg+kRqjYtBuTuXdxRMIsS/QUym9apFE0/VNZqPThjj"
    "0PqsPWNW7X7Uq7Nj8afP4/AxB6Gw7bJhTasrmUZc6cKcxvyvV+cP2v3R63xC3SW9w6Fy3Pgw2XTZAIo4sw9T4uN227+6wLMz"
    "nsetkTyaMTyaUltfOtqbNOJ4THbF8bBsmBmKs/Y2ZpU68H7gQmlstJq6zRmyTzISVuEwm6VL8tIZHM/i22Kct3iWtu0bLrLD"
    "bfVzHU0r62jqO752Op2afRGYxbNHotkPnrFpl8OpsWt9XfM3vYu7qTFzfV3zNv1nOJvW5/RPuJ56Hqah66nnk7r/uG5Hyjk1"
    "Gzqv+iWmyqpAr3NVV0BIKVivIOMJfsgyjuTdc33HczsKr3joKG1HkL86v2n1SXUeqTV/VKvtafFIvUPGUcb3Vn+a+XGTvbWR"
    "FPAgcW2EmtoUcBEa7QtEY0Iu8xcSGMXFuvkQEWJ90VdtNHs6iSa6za9XbePakvO0SBM9v15us+kGGDGw8XQgcsVb7e2a+vMp"
    "NGjSYfzwVei4HgC5Lxe0pHVpGdWniNDYLlD6HAjSahKkjKMc+f6x2ezICxzpBfJVfb34tnAs3Q61r5FzPBpG//f/Ee0Po1PR"
    "XnNyQNZFi/WeAxdQJkRKlMlqDogZ0pxXdI1Xea7hQWkK8APhSTEdNmSpB2ySXyNvw+8mEKkSPOf5mkhriUkqCnybCVAPT1t2"
    "80FIGbM9AG868NqiqQIBVcy3lvK1UVZtKgmldy2tK9HHTYAOoc3FWNpSVIcwU8jKuQmyKijHtqnsgqPH4a7nydrDsL0DT8Zv"
    "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"
    "f1Iz9O7FC/uB65Ao3JdwzQ1GTFMEtFU67zUPN+SutqG9f7/R/aGTWDQaONx8rx78C567NY7TxS8QkZvDbCzegBbw00h46iWS"
    "YqjTS1ehJ8sAsoxbeK5Kja3Hn2wbswk+Fvm81B6p4grMWAZ4ICyBHpAbKkQGXY+ug7zK2LP1+6O//0/UAsWw8aDTGK3Yw0wu"
    "GIiH9OzjpKXrgtNbYTYN5W+jtC3aQarSkXvF+TT/tM7uCYqDrujBw7//z769yrD+yivcjC6zWbFRkoGqzviYJu3G0/WqmKWr"
    "fNA3Rh70iVtgPuH4B9PgwC/gSHd4mamVaQrgXq8jrXPRO8YBeu3ynaKYlKQATa8bqheqXfVUvHYUL0jkppZnY420TZnLO1DR"
    "uwVDfCkqmfGJfXIyAw9l9eWcdCB+0OUH3VMAmsxaDf7Dlj4MSx+6pQ/VX6daNUcgK8X2gEKbRPnNBYgbzkruw0/0UPYJAeOr"
    "Pn9YXdNOYRwvmQ1+qJjCvVQtwj3+goGIzReKNfBbC5S3TCE1UlupIm63ekP+y1TTP0xJP4LwnjQM3SjV+B5m996xJ6oXZmKd"
    "aQ47Y+p4oo7XvhFYNNZMNlgVyxZxlBf58pSHjHqMmqthsbIFjDBKEGvyR15Fi3w1uHC3fMpK5jkvUso6JgsW33Flxx6ZwC9L"
    "esO+0lRG8SN+1DWPTu1CzYvljHSVk8Mh4W21UJH6/NQuDZdBlT8izsz5xLxRhCN/d/WXah/8aT07z5ba5913sCev++aLTld1"
    "6kWnRwnLdeot2spKt5lneobgKG/O+9+VdnNRxeQboQUTFqFl4yp2on4hb1Exd/fUeVpmr+dDNigUciRzJEhzoZZ5wVMvSyqe"
    "YbwA+lkH/lfkxRKoSytN87rtjAw8j7RVVY66J0DbgC8XzHZkOCRe1a9TYkjv+KAaL8nKJ7X0zQiVxKQ4JyXTodq0FGUIsyUZ"
    "1opiOczxwdsC9m5r7fWVKp4cVXtbevkNJ2kCUxhHJGpXebMiJiTRSEhskUYtxiOdpcMkqbAqhAJ98toqRqMyI49fro2d1rnr"
    "lrIP9POu87xrSRmE+gjiYLaksAlDgMLHjZQou2OhzllE1BSeFJil83cjBBxgAkQOlCO/kpKk4cyIthNMrNTIzm/sszAJ5wWk"
    "BYWFx96ErWBA6a0pGEDkBk7qYbR9Et6XGYUid+3mp57qrMuPYNL1BdbZRA/HQayZGMSaiaDVONWA0GEx5clwPyvMZ0X1My0i"
    "6j5W2mw1cgsrNEG65gmksKY0dJKDPmcFMkokxxWbDcJ9COwKtT9udCQdzqFnahjlS4ZFQ2EuhG6cuJME92r1IT33+f+c2OE7"
    "J1VFkys8bFBPqdYD37hh6KySpWKX9cXVvnQtrFHR3safZls9MX1r6pcJ5wLS9I9cYIrZKsbcb6xFSOO7pg28s4uS4pZKl9rV"
    "7z+mq71Cdyi7C9eCY4W/pMNI/vTOIy6pD0favPKoa85LOqmcKs036qQyhb3VoQsAjtDa9wx24kmEiKbFgpUDcwrft0OAM7s5"
    "is3zLlXlfWEH6H9hOu4xuD9TABPztmqLi6Ax98u/VL50W14EjR57Fs5/+5uEcbAedLIOR+rWxh79pg/+62AdSFvkEA1zWeWz"
    "nCBg/MoJ+64JEje5GZNA0TQh4CYC3AaAS/z3E8UD4GktSqzWYbXe6Xadcm39dl1HOD45wpq/MJxLfvRj48Km/7K9pvOiG+k7"
    "ib+jNxhlDa+Jmeq2WXvc88ipXRuAHrlcyZyzf1Qy2zDLFvDkRyYxtYn5LGU8W6W8GnNMOi8W19YSRGYsJQ7pmii+hihMK/gc"
    "cMoFyTe81EqoBWf3OBge/FJMtTwnP7NlOh9kjSdPyCPJm8Nssbp4jWMEITNdwc0bRehgFssTaFbYTG/O0wv7ylz9w0boip0z"
    "Pz1WFeJ8U4/oWIKXkzPjU2b1He/KyC8Ao0KPZVSTCM4ImUJx2XDVnPVayG2rgxSM2UVf0K2yuSNr+YOmW8QmTGGgiRbFDP3F"
    "/NaDpzxCEBA2gALDv5VpKZ0suL1j9VMpe/oKlzj4MYs/poydthNszFMKYPm3vxH7AqxbcmwhMJ+uZPNiylYMASPgl9wtfGDn"
    "5lfq1MHeneI5wgeVBdNwapHtninh0lkn2sgVFGBo1V1InFZoXAF0HnnImkcPVZsr/HX4wwP95++hfivF+z4+YivUHQ+nknbj"
    "q4LiCVnmYEFo/dM6Kw0ypE83kneYLj3dCRV4Z3r1+zC8k8BpM5LW+E4fP3BUcksVmlqbVxrzWOLn5KEAG2Pqs8M/SMQcrvLd"
    "y2pUxFzGS6ng0cghhkP0wXl1XbKIOdzUEUc1C5rmCJpYQgiNbbxBrwX13r3xvLnjJ4BkrxXhvCFXVbO8XjE/nmWwooNbX6ZK"
    "2aO7stIDeyqKRpMMX3Nk4z1Xn4NRs/UNmONgyjDdCY8mo8A4XSRhuG1Bi8zd0DFnB+LSRBT3c6YOmdU1c2yyHNtLdUSY0gI7"
    "s0Umd6IdyAOo/keiHraeYfkJHJeJ6wQFFAM25HUcXAyb3UglzFbkc3Q9H3AcR7q5lv2g1KMM/5ak8rfMprEg/qEKBjexhR95"
    "6lywk+eOVKmRnEOtWbjw4hSGPfnxCRY9hx6FGoWlNJBZckWuV5GE2ilNDffrhP9x1SN/qqk0IbHK+MhrYZVIxlIEuOnek/cK"
    "G0C9wrOEUdkLCnjFJ4oruoedPWZUB6jAJrEI4Yaj+PvPa5tv0P3EgHhP4bduQUclnRl9VI692Wmi0VvdT6r5oriLe1X9ly+p"
    "mk4yw+PlUUtNIHLQyvw5B4I8wsq7MVtst5b1FScMabjfIGQAeM25J37fOfDRHB3x1vvixmHeHEstNGT0vyoikjTPlN2NkOxi"
    "mWEmJ0pNr6FCfUfHoqJ36dUct3Hgow6QzHcesRxW4srQ8U981/8JtCh1ql8hWKVG0HmkCir1TzcegnjZ6GruAlKyEqCv260W"
    "nV70Y20LrHUB35tuUjoujZq5sCKHs+UmGpQ3qaBb3fgsSGAWHvgndHDLOF2XK0UbWFe1FljU2KpS/OZk24pCi9KVneg/6hnM"
    "hOZqcDLwIuA3av/jOEHoD4Y2ScRrw1RIjylHazDmSYTZUrQ1IXTJ52636b7UWNEgvFVJYcb+UzVmtPZGdvcgsqVpPup4e3NB"
    "lPBJkQHogk0UpDBqC1eSmCvpHHbjaQTuDsPbsJ8j3/1AgsUxiz/vw/IGrJ7p1JnipBLDmEoORblJt54GclVPKtsiLcuM0iG6"
    "yl4lVQ4vrHAb8PoJeNah+genANDOE+qwYXITYmCGpU2I4W1CVzaD5dxutye0Hzznr62bA/h6NfvjPBvnc7c35g+kZWQORmvA"
    "N8Bj1TlkiaFNAr6jprP90N9NbAUN8upKnsdIhYpzc4U+E0TlB9XK12TV9UzUEwot38RebKqaq2WC6hOTI5G44GaiZAu36/Sg"
    "Rc/Zesp5gE2linKeMm2Mi6w0bhurqjuIOICwt3+fn8HnAqgaDhLUlKCUXJ95xlIyzvjWRd/cOIlbSFMtY2qrwqe4C3Zgnkgs"
    "zkvqcN+lYbXv5rRz6L5YMJw84CxyD7G318NlqrZJCtCsZcoxBabPxi5MYQaFYltIJ1fautQ+w715rlhRChdhTtZOcuQkWxjv"
    "H3zlgT6R2KwbfEPCuLkhIoPSJH4bYK5bMLpX6YwQZdjN+KtMEihlCmgtRb2owpEXyyGZJJX2uiFvecx8hZUZ8wLKkZSzjmcU"
    "r+rzjzRayp56PHRhqLQd0eTvcpd8jT7xrQv80XEcBvtbdOngNl2S+fGNC7UGhiiCPQjsTaEzu6sff8yDeIdJ6Vxw6VOrumRo"
    "KNRozQEQDxIpLt3ENqYwMy+2fLAtpXpUxZmMYQh9XFXTRSAyoc3Utlk2ug8VXXKhsDbNRC7zIYBQzaU1MxWvtKNgEUk315Y7"
    "qn9kT6nqSPGZxl2xxWJiDhnWhamxZ+A6pAl/39KhCE06OXHkBJ79RJHErg/FVsEUQY8O+JF7PLIA9fkLBnRX8426IQ0I04TT"
    "8Zn+RrprThXdbw7vkLsnMyPSCfXJAaqGrcpReXQB9kELP9eVE/84lM9rIkWY/Fk+IAR7lr/u8/CTY9kSXICqCwrcicke5soS"
    "DM/j2B/y8SzF+hLmjm8tm1NOeXsce2arJb/iTC96F+o5NH2q9GF+7n3o+a+ZtPHHloV9ZIv2R1i0J/jDtWkPCUVqcvIRbW6O"
    "/fiH4YaiG474ojRxcr2ffJSVya4WzUNeMFUc3i4oSMmFMdU3mm1RfL83odbiLjca+cpk+dZT+5p0F0xtj9jnKq7SzZcEBUXz"
    "cWbYmXu/f1lyG27yCzWEy4+U9kLNKBZeX+FffsQqqik5bkjk00dnJLjDvyyPQywbcD3pwpXugqFspjiiX0PPTKb07CpJolKu"
    "J9p1WLSjJSShbi1C3ZJ3jUfEoUisGOLPsPmAA/IFOHgSmYFYMzC6gCTJ0sLKMF9mbC9zkUs9cUjVmU1H2pE+FbEHLrvDij6R"
    "kGN0vnLAOJUIxXJZToIYxDv25GhFZK/2HR+jiXDQobdpwGtnL5Qdl7l0kSSCCVVjPtkVVUReQjlrliB1UqzA9MoOruRhnVZv"
    "6fy/cQRkJfULJqgBeEWqYjmzRnk2HcKRi9SG3DpY8tGW+AMpXqpPyZvIlafU5AcnMzn6bbF0qwHKPUin0xERhQywO43ckDjg"
    "z3bfEBIJLfZpxbZ9V+5eRYIRAcYFCIDbRhRviOd9yLze0KqvD6lVehaoLOxKCb2u5U4ZjoYOH2/kbOAIF6aql25VoMxDFLeS"
    "NWejCypdB68jX1TbwiZypS1EC1AuBU3DfbtHBuvlJvMEH3z+u1IjCc+LxodDbCRHyVg5+LWUM7BeUQmRgJFV1akJsS76a3jx"
    "VStYaJ6BGaM6/A04n7inG8zlAxE7V6mWA+WmdVJVWCfM1SbE1TjTzASMTdXCpwzWYMJX14dgdxPx23CebujpjrqlXt84xH5c"
    "5IDAf3IfcAqozp9QdBZthWMHRy+EOyUeo/p7MhF6KLv6QWARK3FDp94iWtBroxOv37zHl4q7dxPiQKHZP4fWd2Ry3mozB05v"
    "xb1aVo8rQakxrD9FY8hQUF4oirwTDYPkrgAiUTWX2J5xLth8PDcl7qNE3BzDpA4YXjaRbQLOJlZXFpdYWu3K37TWrcYf8ROu"
    "GmCN2y5PmxsqhAn7Y1IRnyjRBG5OV6oWgMmswOod8CHhTnQCqAIM77nq8Q3wfSOJN+ldT79b4d0f8UvNBRc4lBoOpAb9NdOu"
    "roGfmy/pZQ1LnJ+7oz5y88ASt/yOAVuS6O19hw+G+bn615MA2T36GSQ4zC7km/k5LiUUKxJpby28Vx8NQ2G26tHw1EEBDj+R"
    "YqdxY5WRaHjvB/Z6IRsUJNppVjPBlNnYOP5uinz4ju+Bje3kCbva9ZEwepWZyz5fG1pbCnOIbfApuJ0sESny2RkhJVlE2Ec9"
    "d6in5spI4GRcu7hNIfukssLBGnfFQkHV6XWOXm3Qye5ULGIPf+hhIZoVHXxq0n3FaURW9ff6PNtGpy/NKNTkv2HdtnLSvuHZ"
    "8o/bKF3rMc+zCGFzezI4lHAp9I0jNQ+pzQOfnEFTJGQ6Z7k9U+mkn61x3QWkdxvaJhgggiI2ErRpErYxv4m7Wdj+6Nh3zR2C"
    "tbQ/aZz4JFuxu1uXb+3K/awoIHSEHmxoUzviaq+tVzCNACWbzLaA2OehOaFKfFewSAnCLOUUTOKZgLND11TvpaCUw+t5OssH"
    "jcWyGC/T2YwSrW8kbIDJn50T7zg6Bru7Go+GGXnPk3uAthtr1YGCGTjkh75cl+t0aiIGWdq6TNUJ5ONr5qUWy1UDSZsmsU9+"
    "Euo8XK440ErxjrbvOxDwExHX111s8xazq1fz0GeAVG0QvhLs1/JnwN+Di1Bn2w83XiTgXK37A4gu1q0CZdcdnNI4dobw/pyb"
    "aD8XJ0KuG95499Fr6n9CxpvfO+aXmS7EjMhaY6T8gVceXXhTUH8JlbrrtzlKyQkvZltusSefcHDt1neCQeGrIHhSEQRH+/NM"
    "YyK0JYQUGTsb98ntqumIBGZWE/e2gzRUEtNZUBGllsA+85W4LmpsP5eyXd8nZM3IKsm5PdfCrrgWckpb/kC7Gd7Zh03TtZ/S"
    "f3z1jSaE2XM1Aek3cICV+9S9/F/tQrpOsNpN6LF28Uyc+XJTEd24UbhSZAsdiy9SJdS9OdcC24yY4rSpA+oRqp9SGlxxue1+"
    "X/3Eb+RjJ9iZdGdgSb3V+NgNS3S9Era+cnXysXPq6vARAnlknU9vQRXhyq9fssuhpo6I1OAYB2faNHgnliQe/OKg8ZFYy/Ee"
    "HnMRl4pvQ4iGnZLX717e5B11PG6nykNNlYw/T550pZNpR41rKrd6KFjpDB0qiCd+rYNUulgjcsvEsz5ni2PnJ98VDmvk+zhZ"
    "N0oLB5Eux77ZJ9B1XV3oI/SShJRfh8PPML14ecAvq6lcHU8PUGkzZ31YbQdGK9GW+Wbzo46fAZuHsTcCSC8efI7f3uBYhjE5"
    "rk2NRS3n0uZHUBR9fEAzfOxZ+QcTv6Cqu14wJW7CwZHtvHyFWc2a1Nhc19E9Tbadl2WMy8w97yFvI89btLk+0oZ+zBbuw0NM"
    "x7pkx3FvS1HhYFB2mtYlo1qsVS3q7zn93XXOrylZccyNcwM2Lph6rxTz9MZR4kR3x8BYJesy2XpeVbhRJXjuzk66zKN0OccL"
    "lybZaqlGOTnVMXSm7nLGU8eBdFUjd6r2+QrypknUhNN7qdif+knJDMqIbJrPGb/bMbOJKKs+vizW0yEERhY1+SdVPwO0h9EE"
    "1koLHF77kw0BV4s0zfUoYswtauZu69x/iYSwHjVzcTj7FVnzryAlFHHXaJ08ALtnU2MvKZ6L0cmsQ5MnAeugD0Ukb/gZmo1f"
    "jP3AQG6e4M1FHwsnlwaQlIv/OpS6OI1FPTcJ00VsFQFMXoYT6k+rds1PQ3VPfRlqe891JrVRLDh8z+xuNrdbSwqbog4OdkLQ"
    "F9iT2p787OWrd+9fmj1jdD3435DrjcbNUW1FvY9m0M0CkBndOwMqoZRvQrbQqh2G1jK+BLzniwVFr2deOFIpuByFo4POfK8l"
    "N1ZpscyelrB/1gNh2QhKslKpSUFU57t54OUTqoqvxxQzuuk1UyWsBsIojW34S5x+9osOwwZ0E46ZgHNy15YGQv9rm9r2J9VI"
    "z9icFXMXZdFx1EVYgpLcV+VPsdS7zjUsJfWTFqvS7iRIVLsqT8pTDm2ljthU0dVSXVsqDDSQrpE/B9LnmQcI3Z5w6jfziXkJ"
    "z/uJZIEra7wOF+lSyWsccDz0MU+HnhOV6JwC+iTOdIamJeC2FcDMJi2aOrZhwNBkrmx9S4Oq5heq+MvmnibJVEKzZAIHS0hJ"
    "9h0mzHkXTJ0tN/bLySyeiGt6Cdf0P5u/q+6/u+g76LtL6kgyzknxIoCb4jB1VSWzlUdmipyvo2WiREYoH5y06VCJ3OQ1Wk0H"
    "ZcY0vKKCx/jjR2L/wysPgW14rQtc6wIWos1ZSo9Sm+fAcxpe2S3SVGNQT3BXqyYlnlzFOLhy+ia74Q4tigNnIeQddmhBHCqC"
    "wBDCC5LZ8gwMj/WkeJ5tN97SUQmKJK9bLAYQMSvEHcBnpw4qqVkc93XV7L4Y+Xgbph5gjI4DgNKxh1D6Kf5pVz7tBp/G8EqF"
    "xaEiSvftoXp8GnmYHl5eQ8GP89zjzNkFmVUQnTizaWgUWIUxoatoUKg4AWaUDRFkzBjb/Pdw6mmnw+V/IAp3YmJLCk5aK9F+"
    "6ETG8tPNKqRnzy+KErM/8Ny90CZq19fKtvya88sZO2kc+5f2zNq47q7VPhgusX1oeqEj6iUIXEPlwx/5w0P+ckpfYp0O8aks"
    "ZM3W095tQzozTlBYFtzmqy8lNAIjl/AFfqTPPP+QqORhpduUR9yIvtVR3eXf5NOgoW4WNIf0gmAfnpi/HXgQq/HpQ5OuJtKp"
    "0vYUFalDS7uHkxiYc5ZfVt8ucBOwuizu+Cm5yNlcn6oDrfcpIdSIoOdKcZTICUr9VCDDM901TK+DyhrZbLG6jsA44p2Vszl/"
    "jOqrl1Tmrxe5EgeoH5f5dEpYaGAwWSlbXDzt8UzRvJ6C35V+TkJV06iAV3pprqUkU7TJE51LjDDJ2Xlp/GYxWYNilnkBpMaL"
    "lsUWka8Jre3jmmIuRgDIJ3RV8gaDk64z/Uk7TMKjB4X7J0BlUhRKs0ZQDxkavBd0IGsJcZyxX+mwXi/wRRemXCspkY//XNL4"
    "+KkGECWT0RDQ+fmQ/CJIdacvQU6qvnG+oaui9p3QWvevvv13Mko4/vn5ds44qXwGhwPXRmJfHNCLuClAHD99gcsgzEySqvcW"
    "gz/H+BAPCbEMOpA7PS9Ijw7hxsOcAgIoXTEGZoRk9ghHDdJa3+XaFYukF4+5NUa50w1nx/FaHptKuFO6kh+5k1yJ7m9W5yNM"
    "4edPV5EhoSHuAzQjqkf/of3PuK+qxTCSlWvkAuzh0eQPu/bDLrnJNDcrp6qEQ7mqbXirxo7eWgrSDVGAyMrvV7ftxwQqBspl"
    "+9ERbVbxb7gs3yyzZ3Cmx5jx+PTtOQ+HJr96MtEXAnsUJ7jABkWm8acrOplB9U0+OrQtyRw3LnlTC1yOHLHJ4l61TLFN+rk2"
    "Sj9nq7Tev5P44ehDOZlW6atA2dkNuEJA4NZw4IFrx4DNz69fqCqqitwXaPw6xzg0Iq8HTrwn68fUpq/3USl54Sl9K1/pkyJj"
    "v0jiGxNufACTmtH5mTCkYpsKIwy4Z+aDEHkCYGsJNEqTQgW0RecdWWrPlwWyu3Lwo2LxFmfVnHDsJS2HjnWUhkZeGvuQoF1d"
    "XuCAEjxxdW7ks3Sstorj/EzewK7taL1hE9C2kZ8E6ATkqw4AP5kKBJfkmIsqJB9vbMYIDfGQND7PIxccRN/Fu1geOt7fCejx"
    "3q83cRtUulpB9mG61D9+hPu0/PCPMKrmdmgMoWx5ka9infRnE7eSNIXrzTYcGPboioJ7WEiPBE1Shoapx/EiU6YjoLQYzUFQ"
    "d/E9gS5wNFRoojJYC8ikEPhWm0shpiEid+OLsvG2mLE8DrPz9ZiKJ43gQfPeMr281yLKkioFRulkfo4aTxlujH/IG5pMDucy"
    "E5ecsosp5F/FG8nDZ67kYr1bmOLnEPV5UfvWm5kBp2Q/GaHSGIrJAcNDDl4Os6UowPlSmHJJOMnslwQDPcvXfJ6VjIdsYMcv"
    "zSZmlUDsbRDwjZTH7VN1C9Dg+VQpCJeQOGHqzstiVczzgcE3z8ZKlC2ViH+sXaiU2D5nKKyIbxg1U1I7yETlJRz48MvLn423"
    "0Dhd/NkEYlUB8NLZYhrjJCRevqUYaKupT96mpIv7ImmM6az1ZhGq8bnIpOM5Bq03Yhz5jrqOsOeuby9TJRTxOM58Tmn3gkV9"
    "96OqvSooWFeEyi3sBBchk45kP7hc7SrjMrGPUU8MCrnuiBg+6VbEbnv/NlFrgkHY/atOA9AJXCrcEfdwBKtZ+EiiAQMmKkb0"
    "0b3e0pwqVdsppTteVR7n61rtvc25PDFXp1RVy2ulm1SMXLAvbc4hOKdKNnIdpd3LWfaQStlD6pz/TgSR0nBqcnOf8E10qmEv"
    "1ii9TitMOWCt+tuYi9IWHKXLldNoXVzzIwOnxJ4X/VCiauht4jHLz6paUKqaQiIbG9Il+8YiGbxFzMkEQWiygyyqgfqNSiKi"
    "DTMpBt3V7QOktLkdRNfAwn5Yzyx1+rSp+qOpU7WPPw8eNR5YEqUMUYzC4mo4zkaX+ULidnSzSt1gKbrXVpsMDioEWKhBSRjF"
    "Y/zdrbfdYveWbSYFfBXESKDDiR36gRtioKFVNHindF8eeqqSjNtystN68bq0ELRo8kfy3SH0ya4BSCwEImPXccUpKXjRYe9Q"
    "Z2CBJDlAK3C9WvlEMQdJQn6tM6D4KQm9mDbS4ccUkcONTV5MU1UHAvJ0BZRcYjz/+/9ZN/7vf/vvOBv+/n/0QC/JakQuybBb"
    "DbMBRFKJNZMVmK9MOoV1mf2V+lB6jgk7CC3GCNNVEGC+lc520JfQ1eNHTFjAGSGfdqKsVF+F6zGXvspJas+ufHiiWcG3sKzv"
    "uNMEeUWvYnaXnI4ApmYkgQRnNNMKw2X7Eh4rfT8dFzetQYHK9ayPTwEVM4Y7ErYIQSfTyX9fI8Wp+lGu1Zj3kUnIU2/l0JGK"
    "LewNpo8fnvjv1IGkmJAi+e+2vL+EbSb+trvja/X+Mu5ghnwtj8wUFEpeaRmrT21lvjWIilP7BzheFvjzWJ5e6meX+slcP5lv"
    "zyrGXDMfXkV8cjTJkCZKvYtRwzUNwUxM+zLqr+fEvqgyc4p8aaFdN3rBpb5HivhO1PvTY597MisA3eDawyeW4/q0aYaxKb4D"
    "6TRVLC1F5CoE87//r27D9Xkw9mTtPkau9I+QWmgZAY5kx5aK9KUKy2qKR9GDbcCrtnTc7YyPxXrHs48RxzPMyEfP8cy2wp6R"
    "8EKjUuKFxtSlSlEkeOCIpsYpnmi01hVftIoDTkXQpvsszf+cBJ9gGaiXJrqpflDaufa6E/XbKwE1Jq1q3u5mnZMjhy4pGeeG"
    "K4arDq0xXSOQstIcqzNHMZFzDsmeFzhEBhdK9m85VcGfh8whFPQyXistdb7KstI5+9Jxms/3Ha7vuJBNV6k7+DWrm2sNNL8m"
    "0l934Ja1FbsXn3okJDz1kf4c94Ro7cRgtfpz6OZnsGhIfhzAtniXtIGjnz4ijdcIExsKkj9XE2AsSvzLltWFUENBWbXMXU3u"
    "ygLlWq1dKtpkvpL4GBMu0/ScRxIb0MNXLiTD2JVNqXLyc6RlJqgiiuEvDc0cO6sso2b1vDk36rQ21ySeDEQkQ/dCU+rbZT6X"
    "WGIkQBN1eaZUYbpHWhUYmRtS7w61b7w7eU5IgXezjFGWZkrprOGkxRSnCg8LvmPSOr72OJOBIIwoCBtS366ydEjp5ovhGBxT"
    "Mhul43E2dJCsXUGLqIGByY0cGsfa3eI7/+2cM7X5e7uvvPFY9AJYdLi+E5PTjMXwJKdbXTkjkMgQgNwZAspmulRT7E0TtjYn"
    "laoLmqjb9DGFvUlS5QEYCmvuLaOW5IE+clcW0V/BmERpHRl8vKghZ4f93hhuAHrgo0ZVASqXWQbjSbV1hNuyDy/K9OJlDrhM"
    "ePfBtaqVpk/5nBGOyK+ecP1I2EFL0OhzZep//OCYFwHnoGBV3dS62wvJ0AfVA5DAbwjujZaO/9TYLzWRvFDq4SNOhd21rZfY"
    "5bSpkoFcU9eepz4RkJTuUChvrHq9Yq9DjLvgHWO3PchMFZWjzFd06440Oe5tzAC8EImxsTigBIBJPp/A6ZA5PkDsdi5ib/sS"
    "1YgBJYH62hkzkxLtvEnD+QanyqXqXX+PM7mtDnHW24u5d5444rB7fKrj8qNE9woIBs6ipjlCl1RUElHiW3XgWe9oeKHIgNRZ"
    "RQc6rlgRacsyFz6WkAI6yPj8IeVdn03mcC4Ea5FGg4Uxh6xtfagTIF0U06HNITrI2nf2XYPLmOhuCVDi6CoZKb5kKz+IbWW6"
    "PJpfvxVeX42ivN02J6F3x5BuufniMRh7Him3PVRYNlKrTa5GFH2tkyA2Zn0JWRGCb9lAFIqETmKKZ+xwMupQr91pacB/dlLb"
    "fli5x9WbPY6r93sfVzJPXDHiZOnz+Az5djz+xDfghd6jq2Vae3KKNs3AREcUKwN6bR/5dnCyJ4O4RKOwX3H1wWgqwEh1podZ"
    "jPC9DOe9X4+c+dpRRhWfbN0/N1zXWQLieYEflFHDiSFAn4Bl/cmTxlGrYZ+9SpFxiR57uQJMdxJzVSRn1zIdrtMp66qqItKX"
    "9NXfUWMWJf+YWUA6XTEO2KmyxgGnbF3wJO5aDGUIoIoZDaImGbVZT1UECNNzNB3Bb9Ti8BAFnHzEdJi/QS51Xk5ifHzU6HWO"
    "a/yt79KaxpYb4Lhy6cqnm690STAdAXIWtWHxRhuamzA4Yzdy4t/2swLRZXawa2GeQUhUhQUnHDlmDDHHVZxUeE8MmU9XJLQD"
    "PcP5aSAPylCa9HVCoWBgmfoxhBdI13vtPwKWMVvOLFc3DirjHI3+Dryd3Egdr8iKcfORjqeQlj1HhJtAT4bpj8HGCc6DE5OI"
    "PUNnFKmQyFeIknuLAbO4aPltj/ydx7yO3SPX9MpLWhwWBKylauaJDzd7rJdZkzkQTNLlgkX4RpP37UAH/7Gfg44jIEnCoKDs"
    "XJejLxLxv3BZuuGyMFhU9Piy5wamkIHNGaJL5vSA/6AbeN66SCa09YB4sNfR8EAfCqj/oyzex30WL6rXqOOhVNpZej7NgBZ4"
    "CdvVKIePuxPhzfnA//6/lDCIqyHNmMnANS+MZW+xXlnZsBDjWL2WdBwqe7xa1AXm3q3Glxn+v8bi3/stDP69vez9vdua+79s"
    "F92Ku2F1mvQfRX+txg8J/ncc+KiLzUNND/vm1weef2NWuI3ZcbdcH5iab2bmE+p9PCg8NzC0CCJipG0ZOCU46DIr0BR525uD"
    "2bZTs95AgctjwEo5xhUC0NqdVcExTjDrNu5rFjSLnLRZMs8p9XJL7gkW6+lUPSSsm6bjOofnFQMFZwuHnomTWv2rlLaUYGux"
    "sckEQuZ4jQNq74QIf2tOsmLu3BmKv51kSthqmjBOiqqC54SYHg2Nvu1muuV2Mp5tc/KNEN9KxzEB5/bnm0Qg2yi+rKoEVTwh"
    "9AUH3xKM0iVPdQAjB3poNbL2uK3YvRMcdJ7BTORXRqnXw6iillzewBqm+NsQdxPucmiEZb8qJ3oGKpRcrIBASgmgUUvyNr1K"
    "KsNhQY97Kpcy/AHyAhKGnOt82eLgBaJLvyokYaK7yFWhplpNSlKZUu6CgV2Xn4Sn/yCxSGXq+QfqVDQ3YUgOE8UmJ5sWj4NI"
    "YjKPq3R5EKGz0aqo3MJX1bfchCoEkoubeGMiV5kk+f8ILkqdf2LH0deT73i95KrX9sPTZIvmdytbnivrrwx7jNkA1OstzuaR"
    "qI5iudzRFWI1mT/NR6Fv6BcdUno9fiQPCgrnTahHzND5GcXlutIDXXDp8lpnRxcDVZ1pWi+4+sA6HpGDa6Wx4H28Ycgu8boi"
    "nbjxdxO8vYZVXt5ojqYp8SY6XPCoDw1BdW7OCIzLrLrvUNkHSqxMy3ylOntFWUhFpr5KatGoyHNWCdmQ7FZmeWohp05ygJ13"
    "ZJYmcHjNEWTc1U8CzNQKLeQdgUzJtWpg5z3vKkE/h1vfkGBThpjloYEG/NCkGc8ZGzDv8t9wfB126mb6y+lwFxlyj/Yjxi3U"
    "+KXEWNP8TpKspcmqqoc+3H5Ta/2tohT7XD03nnpbVYOjvXS6I6vTXbJOdxQ1o8XtJjduxioIKeaCijZdyQHZ7tXPONVRB/51"
    "j5evSgQpxrzR+nwoViXfQmLaqcZrk3H7cltUX72a37iFjYwtY6rXnd32ke0rcmNcrGnW3/p+BbvHv3effWPEqeO4RIaHzsPE"
    "6YFWZyhCbXqtZe3SgeOF7xYy8sjSV3JNuOEj0/1jy7aHeezwARcBbUX6Odxd/Zv0VsOaVLVzS8xQ298RV+TGa+Qd1mHzLuN8"
    "9Fyz0B3PGVYVpSOB7pB0sgp8os4bg2PCPwj7L+9YrVJX0cXRnnciVXTdKrq6iu7h4fHunCsmBqjlRAqVTiYRzh7iQE1C6s61"
    "ObVcKc1KFV8yjXiJRTyaSNpMTsYDnV3QIEavlvnslzSflgyeXCxKE70vukrMrU18ofwsEJOcmIuAN3zeEtux4CnzwWKIi+iw"
    "OkBoVGM7OsYxqVPB6d+ayu5RTSo7k4izghwHGht1WKbwaWHkkBOGzHQwwjEIbbWreINDQFRPl+SPsB6Hppx6uk49DhUZ5dgu"
    "mPqO+zLqVni3PvJGHbaTfuS21B/AniTPetPkx1MOmz6kpBe5ly5n1GGoUByCFW8fv6UuAT3zgQr566PmgLUtdd0zedRlxNJI"
    "S25aAT9yxwp1J8QDOBFrznwBt2pwuTdRyRb97t1odEhId3JyrjSEo5MzBfaKqJFm5rLYi3y1J4dFXPIpBygTh3XKLPJs4AAn"
    "uF+zUyDwuVAm5Mu1AVMcocpJRuIRdDiP0Xni51PacRWMsJk32c1ZaAYhMFurO1J1+GLaMtDJN3tFqXth53uidkmkkBu/7uwC"
    "tchAHcyGFvVD+Bo8RdOScQaP2dKRj9jxlYFW0pIcEg0MKDPL0YghSQp7HerYtxJ2ap0zl/QoJl+1bKYr9uMhgBNqsXBQWiLr"
    "sgqyUmPMM2z8GQUE03E0iwTofw7AChB5NnMDxXWqY5ur+BNE/8fElNV2+gSzQ5IAcaoKU9ALUlx4y3JzZ8/RBGP5xp2uXpw6"
    "eModxmR5EcFk+QY7N8i0QkWMXFgGAXp1gGv4KAq5pl/Ugq5pyDUaoUa4eMHAazJ+HliVwfrzj8J7M8698DSJexqDsZcFuKnd"
    "0UXocLzjtLcT/Oa8UG4vVJrhQHbgLZISlHFfXXnfnK4pgvPn6Sp7j7jzWmBOYh4auQF0ykeTQ6s6zP9EyIfz8SJxYt012FdA"
    "GuKvu+7iVaE96mMxGQeOrNkeEJx58msgwTn7QwZbxm5ySxIXVjQ+M/IyGG95uk3xFJw3Hk0F6M19/E+kt9shvdXhvLE5wIV3"
    "U1PDv8tilll0t7seuhtoNqh3JbKwLVSdxf3Y+g7GbvdzlQalU1rsebSjP42vPx5qPCy/3RHxhcdEnc8iY2Uxxn/kCHE4rnuZ"
    "fhMiUl9kckNpr7WW6ynCnsz9VK6vmUodtsYROOoAgce0ropw/cxRJO4tbXsCFJc/p0oTZ5LBdk5nmT0MNODS++Lym+AtaQE2"
    "wFZS1ZPkxq/ZbsBPIUKbpwGqEt6P3fdRSKUobhI+rRrw6EwKr3JU5QtNWOrvT6fupY5n4u9yckXnZKsY+PV0u3f1zsYyFIM6"
    "TianoiPG35LJmRI08KrZOkOzHqMZDt9RInK1zMUckNsRHIUvUEnu1ukk3IjtFD5uBvKvp2klEeHd9lxXZDcJQP0aFxRW3mi+"
    "+PA2sdEatC8skV9I7LkL/EDRXxczO7j0Ms05G1nznlu1X2nj//7X/3EvMPYxIanaHVPfZ0mp2+eJpJufG6U3Mm7Q6VbPH5Pd"
    "INRskSRDIy6pzVTJ3XExY8dg7tDJ9JT3gzy37MbOB5d002Y3DXLKE4duZ5lal3kfyGCAYcinGdJzP0Q6zcUfOuHzP+D5qlhU"
    "nv8haTlVosKN0WRKJH3mQ/agoRR4un3SndGf3ZisbiE1HAb/pzbr9FpREl+uu4v7br1SIqr2BBAW6iyrwQTDc+u5IEpcPn/t"
    "GoWUvKs2QJhijid0Ye6CkVFBlf4vt/2ya4SJr9qiTeCw9TQC332CZYMk02uTqMYpDjTyvVsU9sekzmqCCS7JeTYdZD/LbDeZ"
    "UFuNSyQkJSye8+JK6tOjU2UxQItOoh7wqyVFyXomU8/uns8W03x0/V4Vo7Jt9W6Qrpon+EFpNRGw8b2fh1xSWos5faBoa3At"
    "AF6DKx2UqfYmDOaD60Q/8dPe+96/NPZapwBxfJ+mYx9LQz5lfkmveYPee/fq1dlP71+/uOe1ks8ZvLVS/PWfXr18//7li7NX"
    "79+9Pfvw89Pnr//0k/+tXlL/w7+9+/Offjr75f3Ll35pEpNIe3pMxlbnk7cv3/+kWvrw85vXv5w9f//ur3/yPwXeN7laVRp7"
    "++4v6sOn6psPH85+evpzyDYvxqBlzYsU6cmfQiymMHt5MTCIPTSH/ca95wC4/rDC2suB0l6kww+4Cmg+bDXude45/IaWzP4E"
    "nfcbkY3RqoBc9l1Jzr4mBqc1Q/sYJoEz1ee+VkifNIyAgkhSsIEDMrUTkKz/HfshhB925MPX0Q9pzvsy9TD0Ne8duwPnaT1j"
    "Fn4269PEP8F/2/ysUqN8oTi7X1w9qCurJsMvC0iTmrLg/GE/Kr248YQ4ujrA9c3P2Jk6yRJ57NonAaeMnqhuJQd0/uXk+O9X"
    "pt7M0nJCLzzk0SxdMk6kPi1FL16l815zw+l9Bc+2+wOdYPT+Z6zc0fcdmF/Uc+6o8KXPHp23AvOTWtioTUXACEV54/17xExN"
    "yR7+875+DzamG1CCNdQK9dLZVHMQ7pnYlLBE7Ejm0+gmQ2RSidef5RO1kVoC/dRvnCtBxxV43fqsvcAvIhN7NszGfT3LzTSp"
    "7sV4Se5SrDjhjp7R1Td/wrfg9R84nV3kq8FFpMhA8dZVX1/B5P+exapZXcNRHb8/4E+H8Wiz2hmpbZg+NWnbLG72W6aRM+K3"
    "ZzzfZ+NUsSHDhV1xSklcKyNRNTec18ZKBnzfB1GtE5sIits7I4/aM4i01GrJTdG7DxTSYt3BR6Mzog/ppNV0FXM4M0PrOyqw"
    "/ZiOK+J+5Vk6HGaKuwvf1waYAY3kO3uCfdcGomZz0CbGp7QHX1Jk1xBRng8N0nypKGagpPJyAY/VEbuhzBvQk6cjeJqcZ6vL"
    "LJsbL9Iy6CMNUdeCHx/ob0yuK7NQsVlKYDGK8GB8dIu38ee7Scs2I3Vi5E4xeLFSJ8+K+ZkpGxR5hQItF0uLxxCvUl7qeuVn"
    "rAjXi4VVzII4Ay2QV5S8dD1ZbY2kV14ZedaKrbJTjJ7Yum5C6kAHLtPphLjIX9UfYYFhupyYpcEPrjhajCvCn7GaQAxn5UU6"
    "zKTgLB/Gyo3UgaUIvSymayxMHxEpWZwgYgttS9cutFdkj4V2ytctdLVI7UI7RWsX2imzbaGdYrzQjXMlbZ2VRT71u6SfRikB"
    "PPeMHb6EC4K3IKb+TMf/9x0fIv+Qsa7dfWOIqStxVqaz7IwZVV9bWCqMWMwTZ+fXZ+RrdmZFRrGxPHGNL9/pp4GoY8/kMTAP"
    "U6Ylx2o0Pj9j+eRs1uvj53P69bZne5RNsw196HQyLVdExTigkWMT60AHzTLD5H2gV8450+DCZ/x6SI3xo/f8xHbV/EEC1RkE"
    "Zyruyle+PC0DMCU9gcsvCqaAHl/mQ1hQlCrJkqMSGQVrxLZPEpz+AaHNDBHS1Xs1BZwvHq/gVTUfZqA28wnrqzUf8cvYZ6Kr"
    "RL9DcseLGZZe/bfyLbnVkN3iMp+rqW3/nKmpeKqI5rrMIcfyuqd4oEQLNRnYnTRPH3J+oFgm/UX77RVmHJDgmTpdf6K8fuye"
    "9mG9HCnNvNV48fKXp6/fnP38+t9evvnAnfjLy/cfXr/7k9Khep3e94fd7mGvcw+vVO9uEhh4/x9HiXQr"
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
