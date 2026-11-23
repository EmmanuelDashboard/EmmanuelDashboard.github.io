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
    "eNrcvetyG9eyJvjfT1FyzDldJYEQQF0sk6Icsi3ZipZtHdnneKsR7L0LQIEsEjdWFUjAPuqYt5g3mN8T/Qg9bzJPMvll5rrV"
    "BaRl74iJ2bEtFqrWynXPlfd8eP+z6H70arFIl5tsHn2blufjVVpMo9mqiN5lk3QZ/XKeL5f58uwoyhfpWRaly3S+K/MyypdR"
    "dZ5F42J1U2ZFnwAB1s/pIosWWXW+mkZpySXWDIf+W6130XJVZePV6vIIhaNo2I8mq/lqU0QPosVmXuUH5SSdZ1GVbatNkUWz"
    "LMXf8liKH/ajqkhz9Cda59tsXkbnaXGdlVVGfS5WC25QOppXZTafRbE03IvOitVmORU49L/r7Cyr0ipfLXtRucrn0cOoWKVT"
    "+nGeTlc3CQ2URhBN5mlZ5rM8K2ROFFyiHXrUj27y6pwKR9k8u2Z40SynEcTf/vxDcmSGfZ7lZ+dVdBLR2+iAhldgGMfRajnf"
    "2S5p2bSK5llaVtEwovGk83lEs53SwDOCelZQf7hvN1k2LaOzVdKLJrQG1IHpWVZaYOFsaHcf03QT7IKaIvgyUgKcLxb+/Hng"
    "qMFsWc0zM94nfflaRuV6nlfRmLobLdLiMisOJqtlVazmcwJ1k9IIy3N6itHtarUuHXiagp7tJs2pNDehdmihZTrdKmKCTONP"
    "+9F6ni4rrP5ZkU9pn6Ul7Q6v66a1YymwXpU5L8lqFmXXWeHNNRrF/JvhrmmFLXRbL071My3BEhCiyabirX+zMt36oh/taGud"
    "RXGRMYRsmnBHyiOanHJBK9iLePPR5tosMW0Ck87HGRVb6kGxfZMDQ+tEG5uqVLTa6BR1h16/+uHdLx9aekr/H6d0YmSbuwke"
    "Z5PVIsMGkkbj2Tw9iz789O8/fvf3X96/ekUbfU6ndXN2TicGw6TaNMzDaPG//iedCdqDxxYWdXIp8xhli3W1C+YX0/94EG2j"
    "w0E0q2T+CVxJ222CnYDjk1pQZUr7B3OWYm7n2NAEQ8aMsZjFpOEX+Rp7nIaFU44NEd1QR/xeuZ0g+yKtqEnBPtOszM+WfGLw"
    "swR+kj7ROV3SCZqlhTs02Q2NfY5dUa5XFbDLfHaT7mgWqxuslBkwHQxdajMSWinCbRYSn87kmKa92PD0yBbhbbEHKp90Wie3"
    "F7QVVMFuWqcT3guzGQ+JypuJqVarY16hHqaHdoK/pVZFfpYT5ja4kXZefk0tx6t1VfYn5zzFjKxtV8oqJ9Qj60wby4KaE/am"
    "5lI+BrMNlaFju5pvUAsbXgYqm43GbbHZoP+I9xQOuAVGSDpCwSmhEF1vQUpjWrYpkHgUC6LDaAX5zXO6F+ZpZc7fs3706j9e"
    "/Uib9advv42meXq2opGWOFU36fWOyi95LxS8yS2CowXDOm+WdINdetsJxRkbGEi0R+flKkqpFrV+c55XGSYkLWnKMjmaehdO"
    "8yKbYB7cvM/0g4DqGSQCUKiYpbQPV1Sk6EWXWbY2a6tXgVR30FDQXJi8yABW8WrzzplhBtdpdU4bj3bxmR5Wf0MtcRONCUBu"
    "Vokqzmk34g9h7SWh7igeRuP5JusRGsCh6vf7NFZvFiw0Qv65jASLByIgerNcb6qj6P13X780dzQwV1RebQQ/5XzU/8Hf/gFa"
    "AWg/Xq4KQvwb7MNUrmnamjkmjc77P+hwz9JJ9o8jvkDdVauYgk81Q45/TH+ka3a5oi1VpQl69PCzOE6ikxfR759F0eebUnDK"
    "pPr8+DN68fBhdPCn/xfRxsSNR/BmmyVvgWi82mbl61XxXbopy5iQ0CKl5ae+PUq4J3QH0TGjUzTNaI+dRD/QsvXLq6KK4+Eh"
    "DZ9rmL8J7e4l0UhDbHps0qyKbmytGZ3JImZA+j0nwoe+/wst4MnJSTRIqPTBwbHX7M2GalORB9Gh/3phYPI10tYVIl+W9ExV"
    "+Z+D6DE9LM2PR/iB3sYHj22BJPGbKPPfaMFPotGpvMXFEGNAOb0cHNOf5xFdzfmDB4mU7a835XmM14voK4A8ot4ryCIj+nAp"
    "5fDm42e1NXiXYvaLSS+alhWhxl50Trcx/Uso8TeiWGjKdDkwZ0X0nGfrd5Tul1mFqoTGpZljAu/GQTsWszXEYDFLhbc+gNVs"
    "wRvrTsa6o0HRYd1hrKaMgb+iIjvM4bH9gJrpZMJ17UsL85LeHxTH9JcGgb+AiuIPTiIaxmhFHeS1XeTL+IYWZtjT3+k2HhAC"
    "SpLTFrBb6eqWunpDf8KuRpgohrw9pXJo7T7PzLFX5JY+bOmdTN4pvQhLSc+29L7wO/fxM//vR8IIdKxNr2T6ZJ6W2U30er5K"
    "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"
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
