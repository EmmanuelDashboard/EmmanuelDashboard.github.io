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
    "u01gV/5fXF7aSQ=="
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
