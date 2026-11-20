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
  const result = await P.analyze({ rgba, width: W, height: H, pixel, surface },
    { designFt: meta.designFt, directionHintDeg: meta.hint, youngTrees: meta.youngTrees !== false, returnRasters: true },
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
    "h9t3IIMT+LGjUXNvO/PuGfUWS4/6Drixp3I8mr1594xGE8uI9i5y76oCdHrRiTiBBm2CffXr3v26C0nW6eVRtFb2eh+S6u+d"
    "EhXwj2CruIMIbtcT/fMIRHiPN/ueL1t31ZeVk3CEgDSji/g20ENr2Kq6G+zZfXkEwZ1xU8BwKmIhmzOT+9eU1vSDRnyDUSP3"
    "SW2sm6IivUemXDA/oUwEleiXOqdAAEwlcZjMFd/I5emg3orjwLUQx2ZkZn6kM+eJX9HER+ygsau5XXnstWPn/YGnU6GPdory"
    "wXJ68Wkwkpze3JZl+sjKDU1zFYQvxk3sTkV2cacD8XmVMfLPlDLscXl8lTkOe91HYCjFis+1gLB5HUW0Zri9nO4iK6997fqB"
    "9vTmlSs8MF22EQyQVdUPH7vdQEcnyX8RcUQlZZXjVAgyj7WPsVpCXh9v5X9QJzILrFK9ts1Yz2FsLLNb79U5rA2snirv3QAq"
    "t1P04gXZRnN+3ik3natzaLejLx4DTvbCGlyTxCRTpgf3Wv6Zp5QtGXMTsgDb2rqojzhtnHylNTWnE5LGJW5D3BNC7sOoSaOQ"
    "SfQAG2FfU9JyJATbjKNdLY84XIvsSdcqOZGKBAdlgvouS0qPH0PQ1ghFusFzYJovDhcjYAfMlXIjKuBhGGMiCRzBEnruPzYi"
    "hAeRM9wqCTKNMfYeK82jUXXcUa0ZN5Ev9ykXa7TY+CmV6xdAfOLdvewSL6LxDFFgeXmlZBuxcLLCbwq3PhEv42DQ2x3RLCyv"
    "XhIh/vVCXmxWAZU5VX9ncvkKIEBDq1B7qomAW9XrWPzH4dh7zD6+tl+2gHkcApo6tDdRTmOn16reD8LuftV7QtXjr9q8Meeu"
    "fRjtmoz1RJzZhs3vWdhIryrwbHSXvtVRuu4iXZfautLR1qjmJj065CZtr2FmKqVZtrF/1Im7xo+lMVVdCTmhpe4zEVbhMNuP"
    "Sza+CTzE4tviv6IVxzy9SSvLRaqBdGXI125SPasjzNXhaJs3p2oaGB82ad1k+nsSh0hcitU0eOHZr7P/tsEHxyrUnfpGxaFu"
    "brRXiGHdw3mW7sxPFkSle18h59ftS92+X7cfqvvxs7pBgkPlNiKhanMsTef9NnfMoB07FD/I7oYqXvuvtnGA00GGEmhWVzHT"
    "iofATPthy8GQ+cPx46rBJ1R+dZ1X9RBTZZWn17muKiCkFMxMEA0eP2QZ+/Lumb5j2Pb9Ix7KRDuQdIKOqjoTxoCV3YXXfudW"
    "21P68N/CJnnnAtN7q+jM3JiSztpITjKwuDYAQAUEXIRm+xzBLpBc67n4nZehBD5ERFift1VtzC5JojJ2NMS+eDWbhBIXBKSJ"
    "xt8eFAx91v31Im5sMO6KXPFWw7i6onvc5dSzGDB+uLpuHA9ouS0HtIS8D8zqj4jQ2IBPxuwJ0ioSpIyDSLiOrNnk1Amq5QQ5"
    "VMW6OKFwnMEGta+RczzqRf/f/x0d96KpqJk5cRKrmsXMrmzO40yIlciT1QwtZsikWCB78HKpodMIBPiB0G0Ahw3n5jQ2Wm4R"
    "0/pXE6RNGZ6L5YZYa4nXJpp2myVJN0/ZhjVQzLs+Z8yKe0Y63LVFUwUGaj3bW87XRqCzYbaV37W8rkRmjRDZaJYqby5WzZaj"
    "OoE9QZbPTAA6ITm2T70ulPw4/OocWbsf0rDryPiNryVH66alGF/lbhKY0vtYjAJowuvNIrpIZtObZF+2Y3JN1pX+bQ2PR5w+"
    "+w7dJBbA3eidrCzH8SsbkwTQOqd6i6cadIxv0JtVdLlhtltxI1vURmVzdy1oFAtrG4p1yx3zie7daEBIHz4O6cPFi/7u2m0p"
    "H3RakmM6VN/V1NcbED09nfpMYOrGE+9W6jNbNb1Yv5IASMf2ztqO1j+zXAefjsBItVl1XbpI8ffS0oxlXNBMv3VpoXGyTnEM"
    "tsR7UgQtV56g5aquu7wa1nWWIYouEqQbV5UlKKguxonSH0uglEvTa0prRQRvT6tuEyMls/izKuMv2nPL3lXprUhvOAFnz2c7"
    "VkHu+9ow19eGub72uW8sVOfhKR7EGWuyCjsYsb+dIAKz3SuIEycrh+2+xovruO7Qkh8SpU1XzDZ0IttiO5rysCfXbunUpLkc"
    "WL5puooNFhlTDH59jdeMhUSh1M9pBSmUPl9jHSXrrjo2yde+87U/HMZNqTvSikO9Ct18jyBmKphXT/u0UY4loEdfOK70OMyv"
    "s1ovL1w2vJa7YeAxr6noclNpHhcUZWcPlTrWUsOakNCmVJST0Z6jRmvwSq3M+GxqYcYnTPv4Mg+hi9RAHBsft6fZYqJUGQie"
    "7wj4+1rqVt2AVa+x3R0MTfN9o6/YGL5ib1lbszZPY/UV69gPz6TZtw3rGbkhQa1p6njMEmNVnHmvSwkdQsc0Neof3bhGWmEV"
    "L4Qcj+NsTcd/DVCjcyUGo/P3Nbt2bO9kQf1taibxvqMV1BFvjbMV9Apvgw5XYykmTldjKdboeFW5XUPbOR632Z9pvT7gzjTh"
    "QbxiO90JbiRE+5tMYXVyVePw0oSrwfwe/eCgYCOsDwQ7V+Z+Bck6gdMTt1/Ft7i9jM5LgtDrwnBJB+eve9PtBvkQEc5V3Z95"
    "+avp3cvtTKMzYrm62VGoO3N5MxYU/5orwxZAqMc9/xo8Oq9cg60ZwoWn66tUgbyY0xtb45BSdhxQp+Sv6Wt5cOvqsF+yrora"
    "R/h1cU6LlYI/Qr68sJ4dTgySJLi68f7z98AVZxcVlEfRUi0m24HfBXJN87QGV72hFRIc3ETYJbJh7gkcHae9Ztt+28dZdHSU"
    "udYCijF6A1DCDzHp3DK4LlE61YvF+VIMlKvtjNL9kolZUsi1ZsSRrKcqrSfw3kjAdtXkyBkR8uXMLO9jPfeanDpZkZUmV3U/"
    "mDs46d3BTe92xxWOo8M0ZBMfsNAfbOkg3O416slmJxrezR6xTzY7UeGaX9xph602zS8YbQ2bfW8gDdkaGG13dScaVkdCMS6Y"
    "zblloBMX5OSfUIfL9tCfzyw6f2AYw2qtFOV8bBjKFdaitQ3thm3jdqhuCVHPGjTfHvCPckRvsnm2Zvdsw9sHxOrJ3adkxwNJ"
    "xnHAa8lzmTzov/MxgOvPTSQmz6m2Tiue83lz/NiItNTA7C6yNRjPyV4TW7zAKM7NKMA8MiEOZmw7p17do6Db7Z4H6BoX68gR"
    "wwqlR3cUAeKAhDGZK/hwmKa6Z5OjWDOay173C/8QCnZ9FyWmdVja6fky3x/XyUzVX9WYqqDWkXHc4apHPhmpoAPKP/BwQWpV"
    "3tVGxxpuHSDUvPrIauvjJg/JbPE6y10/ah71nSZ2i4/S7fTSDthX0h+3qRLQhgBRmscxHSQwqL/67R5QSlaVnvoeTh+CpENg"
    "Yb33mukMQzfkoOiYcG2NCVeg4Mdb/PQqhhLHf56ZRChA3AF7iWPHWqIJINY04rg0jGAA3Wmrq43CbZYQdtw44D+hG9sR/Pa0"
    "E5cnCXlB1owExBYOZ0bAK9EzFLgleETNZIDNBEKF/zQLgIANwOM/oI4RVzCVGpfC4pDLtQmpXpL+ALSYQTfBgaGXod9d/hEb"
    "5v1DJPDWWLqi3Ri0mJqGdOIgkRU1eCnjqO6zMr09ddZmPehJdE+GVYlCxiOz6k8IMcpXoiXex+YG+7HBAkuk0O5tXaUeEJLz"
    "XBGok4j1cvFSLCp6bSNxdp/Nx7J9l6+8JiYRaUgvsiEzl0YzWIs9zn7znJoARPI0YjngRSZhrVchWzlPleco/OJDYr5c1cPP"
    "CxFungYzUOcAuCMqauVWTFcXK64b+gMcu8tF3aB23cXaxrYEQzBUDIsay0L5XqSqDsT3k0jKeY6jH5tgy3AlsAZAWoeP4PT/"
    "CigpKt0GKhfjDpkgG2V0kf4MMuXB2U2S22/bb5DoUflvVM9zFugxWFML2Pl6DqYfIuJBuVzbKGFO/Hoc0VkyfnqDpNdsxuOO"
    "6alVWOjMPrpxBcuOu9pVQ3j/KQcKRi/1LTVlVOrySZwv52mLbo0cJMq4O8f/KVFUnG4d/ezG089uKvrZsNp56qqdtVlX9+xO"
    "UMbWpIU+FIOzWBbJrGGuUl/e2ECaqQ2kaZJwFMYlTzVXEkuw8k4i3MrLHnwblrYE9F6nNlqeOmbSDMo0HxJt0QzRdYf3kwSw"
    "2+9zzoVRMb/mad4xTqFpRaI9GlfiQF1TCVJDtB+s3z/1Yt9O0/U6lKNABqgr+AfG3zAGiQ/OSzcocE1HsaHkLYulEzM8sZwz"
    "I7NvP1YC6Y+Wu1TySCXs9tZKpsQrSZRreZ8j4U/uRZVE0eOQO5SgnWZjWO6aQqPbYmV49IHxnmpH5VMn8Dj8U2MHHwx71uSF"
    "yNP3fExUJjyCTc1yN5h5TaD30YgjifEf34FiNOJwYvyndKOAaIleHcuXY68Of3koXx5KHT/C87ciHZd8epqOT/T/MLqA54Dk"
    "3dtLcLnFUvh7407IiSxz01hr74cWkByAdRuTdDdOV6Dp3cbd6GIy89UMRz9/AOuAWPFssLm683p3CL3acsP+IcSE2j17Zllt"
    "lvWbthANEjecgnUM0SMTpyKJEHIKhhxUIMujzcI0IUKVuF1mHzRt1Qxt8mRvcstZJXfV2MWxcYHHe/GZvX9p/AmCqGP4AisO"
    "YZDVtMJoxmCjA4cm68lkGnIF1DRqm/8OpiR2cBrgW+FlLeC7xFisr8oAF2oaMk+snYse6ZAGSfwOl2CwYdI7DWJv2Hl/J14k"
    "29T3bESu8oMOGA2odUv8CEGkEJeAIdzF+ONgwAiMuhYsgl/+iYEiQuYjqvBt8CnDqtecysqUwN+VScJCjrKH7kuY26G70pHe"
    "lcI8PZaI4X63K5LGxX2NvACEevT4A2zvVoHogvdssy2p0XFDyKIDft3p3Fr3h8wPP6t1f8jqtwEp3dEx4oR7Fj3ikFacL40+"
    "NsnLfauVoP7tT7FRca9RPEuxUbmLiYpGoYlDFiaTsIXJpGZh0sHg1MKkg5l4Fib2a9/56luY1HXK6S1pG/6wkckdzUxq4YL/"
    "fHMTP/nUwfjAnxge2EYHtsGBy9jAauoB0qnWHMZqo8Hnz7hegZw4hW/1warscD6/qmGpo6onxHptXCGI8MGhhGoZn4iKVwUX"
    "3dWKBhTSbFqxoyFO2FkCTOFENXn3WRl2SEjJmm/2sRDP2jU3Ud9COwPqvXnYWZjvD4Rj/sPBek1Adpx5M9aIxga4lejc9V2F"
    "crFzcptbrFxWHSnPwSupYRg6lf9Fkyw5h2FfXkul8906uTFJq5xoWbZCPZdOsjjnxFUq4W2pJwExWsetzpaP061E8o36X/Zi"
    "IyZ584oA9/AJvfgXvPeSQCWrX3DGtCbpuRcr6NxsQ6Zu6TkBlZiqwjRhRKT2fD9v4z1OhDIGkEq0V8KrEuEYDNTgTiih2tcx"
    "4aPDluiVPAyUaDqlOm6pDj0Nq24vPJMWQ6gFF5g4luPJvmSHGD8RwGCbjovlus0L8l22HkqCbAx30PMTE1GBH5UFkUp+eqys"
    "YDGThVgiXigjGXsiTihyN3ogjbnVJxn7X0nDQu24DC2nvOrbV8My6+diuZ6z8GPQoQZ49fG3Z0zt2YeIy6BJztfiVLFfCJ76"
    "3Dc14cHHKWCBntYR00+s1vqu16dBfdc7ljC6JnodTLjHy/UiNRCixjjwCPPo/5pb/Has4M3dSu7kGvW2HjZOb59Jnr5aTNjx"
    "CGIYJ90DHe9fi//TfbOk6nwkC2De9YYarj2uyF8KkyHdJpwg+u0kSddN/zXiIV9kU5YispiL0wmfNElFWJDxDt4CLJ3XVk7K"
    "tBbPcfdCVy1uzdAyi5ixL4WRyVPtro7iTwTCVbrPNbeLUfpaiFsN05VJGCFxGTlxvEuDEYyutkNX7NTkkYbpVMJ0a+Sh0uXu"
    "flRi7pHjdWff90tUBSKewWc4XfOV1iKYyWlmbniK/aslrRDyQblYNU+Txc/TKaeT3CtSmYw7HENX1OVjuYo5EDGCxSsLExFe"
    "XUnkzqsqXIA6UPHL3BHCDdkoiWbieDFkNYcDZZnWhy+B65SVRP1yc/NIlSGE9Loq9Jlfmek4eeWvbF75K84r/yBymgEiSwpB"
    "AMOttrTVlvVqNtOqjrHWJ655pn7rCuq4KxxSLe0ILj/0bhmj6XpmNaQuZDYJrUvoFgnl4mVTy9a5pNQ4X2ghDGPgAgmMAVXk"
    "9z59XzC5+9mxOGxJg52IR8qtHvnSUItnNb+A28S17i3etCJCH97beLTb6ms7tpb5GEsIqM9sjBcix9M1Ed6TaKORw4l+5nBg"
    "TIjq5jn/9FImLm9+4GwXctQ+MI4XiZw0/Ms5abQVHBtSkw8bffTOGylpDj/evPqqb89DPomcJm0dOolsYT9LDu5VIrC56xnL"
    "U/BO8Bkxesw72VP2fjkFWMPZo9a+73NTXo1ygn4NO3CPwP21kvfQ73FV6cyt+bdaTbfnVaVTPw3d39+zZQ2tv4icN9WZuq3B"
    "7MUZg/+5sg7Mi4s/Cjj/QJbNcT3J9i33trqRvtLAO1zYviYaQHtBrwjlPcxmwXQMH3Gv+183dBMeeW2fxhyMil8h0pQNqujc"
    "HjmZTGDsIBE/T19rktMQrQklmzskkHbPI6d1I6M+c6mSEhqbxSpdOCe2D0LYx/61xfozESjC3PRv5Qs3n/oMB80W4jX8rck8"
    "2bmI79IPouPTiOOSGwkLU4JTOUZtmTLs0wALLMHL/v6etwHMMOLT0v7qeaFIYJO8xCaHuQwLFU7r7q9/7qCO7jwogREqlIMy"
    "vj8qogygTZqwWMesFCOEG3nK3t364FxK5qOADR08FloPH1OfBZ46Xz4yj1/glnfco8cijuWy95kXWZI1Ei+X2DQtObvkQN18"
    "v0nhIroNYA7L5s847I4H0LaNWtjrfmEEKi4amehFG3DL+AGSKz3VcGpjP8HgF+2e0run9uWRdkagTztf4dvRmZpCffDCky2J"
    "m1wHIpEZHOlgOowfZ5F5NGgRMvVoQo7PGgUiBllqEeKY9SHi4u7qmWEnZC149CGjDd9xf+BI1WXCRrp1P+ozB9g3SGRiGQgc"
    "42qwQF4jotrUrtxE8WIg3gwtfQnaZ17T8PG2b4v2h+KXI2tN307LpT6t2ZwoGnMJPx/ydLNgN8voJtnuifkEPHK+nrUt4iJJ"
    "0CqBXUeVnYbeaRWID+y412qTJutg9QZkSOFqCOGE+XUNsUTNr9NsbAhmdxK9oB7Ke5YwRGRkA/njMrv+QnBpSe0jM2TJfxGr"
    "NyzMus342eJkLjy7W3gec2mikk/PuPcu0SZjEgc0cKg9jYBLbNVE1NvY/jbwOq+FLdfvm+jML+jcMOb2emHOn/kw1ouKV6du"
    "+yhjvFPbf/uktvlIscRWX7UJhkQ8DQgdyqyvsPhxLUKQLrGKKLXjk4i1cpuTyuF74p696JBP23p4B46Lx4itiNQbhjD8XOxh"
    "eQCC4P0A3mrIjSu6eTWgoklmKZprz2W5dd7F2ctxOLaAr4swHdd0KZgOUdu8PSWi6TycFpHjvvAQTIpDb1htCRiDH5uywMYU"
    "8L17ruxibU4sjRHeHGQ+bCpoKp03KEQlY6baejvch8/PzjZ5oakHaS2wqKFVvcKqXh1aUTDGprGBeWimMlcMqzECi5RFuois"
    "RcOOEY6Ec9PGarljGxRF21VdCXzVEJOdQyaY6lV1uO39GfioOirMxUSuQTLS3er+Hgc2NcOjicS3VowJ14QGHEWMb518BzBC"
    "izhW6VI7yiDqk+SPLbZQvOJdi7Gx7BosJCQmeLwPEcoD8O4zB7AxYdBBFILZYwMWjdJzDvZsSZN9gL+O7HMeqWgGzolqxYS0"
    "jErYnR04J1dU7IuJS8o8B6BAg0ThpEGfVKDxo3rjGxZnebK5qy7CLW1DH7Z1O4qSVFAV6zzDtGJ7hUz2ztD5RZvfi9joSIK3"
    "OW7xb3CcQn4tkiKYp/AJK1YsEDjOskWKKCZsUKRmR5NsnY41Wo3vYm/lSdRmOpsaE7UkyudLYsBovWD1UhqsSPheTc5SNkWl"
    "colDgrEla4jSReTZdtLKmc786L5Fks3YGb8dGWcFRyeU91xs4bgcxpOXyzriSyLGOVC+lUN4xoiLm1Leg+wK7Dd95TtUNV1E"
    "ukD+CQK7aICAojyWF+CDFtgX9M2unkd+jFPoIg3Y0i9G7uo+bDvRRLjPBxJqKw5eRHty11uM6K+YJ5Y+7VybdjpkAxDbLkY4"
    "1mkQStA2OhWhBPcZxK0Jz+kbxl01EHAjZFera7FhGKV7FZQWq/3rUHCmbU+Ay6jd12e+89dCDkJb0Qiy1hbT3TKJ9OFWgVxf"
    "b0HcnIFe8MDdItaH07CijVR0WSoPNtdD9v4ys8AdBC9RHdffFkdSk3HapgH1YXybzTpjUSPCvLATJ24ACSOo242HldTJawFw"
    "GFuDYFqkAQyT/hQeKOGC4DVuinXUQtKmLb55eKXbE/VPXb21ngf774go3C0x+gGJEq4DalFml1qugdw+XyMc8RLS3sgHqEOc"
    "D17+Gy1y7heppMH56AuqGmbi+I4j5JI0XAZdql7pxBQeVzBW17RVBNaCxMSK4H8mbNksRuvlVSrghfWLpbHtUtUpR0Wx5B/l"
    "adHmqFg2q5bIxm8ukKdezZUXSwmUFddNFp1Zb7avqfP84MwHIgdom9vD0GRxV1DQ5NYZJ3I/FFmyejk2rC14o1IEZOQ/rsTG"
    "3CdDuRpYSrMNs3tJUcBIQ3DQ/HiKM0R/+LSFm7nbPb9K+wjuocHV8tRtGXSb7SEpX3zaKLopBTYxuuQoXzMvrEgAVOauaTSx"
    "Ity7h/p8mecrTTVMcXnhh++FEStWyJ7iDqO5QZHNNj6tGLgIgqktLGB5AtV/ynH5mM0w6Ktq/NxyQrlN5GytazljiMSQyNYq"
    "oModk11IDVx7XeJnYMJtY+zd2D1zw0H2uGeY5BOxPSeKRAiwl/65uRWWfgTDxxvOqpfCinhZLBfZ+DObifd8nVKp5YLwyjJw"
    "C7HTIRb2/MLGIhY+D93k3M8Msi93N7775cUbjW3GJpx/lR++GFOP22S+moU2Lt/NJSlL6SRw9SP711tfgeY9vjE4ystW3bRX"
    "PS+s2marsVYf8NDBmvd9pQiV2Gzd08wp7boCUL2n1HrduqvUE9R8Kq5wPF/11KvipritjEszLl1hrBfr/6qnyT+v+rVA/yXz"
    "cUVrgkmU20YNZJHtw53xMU43gsIlsxsioqT9f+mKUw2BSGj7JyAGKM8hOkb0ZqRvLB99qT7oLlzjGj8G0cV2FCGEHbHVruje"
    "FekLu5MIuzOS5zh2MwbRPh9cMRB5HIkRYGxQepPUbb99imbqMmX77M7C6ZvC6bRJRn1mZdS4K/do1P0qjdJt4tGoD9QsMJVA"
    "yGhTilR031hg4XcbaHlqdpCFHX6jkQAnYdIa4Yg0/cOCoHXYwsXabLzbzEvs9HGTxmOwk/rH49EZm1E7qK3yNDdYkbPRFV60"
    "xXmYdexmL1Id9bmFWuV8QDYMmhQrWeFAT8/95jwOHAWnK6iAWhX5DwYcl1M/UijDgqJlhGRGs67D15eebbLOu6Rkw9NGL8C8"
    "tA9Bl09Z2sKq4f5p6clBTFly+3HVllQ5vOiIQZrmxXKt8e4+c27rcqLYgyRmE7A5lKOt1XI5i5LJZQL752ibLWcJtZGbUEjU"
    "AIfGPl/8x//YRP/z//y/cDb8x/8wEyXKM76Q/PTwZJqkY3CAGjxPV2BRWFPQTZ7+ymPIPbXaLYgWIoRJ0RwTvI5nt+CX4tWz"
    "M0EsqHT4HsiYlRizejPn3M8PxTeK22zd9dIyW46vDgUzL7sQt6fQ/TPjI0CwGSFyQRktWAmV8u4NTJROfN9Z6dqaC2/mJ6gK"
    "oR+h433eImzXxCf/faN+o/ZRrh0tTuAXGAe8MrThUoAJ8MnLgf+NDiQiQoTyDw58v4EnXfhr/5ba9P0m7FZRLCGCNiCQaFji"
    "iHfW3Jhvus3Fuf8jHC8rPJ7q2xvz7sa8WZg3i9ti50ust11jpqjRjC9+PLoQNux5ChYw3Zugbb4jeaEyCxa4tNGve313se+M"
    "kG9A34enPvUUUgC8QfQuH1lCKX0+upp+DhW4BXeaEElDjlNOfPgf/y+nOLHSPvCxLr0UgeIZXBPXAW28pFWqcV9U2MQwEZuY"
    "R4dy7JWla5HC+Roox6IxqayLjy4r5gpwNwJELv//2q6st3EbCL/3Vzh5srCO13YPYBVsgzQXXGziwN5uWwQGoViyo9ZXLcnZ"
    "tMh/L+cgOZTkrJtFX4KYIimSGg451zfAR+yZ7t5CiD4QwoS16DMwdela+gpmPC1lsijOdYrfenJcb0m2uZOeKxftDCPDmP8J"
    "9HZgGZhDFRa6qX9gEHm7gAu3XFOKGM7AaMRvNbxdxpAbDFZw9kUEa+4YwhDxG6PbAgorzZk+czQTuSc18HIFhwgkSw9aoivA"
    "fkTtA5yIjVmhhcNlDmGg7uyLZhEDXO8xXY964mSeR3LyBWnmC+MFWiDpFx3IcfVimkZo6pEQ89T3pjkouOBtNkFhaQ3RaVtc"
    "yFAMvQZf7dKV7MXp7TU0L7GD5f1mmpwPJhAjMCIDqo/mJKYmmBpnaiRP+NKTIje+9QYCqVb1M99f8fOyUPiFGyNbefOsh7gj"
    "Jw2fjlsN952NEqSJV4A3UEpiFookWuBBsiBNitdgG3jSXdohTpGCNApvlorub7yjU1fV3CHFE/OAff4gk0vvLu2MTQYX+oFO"
    "JWnHSW6miy7cWtJOTRdd2UXXdNF1oZXzigYFeoPOThp3OHIyG6Y0GxBjMfp1XFGzDKbTI4yzYAhYjGD3rESPD6tMmpiMekJX"
    "WkjCeEjzPekCVJ1j0nkiXYg6lIjXqkNlaxIQ9SOssxN4oCwUkvILkFR2aQngfIDBcxYsEE4qMdILb7GbC0mx0ACz8DldAXbH"
    "OUkXnqy3b7A1qtOkthsSpbC2e49p5N5wMH840NmCFXP/lNzBQeZcSI2sMVc7e/NfoIT4kTyCT/R1JGysgwC822vib4XDXIe8"
    "5c6FXqkuCPqVpFEynFEeUcMus5KUuyv/EjS6y8aVxEL2gZ9XqCTTxuw8dm48x2Cq8bFzGISJVZ3H/PWHynvvTAKUaKzWGM+U"
    "gJbfmmlBY/o556KFVfTbCxoYaTHhU2INyIA1Acc6avsZ0wDQwD19qKdvXEcb/Rkxmg9pw8XvQdhezx2Dm4TGKo9BCoWshD4d"
    "mNCnyim/iR6XxtoAdEq8T9CqUVHfMfkQfpHD+3kt0EHt7sPgXWka2q2+yCteynm9m7JBDZ7mGI/e4lQp9H8891Sy8aZmJ9Wb"
    "kXlqWR1EAMWa5T6KQ1aaXfZiJtJYemFn6IZX5HqA0hebird52X3UUcw2oLTl33keRgjCXuAYO17CqriAK073S0HPuDELq/8r"
    "9GrEqEzA5YuLY7vE2XElUBerY8MjajnHlnPMDKGb8ofK6vOgmlCimAgWKvMHdZSasQcQzJy9dKiIdlZTfoGKiQV6NhcTwC7E"
    "3+RIyXEcuJmwHINUgHZL/QLd+pWqq7gfE/8CG3e7t0qDPCjruv2F8TS+/jDo7ZeQ7/UHwisPhZey80GT6oGNB4bgr1LeLIeK"
    "nyG8Y9MLrd0U8yQIHUQOlJkkACTpoeYSjgvQEJquEFTGHjyMbNR2/H71eBulG45hhO0MUbqW9Rvz/xC1N/+b9V93T+gnJmEf"
    "mv6hFG5ktrRk94fnM/m81uhfa9mHplUhlSH1fMX9CpyymbBW4JItldGeLrVLyaM9/OWSOtUstzQKiY1lKQb6uPsT4Nphj9U/"
    "Rb0ZrBR/NdenVIOjOySosZN4gOmzESwUAwOrpodX3HAPdl1x6SVuUNC4Wbrtehf3oMb/w43cdOQ2yUe47TygJrbRPB9dB5iF"
    "CjcB7gtH5A+srpW2EvQEfVi4yUn8Bdm136lEXhABZAUeHXsjvY1f1G3Y/HlVrIt0fOxwLSgQv51ml8DfE5gO6k14QHfzMe0H"
    "Lnfsxq0H1aTzjJEjrLHxRNDtItHfZRl6OS8NmMu7Trn8HZTnq3Wl/B2rmqhL6HAr3Ww/Wzfbz+xkawZjmj2XAtNfQNdYr+ZP"
    "mpKq4BqDItcXUgOvwSy0AqjB5c56YJyDl31pONS3W0Aa4GvCT6sVZC4OZNA9IU60oOXP/7Vl114mvmqL1sLMQVh4G69qTYua"
    "CAHpsio4oAS7pFdY4AyNa9EkueXVZhRzBM/KyXwNcIDUn5mdrgsTdAY9XUCPNujOX5voCZVP6WI9T6dPQ/CUR8Bq/WwS5c07"
    "+IF+kFWcbeN5yjqliaatyROJHPpHCQx68uRQoKUjYOBJ5gc4952o3WyXn0cz3/wkwNgCekwb9HBweamuhv3zQ+8tArnSr96/"
    "ubwYDi/O1eVwcK1Gt6dn/Zsrv62DDpQNfx/8cnOlPg4vLvzajEOaj8knXja5vhhe6TeNbj/0P6qz4eDXG78pQE0gxEXlZdeD"
    "T7rhqW4zGqmr09sy23yYAS0bXqRJj/9lYrGVcSOyLU2kPg0bh2eA0zXK4dvzgdJeR/Eoj/QJ/n2rcdg5FPwGP5n7CXQeNmo2"
    "hqtijqJQ3uTcY2RwRjJ0xaAAUHrMoRE/TxouD/ERsYE36PCPkYl+O5Dgqw073LBf2xDXPOSl/2OlWcvhsZw4LasiFq4WIS78"
    "CfxtU1mlR26hObtfXRfsqqsXw68LVsAddYHzl8dRGcWzd4kDzod67VvYmQZxA0EWXUmJU9aeqLKTN+850BpOVa8z/YTzzR57"
    "3gT3ScTo/nU4S1vyx0YPhj1glgyUvEfnrZKyCSK46jQoxMuM8Eb799uxiRfxy0PzHNiYeQGiWimLFxpKmCsByKY7+xxy6k6c"
    "3pP9BfdxR+yIduyTeEvSND33KVkMR9/zQcrR4xB7fAldK1ZoAcUQFoW8TcunIjbYq8JfTcXJLDSfsBkF1Y1eX5NBnWqqo9ut"
    "QnsSNSHT0u4GYrCIC1VTBUFEQ2MuSP9OBMsy6jeFAh8sjl6vlzRzri1Rl0JOrcivRc2iNQINEf+WF7GtAaxBVJZtCaiG45U6"
    "GOlYMwXENFLonq/gMoxvzehVDu/INVxNpwophQfpZGTNVpSdWiiEZ9cYDzqksUxFcYzg7XRiGNXNBGfy1p19b9uayPPmpI0s"
    "MwhKd0y8Thqx+8giUFEis5Az9tr0wG9NHmKXftjsqdIYcYqmFwFBrxdX3nZsc2USasra7dWyRS9Xq6Xcvl4VxKdvSYcUGlt9"
    "lzZDMfVr00dUq1C/Dia/imfvXvrs/mVew0u6g+mUPwLsAEXRSUy2QAzbFdEJOmqHwmbqb3kHJBxamXtXDQXAZIooKzTCdGXn"
    "sCSq7p8UuiQrdztgcfpEytlvTWnpVHP8bgYeYQhhJnneZnav6ChSi14IP8/w13VPLOc82WJDMcgoyxXladS9gYtSzJxhk8Di"
    "jfCRYAyc1FHR4xhfRkVDKnFDtf/g2angjoTV5VHqX514Aramd7b6VYGCYMSPaQzCMkDu4iVB3w5ajClg30+wvMa8p89nO0U4"
    "SIcRxbSe4COwIi/jZKrpwzYh0WRHI3pY14yvpbXtMCfEAj69/ltpi04hKKI+pkvIeX4LScJONdE8ZSlcWei7R1CgGT3DV+M6"
    "jVIq0PsL/8PtdQkrDnEKiWaHQF0tNsePis1UC2GtxvnFx9P+B3Xb/+3iw4gG8eliOOoPbvR1udfp/XDU7R71OofwSI/uOQBd"
    "3r+GHpYv"
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
