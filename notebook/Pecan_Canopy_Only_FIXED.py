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
    "cECocUqNVQu4DVXWt8plgDQ/9z6b5cvM8EAczfPdX5oe73KJvWiVapIQCou8xCTcLEig1vXvL4uEbXJHGShG1G6XQJrmuQHb"
    "5zrFiMCOacGl6SYADDg+TIrrrETF9qk9IosUmh51tEMCZ0aSik94R5lA6jQfk/PFsQhb8uWswARHVsDADlrDVsXyOPoHLgjD"
    "q6N3LPcJjfrm2+j/+mTzq6S+Vy0nn/9NpbKntG7/4LxIjiroxj2+JZmbXjsmpZ/CGqzhI1fFXf2okiIjuKfoUUcubvDjieZJ"
    "WO67MOhR1WzBLppA2USZBQfSGXXEuJaUJlCXUt2rtGaIZpPYzC4tmVpaV+9tGRSFc1QrQlbdO5WMi01OVY+cnF4s0nKyxDU5"
    "pLvfZBy7bKn21Qvt21pnXST0mLSLFCWc74fcxrcqwZaZtq9MAwuVGZNmjyL51izAbWyFLbW/p5eoAQvchbPg+RKqkyhCOTuN"
    "ohVy2KchDDGx9jeBUpwy6Sv0LqJPnei7Hpn4dC1l/tfOzbLgo+Wc9kxlX6b5rFFs0D4RBaQSnQjwhZbr5BmP92ciAok1eZ9t"
    "1IyzGac+wK5p++ZKxSZAx6DrjheRDP6d5eVil4Jr5+yr6jxZFvplAp+eEj8XyEIqoshpPk9xdhSg9gSy2AXD+n1hRynKttdF"
    "8o6R5AzW0nP6yg3i1YZSO/6IUdhieHmnBI9cWSWQtGohiTGDU9LgVxQlQZbDKq0n59FiTfMqfMkxyYW/69Xn+2atyooiTEeL"
    "rpOQQ0XtndCR91nFajZd8X2911mrCpNhoA7jT8S63rUxpf2hLizyYfl+e6Ay3UNrwcAZ9g4xmVUBnmXODAR5JlNlz0Ga4kXN"
    "cdXcU33TKerQEdfQC/2ZaUZo1QuRF5iJ1TNX8X6nVewglEFqrIUWsMlNGl67g8bavZVWG45O98p14J/VKu7grbRIEVVBFO7R"
    "44h6ivRdoY/pqyVBQTqLepYOE8bfFdt0rBkooMrV77xBfi+Bkc/KY83udDSkMltA+env460quxgcoxKL6Ok+DwXvI4ggVCPd"
    "/L7iqftdqh5b06ngfu4Yozh173hJ6QsnKZnTp1NgsOoyXyD51T4f1vUAs1AsazwI0kr8rxiMFqAeU87M6PnP38NqryrJt1yf"
    "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"
    "i5QHPPmdXrvjbnbDvwpF4Sge2kNLFtRfLS9cpDmaxaTVdjmx7CxQkADXfpImksi+ottOh4032CiP1ST+FUhCXjDAuM42Yu+g"
    "a/K7kyi9SfOaNtK7sljkVQbcLdvfZ/Wv+SIr1nVcdkip7GqYPkVsvk2zfkzGT2J+qWy36W5GfT/5OuNtztVBJmJ11VVpchpE"
    "YziAA/Gwh0psp6CbX2xZa0AqbcmP8PL77IwOx/6h8bLh6zybsalbu7qOowTI2MKzNaUJjAwXhnUVYRbgrQI2nqN66wEGv4PD"
    "YkwCUmRq1pPznHM8kN0nm2zybODkxd/CGp2qEiwk+v/+n//1rUXcyChst+ku8PE/4SAcNny3vZm2Kz5tsH5PjY3BiW+XZlqi"
    "i4VtTKs+AZppNQBNx98NN/t5Wn+tv6ZPR/+0WX5E3REG6D2aUh7QbUc/vaAnJmPvkYgOjgZcwjy/kOfEd/NxdVPtTljYh6B3"
    "K1kK3+rddEvgAIKivII2I9Hh2C9j5h+1J7+n5tkZQsEBRMEAQi2oKAE3rNlpgR+YcWuZOpbwriOZ3VQ+8j33xDXS3nQvybaP"
    "dRRAINc2D8h0ytqEPBevcXdlKZatxKUl4IR/0mzrbZaWaPnKaYdWGdqLi2sfmo1XbDdOlnh5JQSm0TqcTVk2hbVT3CzJ4aA+"
    "h36cF/NpRQvpPC2vs6qOX7f0yraym5LEtkYX0/+YpliFYIf6bhMnp+fxsgBmOa1IL5TP0T+VLYCTZt+ry3z1dcbGWJMnn4et"
    "/AN6tDdfdz9Kqj+Ni+6YwtJ/5M9176jHVawS9X9hYult00wZEyutymyaT+qXNMiYUKURip0BVly6/U670Lxrt2Umixcml18x"
    "emXz6y79d+JMi14SzhjgEPoIlOHMivyVl9qJNM/QTgRF1JxZW5y6BtHi//xv2gfz4kZeKmgY6kIMRMkZrGIxMBuqwl4rbsg0"
    "Gl8uuhhIY0vw0S0DOBLknwCuAsZEizyab1SWri1r7SLigfg4NRGFQ2GAqevqfN/iYH+lCMLIKtDJZ+4Wk3mBRsKWI0TQx0lm"
    "58tKK1N2r3S4sGfAbtfZ6TIs1Y3xeUvNP+WwaZZYM1f2JZLE16+e//qf7199SKLXbKHgh44CSsi75afB7f7mYnluOTq7yuFj"
    "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"
    "2WjWm384bzTOm/Wv8cqM7nMCSej/BSYer08="
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
