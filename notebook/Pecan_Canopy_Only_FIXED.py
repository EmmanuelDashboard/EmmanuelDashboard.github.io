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
    "mTKiZk79HCxkducJWvMX/bHHrDXVgDFQG3sPaMMtV2VMrX4aBR0WXc8ca8OfQTs6WtylnmiEc4oTw2nKuzI5a6m+86ZLV2fh"
    "C39hVZuc4XqceSu9xN8jXHflGpaXX6Nlh6ug8NC3CIs0XFyHWrGCYJO1RavaldHoZwpWwRlV7XvLkgU8OXSjPmIF0BfwFcQu"
    "6KcQr8F3IHq9S/Suuhmo7jaXALjbDlRv+CfvtSU+wkAtJ9VsUmt2818DWBFTp35Cr0FMpihP1ySEpbN8m6nf2Pip8fKMxm65"
    "xk/6GoqcoPfBroz7njllfxS1/lAmiYd09DT+okDlMzpSu3Ken+eLH9IlBbU9NI5+Tt684vqvig7pCIGhFVXiG/0zXzyZzbJb"
    "vjBnA7KGxGV6q9EOze7EAdSncMQEP2w0yQ8GlFysiG3AP/vW+ksMqNjby3xM06RKuXVJNiSsGFnlMLi7nGPMJxrDGWQoKC1D"
    "0N5rHgOBzDKfbTWmWaa2LWxWdHKwm6CbkRA3keyQ0zM2HgmSeTSGnbTP15RMaaOqZ6uOUfEXdH1qMuEVbWCsbbCozpLKPsx6"
    "8lv1JxErcQdVclU3UiaatmD7hFYgWxQco5O0Dk3rJ26TivZWzE5Hdngi6RAk16HvbkHKBbBdtYtgQyOGR+oGIgPPbrkulkqa"
    "vlXS8NpkuKK9ooE2rrwQRrOMN+Kcr0aCBqxvUeQ3OIzSM9dLBRbnNnDrvJJHnNjLXZ9oMdmx1cged9rfICDqm7N4dBuZiQm3"
    "FYOEF++HxoowYaY6/G0cvrw1fBeoesKfnfs1ekpymbRZNVc70nkQge4sA3xdH9nuwAiNJVYQ16vYIaYETHkaoj54Hsmg8V9t"
    "KTUvGMDlBo5qjS67uXnRUzWmb8FaAT1VIxIOBu+ls8dSI/d40Lu9Xdi7vbP9+LjdO+PjRtH/uFrkdolnQtfplkrBQyrCgNDC"
    "sbBZc5viL46hzVrd2N/mYa2dhj9F2/ByWmtJ0A3yuNdt21+QN+51LJJR0lLZPeGxYjONQYV7Kp4JfrlZV/iQM9eKWEXDU8YX"
    "834C5lQuPMzrjR1n0PQaXEiRMa84tevUjiDXwiMYhs2pPuwdW7QEm1RS7V2BtAd4ZrQP05ZAIOpdyNcpiDYAnw1Vn3GrIYUm"
    "4phIQll9hiF3pr9t1Hg7iyjSRmX+qUzwvUA+y67iLSKodonXL3ttf9x0PSzFgKljd9domC2WTZLK/hSIH0Y24VMNzktYfzjD"
    "yGFoYDPNRL6uBafw+5wpHglu2ck1moZKVyICF/iq3ymjdQ2/RprXrnzDGwZXt+VMIVOFoQeZTe/rj0WcC7PmHrJUqmrbpY/X"
    "1vMA23pnyQ5Pb5zUqthfNjdWLudJHAopXaASl5Wg6mXKrGwPyzISQ2zS4tBBhmWi5QbOLaedCLMvCDtRRPeoEoQQSDz3BlWc"
    "OW926zzqmkd1zrcbjHYhqUnVtL5Uf/jElt3oSzqvZUom3W5N4S4X7vqFu6Zw/GvLpdYPc9z0Brz13Tv/5nYbVdMYbrihdboU"
    "l6W+n+xZNfFY1agfkoMPDjeBNugdjoM0V6dL2DffnZ3siShYckoi7zBWgbr7JLA/X15471CxB/XXO1QvbID+lkGewTsA/OnF"
    "G1VuCdcm51Ja06KGG3IqzzZEuEcgxoQ+GMQ6at9AOnBjNchhW5a9d+I2fdhheHZPc7/oXE10EjN5nrzzXNN170PgCbvFY+l/"
    "EcpKfsY+UrndkHu+HBVbQYoo5v2caPOLQDlLaKS1EJXVE/ytxgrNYhzBnQOrzNAH/YicgaIEH2OYZuELG3xI/AuUADQRLLQh"
    "2MOfMltCT13mzVzGemNMXrZ77rxjfr8iZ2HqjFbhU5QJdxY27iBu3FWssN3aLVn0WwbBtkVW0LpaDpUQdtb3ITLsjtgQNlnL"
    "DdVGWMGh0s3aTKsR2aHShseaSKrSw6fJA4eI6f/a3/+1v/8/tr+bBXYgMq/9V47RxTqd1YwG18V31Dg8T4eXzWamtZSa8BEY"
    "yKBKsuQaU3U6Cu+Rg7Dc7KhrNQK6BCuws1k+yS9mGd9FX01PKeNNeaqbOOXx5a5IITElQDP/hHBBAtg/6qaU3aft17VA3f/I"
    "aPsj7+qXgNiE+qPvd90G5otxtloRK442UOb6Du2vacN7Ce9VU3e6hr8+ip2xKCLmRN08PhPplpm7Io8pAYq0Wo2L4kYSMUEt"
    "qKSvdAzM43m2msj9cqlErFIMYzIhqmgvBtnIZAcnmRZqhmc87rdhmGsC8Qb0N2iYYo0PrQaHL55qvV2rYX8dR36eeaRRbubf"
    "/21fS1Do77sLBVMg+HeIPLqMJSTsUC5C5Pvywu9EgoH6inlINXMfqXJn1d2OhvR0dgY3U+qU+g2R4YYv/y6Xhr+L+f5CfUsN"
    "9OnM+wABelzAYMB/GlQPntzyky4/6eKJoHcQ2MfFaY+f9Lx36Emfn/T5HYfjaFoqJXC6B5st2bMZ7lo9ImBvJYXfkiIMeYlh"
    "naFsIVCuUD6tUlfWdM3irPLNVxGNQHYzzJYww7Zr17+7dtAVHuU1+SZYbXmvTS5v9/T6Cde6xMHJ+WWej0YzPg7AIi2RTWy2"
    "wF1yBtisocBMWHk0ytNJAXRy3DPdXG3QNbX1Ti4B1d34q7GrS10ICqaqjKZKvpWqgV6sdT3lZbrMFJVr+z9MK3nZ2JTZyF2z"
    "qtnYdpmJkx5VCXVrj+Vb8l27OyaRrKZPEyo19SmRAIEM4X5l6FkgggwN2yfB+7kvjAxvPPhmuWkjU+8G3SzL0cNohm/JrNod"
    "uT7lQNtIghoKFo7kp8GepefkOPifIQGefddZ4Nkqvc7FUPb81+c/qRF6/eyZfcF1FBDuS6ifJoK6KRKaOg/3msdbMiNvae3f"
    "b3S/6SQ2VhuGsK/VjX/DfbfGSbr8BTJyc5RNxEpv4bCMiKceAjJa7V66Cj1YJlx50sJ9VWpiLfEigi8ZP0EJfqfs4HDUuCDB"
    "OWWZFtwI4aHqiMY/uNSxW+rYLXWsfp3pgx3CAchDEj1p0gg1lxgEGJvcm+/opjeep9tsuC5WLZqQZ/nqjHHy0VxzTNAx99ny"
    "R3D3bNHkl7yKlvl6eOmOWMpC+gW3PWUZnfnyV1yZ+/ooh7lJWsMuIFRGTSff6ppb0irCRilWcxL1To9HFMzfQkXqdSmDhcJl"
    "UOW3cGJ1XjFP1HjK765+U9HxT5v5RbYyrjVeqnEC0W8+63RVo551epQNUeP6k7JFiYaLTI+Qqsyyyz+UlidSxaTy1XydJRDR"
    "zCtWrK4Ail4sXHZ2kZbZy8WID2SFcDR2M2su1XF0yUMvUyoGL54Afa8DsxIp5wNpc60TJehvZ3RChs6MtBjCKR4jlA8mKig0"
    "SKcyL7b4XSMDktj2Vn28JP2H1DIwPVQbTrGYEFI31aY3IUOYLUnfUBSrUY4XflTfK4mRUbsSXyblwVG1t6WVn3GQplAmsLuz"
    "9gAyM2L8nc0GM6U2ohbjaMOba5JUVjD8DN953yrG4zIjRwaujX1xuOmWso/0/a5zv2tJGYT6CLtptpoXtNqEAMWobDZZWR3L"
    "Qs0gHBW8TTRLF6/H8KPCAMg2KhyzgnfccEZEH7OmdtNlmx6rYqfhuIC0IO9x35s4ag0pdx75OAnbZcRgc1gi2WeVUZxD1y5+"
    "aqlO6fYIejJ/v59PdXeccNipCYedSiisUw0IXb0jg+G+VpjXiupreofVbax8s9XIbczyFLngptjEmvKh0xz0OS8AV5ucVI68"
    "ZT5ZUCQ9av9OrTDG2j72TmrjfMWYCyjMhdCMU3eQ4DWiXqT7Pv9fEDt87eDgNrnC4wa1lGo98s+Ghs4qELj7Dq+u8KprYYGU"
    "1jZ+mmX12LStqR8mDDSu6R+JBhSzVYx50NiIcw50rEhZnSquXJJ7ZOlSu7r+c7o+yCORoKO5Fmwr/CZtRvLT24+4pN4cafHK"
    "ra7ZL2mncqo076idyhT2ZoeUqewIeuge7LjJCRHNiiXLVmYXvm+7AB8dsxWb+12qynvDdtB/wzTcY3B/Ib9M5m3VLy6Dj7lv"
    "/lp50/3yMvjoiacg+tvfxTuNxcjTTdhTtzZ2VDJt8B8H80DCNnueGTcQn+UE0Sg3TkxJTQSKSfySBHK6iS8x4SU2ukSCSx4r"
    "HgAHEjkD6COAFtvdphOQ/+/XdMT6kH3f/EJ3rvnWt41Lm1vAtpr2i26k7WARr8ev0MsaXhPTdOw6LLv7kVO7Pj8/crmS2Wf/"
    "rGS2UZYt4aCENAUJcltjL2WwrHxhT7Ppolje2oM0aQGUOKRrIrdBojB9PmJvdi5ILi+l9rS3yI8eB8ONX4qZlufkMluli2HW"
    "ePyYDC3eGGbL9eVLbCPwBOwKKMc4QgfzWBISM8NmeHMeXhxPF+oP6/AqaqL87ERViP1N3aJtCcYbZ8RnzOo7nsrdL4AzWY9l"
    "VJNlwgiZQnHZaN2c91pInKV9r8ypVRs41tnCkbX8TpMVpglNAmiiRa6Qv5pr3XkCKYeAsAXOAP5WhqV0Umz1TtTlt42eNocR"
    "Bz9h8ceUscN2ioV5Rn55f/s7sS9gRiROvuEna1m8GLI1x5cKsg43Cy/YsfmNGnV0cKN4jPBCZcI0VkNkuWdKuHTmiRZyBWIM"
    "Z/IuJE4rNK6BaIkkB83+Q/XNNX4df/NA//wjTu+9jvq5ThI+xH/hgeDQanxRkJs0yxwsCG1+2GSlgZ3x6UaSmpHRyB1QwY5r"
    "6Fzkntc6IV9lnF+b7KO4wFbJX6rQ1MY80oBq4hYsNwU1DUOfHf9JHIFhDHWNfaiIuYyH1+rRyDG6Q/TBSbtcsog5TtURRzXF"
    "guYImljC+LxdvEHPBbXeNRiFoQJTslSeOmo17rB2Yr3f6JLk3tVERLY/+KfTOCZnfprPezRHardV9bKGATdOOVdCU26CLenb"
    "qvm42zVFu2cEUCVzrZ6d2Kk+CWxcZmVQCbMseE/bLIbsKpZub4U21VElw9+Sjt8tQ8AWrTM8DgHtd+k7tzvGQnJNkCo1ZFt4"
    "ghWOuDyD7kku3kHp5NCGUIYs7wZSyKzJQySSOS+lceF2nfIf96jiTweVJsgl6R9ZYNeJpCaCD61uPVni53zicgvPE4ZfLMin"
    "Hq8oDuVuPJblqwZQgW1ioQDN6vbXgvdtNgb6GUDwnDz83YLO8XBuzoayBc3PEg3T5L5SBYbnJh5U9a8fUzXtKobfyq2WGkAk"
    "m5Lxc5iz3MLMu26hbLCT+RWDsnx40KDgIzj3uLvvwNl88Tnabq0l+YPDSDlcQ2jInMWqoc/yeabsboRkl6sMIzlVR+YaKtTm"
    "BhbbPP19c9LG5os6QDJfecRyXHFdRcPfsdnyHWhR6lRXISqNDpV9pAqqo5j+eBitbwM4uAnIvUTIXW6zWrST0MXGFtjoAr67"
    "0LR0PK80c+FDFfj8h6jf77QSxv7BZ0ESyfXA3y0Dgwlnp2WN+pQmNTar5CI+3TWjONHoyk71j3oGM6WxGp4OvSCbrVr/qtkJ"
    "vAvRtWkiBmhTId2mZExBn6cRZksBHRSKL6+7zSbTj9FoQZCqksKcfUFqVFrtrazuYWRJ03jU8fbmkijhnSID0AWrC+jwprVN"
    "SWKsazl0uDN+j2BgprRo0TYySkCGhKoLP+9D94VQ3NnMGdhEUdBOEgLcRA0VXWSTfOEyJvMDWUp4nVNL2eQzUTwLoMlESlid"
    "6vPthz7Nsd4uSDMlaU8iFSr+xhX6rAKVH1Ur35Ae0lOqTinGYxt7sK2etSyrUK+YlCHEK7ZTtQO7TacbLbrP+j5Oi2UqVcfP"
    "J3wAnRRZaey066r9Vyy+7Lw64HswsiK8zVamyAkxxiYkjuLHOTm2OLA66am1jUTswE01jamtCq/Cq9WJfyaQzbykBg/cI7ai"
    "zgVRGoFtCbimF0dO9mAy1q/hYD9apeocnSKGfJUSCKdts9FkIsD7slCLm5KIOumuZzD/N3O1YFM4BXLuQpK2pupgrc39eMuL"
    "s6azuf7gK9jIrU2DVCDTuP7aGAjQuxfpnEI72bHwkw7RKGUKaLlaPaii8xWrESnRlKi8Jc9XjHwlJNQciFGOZIFNPMFe9QT6"
    "SIctHnjyxOkNh7COnD3vcZP8M2jin4f5pZM4KtznaNLRXZok4+Mfh2uPxFFARxDYq0InOlQXf84D52XFTaxJRvP26pThQ+EZ"
    "bKqlwbjHd3Ht4jybwsy8+KzOp//kpAoSH2MyhiEMYFwl0xUSA8zVslk1ug8VXXKhsDbNRCh3c2LNrMxUvNLOMYRIurmx3FH9"
    "kTWlqqPjwSzufClnfLPJsKmSPvYUXIe0bV+3tPtxU83zMUTk0JeXKJLY9bGcrpki6BYnjvacIVnMeP8RHbqn+UZdl4YUXMjZ"
    "KUx7I801u4puNzt3i7XEjIg0Qr1yhKqhXXEOBroAO52Er+vKiX8cy+s1DuJM/pL/EoCOLKXc5+4jOyYtCS5A1QUF7MqqHEbL"
    "HCK9z7Hf5pN5ivml4Fdfv7OgFIt2O/YULSt+xMDHehXqMTRtqrRhceG96DmsmCyKJ5aFXbEO9go62Cl+uFrYEYVzT0+v8M3t"
    "ie/xPNqSP3OfTXuJk/rw9EpmJrtZNo95wlRx+GegIOXaoryjmm1RoI03oFZHLDr4fG2S3umhfUkSPoa2R+xzHT/4LFYUk03j"
    "cW7YmWuRvi4lY6+DBau6cH1FKLBqRDHx2uh8fYVZVENy0pD4hyunJ7A6X5cnYVApuJ404UY3wVA2UxzRr6FnJlO6d5PEpVxP"
    "tOuwaEdTSELdRoS6Fa8aj4hDkVgxxJ+hGQEHZJMteBIpS9h0YHAKBDNeCyujfJWxVskFk/HEIVVnNhtrz9lUxB746I08cwe+"
    "k5AnZO5g3KhSJctlOQliEO/Y96AVkb3aX/jB0gQLiNONxn9z1kLZcZlLF5ipOs0rB1/bGVVEXuII0yxB6nQQAdMrOzAiQ5+q"
    "ntL+/8ERkJXUT+128I6QuUv2rHGezUZwPaJjA07jMqK8tSV+R4rn6lXyf3HlKTX4wc5MCaF36GZVB0Vz3+l0REQhX729allI"
    "HPDAum8IiYQWe7eijb0n1kKRYESAcSN14GgQDfzlcR8xrze06p+H1Cw9DY4s7A+Hc13LHTJsDR3e3sg87ggXpqrnblWgzGMU"
    "t5I1J2cIKt0EjyNvVL+FReRKW3APJmhRTcMDu0aGm9U28wQfvP6HUgNrLYrG22MsJOeQsXYQniiFRv1BJQTGQpIhpyY4t+u3"
    "4XdWrWCpeQZGjOrwF+Bi6u5uUCoPRexcp1oOFNvgtHpgdfNpTxl4mXJqq1p4l8EcTNnYegx2NxVPA+fulu7uqbuSq5sEWdLK"
    "k8mcf3IbsAuoxnOie1oK8Qy1Do9R7T2dCj2UXX0j0BuVsCmpp4gP8r7RiddvnuNNxd27CXGgUDme49TXNymgtJoDu7fiXo4F"
    "owSlxkA3FI0BsLO8VBT5RTTwiZsCrBL1ucS2jFMj5ZOFKXEfJeLqGCZ14GGxQmgbcDbRTbK4xNJqV37TXLcaf8YlnAvAGneZ"
    "+5pbKoQB+3NSEZ8IdxW2vrWqBVGda7B6JwpYuBPtAKoA4+yse2yzvG8k8SY96+lnazz7M67UWHCBY6nhSGrQbzPt6hr4vnmT"
    "HtawxMWF2+u+mxaJuOVXHDmZRO3NHd4YFhfqrycBskPvU0hwGF3IN4sLqO4VKxJpbyO8V28NI2G26tbITS8fviLFzuLKKiPR"
    "8NoPtNpCNihItNOsAiOX2cS4qm6LfPSaLZdGd/KYncMGyJ+2zoxJzD8NbSyFOcQ2dL/GWW8mFBxnekg5R+DnXc8d6qm50hO4"
    "xdZOblPIPqnMcDDHXdFQUHV6nqMGANrZnYpF7OEXPVASM6PDd03S6p9FZFV/rdMOWEunz00v1OC/4rNtZad9xaPlb7dRutZ9"
    "XmQRwubvSedQwqXQV47UPKJvHvnkDJoiITPwXEIl2gFTe+u8gIIBoG+k/AScI4sCIhoilw9HwyxTishPGddbQNvAgXVN1i+o"
    "4vPTHN0u0nk+bCxXxWSVzueUvW8r7uJMROyU9oUjqbObIwTx6/S2JC/tUtWIQ4BoX7UATk7sJcUK05ubcpPOTKANyyzXqeLj"
    "PlxMXmrhVn0gadOqQ5RbgeinFeUwohXY9u3UwaoUoXfTxWJp8aJ/sQjt03RgBfko8XgjPwMuGRjdnMUz2noBNAvFfR5AALB2"
    "fpTddLDXgXmP4PW3MEEybny1KO1febbPDbU/IRXIHx0lxlwX4uVsdRpS/sgrjya8Kqi9BLLW9b85Tue9RlxD22IPLuGD2p3r"
    "FJ3CW0HMkSIIDpLlkSaHBdmX6ThgR+M+uds0nY3VjGri2gzonEfCLm/3cjQk7Jp8LS5rGqrCpWzX5wVQrFkl45vnUtYVlzLO"
    "k8QvaPeyLw5hdmRsUqcI/xBEA8JMrprV5jM4Port7iC/RzuRrvOjdlL5Trv2Jc54ufjWH9zgNSmyg46JG0QiRJsLLfbME8Q1"
    "zJo6DhURrinlVhJXy+7X1Vf8j1x1gpVJmndL6q3GVTcs0fVK2PrK9elV58w9CUcI5JF1OrwDVYQzv3nOrmaaOiJ7r6Nim2sF"
    "2xexzIPgF0eNK2ItJwd4SkXM95+HEA07JW/Pg7yIO0qQ2k2Vx5oqGU5xnqkjY+nAN6t+zcQ2hoKVxtCmgjC8lzo4oYs5Inc8"
    "3BtwCgJ2tPF9szBHvj+NdZ+zUdTpahImIvdOjO6J4grSfUJHSIfDzznN4xWalVTOlYFXAai0mfOpUi0HDvLX+u1m80rHTYDN"
    "Q2UawVcUJzHHQWx4It2YntTirdOXc/nmFSiKXj6iET7xdOXDqV9Q1V0v3hE34aC4dl6+wKhmTfrYQtdB3mz1+2UZ4zILz1PF"
    "W8iLFi2uK1rQ37Ge+PgYw7Ep2WHYW1JUOOiUHaZNycHgG1WL+r2g311n/5qRLsTYbRvQFEFheqOYp9ePEju62wcO8d+Uyc79"
    "qsKNKkFTX+ylyzxKlws8cGmSdX+ql9MzHTtl6i7nPHQcQFVVFadqna8hb5KJXu/eyHipLgmbs4zIpvmC4egcZZWIsurl62Iz"
    "G0FgZFGTL6n6OSLijfp5o85So1t/sCHgapGmuRlHVKJFzdjtHPuPkRA242Yuzk2/IWv+DaSEIu4Sq7EwsXq2NVqH4ntR3Zh5"
    "aPIgYB70pggs0p9xsvGLsc8RyM0TvLnod8LJ5QNqr5Ffx1IXo7LWc5MQ/XSnCGBgRk+pPa3aOT8LA1XUm95pT6zst8/ydFLN"
    "s3PH0BLMoqCnG9pkoY3qJ5OzE2cCPTs/QFyk84CB46UCKTLxiyS+3+kHP+6hpic+KplUbGHJQt9gBhGCNy/FbbYkoqJJ9lp9"
    "knxNC/1iVQDrlpb9Ui1u3dKWNdazqYpS4JaOtYrzPujaJEju+lJtRhrFZVFwuobEtUCRScYNj9ls2dtmV89PA0dqzvfcbrdl"
    "KJw8R2EkLyvIOOI+DKPSYT2P3JgCfZRzQwC0a3Is1wC5/W/jSyBdr7P5UmhQX3wLG5Zc+MyQqrmb43goJ1/m63hCBHc0IdTS"
    "EG62u8JHWK0WjQmwkQAJPknAWDMvJUBkyLQbig7NZk+Ue3if/MPZJSU4qFkfcgBYBQYuI1MwDRG5G1XG1ltixsdrlF1sJlQ8"
    "aQQ3ml+u0usvW0RZUqVEX50uLlAjZ8CSC3lCg8k+NWbgkjPW8+M0rmQ+UhCpbdasPaZ4zgBCkzqwJiWOU5P1pDVHZm8u6fxu"
    "fG2AR75CvgBWVOUrCcEonVwE4O9uhpKS0xIYsJdrs4hZIKAvAzpJMf9JviV9En+fqluCBi+Q8P06g6SmzhF5WayLRT40qDLZ"
    "RJ05yrxQbD81Fu0FR9BpsBnOVsOGb3ympO/MEN3hsoe3vzz/2SibJunyL8Ybpho3m86XsxgnIQnuR3JEtceQ6Y8poAmOfaz2"
    "GNPZ6MUiVONzkWnH0ytttoKa9xU1Hb6nXV8iUSUU8TgaVae0uz+r975VtVdhmuxJtiLET7GPTjuCOXW93lfGZWJX0YM8+b12"
    "RNKddiunLSu+TdWcoBN2/ardAHSCE7nbY6QjVzdOr0gLz0E4ihFdudKR5lSpWk4pHRFUeeyvG7X2thdyx0jeVFXL+0o3qSj4"
    "4Ru2vVBca5uqY61rrXJle1awpaxgu+DfiQSyG05NtsYpH2RS7aG/QelNWmHKAWvV78Y0XDvCr67XzkfrnEsfmSgsPrgPrOJS"
    "M0tZJh6zfK+qBaWqISSysX41sm7MYOG6BbI80SvIjB2uUUlEtGEmxVgd+vvANmjuxt4waBJvN3NLnT5tqvZo6lTfx0915Hpg"
    "SZSAOTlgxBVenYUu4wUYezSzSt2U+VdaPTGjFmxUsHKrTokt+zv87sYARe3qLdtMCngrMFSjwYnt+pFr59VRIDrmX5ovNzuu"
    "ZUj6bTlZ4JPkQVdY5Ap88ltS/VDQetfEVcO0skz3b1cMBMaTnpdqPyjXBaAJ4TLuGkV4RzEbSUJmkTmCf5vLopg10tFVCvfN"
    "xjYvZqmqA15RugKC9Jos/vFfm8b/+Z//C3vDP/5Ld1RxnuFlY77B7BYIqR5CJBWHH5mBxdqAWG3K7K/UhtI71+4htBgjTMNU"
    "EDvpbA99CV1994gJC0GLZFgkykr1SUr3uTzzBDA64uzDJpZTFFTTZX3DnU9wjrTYCT6nLYCpGZDX4IxmWBUple1rKDwGPgoq"
    "f1rHL5Wb+QCvIqplAm0WlgghrtDOf18HmKr6Ua7VWAyA3+id22XTkYpthA6Gj2+e+s/UhqSYkCL5r3Y8vwb+YPxpd8/b6vl1"
    "XD8JlLxHZggKymjG8IWP6ivz9ZNUnL5/hO1liZ8ncvda37vWdxb6zmJfhm46wI9uIiodTTJ0EqXWxajhlrpgBqZ9HVX3Og4I"
    "qsyC3A9a+K5rQnap75EivlP1/OzE557MCkA3t0gD6xHLST1YrWFsiu9QfjzF0lK4D0Iw/8d/dhvzIIOUyy/ZEvsIgI6rSLw5"
    "60Uq0pcqrBPPsELqwS68Bls6rrXkbbFeb3kV0VtiRK48vaX9CivWocSkUqLEZOpSpcgdN9Bjqn6KIpPmuqLKrOhvKoI2JV7U"
    "/M8BWwfLQL000E11QWC/7U0nqvYtERUpX9W83cX6lS2HYMg42IgrVtuSpEVE1BSnCZsg7WRG3rNqL1sU2ESGl0r2bzlVLXC2"
    "oROR2hEbk406pS7WWVY6e186SfPFod31LezZbJ26nd/wcXOj8ak2RPqbDrR6OyE/8KpHQsJTH+nX4XCBr50aiAd/DF1YNxuS"
    "5puRd7lLpA1s/fQSnXiNMMH5Ly/UABiNEl/ZsroQaigIy5SdHnLKwGJlgXKDvJxymszX4l5hvC2aXtBbYv1BUogmLMPYmeX0"
    "LqQmX3PCl0wcqUtDMyfOLEuv+XjeXJjjtFbXJJ4MRCRDWMczatt1vhCHTs6CSp1HWrqrDW0R6Jnr1+x2dWCMAzwmdIB3sV3n"
    "BfmepguDQiOqOFV4hKBC9Vef8TU+rnQEXiiB14l6d52lIwLfL0aT3OYSTSeTbOQA4LiCFlED4xkZOTQO0bHD9Pr5dPvaorfb"
    "1GoU3p7/g/aZdlw6mjEXkORspyUggqQCAcgdIQACpCs1xN4wYWkTvmOtzb1u0ccO7E2SKo/AUPjk3jLHkjwE2JBJ9GcwJlGa"
    "1oz8oL0Rg/J/bZMgjsLQvWos/SrLoDypfh0+j2wCQplevMwRl6kkGKda1UzTq7zPCEfkR4+5fuD80RQ0BlyZ+o9vnPAkYB+U"
    "gMEPtdZaIRl6oboBUgQSxdzS1PFPHYBT406JQ30vkcLu3NZL7LLbVMngIhsXZO6v2U99IiAp3aFQXlj154qDNjFugreN3XUj"
    "M1VUtjL/oFu3pcl2b03OiKsmxsbigBIApvliSkmjieMjknjvJPZ2T1GNGFAS/ogdMTMo0cYb8PNX2FWuVesGB+zJbbWJ87m9"
    "WHj7iSMOu9un2i5pM7KRCNiLmmYLXVFRgf/Gu2rDM1vHcoUtgzuEHN7Y0Lvthw04arLMhZfFIk0bGe8/dHjXe5PZnAsJeKfe"
    "YGLMJmu/PtK4qZfFbGSR24dZ+4tD5+A6JrpbAhQ3rAqQ3ccs5QexpUzGo8Xtj8Lrq054d1vmpSQ23tWlOy6+uAn/wC3lrpsK"
    "y0ZqtinnHjnvQpWkWHOvMR+Ix4MQfMv6MZAjbTSLb2xzMsehXrvT0jhhtFcluzcrd7t6dcB29ebg7UrGiSuGmyW9vjtNEevx"
    "+BVfgRemEVqv0tqdU07THB3WJ1cL0Gu77+vBSZ8M4pIThX2Lqw96U4lOq1M9zGOE7+WV6f125MxmR+lVfLB1+1xvT2cKiOf9"
    "u+9GaY7hxBBwnoBm/fHjRr/VsPdepABqpduJ619nmpMYU5HsXat0tElnfFZVFdF5SZv++o15lPxjagFpdEU5YIfKKgecsnW+"
    "d7C1GMqQqBbTGzjdMdSMHqoIGoEX/jSGN7gNhiIKOL3CcJjfIJe6cHRRPj5q9DoxDxyacZrT2HQDoUSMrry7+Ycu8cUiVISi"
    "1qvanIYWxovK6I0c96nDtEBkzA5WLdQz8KipsOCEHY+MIuakClYB74kR8+mKhHakRzg/C+RB6UqT3k7IkwgsU9+G8ALp+qD1"
    "R9E9ZsmZ6erGI3ucrdFfgXeTG6nhFVkxrj7SKUjky54jwofgnAzVXzlPZzOOBmk1LjZrrc9YiO6hQiKfIEoeLAbM46Ll593y"
    "927z2vUrIW+z8CFNDgsCVlM198SHDwfMl5mTBQJg0tWSRfhGk9et5PmEDptigDn7D0sSJohm77z0P0rE/8hp6YbTwhF70e3L"
    "7hsYQkaX4jhJGdMj/kEWeF66wCDduUE8OGhreKA3BdR/JZN3dcjkRc81anso1eksvZhlCNm+hu5qnCMs2nEQTi+w4v7xn0oY"
    "TFqWMZOCa1EYzd5ys7ayYSHKsfpT0kl42OPZoiYw9241Pk7x/yka/97vofDvHaTv791V3f9xq+hO3A2z06R/FP21Gt8k+M9H"
    "fDE6DzU8DJdR77f8mVnhLmbHzfKzLEbfmZtXqPVxn+LcYIGoee8y3JF0nFDmuswKNEXe1XIw37VrVhcyIVnhVP+jr7zczysP"
    "Fh58jnfmWEeIu3UeJk4L9JiRG+yMvdtIfrYhozAQAXtJkglVUEVcH7XZ4Q6su33J9jiaiKp4TUwANnVfXddqWLlNa9Bj0uBg"
    "j/Oi6xSWd3ih5NDD4svu3vOFZ3FXRYHLRoZ2A0uCV05zSivbapgLik/JO5Z0dRVdKN/zTqSKrltFV1fRPT4+2Y+uYxwNW447"
    "YulgxjBOjBMOBXVDrmW2cp2u4K64YhrxIGQ8mkjaTE7GzYXtXDjbKVFz/kuaz0oO8C2WpV6O2moVs52JwcXH+1ByzMim3Hi/"
    "w4FsyUPm5/omkUz77ioBLOJA1jHWj04FkWEv8n/MHhKkwXGiG0BjY9BY3gloYeyQE7rMdDBGzhRgI3YVb3AIiOoBjeYhWY4d"
    "mnLq6Tr1OFRk3ITthKn3uC1jD+XV23LHHRbGrvhb6gfio8h9x3zyCgtNffGY4E1yDxhp3OFwtn4SMSn4X+pSMDJv6HlX9uyd"
    "X+q6MsG4y1F1kS+5ABK+eyDWJVr8uHFKPICBaXPmCzi6n3HO7TAe//V4fEwp2kTcXuswIwcdBwK3XZMODM/cZbGXWNYHcVgE"
    "P5xxFARxWKcMJfu1ARLu22x5RA5mlKnNphp6ZbIbPMPJxN10IZSg8cTPZ7Tigk40GnNvsJtzl/fjBQq4tM66VB3emLVMeO+H"
    "g8JevNgWN/5FfUTHvxzQjbXXHFQ2xyqZi4v++yBTFJw+526MhgZEtojG7+AF/B0nC3rceKeoaZlQfvTI8d8JKu1wQoZnjmP3"
    "Li35HUkjAG2hIkbwKAM304YDeEGL7fJ2WSgWjJdOyzMOQzFZ41oN86DrPKgw6pGEnj7Tsafo6ujE5qRAx6or2B9/FD54ZcrO"
    "UywpFWLm5bYpCUFDbs1N6I8HKNzUThWyqzk2Hq2zh/XHC0jwHP6X6UpNI0UgW/w+2nAc/Fs6SWTcVlegNOzbz5posxOGbjaA"
    "GtbxR6BT5n0OrepglVMhH4b2tSnUq4e5PWtv1+rDr3tusFi9//Ca0uAQcrIg8a8pEY654x96ZhkZgI67SGiL+Fr+PZp5MRGj"
    "VWQlxSGMpGtlTPvAaSopIYftZxn0rqxRrcvEbJz+lZTkYbNuMfDIr97t7fqkTvkB6MzvCMTeQwQlqWdjkcds+Q0OC12LfBpv"
    "IS3MjXHA36jRGJE3Lw0fAqD1EAfowfLit/ziMb85ozcxD8d4VSaqPInaRjQA6ogJFoVlQi2lloIxj54LDjzf4pXVdGegEmyF"
    "mrWIr4aGrzlLh6R4o8VE9yl/HWg3qHctopUtVB3Fw5j4HjZuV2+VBqVRJjvQnvY0Pn0zqLEKfr4N4SM3hTo7G6rSsAaRDcPh"
    "r64C6EOwkXx/mQ2n7AtgsvKuNjO46i3g+32hasA92UDE1ZK9xtR2ASu/ruoSoEZm4xGVbNvy++L65zRfSaQ+ljMS/BrWrwOC"
    "3xTXv2U8sKqeohX5MR9D+S4kMnM3iATG84n7PBoGHI31xatV3THtQGHkTIGEP0JYBfL9uNEgXjBDl1EZnX2sEs+gh9vVLzkL"
    "y1AM6jidnsmRI/6UFNCEScGzZuv8EGiJyJ0jG70mBHM1zcWCcoZWY38+QsK9Vyfi8kdso/ByM5B2PcE9iUSE25briuwi+QXS"
    "ziWFQjSaz97+mFgPI1oXlsgvJV7CDVYij8XLue2cm7rdrdqv1E3a7uSW3NDW4WiO3gsW76BhEjQ2PqhjCMe6nn1KPkE+HqnF"
    "VIEruZyzMZsbdDo74/Ug9y27sePBJV287aaJ9nvs0O08U/OyGCCaHaFD+SwDrvdD4HAu/9QJ7/8J99fFsnL/T0nLqRIVbt1E"
    "LjcmkcuNpHHRjdGvfQhyVltqOA7+pxbr7FZREvtMuZP7erNWAqnOPCks1JlWE8d+vXDDd3T6mcVLV8egpFskKRcx4WlRACMs"
    "cfNxc7L6Ft78H3d9s2uEiU9aok1gB/R0ds/7BCVAGTfaJKo1b4vNYiIc2isKdVZSd3rFAJdk8E2H2c8y2pz8oaRMlGuOH70o"
    "bqQ+3TtVFh20EXXqBj9akWd3kLLBUePm8+UsH9++QS4mlG2rZ8N03TzFBeFxwsnoax/AXLCwRTs7VLQ1vJWg8+GNdiRWaxP6"
    "1+Ftou/4ePm+xZr6bvT88fEZz9KJH/8lrzK/pMe8QL98/eLF+Q9vXj770vtKvhhnK7U8KsVf/vTi+Zs3z5+dv3jz+sfztz8/"
    "+f7lTz/47+op9V/8++u//PTD+S9vnj/3S5OYRGclyrrkvvLj8zc/qC+9/fnVy1/Ov3/z+q8/+a8iCz05eVQ+9uPrX9WLT9Q7"
    "b9+e//Dk55BtXk5Ay5oXKdKTn0IspjAtRAlms5vmaND48vsvVcm3a8y9bCjtZTp6C81y82Gr8WXnS4ff0JTZS9D5oBFZGLaI"
    "hbV20UgsjjEYnD4Z2ttQAJyrNg/08fNxwwgo8H4GGzgizS0lv/Tfwwm++mJHXnwZfZHGfCBDf1Uo1vLlidtxHtZzZuHn8wEN"
    "/GP82+Z7lRrlDcXZ/eLqRl1ZNRh+WYTh1ZQF5w/bUWnFh4qti6wBP1PqYTHXkpXZ3gk4ZXRHdSs5eiQ5mLGrepWpJ/O0nNID"
    "N5z3IktXjG2id0s5F6/TRa+5ZVxgCiG+3+h+QzsYPf8ZM9f/ugNli7rPDRW+9N6j81agbEKGwJgGRQA05PDG67d/pjOS+fcH"
    "+jnYmP6AEqxxrFAPnUW1AOGeiwYJU8Qonz6NctrzEo/fyytqIbUkXHnQuFCCjivwuvU5GUS8IjKw56NsMtCj3EyT6lqMl+Qm"
    "xYoTVs45mVD5Fbam1r/gNHaZr4eXkSJDxVvXA63Rz/8jc7iK1pCd05kMY6NGZJfyzL7LBHBOzPScB/N8kioeY1isKyttkfxX"
    "i0vNLW3dzrYviaIp2WWkC+RIek6OVOeQV+mrJX+Knr0lHyvrnzAen9PkSyPtMVat/HPTtYFzvrUv015ErK08T0ejbDTQTF1r"
    "V4bUk6/s9vRVGxAvzWGbuJo6GvhiIHueyMn4mM62gN0tFTkMlchdLpH6bEyeEOr2V5RbCEAoF9n6OssWDf1GGbSRuqhrwcVb"
    "+o3BdQUSKjZPKXpRURUABNzibfx8PW3Zz0id6LlTrFi0uJHnxeLclA2KvECBlhvczX2IVykPdb1yGSvC9WJiFSegZU8T5BWF"
    "8ajlCWKK8IMvyr1WbJadYnTH1vUhpA404DqdTYlF/FX9CAuM0tXUTA0uuOJoMa4IP2M1jdUuowi4LGYbDPgArk9ZfKJjE2hL"
    "106gV+SACXTK101gtUjtBDpFayfQKbNrAp1iPIGNCyUinZdFPvObpO9GZxiM8pyjR4S7gWcgeONcB5oMHD8Sf2cw+q5yYLQn"
    "dSXOy3SenTMDGmi1SIXBik7h/OL2nMJYzq2cJ4qRx67G5Ct9N5BP7EY6AbhGyrTkqHomF+csVJzPewNcfk9XP/Zsi7JZtqUX"
    "nUam5fq8vExH2FWBBYp5oA1klWHw3tIjZ/9ocOFzfjyij/GtN3zHNtX8ICnoHNIuFXeFIl8Ilg6Ykp6U5BfFYkeLKS3dAOc/"
    "FveUnCdBbfb7JHbpC0haposQid6knP/2MT2CZ81ilIHazCt8yKx5iR/GXpMDRvQ9gFBezjH16t/Ku+RaQcqG63yhhrb9c6aG"
    "4okimtsyh/DJ857ihpIHKNvygof0bc43FCukX7TeXmDEgT2XqV0T1NUSF6W3m9VYHadbjWfPf3ny8tX5zy//9vzVW27Er8/f"
    "vH35+id18Ol1el8fd7vHvc6XeKRa9yGBVvb/Al82Bdg="
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
