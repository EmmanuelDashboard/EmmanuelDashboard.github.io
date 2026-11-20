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
    "z0IQXHJqswUOBsTLzBFAgOj7bBjIG3jHoF+iD+BGYF5w+AExIXxHj366KFuKQxHE3WQyaV3Fh7MOmXMnX87slr8Ng9N0pigk"
    "71zhDWFPzml5cprCcnos7Tl5G0LpUgE9b92IDvzDZTEaKIOGHUU4hpaHPxm13kXVSXXG/XaG2NG7CbM1uYq5jZENepUzMy6d"
    "yod2JkzCWElUQ8wSfdS4AibLTjvS133nNVbvY80U5u0TDOOGPD42SCaVrNPASneuwt25CnWnUnUVrrqqV61T2D9NOT17HtWr"
    "G0bKUTiElUy5xIkUBz4RWJ64IlPoh1KmjjYOlxtKZpTSvYyoM/uMMV/RgrLMkFLlTmw8lejF31/8RHeZn58/j7t+JHbLeTuh"
    "U2zUlBpL8meYjHJ1YosAGpMYn88AC+/dO+BLWTvl18kk2+Rethw5YEKRnebZ4qkYBEFldtcNnvU5VxlnE5rPdpnAMsChA04m"
    "k+qBDiLAKDC8Ro0H00xmJptXjd89Z86crul4gM0IP9VjoDC3xtS9hZMJpZnaGGbyQ9yU10/0o9V8crYDV9L8FRonqPRko4Z5"
    "LV75bRmoFft97ip/az4f1yq2Z/gs2msY0a17W5f7gee+QthLA7buxpET+cGStFQHWRo5iiO94D5WyVLcLmlRoHCFWA0DTVYM"
    "ONEFtFGG2MxxtAgL0qmV7A9rYncb5W5sZUrrUrqEg+VaX9LjRdilQjn4omRb8rVyWvk4MIqZhE25EhdtosmFhmUKuMpcL0So"
    "KLuurXN/j8MW6P6TI3LG0JKiNrI7AqN+VIn5XLJOT7zIXnVXnp6rmegxjjtKexDSwktgciHuPDlceLRIfVTjGsSxC3HsQbxW"
    "iONDECGlrXkgeIzzWjhneuiHeOexMM/00A8kw7IzV05xmX+l7nwz48uJhOpGFgf+aWfucIpUyQkisykQvtU9gYPbbpZQVU65"
    "6NoxqxSPMzDyzOwr4rt61+Fs2ol0X5WS/NXevO0PrTxfyiJoe60oXpqSIl3XmJEGa3FToB2+3KG8XjFVYsbWR71eH5vR/fzY"
    "nGthrNIKZz6GRXvgW1BqkY4RKk29KzGqOqFJjYQcPD8ajNlaa+G89XZ5zJ7wEmilaferyLjgM8EpFIf7rumcGWwgk7qf0lFK"
    "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"
    "DC4RrwV7LPyVL7p80ZdZK2G63nCsuoIAJJ38zPdf9hZmI047y3+Gw73TxOJKI2WnULlV4XY9xj0OyOrKnhtA5SZBbjCTBLX1"
    "/N2PMYeh5E3A+6Jc5FIor6UTHV/Mw8lEXdA+0FD+UFlIf8TVa3i7HFR1F61saO6ttJnEaaKb5S9B31MMB8lBYu3QYDaU/aDv"
    "S3JT4kNKynmmXj7bMrFguW7nKc3L4sQLem2CB37Tq77/psdZvVa199/EbQckAG5dlejOqkR3qhA1nTHVPlWcCD41e0KtlrM9"
    "raS6I9TPm4IYUuMKpSS05vyk70uZnVHkLl5VBDnwCqnINVwHCfEOaqPmf/mjNfuWmfhTWzToZwYT/i6zai3rNgnnAbcohCNx"
    "0+0VCM7ZUD4Zp28U2xrGhNPMFZpWaLlTeGZ0VBYDLMWB9EI+rdn0IigAZK/3bL6aZdP9W1g1cMQK+jZOitYAP1hnVQ+0YbSE"
    "KtAa09oaq0CLflSiQYz3ZRgIV2kTezfzOzz2xrAdav4xS859WwInm28sn2WDfvnzy5e/ff/21fMvvVYc11W/+KufXr54+/bF"
    "899evv35x9/evXn67NVP3/t1S99Bt2KZutkvrY7IxVAkt26VH1+8/Z5aevfm9atffnv29udff/Krwi2I3ZFqjf3489+p4lOq"
    "8+7db98/fVMlmxfnWMuGFtHS00ddLLYwb0Q1a3Fin59EXz5DLIl3BeZeD5TuKpm8KxI6wR+2oy97Xzr0hqes/Il1fhIFNkZZ"
    "xBxFJy4nV35mAmduhm0nhV02+Y36fGKun99GZSKCjpAByTjHVqR+Pdzg6xV7WvFVsCLj/ERR/2FJpOXLU3fggtbfhIT/Nj9h"
    "xH+Lf7vyrgZRaxBl94vTi6ay/39jV9vaRgyD/8rYp5ZekjaDfbhSStkyCHRbSMrYlyEuuUs4aJKSZCkd5L9PemT55e4y9qWk"
    "tmTLb7Llsx5xZ6S0nHCOVjR/U46WFKfkECeaD99mJrIyzTsKUdJDSkNTdu6ocSFXdz587W1a2NWdAc7fJp/55lXh4H26fGKP"
    "+u0cfhz/4RJrWDLJPM8al03y2q7rBkV1WXpV/OGXve1J03PLFzVmFfDBWswKzowW1UYmLrkbJBkiddSJj69xbvRwOiFx3URl"
    "tcqtzy6Ky/bK6qZ0Hq8d5ECEJzjYKov62p5niISF02wHCRAEcrt/r/9UkY6w+y6ChSWdw/31r6uwwKvDSVCNpP4ttCpe4IWp"
    "CjM++RzNmw8ua8eGF58Fd8Uz0I4mwOGT8HaB5PSJWvdaVXAGDYzb5ZKgbpyQwSjldUy+aXlkrQZm7CxQVHsqyhJwKaqi7a5k"
    "gZYMwmYz6B/Y+L1Y9KGjfLykLP4Ia3Zuz7vnKnRo7jDyPSD/wJD/A+C/ofI3ZEQTrZQI9IU7Nz5eeHYyCOuYur/dZFo5bTfk"
    "aRskQITJ4lidKlt3kT4mgJbrAZvaJFpuAKZpI8iESk/pIgwBI3Jvdp6jIPGjJh3r3OzJ1lx2xhjN3wgYCBQ2SGdR3sem5sBS"
    "G4o9aKDVmjUaPK5jLbRbzUm1Ma2Hufz7Cf99HQaJqufqCMZIyGJ/IMUq5tIEIL10a3VXyXpwobbCUnXAxqTZJSrTpKmmBFH9"
    "D2wfJMcEkMe7SXp6cA3wlMn2kpLKmIrEr3Up9qLEZ8Y+mUt8U3WB8PVrDGf7wsVblG+i7CXTQp/g3iOL+50nT7VkxeZZ9HR+"
    "hkkzu9jcyayTD7hIaxl6/tvixYM7WGmv9UbifkwEKPOBJ83bvpZdW8e9kARWvQ6nE/00qzWBZzx+YcJ/0QcaC+6/QyWzKxMM"
    "gJfnavZ7t2Q7JHv3efT0MH6kyfjn6HGmQvwYTWfj79/4xDi8Hn7s3dz0htfvJYulO13KddZfB4Q12Q=="
)
# ---- END EMBEDDED DASHBOARD ENGINE ----


# ==================== RUN ====================

RESULTS_FOLDERS = run_all()
RESULTS_FOLDER = RESULTS_FOLDERS[0] if RESULTS_FOLDERS else None
log("FINISHED — download the results ZIP(s) from Kaggle Output / Files.")
