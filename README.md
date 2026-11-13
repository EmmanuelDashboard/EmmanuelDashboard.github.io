# Emmanuel Dashboard for Pecan Thinning

A web dashboard for diagonal thinning of pecan orchards from drone images. Drop a drone image of any pecan orchard and it finds:

- the pecan canopy (no grass, soil or shadow);
- every crown;
- the planting grid;
- the EVEN / ODD thinning diagonals;
- the canopy total of each diagonal;
- where thinning starts.

You can correct crowns before you export the plan.

The site runs entirely in the web browser: the images you open are not uploaded anywhere.

## Two ways to use it

| | Drop a drone image | Drop a results ZIP from the notebook |
|---|---|---|
| Input | Orthomosaic GeoTIFF (any projection, 8 or 16 bit), or a JPG / PNG of an orchard | `Pecan_Canopy_Results_….zip` from the Kaggle notebook |
| Where it is analysed | In the browser, in about a minute | On Kaggle (Python), a few minutes per image |
| Canopy | Colour and texture (RGB) | Colour and texture, plus tree height from the elevation (DSM) file: most accurate |
| Then | The same map, totals, corrections and export | The same map, totals, corrections and export |

A JPG or PNG has no coordinates, so you enter the ground size of one pixel (in cm). Areas are then in square metres, and exported shapes are in local metres instead of longitude / latitude.

## What you see

- **The drone image** with every pecan crown tinted in the colour of its diagonal: blue for EVEN, red for ODD. Along each planted row the crowns alternate blue, red, blue.
- **The wavy thinning diagonals.** Each one runs from its first tree to its last tree and ends in an arrow.
- **The canopy total (m²) of each diagonal**, written at the end of its line (zoom in to read it), plus a table and a chart.
- **EVEN, ODD and overall totals**: canopy area, crowns and diagonals.
- **Where to start.** The start diagonal (D02 by default, always an EVEN diagonal) is drawn thick with a START label and an arrow for the direction of travel. *Start only* hides the other lines.

## Corrections

Click a crown on the map, then:

| Action | What it does |
|---|---|
| ‹ › or the diagonal list | Moves the crown to another diagonal; its colour follows. |
| Not a pecan | Takes the crown out of all totals; it is shown grey. |
| Merge with… | Click a second crown: two pieces of one tree become one crown. |
| Add missed crown | Click around a crown the notebook missed, then double-click. Parts already covered by other crowns are left out. The diagonal comes from the planting grid. |

Undo and redo with <kbd>Ctrl</kbd>+<kbd>Z</kbd> and <kbd>Ctrl</kbd>+<kbd>Y</kbd>. Corrections are saved in the browser you used, until you click *Reset all*.

## Export

- **Canopy per diagonal (CSV):** one row per diagonal, then the EVEN, ODD and ALL totals at the end.
- **Crowns table (CSV)** and **corrected crowns (GeoJSON, WGS84)**, for GIS software.
- **Corrections log (CSV).**
- **Print / save the map as PDF.**

## The Kaggle notebook (most accurate)

The dashboard also opens the results ZIP made by the pecan canopy notebook (`notebook/Pecan_Canopy_Only_FIXED.ipynb`, run on Kaggle). The notebook works on **any pecan orchard orthomosaic**:

- any file name;
- lat/long, Web Mercator or a projected CRS;
- 8- or 16-bit;
- clipped to the orchard or not.

Several fields can be analysed in one run, one ZIP per image.

1. Attach the orthomosaic(s) (GeoTIFF) and, if you have them, the elevation / DSM GeoTIFF(s) to the notebook, then *Run All*. Each orthomosaic uses the DSM that covers it.
2. Download `Pecan_Canopy_Results_<image name>_….zip` from the notebook's *Output*.
3. Open the dashboard and drag the ZIP onto the map, or click **Open results**.

The ZIP needs `crowns_wgs84.geojson` and `guides_wgs84.geojson`. The drone image is shown when the ZIP also has `web_preview.jpg` and `web_preview.json`; results from the current notebook have both.

## Put the dashboard online (GitHub Pages, free)

1. Sign in to GitHub and create a new **public** repository named `emmanuel-dashboard`. The repository name becomes the end of the site's link.
2. On the new repository page, click **uploading an existing file**. Drag in everything from this folder: `index.html`, `README.md` and the folders `css`, `js`, `assets` and `notebook`. Click **Commit changes**.
3. Open **Settings → Pages**. Under *Build and deployment*, choose **Deploy from a branch**, branch **main**, folder **/ (root)**, then **Save**.
4. After a minute or two the site is live at `https://YOUR-USER-NAME.github.io/emmanuel-dashboard/`. Anyone with this link can use it with their own drone images or results ZIPs.

To update the site later, upload the changed files to the same repository; GitHub Pages republishes by itself.

*Optional:* you can share a link that opens a results ZIP straight away. Upload the ZIP to the repository, for example as `data/field.zip`, and share `…/emmanuel-dashboard/?data=data/field.zip`. In a public repository, anyone can then download that ZIP, so only do this with data you are allowed to share.

## Files

| Path | Contents |
|---|---|
| `index.html` | The page |
| `css/app.css` | Layout and colours |
| `js/app.js` | Map, totals, corrections, export, image reading |
| `js/analyze.js` | Image analysis in the browser: canopy, crowns, planting grid, EVEN / ODD diagonals |
| `assets/` | Site icon |
| `notebook/` | The pecan canopy notebook that makes the results ZIP |
| `.nojekyll` | Tells GitHub Pages to serve the files as they are (optional) |

The libraries load from public CDNs, so the page needs an internet connection:

- Leaflet 1.9.4
- Chart.js 4.4.1
- JSZip 3.10.1
- proj4js 2.11.0
- polygon-clipping 0.15.7
- geotiff.js 2.1.3

There is no build step.

## Method in short

- **Pecan canopy only.** Grass, weeds, soil and shadow are excluded with a colour and texture classifier trained on the image itself. With a DSM, canopy must also stand at least 1 m above the ground.
- **One crown per planting position.** The planting grid (20 × 40 ft design) is measured from the treetops.
- **Diagonal thinning removes alternate trees in each row and column**, so every crown belongs to exactly one diagonal: EVEN or ODD.
- **Canopy areas** are in square metres, in the projected coordinates of the drone image.

