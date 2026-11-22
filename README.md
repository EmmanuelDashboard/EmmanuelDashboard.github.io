# Emmanuel Dashboard for Pecan Thinning

A web dashboard for diagonal thinning of pecan orchards from drone images. Drop a drone image of any pecan orchard and it finds:

- the pecan canopy (never grass, soil or shadow);
- every crown, **young replanted trees included** (flag YOUNG_TREE): a small, round, sunlit crown as green as the pecan leaves on an empty planting position, standing on clean bare ground or casting its own shadow. Every empty position of the 40 × 20 ft planting grid, along the whole length of every row (also a row of replants with few or no big trees), is also searched for a sapling (an island of leaves on the tree strip). These saplings are counted only when the measured grid matches the 40 × 20 ft design and the same search finds far fewer lone green spots halfway between planting positions, where nothing was planted, so weeds and grass are not counted. A line of young trees one spacing apart halfway between the positions (a row planted half a spacing off the old trees) is counted too. A fainter or paler young crown also counts when it casts its own compact shadow on the side the big trees' shadows fall. Finally, every planting position still empty is looked at in the full-resolution drone image (about 2 in per pixel): a young crown of at least 3 ft² with its own dark tree shadow beside it is counted, while weeds and grass, which lie flat and cast no such shadow, are not. A small green crown standing alone on clean bare soil (a weeded tree strip) is counted too, when such crowns are found on the planting positions far more often than halfway between them;
- the planting grid;
- the EVEN / ODD thinning diagonals (along every row the crowns alternate blue, red, blue, also across a missing tree);
- the canopy total of each diagonal;
- where thinning starts;
- a thinning plan: trees and canopy removed and left, canopy cover before and after, tree heights, and the list of trees to remove with their GPS positions.

Live site: <https://emmanueldashboard.github.io>

You can correct crowns before you export the plan.

The site runs entirely in the web browser: the images you open are not uploaded anywhere.

## Two ways to use it

| | Drop a drone image (and its DSM) | Drop a results ZIP from the notebook |
|---|---|---|
| Input | Orthomosaic GeoTIFF (any projection, 8 or 16 bit), or a JPG / PNG of an orchard; plus, for a GeoTIFF, the elevation (DSM) file of the same flight | `Pecan_Canopy_Results_….zip` from the Kaggle notebook |
| Where it is analysed | In the browser, in about a minute | On Kaggle, with the same engine (`js/analyze.js`, run with Node.js) |
| Canopy | Colour and texture; with the DSM also tree height (canopy at least 1 m tall), cast shadow trimmed | Identical: same crowns, same canopy to the m², same diagonals |
| Then | The same map, totals, thinning plan, checks, corrections and export | The same map, totals, thinning plan, checks, corrections and export |

Drop the orthomosaic and its DSM together (hold <kbd>Ctrl</kbd> to select both), or add the DSM in the window that opens. Without a DSM the canopy comes from the image only and can include some grass. A JPG or PNG has no coordinates, so you enter the ground size of one pixel (in cm). Areas are then in square metres, and exported shapes are in local metres instead of longitude / latitude.

## Decide

- **Thinning plan.** Choose to remove the EVEN (blue) or the ODD (red) trees. The dashboard shows the trees and canopy removed and left, the field's canopy cover before and after, the mean tree height of both groups (with a DSM), the in-row spacing after thinning, and where to start.
- **Trees to remove (CSV with GPS).** One row per tree to cut, ordered along each diagonal from its START end: crown ID, diagonal, row and tree number, canopy, height, longitude and latitude.
- **Quality checks.** Row neighbours with the same colour (must be 0; *Show* goes to any that are left after corrections), whether the DSM was used, crowns crossed by their own line, crowns off the planting grid, and the cast shadow trimmed.

## What you see

- **The drone image** with every pecan crown tinted in the colour of its diagonal: blue for EVEN, red for ODD. Along each planted row the crowns alternate blue, red, blue.
- **The thinning diagonals.** Each one is a single smooth, unbroken line through the middle of all its trees (within about 2 ft of the centre of each crown), as straight along the diagonal as that allows. Where a row has a missing tree, the crowns on one side of the gap move to the next diagonal so that the colours still alternate; they move the same way as the crowns in the rows beside them, so the lines bend as little as possible. Lines run also across white areas (no image) inside the field. Past its end trees it carries on in the direction of the diagonal to the edge of the field, where it ends in an arrow (no line starts or ends in the middle of the field). All arrows point the same way. Lines never cross each other. A line keeps off the canopy of the other colour (a red line goes around blue crowns, a blue line around red ones) with wide, gradual bends (it may just touch a ragged crown edge), except where crowns of both colours touch and there is no free path.
- **Dark young trees in weeds.** A young pecan in green weeds is no greener than the weeds, but it is a dark, round bush that stands alone on the strip, at least 6 ft from big trees. Such bushes are counted at the empty planting positions only when they stand on the positions far more often than halfway between them (the planting spacing is the check: weeds lie anywhere).
- **Young trees along every row.** Besides the planting positions, the dashboard walks along every row of trees to the end of its tree strip, so young trees are found also where a block was planted at another spacing. This walk is counted only when the same walk along the grass alleys between the rows finds almost nothing there (otherwise the green spots are weeds).- **The canopy total (ft²) of each diagonal**, written at the end of its line (zoom in to read it), plus a table, a chart per diagonal and an EVEN vs ODD total chart.
- **EVEN, ODD and overall totals**: canopy area (ft²), crowns and diagonals. Field area in acres, spacing and heights in feet (design 20 × 40 ft).
- **Where to start.** Thinning removes the family (EVEN or ODD) with the smaller total canopy and starts on its first diagonal: D01 when ODD is smaller, D02 when EVEN is smaller. The start line is drawn thick with an arrow for the direction of travel and a small black "start" just beyond its first end; you can pick another one, and the thinning plan follows it. *Start only* hides the other lines.

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

- **Trees to remove (CSV with GPS)**, for the field crew (canopy in ft², height in ft, young trees marked).
- **Result map image (JPG):** the whole field with crowns, diagonals, the start line, canopy (ft²) per line and the totals.
- **EVEN vs ODD total chart (JPG).**- **Canopy per diagonal (CSV, ft²):** one row per diagonal, then the EVEN, ODD and ALL totals at the end.
- **Crowns table (CSV)** and **corrected crowns (GeoJSON, WGS84)**, for GIS software.
- **Corrections log (CSV).**
- **Print / save the map as PDF.**

## The Kaggle notebook (same results)

The dashboard also opens the results ZIP made by the pecan canopy notebook (`notebook/Pecan_Canopy_Only_FIXED.ipynb`, run on Kaggle). With `ENGINE = "dashboard"` (the default) the notebook runs **this dashboard's own engine** (`js/analyze.js`, embedded in the notebook, run with Node.js), so the same image, DSM and Detail give exactly the same crowns, canopy and diagonals in both places. If Kaggle has no Node.js, the notebook installs it (`nodejs-bin`; switch *Internet* on in the notebook settings). `ENGINE = "python"` runs the older full-resolution Python analysis instead (numbers about 1–2 % different). The notebook works on **any pecan orchard orthomosaic**:

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

This site is the repository `EmmanuelDashboard/EmmanuelDashboard.github.io`, published by GitHub Pages at <https://emmanueldashboard.github.io>. Anyone with the link can use it with their own drone images or results ZIPs.

To update the site, upload the changed files to the same repository (**Add file → Upload files**, keep the folder names), then **Commit changes**; GitHub Pages republishes by itself within a minute or two.

*Optional:* you can share a link that opens a results ZIP straight away. Upload the ZIP to the repository, for example as `data/field.zip`, and share `https://emmanueldashboard.github.io/?data=data/field.zip`. In a public repository, anyone can then download that ZIP, so only do this with data you are allowed to share.

## Files

| Path | Contents |
|---|---|
| `index.html` | The page |
| `css/app.css` | Layout and colours |
| `js/app.js` | Map, totals, corrections, export, image reading |
| `js/analyze.js` | Image analysis in the browser: canopy (with the DSM: tree heights), crowns, planting grid, EVEN / ODD diagonals, row alternation |
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

- **Pecan canopy only.** Grass, weeds, soil and shadow are excluded with a colour and texture classifier trained on the image itself. With a DSM, canopy must also stand at least 1 m above the ground. The ground level comes from the bare ground seen in the image and, where weeds or lush grass hide the ground (on slopes and in hollows), from the low surface of the DSM itself, so trees there are not taken for low plants. Clear cast shadow (dark, not leaf-green, at ground level) is trimmed from the crown edges.
- **One crown per planting position.** The planting grid (20 × 40 ft design) is measured from the treetops; where the rows were planted at another distance (for example 50 ft), the distance measured in the image is used. A crown is never cut into two pieces.
- **Diagonal thinning removes alternate trees in each row and column**, so every crown belongs to exactly one diagonal: EVEN or ODD. Along every row neighbouring crowns always alternate, also across a missing tree (the crowns on one side of the gap move to the next diagonal, chosen so the diagonals stay as smooth as possible).
- **Canopy areas** are in square metres, in the projected coordinates of the drone image.

