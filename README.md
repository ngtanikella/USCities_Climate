# USCities_Climate

Interactive U.S. climate normals map built from NOAA 1991–2020 monthly station data and the SimpleMaps city database.

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Voronoi-tessellated choropleth** — every point on the U.S. map is colored by the nearest city's climate data, clipped to the national border
- **14 climate variables** — mean/max/min temperature, precipitation, snowfall, heating & cooling degree days, freezing days, 90°F+ days, snow cover days, and a computed Comfort Index
- **Monthly & annual views** — toggle between any month or the full-year summary
- **City comparison** — select two cities (click or search) to overlay their 12-month profiles side by side
- **National distribution histogram** — see where your selected cities fall relative to all others
- **Imperial / Metric toggle**
- **Zero-lag interaction** — all data is baked into a single HTML file; no server required


## Quick Start

### 1. Get the data

All data is uploaded in the "Data" folder
**Climate data** — Download NOAA 1991–2020 Monthly Climate Normals (by-station CSVs):
- Go to [NOAA Climate Normals](https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals)
- Under "Bulk Download" → [Monthly](https://www.ncei.noaa.gov/data/normals-monthly/)
- Download the `by-station` folder (~13,000 CSV files)

**City database** — Download the free U.S. cities CSV from [SimpleMaps](https://simplemaps.com/data/us-cities).

### 2. Build the app

```bash
# Install dependencies
pip install pandas numpy scipy geopandas shapely

# Edit paths in build_climate_app.py, then run:
python build_climate_app.py
```

This generates a self-contained **`climate_map.html`** (~3–5 MB) in the same directory.

### 3. Open it

**Option A** — Double-click `climate_map.html` in any modern browser.

**Option B** — Run as a desktop app:
```bash
pip install pywebview
python launch.pyw
```

**Option C** — Package as a standalone `.exe`:
```bash
pip install pywebview pyinstaller
pyinstaller --onefile --noconsole launch.pyw
```
Then distribute `launch.exe` + `climate_map.html` together.

### Alternative: Dash server version

If you prefer a Python-server-based app (useful for development or adding more features):

```bash
pip install pandas numpy dash plotly scipy geopandas shapely
python climate_map_dash.py
# Open http://127.0.0.1:8050
```

## Configuration

Edit the top of `build_climate_app.py` (or `climate_map_dash.py`):

```python
CLIMATE_DIR  = r"path/to/your/station/csvs"
CITIES_CSV   = r"path/to/uscities.csv"
CITY_SPACING = 0.75   # ±degrees deduplication (0.75→~896 cities, 1.0→~558)
```

| Spacing | Cities | Detail      | Build time |
|---------|--------|-------------|------------|
| ±1.00   | ~558   | Coarse      | ~2 min     |
| ±0.75   | ~896   | Moderate    | ~3 min     |
| ±0.50   | ~1,673 | Fine        | ~5 min     |

## Project Structure

```
USCities_Climate/
├── build_climate_app.py   # Build script: NOAA data → standalone HTML
├── climate_map_dash.py    # Alternative: Dash-based server version
├── launch.pyw             # Desktop wrapper (pywebview)
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md
```

## How It Works

1. **Load** ~13,000 NOAA station CSVs, each containing 12 rows of monthly normals
2. **Filter** U.S. cities by population, deduplicating those within ±0.75° lat/lon of a higher-population city
3. **Match** each city to its nearest weather station (KD-tree)
4. **Interpolate** missing values (precipitation, snowfall, etc.) from up to 4 nearest stations via inverse-distance weighting
5. **Compute** Voronoi tessellation clipped to the U.S. border (Shapely)
6. **Embed** everything into a single HTML file with Leaflet.js (map) + Plotly.js (charts)

## Data Sources

- **Climate data**: [NOAA 1991–2020 U.S. Climate Normals](https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals)
  - Palecki, M., Durre, I., Applequist, S., Arguez, A., & Lawrimore, J. (2021). NOAA National Centers for Environmental Information.
- **City database**: [SimpleMaps U.S. Cities Database](https://simplemaps.com/data/us-cities) (free tier)
  - Use of the free database requires linking back to https://simplemaps.com/data/us-cities

## License

[MIT](LICENSE)
