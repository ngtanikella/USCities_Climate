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

All data is uploaded in the "Data" folder. Alternatively:
Download NOAA 1991–2020 Monthly Climate Normals (by-station CSVs):
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

This generates a self-contained **`USclimate_map.html`** (~3–5 MB) in the same directory.

### 3. Open it

**Option A** — Double-click `USclimate_map.html` in any modern browser.

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
Then distribute `launch.exe` + `USclimate_map.html` together.


## Configuration

Edit the top of `build_USclimate_app.py`

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

Edit the Comfort Index based on your preferences . 
Personally, I prefer 50-70F (daily mean), colder rather than hotter, low rain, low humidity, >32F and <90F, Where each city starts with a score of 100, and penalties are as follows: 
temp_penalty = (cold_diff * 0.8 + cold_diff**2 * 0.01 + cold_diff_extra**2 * 0.01) + (hot_diff * 0.8 + hot_diff**2 * 0.03 + hot_diff_extra**2 * 0.06)
  #cold_diff is difference from 50F, cold_diff_extra is difference from 30F. hot_diff is difference from 70, hot_diff_extra is difference from 90F.
rain_penalty = np.maximum(0, p[mm] - 2.0) * 2.0
  #the first 2 inches of rain per month are no penalized, but every inch of rain is penalized by 2 points.
muggy_penalty = np.maximum(0, t[mm] - 70) * np.maximum(0, 20 - dtr[mm]) * 0.4
  #hudidity is panlized
extreme_penalty = (f[mm] * 1) + (h[mm] * 2.5)
  #Days below 32F and above 90F are penalized


## Project Structure

```
USCities_Climate/
├── build_USclimate_app.py   # Build script: NOAA data → standalone HTML
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
