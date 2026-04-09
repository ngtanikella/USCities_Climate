"""
build_climate_app.py — Generates a standalone climate_map.html
===============================================================
Run once to process NOAA data → self-contained HTML file.
The HTML works by double-clicking in any browser (no server).
Optionally wrap with launch.pyw for a desktop .exe.

DATA SOURCES:
    NOAA 1991-2020 U.S. Climate Normals
        https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals
    SimpleMaps U.S. Cities Database (free tier)
        https://simplemaps.com/data/us-cities

SETUP:
    pip install pandas numpy scipy geopandas shapely

USAGE:
    1. Edit CLIMATE_DIR and CITIES_CSV below.
    2. python build_climate_app.py
    3. Open climate_map.html in your browser  (or run launch.pyw)
"""

import os, glob, json, math
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, mapping, shape
from shapely.ops import voronoi_diagram, unary_union
import geopandas as gpd

# ─── USER CONFIG ────────────────────────────────────────────────────────────
CLIMATE_DIR   = r"C:\Users\nag55\Downloads\Climate\USClimateData"
CITIES_CSV    = r"C:\Users\nag55\Downloads\Climate\uscities.csv"
CITY_SPACING  = 0.5        # ±degrees for deduplication (0.75→~896 cities)
OUTPUT_HTML   = None        # None = same folder as CITIES_CSV
# ────────────────────────────────────────────────────────────────────────────

CLIMATE_COLS = [
    "MLY-TAVG-NORMAL","MLY-TMAX-NORMAL","MLY-TMIN-NORMAL","MLY-DUTR-NORMAL",
    "MLY-PRCP-NORMAL","MLY-SNOW-NORMAL","MLY-HTDD-NORMAL","MLY-CLDD-NORMAL",
    "MLY-TMIN-AVGNDS-LSTH032","MLY-TMAX-AVGNDS-GRTH090",
    "MLY-PRCP-AVGNDS-GE010HI","MLY-SNOW-AVGNDS-GE010TI","MLY-SNWD-AVGNDS-GE001WI",
]
ANNUAL_SUM = {"MLY-PRCP-NORMAL","MLY-SNOW-NORMAL","MLY-HTDD-NORMAL","MLY-CLDD-NORMAL",
              "MLY-TMIN-AVGNDS-LSTH032","MLY-TMAX-AVGNDS-GRTH090",
              "MLY-PRCP-AVGNDS-GE010HI","MLY-SNOW-AVGNDS-GE010TI","MLY-SNWD-AVGNDS-GE001WI"}

KEY_MAP = {
    "MLY-TAVG-NORMAL":"tavg","MLY-TMAX-NORMAL":"tmax","MLY-TMIN-NORMAL":"tmin",
    "MLY-DUTR-NORMAL":"dutr","MLY-PRCP-NORMAL":"prcp","MLY-SNOW-NORMAL":"snow",
    "MLY-HTDD-NORMAL":"htdd","MLY-CLDD-NORMAL":"cldd",
    "MLY-TMIN-AVGNDS-LSTH032":"frost","MLY-TMAX-AVGNDS-GRTH090":"hot90",
    "MLY-PRCP-AVGNDS-GE010HI":"pdays","MLY-SNOW-AVGNDS-GE010TI":"sdays",
    "MLY-SNWD-AVGNDS-GE001WI":"scov",
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_stations(climate_dir):
    print("Loading station files...")
    files = glob.glob(os.path.join(climate_dir, "*.csv"))
    print(f"  Found {len(files)} files.")
    records = []
    for i, fp in enumerate(files):
        if (i+1) % 2000 == 0: print(f"  {i+1}/{len(files)}...")
        try: df = pd.read_csv(fp, dtype=str)
        except: continue
        ht = "MLY-TAVG-NORMAL" in df.columns
        hp = "MLY-PRCP-NORMAL" in df.columns
        if not ht and not hp: continue
        sid = df["STATION"].iloc[0].strip().strip('"')
        try: lat,lon = float(df["LATITUDE"].iloc[0]), float(df["LONGITUDE"].iloc[0])
        except: continue
        nm = df["NAME"].iloc[0].strip().strip('"')
        try: ef = float(df["ELEVATION"].iloc[0]) * 3.28084
        except: ef = np.nan
        for _, row in df.iterrows():
            mo = int(row["DATE"].strip().strip('"'))
            rec = {"station":sid,"stn_lat":lat,"stn_lon":lon,"stn_name":nm,
                   "stn_elev_ft":ef,"has_temp":ht,"month":mo}
            for c in CLIMATE_COLS:
                try: rec[c] = float(row[c]) if c in df.columns else np.nan
                except: rec[c] = np.nan
            records.append(rec)
    sdf = pd.DataFrame(records)
    print(f"  {sdf['station'].nunique()} stations ({sdf[sdf['has_temp']]['station'].nunique()} w/ temp).")
    return sdf

def load_and_filter_cities(csv, target=2000):
    print(f"Filtering cities (spacing={CITY_SPACING})...")
    df = pd.read_csv(csv, dtype={"population":float,"lat":float,"lng":float})
    df = df.dropna(subset=["lat","lng","population"]).sort_values("population",ascending=False).reset_index(drop=True)
    sel = []
    for _, r in df.iterrows():
        if len(sel) >= target: break
        la, lo = r["lat"], r["lng"]
        if any(abs(s["lat"]-la)<CITY_SPACING and abs(s["lng"]-lo)<CITY_SPACING for s in sel): continue
        sel.append({"city":r.get("city_ascii",r.get("city","")),"state":r.get("state_id",""),
                    "lat":la,"lng":lo,"population":r["population"]})
    res = pd.DataFrame(sel)
    print(f"  {len(res)} cities selected.")
    return res

def match_cities(cities, stations):
    print("Matching to nearest stations...")
    ts = stations[stations["has_temp"]]
    sl = ts.groupby("station")[["stn_lat","stn_lon","stn_elev_ft"]].first().reset_index()
    tree = cKDTree(np.deg2rad(sl[["stn_lat","stn_lon"]].values))
    _, ix = tree.query(np.deg2rad(cities[["lat","lng"]].values))
    cities = cities.copy()
    cities["matched_station"] = sl.iloc[ix]["station"].values
    cities["elevation_ft"] = sl.iloc[ix]["stn_elev_ft"].values
    return cities

def build_merged(cities, stations):
    print("Merging...")
    recs = []
    for _, cr in cities.iterrows():
        sd = stations[stations["station"]==cr["matched_station"]]
        if sd.empty: continue
        base = {"city":cr["city"],"state":cr["state"],"lat":cr["lat"],"lng":cr["lng"],
                "population":cr["population"],"matched_station":cr["matched_station"],
                "elevation_ft":cr["elevation_ft"]}
        for _, sr in sd.iterrows():
            rec = {**base, "month":sr["month"]}
            for c in CLIMATE_COLS: rec[c] = sr[c]
            recs.append(rec)
        rec = {**base, "month":0}
        for c in CLIMATE_COLS:
            v = sd[c].dropna()
            rec[c] = (v.sum() if c in ANNUAL_SUM else v.mean()) if len(v) else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)

def interpolate_missing(merged, stations):
    print("Interpolating missing...")
    filled = total = 0
    for col in CLIMATE_COLS:
        for mo in range(1,13):
            mask = (merged["month"]==mo) & merged[col].isna()
            if mask.sum()==0: continue
            total += mask.sum()
            src = stations[(stations["month"]==mo) & stations[col].notna()]
            if src.empty: continue
            sr = np.deg2rad(src[["stn_lat","stn_lon"]].values)
            sv = src[col].values
            st = cKDTree(sr)
            mi = merged.index[mask]
            cr = np.deg2rad(merged.loc[mi,["lat","lng"]].values)
            k = min(4,len(src))
            d, ix = st.query(cr, k=k)
            if k==1: d,ix = d.reshape(-1,1), ix.reshape(-1,1)
            for i, midx in enumerate(mi):
                dd, ii = d[i], ix[i]
                ok = dd < np.inf
                if not ok.any(): continue
                dd, ii = dd[ok], ii[ok]
                vv = sv[ii]
                merged.at[midx,col] = vv[dd==0][0] if (dd==0).any() else np.average(vv, weights=1.0/dd)
                filled += 1
    print(f"  Filled {filled}/{total}.")
    keys = merged[merged["month"]!=0].drop_duplicates(subset=["city","state"])[
        ["city","state","lat","lng","population","matched_station","elevation_ft"]]
    ann = []
    for _, ck in keys.iterrows():
        mo = merged[(merged["city"]==ck["city"])&(merged["state"]==ck["state"])&(merged["month"]!=0)]
        rec = {c:ck[c] for c in ["city","state","lat","lng","population","matched_station","elevation_ft"]}
        rec["month"] = 0
        for c in CLIMATE_COLS:
            v = mo[c].dropna()
            rec[c] = (v.sum() if c in ANNUAL_SUM else v.mean()) if len(v) else np.nan
        ann.append(rec)
    return pd.concat([merged[merged["month"]!=0], pd.DataFrame(ann)], ignore_index=True)

def calc_comfort(merged):
    print("Calculating Comfort Index...")
    t = merged["MLY-TAVG-NORMAL"].fillna(65)
    p = merged["MLY-PRCP-NORMAL"].fillna(0)
    dtr = merged.get("MLY-DUTR-NORMAL", pd.Series(20, index=merged.index)).fillna(20)
    f = merged["MLY-TMIN-AVGNDS-LSTH032"].fillna(0)
    h = merged["MLY-TMAX-AVGNDS-GRTH090"].fillna(0)

    mm = merged["month"] != 0
    ci = pd.Series(index=merged.index, dtype=float)
    # Change equations based on preferences. I prefer 50-70, colder rather than hotter, low rain, low humidity, >32F and <90F.
    cold_diff = np.maximum(0, 50 - t[mm])
    hot_diff  = np.maximum(0, t[mm] - 70)
    cold_diff_extra = np.maximum(0, 30 - t[mm])   # Extra penalty for very cold
    hot_diff_extra = np.maximum(0, t[mm] - 90)    # Extra penalty for very hot
    temp_penalty = (cold_diff * 0.8 + cold_diff**2 * 0.01 + cold_diff_extra**2 * 0.01) + (hot_diff * 0.8 + hot_diff**2 * 0.03 + hot_diff_extra**2 * 0.06)
    rain_penalty = np.maximum(0, p[mm] - 2.0) * 2.0
    muggy_penalty = np.maximum(0, t[mm] - 70) * np.maximum(0, 20 - dtr[mm]) * 0.4
    extreme_penalty = (f[mm] * 1) + (h[mm] * 2.5)

    ci[mm] = np.clip(100 - temp_penalty - rain_penalty - muggy_penalty - extreme_penalty, 0, 100)
    merged["COMFORT-INDEX"] = ci

    for (c,s), grp in merged[mm].groupby(["city","state"]):
        idx = merged[(merged["city"]==c)&(merged["state"]==s)&(merged["month"]==0)].index
        if len(idx): merged.loc[idx,"COMFORT-INDEX"] = grp["COMFORT-INDEX"].mean()
    return merged

def get_us_border():
    cache = os.path.join(os.path.dirname(CITIES_CSV), "us_border_cache.geojson")
    if os.path.exists(cache):
        print("  Loading cached US border...")
        return unary_union(gpd.read_file(cache).geometry)
    print("  Downloading US states boundary (one-time)...")
    for url in ["https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
                "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"]:
        try:
            gdf = gpd.read_file(url); print("    OK")
            us = unary_union(gdf.geometry)
            try: gpd.GeoDataFrame(geometry=[us],crs=gdf.crs).to_file(cache,driver="GeoJSON")
            except: pass
            return us
        except: continue
    from shapely.geometry import box
    return unary_union([box(-125,24.5,-66.5,49.5),box(-180,51,-130,72),box(-161,18.5,-154,22.5)])

def build_voronoi(lats, lngs, us_border):
    print("Building Voronoi...")
    pts = MultiPoint([(lng,lat) for lng,lat in zip(lngs,lats)])
    vor = voronoi_diagram(pts, envelope=us_border.buffer(2).envelope)
    pl = list(zip(lngs,lats)); feats = []; matched = set()
    for poly in vor.geoms:
        cl = poly.intersection(us_border)
        if cl.is_empty: continue
        for i,(lng,lat) in enumerate(pl):
            if i in matched: continue
            if poly.contains(Point(lng,lat)):
                cl = cl.simplify(0.01, preserve_topology=True)
                g = mapping(cl)
                if g["type"] not in ("Polygon","MultiPolygon"):
                    if g["type"]=="GeometryCollection":
                        ps = [x for x in shape(g).geoms if x.geom_type in ("Polygon","MultiPolygon")]
                        if ps: g = mapping(unary_union(ps))
                        else: continue
                    else: continue
                feats.append({"type":"Feature","id":str(i),"geometry":g,"properties":{"i":i}})
                matched.add(i); break
    print(f"  {len(feats)} cells.")
    return {"type":"FeatureCollection","features":feats}

# ═══════════════════════════════════════════════════════════════════════════
# COMPACT JSON
# ═══════════════════════════════════════════════════════════════════════════

def rnd(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)): return None
    return round(v, d)

def build_json_data(merged):
    print("Building compact JSON...")
    cities_json = []
    annual = merged[merged["month"]==0].reset_index(drop=True)
    for i, (_, ar) in enumerate(annual.iterrows()):
        obj = {"c":ar["city"],"s":ar["state"],"la":round(ar["lat"],4),"lo":round(ar["lng"],4),
               "p":int(ar["population"]),"el":rnd(ar["elevation_ft"],0),"m":{},"a":{}}
        for col, key in KEY_MAP.items(): obj["a"][key] = rnd(ar[col])
        obj["a"]["ci"] = rnd(ar.get("COMFORT-INDEX", np.nan))
        monthly = merged[(merged["city"]==ar["city"])&(merged["state"]==ar["state"])&
                         (merged["month"]>=1)&(merged["month"]<=12)].sort_values("month")
        for _, mr in monthly.iterrows():
            md = {}
            for col, key in KEY_MAP.items(): md[key] = rnd(mr[col])
            md["ci"] = rnd(mr.get("COMFORT-INDEX", np.nan))
            obj["m"][str(int(mr["month"]))] = md
        cities_json.append(obj)
    print(f"  {len(cities_json)} cities serialized.")
    return cities_json

# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>U.S. Climate Normals Map (1991-2020)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
:root{
  --red:#d32f2f;
  --purple:#7b1fa2;
  --bg:#f4f5f7;
  --card:#fff;
  --border:#e0e0e0;
  --text:#2c3e50;
  --muted:#888;
}
*{margin:0;padding:0;box-sizing:border-box}

/* Desktop stays app-like; mobile gets natural page scrolling */
html,body{
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
  background:var(--bg);
  min-height:100%;
}
body{color:var(--text)}

/* ── DESKTOP: side-by-side layout ── */
#app{
  display:flex;
  flex-direction:column;
  min-height:100vh;
}
#header{
  text-align:center;
  padding:4px 0 2px;
  background:var(--card);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
#header h1{
  font-size:19px;
  color:var(--text);
  margin:0;
}
#controls{
  display:flex;
  justify-content:center;
  align-items:flex-end;
  gap:12px;
  padding:4px 12px;
  background:var(--card);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
  flex-wrap:wrap;
}
.ctrl{display:flex;flex-direction:column}
.ctrl label{
  font-size:11px;
  font-weight:600;
  color:#555;
  margin-bottom:1px;
}
.ctrl select,.ctrl input{
  font-size:13px;
  padding:4px 8px;
  border:1px solid #ccc;
  border-radius:5px;
  background:#fff;
}
.ctrl select{cursor:pointer}
.ctrl input{width:170px}
.city-a{color:var(--red)}
.city-b{color:var(--purple)}

.btn-row{
  display:flex;
  align-items:center;
  gap:3px;
}
.btn-row button{
  background:none;
  border:none;
  cursor:pointer;
  font-size:20px;
  padding:2px 5px;
  border-radius:4px;
  line-height:1;
}
.btn-row button:hover{background:#eee}

#body{
  display:flex;
  flex:1;
  min-height:0;
}
#map-col{
  flex:3;
  position:relative;
  min-width:0;
  min-height:0;
}
#map{
  height:100%;
  width:100%;
}
#legend{
  position:absolute;
  bottom:20px;
  left:10px;
  background:rgba(255,255,255,0.94);
  padding:8px 12px;
  border-radius:8px;
  font-size:11px;
  box-shadow:0 2px 6px rgba(0,0,0,0.18);
  z-index:800;
  pointer-events:none;
}
.grad{
  height:12px;
  width:180px;
  border-radius:2px;
  margin:4px 0;
}
.legend-labels{
  display:flex;
  justify-content:space-between;
  font-size:10px;
  color:#555;
}

/* ── Right panel (desktop: always visible) ── */
#panel{
  width:380px;
  flex-shrink:0;
  display:flex;
  flex-direction:column;
  background:#fafafa;
  border-left:1px solid var(--border);
  overflow-y:auto;
}
#panel-title{
  text-align:center;
  font-weight:700;
  font-size:13px;
  color:var(--text);
  padding:8px 8px 4px;
  min-height:20px;
  line-height:1.3;
}
#chart-wrap{
  flex:2;
  min-height:520px;
}
#histo-wrap{
  flex:1;
  min-height:155px;
  margin:0 6px 8px;
  background:#f0f1f3;
  border-radius:6px;
}
#stats{
  text-align:center;
  font-size:11px;
  color:#666;
  padding:2px 0;
  background:var(--card);
  border-top:1px solid #eee;
  flex-shrink:0;
}
#footer{
  text-align:center;
  font-size:10px;
  color:#999;
  padding:3px 0;
  background:var(--card);
  border-top:1px solid var(--border);
  flex-shrink:0;
}
#footer a{color:#777}

/* ── Mobile toggle buttons ── */
#mobile-view-switch{
  display:none;
  gap:8px;
  padding:8px;
  background:var(--card);
  border-bottom:1px solid var(--border);
}
#mobile-view-switch button{
  flex:1;
  border:1px solid #ccc;
  background:#fff;
  border-radius:8px;
  padding:8px 10px;
  font-size:13px;
  font-weight:600;
  cursor:pointer;
}
#mobile-view-switch button.active{
  background:#2c3e50;
  color:#fff;
  border-color:#2c3e50;
}

/* ── MOBILE: page scrolls naturally ── */
@media(max-width:900px){
  html,body{
    height:auto;
    overflow:auto;
  }

  #app{
    display:block;
    min-height:auto;
  }

  #mobile-view-switch{
    display:flex;
    position:sticky;
    top:0;
    z-index:950;
  }

  #body{
    display:block;
    height:auto;
    min-height:0;
    overflow:visible;
  }

  /* default mobile mode = stacked scrollable page */
  #map-col{
    width:100%;
    height:55vh;
    min-height:280px;
  }

  #map{
    height:100%;
    width:100%;
  }

  #panel{
    width:100%;
    display:block;
    border-left:none;
    border-top:1px solid var(--border);
    overflow:visible;
  }

  #chart-wrap{
    min-height:360px;
  }

  #histo-wrap{
    min-height:180px;
    margin:0 8px 10px;
  }

  #controls{
    gap:8px;
    padding:6px 8px;
  }

  .ctrl input{width:140px}
  .ctrl select,.ctrl input{
    font-size:12px;
    padding:3px 6px;
  }

  /* optional one-panel-at-a-time mode on mobile */
  body.mobile-show-map #map-col{display:block}
  body.mobile-show-map #panel{display:none}

  body.mobile-show-chart #map-col{display:none}
  body.mobile-show-chart #panel{display:block}
}

@media(max-width:500px){
  #controls{
    flex-direction:column;
    align-items:stretch;
    gap:4px;
  }
  .ctrl{width:100%}
  .ctrl input,.ctrl select{width:100%}
  #header h1{font-size:16px}
  #map-col{height:46vh;min-height:240px}
}
</style>
</head>
<body class="mobile-show-map">

<div id="app">
  <div id="header"><h1>U.S. Climate Normals Map (1991–2020)</h1></div>

  <div id="controls">
    <div class="ctrl">
      <label>Variable</label>
      <select id="varSel"></select>
    </div>
    <div class="ctrl">
      <label>Period</label>
      <select id="moSel"></select>
    </div>
    <div class="ctrl">
      <label>Units</label>
      <select id="unitSel">
        <option value="imp">°F / in / ft</option>
        <option value="met">°C / mm / m</option>
      </select>
    </div>
    <div class="ctrl">
      <label class="city-a">City A</label>
      <div class="btn-row">
        <input id="searchA" list="cityList" placeholder="Click map or type...">
        <button id="clearA" title="Clear A" style="color:var(--red)">×</button>
        <button id="swapBtn" title="Swap A↔B">⇄</button>
      </div>
    </div>
    <div class="ctrl">
      <label class="city-b">City B (compare)</label>
      <div class="btn-row">
        <input id="searchB" list="cityList" placeholder="Optional...">
        <button id="clearB" title="Clear B" style="color:var(--purple)">×</button>
      </div>
    </div>
    <div class="ctrl">
      <label>Click → </label>
      <select id="clickMode" style="width:50px">
        <option value="A">A</option>
        <option value="B">B</option>
      </select>
    </div>
    <datalist id="cityList"></datalist>
  </div>

  <div id="mobile-view-switch">
    <button id="showMapBtn" class="active" type="button">Map</button>
    <button id="showChartBtn" type="button">Charts</button>
  </div>

  <div id="body">
    <div id="map-col">
      <div id="map"></div>
      <div id="legend"></div>
    </div>
    <div id="panel">
      <div id="panel-title">Click a region or search above</div>
      <div id="chart-wrap"></div>
      <div id="histo-wrap"></div>
    </div>
  </div>

  <div id="stats"></div>
  <div id="footer">
    Data: <a href="https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals" target="_blank">NOAA 1991–2020 Climate Normals</a>
    · City database: <a href="https://simplemaps.com/data/us-cities" target="_blank">SimpleMaps</a>
  </div>
</div>

<script>
const CITIES = %%CITIES_JSON%%;
const VORONOI = %%VORONOI_JSON%%;

const VARS = [
  {k:"tavg",  label:"Mean Temp (°F)",          labelM:"Mean Temp (°C)",          scale:"RdBu_r",  convF:v=>(v-32)*5/9},
  {k:"tmax",  label:"Mean Max Temp (°F)",      labelM:"Mean Max Temp (°C)",      scale:"RdBu_r",  convF:v=>(v-32)*5/9},
  {k:"tmin",  label:"Mean Min Temp (°F)",      labelM:"Mean Min Temp (°C)",      scale:"RdBu_r",  convF:v=>(v-32)*5/9},
  {k:"dutr",  label:"Diurnal Range (°F)",      labelM:"Diurnal Range (°C)",      scale:"Oranges", convF:v=>v*5/9},
  {k:"prcp",  label:"Precipitation (in)",      labelM:"Precipitation (mm)",      scale:"Blues",   convF:v=>v*25.4},
  {k:"snow",  label:"Snowfall (in)",           labelM:"Snowfall (mm)",           scale:"ice_r",   convF:v=>v*25.4},
  {k:"htdd",  label:"Heating Degree Days",     labelM:"Heating Degree Days",     scale:"Blues",   convF:null},
  {k:"cldd",  label:"Cooling Degree Days",     labelM:"Cooling Degree Days",     scale:"Reds",    convF:null},
  {k:"frost", label:"Freezing Days (min≤32°F)",labelM:"Freezing Days (min≤0°C)", scale:"ice_r",   convF:null},
  {k:"hot90", label:"Days ≥90°F",              labelM:"Days ≥32°C",              scale:"YlOrRd",  convF:null},
  {k:"pdays", label:"Precip Days (≥0.10 in)",  labelM:"Precip Days (≥2.5mm)",    scale:"Blues",   convF:null},
  {k:"sdays", label:"Snow Days (≥1.0 in)",     labelM:"Snow Days (≥25mm)",       scale:"ice_r",   convF:null},
  {k:"scov",  label:"Snow Cover Days",         labelM:"Snow Cover Days",         scale:"ice_r",   convF:null},
  {k:"ci",    label:"Comfort Index (0-100)",   labelM:"Comfort Index (0-100)",   scale:"RdYlGn",  convF:null},
  {k:"el",    label:"Elevation (ft)",          labelM:"Elevation (m)",           scale:"YlOrBr",  convF:v=>v*0.3048, isElev:true},
];
const MONTHS = ["Annual","January","February","March","April","May","June",
                "July","August","September","October","November","December"];
const SM = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

const SCALES = {
  "RdBu_r":  [[0,"#053061"],[.1,"#2166ac"],[.2,"#4393c3"],[.3,"#92c5de"],[.4,"#d1e5f0"],
              [.5,"#f7f7f7"],[.6,"#fddbc7"],[.7,"#f4a582"],[.8,"#d6604d"],[.9,"#b2182b"],[1,"#67001f"]],
  "Blues":   [[0,"#f7fbff"],[.25,"#c6dbef"],[.5,"#6baed6"],[.75,"#2171b5"],[1,"#08306b"]],
  "Reds":    [[0,"#fff5f0"],[.25,"#fcbba1"],[.5,"#fb6a4a"],[.75,"#cb181d"],[1,"#67000d"]],
  "Oranges": [[0,"#fff5eb"],[.25,"#fdd49e"],[.5,"#fdae6b"],[.75,"#e6550d"],[1,"#8c2d04"]],
  "YlOrRd":  [[0,"#ffffcc"],[.25,"#feb24c"],[.5,"#fd8d3c"],[.75,"#e31a1c"],[1,"#800026"]],
  "YlOrBr":  [[0,"#ffffe5"],[.25,"#fee391"],[.5,"#fe9929"],[.75,"#cc4c02"],[1,"#662506"]],
  "ice_r":   [[0,"#08306b"],[.25,"#2171b5"],[.5,"#6baed6"],[.75,"#c6dbef"],[1,"#f7fbff"]],
  "RdYlGn":  [[0,"#a50026"],[.25,"#f46d43"],[.5,"#ffffbf"],[.75,"#66bd63"],[1,"#006837"]],
  "Viridis": [[0,"#440154"],[.25,"#31688e"],[.5,"#35b779"],[.75,"#90d743"],[1,"#fde725"]],
};

function lerpColor(scale,t){
  t=Math.max(0,Math.min(1,t));
  let S=SCALES[scale]||SCALES.Viridis;
  for(let i=0;i<S.length-1;i++){
    if(t>=S[i][0]&&t<=S[i+1][0]){
      let f=(t-S[i][0])/(S[i+1][0]-S[i][0]),a=S[i][1],b=S[i+1][1];
      let r1=parseInt(a.slice(1,3),16),g1=parseInt(a.slice(3,5),16),b1=parseInt(a.slice(5,7),16);
      let r2=parseInt(b.slice(1,3),16),g2=parseInt(b.slice(3,5),16),b2=parseInt(b.slice(5,7),16);
      return `rgb(${Math.round(r1+f*(r2-r1))},${Math.round(g1+f*(g2-g1))},${Math.round(b1+f*(b2-b1))})`;
    }
  }
  return S[S.length-1][1];
}

/* ── State ── */
let map, geoLayer, cityA=null, cityB=null;

/* ── Helpers ── */
const $=id=>document.getElementById(id);
function gv(){return VARS[$("varSel").selectedIndex]}
function gm(){return +$("moSel").value}
function met(){return $("unitSel").value==="met"}
function isMobile(){return window.innerWidth<=900}

function val(c,vd,mo){
  let v;
  if(vd.isElev) v=c.el;
  else if(mo===0) v=c.a[vd.k];
  else v=c.m[String(mo)] ? c.m[String(mo)][vd.k] : null;
  if(v==null) return null;
  if(met() && vd.convF) v=vd.convF(v);
  return v;
}
function lab(vd){return met()?vd.labelM:vd.label}
function elev(c){
  if(c.el==null) return "N/A";
  return met() ? `${Math.round(c.el*.3048).toLocaleString()} m`
               : `${Math.round(c.el).toLocaleString()} ft`;
}
function ck(c){return `${c.c}, ${c.s}`}
function findCity(k){return k ? CITIES.find(c=>ck(c)===k) || null : null}

/* ── Mobile view switch ── */
function setMobileView(mode){
  if(!isMobile()) return;
  document.body.classList.remove("mobile-show-map","mobile-show-chart");
  document.body.classList.add(mode==="chart" ? "mobile-show-chart" : "mobile-show-map");
  $("showMapBtn").classList.toggle("active", mode!=="chart");
  $("showChartBtn").classList.toggle("active", mode==="chart");

  setTimeout(()=>{
    if(map) map.invalidateSize();
    window.dispatchEvent(new Event("resize"));
  }, 50);
}

/* ── Init ── */
function init(){
  VARS.forEach((v,i)=>{
    let o=document.createElement("option");
    o.value=i;
    o.textContent=v.label;
    $("varSel").appendChild(o);
  });

  MONTHS.forEach((m,i)=>{
    let o=document.createElement("option");
    o.value=i;
    o.textContent=m;
    $("moSel").appendChild(o);
  });

  let dl=$("cityList");
  CITIES.forEach(c=>{
    let o=document.createElement("option");
    o.value=ck(c);
    dl.appendChild(o);
  });

  map=L.map("map",{zoomControl:true}).setView([39.5,-98.5],4);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{
    maxZoom:18,
    subdomains:"abcd"
  }).addTo(map);

  geoLayer=L.geoJSON(VORONOI,{
    style:()=>({fillColor:"#ccc",fillOpacity:.8,weight:.5,color:"rgba(255,255,255,.5)"}),
    onEachFeature:(feat,layer)=>{
      let idx=feat.properties.i;
      if(idx==null || idx>=CITIES.length) return;

      layer.on("click",()=>{
        let m=$("clickMode").value;
        if(m==="A"){
          cityA=CITIES[idx];
          $("searchA").value=ck(CITIES[idx]);
        }else{
          cityB=CITIES[idx];
          $("searchB").value=ck(CITIES[idx]);
        }
        updateAll();

        if(isMobile()){
          setMobileView("chart");
          document.getElementById("panel").scrollIntoView({behavior:"smooth", block:"start"});
        }
      });

      layer.on("mouseover",e=>e.target.setStyle({weight:2,color:"#333"}));
      layer.on("mouseout",e=>{
        geoLayer.resetStyle(e.target);
        paintMap();
      });
    }
  }).addTo(map);

  ["varSel","moSel","unitSel"].forEach(id=>$(id).addEventListener("change",updateAll));

  $("searchA").addEventListener("change",e=>{
    cityA=findCity(e.target.value);
    updateAll();
  });

  $("searchB").addEventListener("change",e=>{
    cityB=findCity(e.target.value);
    updateAll();
  });

  $("clearA").addEventListener("click",()=>{
    cityA=null;
    $("searchA").value="";
    updateAll();
  });

  $("clearB").addEventListener("click",()=>{
    cityB=null;
    $("searchB").value="";
    updateAll();
  });

  $("swapBtn").addEventListener("click",()=>{
    [cityA,cityB]=[cityB,cityA];
    $("searchA").value=cityA?ck(cityA):"";
    $("searchB").value=cityB?ck(cityB):"";
    updateAll();
  });

  $("showMapBtn").addEventListener("click",()=>setMobileView("map"));
  $("showChartBtn").addEventListener("click",()=>setMobileView("chart"));

  window.addEventListener("resize",()=>{
    if(map) map.invalidateSize();
    if(!isMobile()){
      document.body.classList.remove("mobile-show-map","mobile-show-chart");
      $("showMapBtn").classList.add("active");
      $("showChartBtn").classList.remove("active");
    }else if(
      !document.body.classList.contains("mobile-show-map") &&
      !document.body.classList.contains("mobile-show-chart")
    ){
      setMobileView("map");
    }
  });

  if(isMobile()) setMobileView("map");
  updateAll();
}

function updateAll(){
  paintMap();
  paintChart();
  paintHisto();
}

/* ── Map ── */
function paintMap(){
  let vd=gv(), mo=gm();
  let vals=CITIES.map(c=>val(c,vd,mo)).filter(v=>v!=null);
  if(!vals.length) return;

  vals.sort((a,b)=>a-b);
  let lo=vals[Math.floor(vals.length*.02)],
      hi=vals[Math.ceil(vals.length*.98)-1];
  if(lo===hi){lo-=1;hi+=1}
  let rng=hi-lo;

  geoLayer.eachLayer(ly=>{
    let idx=ly.feature.properties.i;
    if(idx==null || idx>=CITIES.length) return;

    let v=val(CITIES[idx],vd,mo);
    let fc=v!=null ? lerpColor(vd.scale,(v-lo)/rng) : "#ddd";
    let isA=cityA && idx===CITIES.indexOf(cityA);
    let isB=cityB && idx===CITIES.indexOf(cityB);

    ly.setStyle({
      fillColor:fc,
      fillOpacity:.82,
      weight:isA||isB?3:.5,
      color:isA?"var(--red)":isB?"var(--purple)":"rgba(255,255,255,.5)"
    });

    let c=CITIES[idx], vs=v!=null ? v.toFixed(1) : "N/A";
    ly.bindTooltip(
      `<b>${c.c}, ${c.s}</b><br>${lab(vd)}: ${vs}<br>Elev: ${elev(c)}<br>Pop: ${c.p.toLocaleString()}`,
      {sticky:true}
    );
  });

  let stops=SCALES[vd.scale]||SCALES.Viridis;
  let grad=stops.map(s=>s[1]).join(",");
  $("legend").innerHTML = `
    <div style="font-weight:600;margin-bottom:2px">${lab(vd)} — ${MONTHS[mo]}</div>
    <div class="grad" style="background:linear-gradient(to right,${grad})"></div>
    <div class="legend-labels"><span>${lo.toFixed(1)}</span><span>${hi.toFixed(1)}</span></div>
  `;

  let mean=vals.reduce((a,b)=>a+b,0)/vals.length;
  let med=vals[Math.floor(vals.length/2)];
  $("stats").textContent =
    `${vals.length} cities · Min: ${vals[0].toFixed(1)} · Mean: ${mean.toFixed(1)} · Median: ${med.toFixed(1)} · Max: ${vals[vals.length-1].toFixed(1)}`;
}

/* ── Chart ── */
function paintChart(){
  let u=met(), mo=gm();

  if(!cityA && !cityB){
    Plotly.react("chart-wrap",[],{
      paper_bgcolor:"#fafafa",
      plot_bgcolor:"#fafafa",
      xaxis:{visible:false},
      yaxis:{visible:false},
      annotations:[{
        text:"Click a region or search",
        xref:"paper",yref:"paper",x:.5,y:.5,
        showarrow:false,font:{size:14,color:"#aaa"}
      }],
      margin:{l:10,r:10,t:10,b:10},
      height:isMobile()?340:400
    },{responsive:true});

    $("panel-title").textContent="Click a region or search above";
    return;
  }

  let traces=[], titles=[], cmp=cityA&&cityB;
  let cA=["#d32f2f","#f57c00","#1976d2"], cB=["#7b1fa2","#388e3c","#00acc1"];

  function add(city,col,pfx,dash){
    let mx=[], av=[], mn=[], fr=[], ht=[];
    for(let m=1;m<=12;m++){
      let d=city.m[String(m)]||{};
      let a=d.tmax,b=d.tavg,c=d.tmin;
      if(u&&a!=null)a=(a-32)*5/9;
      if(u&&b!=null)b=(b-32)*5/9;
      if(u&&c!=null)c=(c-32)*5/9;
      mx.push(a); av.push(b); mn.push(c); fr.push(d.frost); ht.push(d.hot90);
    }
    let tu=u?"°C":"°F", nm=pfx ? pfx+" " : "";
    traces.push({x:SM,y:mx,name:`${nm}Max (${tu})`,type:"scatter",mode:"lines+markers",
      line:{color:col[0],width:2.5,dash:dash||"solid"},marker:{size:4},xaxis:"x",yaxis:"y",legendgroup:pfx});
    traces.push({x:SM,y:av,name:`${nm}Mean (${tu})`,type:"scatter",mode:"lines+markers",
      line:{color:col[1],width:2.5,dash:"dot"},marker:{size:4},xaxis:"x",yaxis:"y",legendgroup:pfx});
    traces.push({x:SM,y:mn,name:`${nm}Min (${tu})`,type:"scatter",mode:"lines+markers",
      line:{color:col[2],width:2.5,dash:dash||"solid"},marker:{size:4},xaxis:"x",yaxis:"y",legendgroup:pfx});
    traces.push({x:SM,y:fr,name:`${nm}Freeze`,type:"bar",
      marker:{color:col[2],opacity:.7},xaxis:"x2",yaxis:"y2",legendgroup:pfx});
    traces.push({x:SM,y:ht,name:`${nm}90°F+`,type:"bar",
      marker:{color:col[0],opacity:.6},xaxis:"x2",yaxis:"y2",legendgroup:pfx});
    titles.push(`${city.c}, ${city.s} (${elev(city)})`);
  }

  if(cityA) add(cityA,cA,cmp?cityA.c:"",null);
  if(cityB) add(cityB,cB,cmp?cityB.c:"",cmp?"dash":null);

  let shapes=[];
  if(mo>=1){
    shapes.push({
      type:"line",
      x0:SM[mo-1],x1:SM[mo-1],y0:0,y1:1,yref:"paper",
      line:{color:"rgba(0,0,0,.35)",width:2,dash:"dot"}
    });
  }

  let yr=u?[-29,49]:[-20,120];

  Plotly.react("chart-wrap",traces,{
    grid:{rows:2,columns:1,pattern:"independent",roworder:"top to bottom",ygap:.18},
    xaxis:{anchor:"y"},
    yaxis:{title:u?"°C":"°F",range:yr,anchor:"x",domain:[.42,.87]},
    xaxis2:{anchor:"y2"},
    yaxis2:{title:"Days",anchor:"x2",domain:[0,.3]},
    paper_bgcolor:"#fafafa",
    plot_bgcolor:"#fff",
    margin:{l:44,r:8,t:50,b:28},
    legend:{font:{size:8},orientation:"h",y:1,x:.5,xanchor:"center",yanchor:"bottom"},
    barmode:"group",
    font:{size:10},
    shapes:shapes,
    height:isMobile()?360:520
  },{responsive:true});

  $("panel-title").textContent = cmp ? titles.join("  vs  ") : (titles[0]||"");
}

/* ── Histogram ── */
function paintHisto(){
  let vd=gv(), mo=gm();
  let vals=CITIES.map(c=>val(c,vd,mo)).filter(v=>v!=null);

  let traces=[{
    x:vals,
    type:"histogram",
    nbinsx:50,
    marker:{color:"#90caf9",line:{width:1,color:"white"}},
    showlegend:false
  }];

  let shapes=[];
  [[cityA,"#d32f2f"],[cityB,"#7b1fa2"]].forEach(([c,col])=>{
    if(!c) return;
    let v=val(c,vd,mo);
    if(v!=null){
      shapes.push({
        type:"line",
        x0:v,x1:v,y0:0,y1:1,yref:"paper",
        line:{color:col,width:3,dash:"dash"}
      });
    }
  });

  Plotly.react("histo-wrap",traces,{
    title:{text:`Distribution: ${lab(vd)}`,font:{size:11,color:"#2c3e50"}},
    margin:{l:10,r:10,t:26,b:20},
    paper_bgcolor:"#f0f1f3",
    plot_bgcolor:"#f0f1f3",
    xaxis:{title:lab(vd),titlefont:{size:10},tickfont:{size:9}},
    yaxis:{visible:false},
    shapes:shapes,
    height:isMobile()?180:155
  },{responsive:true});
}

document.addEventListener("DOMContentLoaded",init);
</script>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    stations = load_stations(CLIMATE_DIR)
    cities = load_and_filter_cities(CITIES_CSV, target=2000)
    cities = match_cities(cities, stations)
    merged = build_merged(cities, stations)
    merged = interpolate_missing(merged, stations)
    merged = calc_comfort(merged)

    us_border = get_us_border()
    ann = merged[merged["month"]==0].reset_index(drop=True)
    vor_gj = build_voronoi(ann["lat"].values, ann["lng"].values, us_border)

    # Cache processed data
    base = os.path.dirname(CITIES_CSV)
    try:
        merged.to_parquet(os.path.join(base, "climate_merged.parquet"))
        with open(os.path.join(base, "voronoi_cells.geojson"), "w") as f:
            json.dump(vor_gj, f)
        print("  Cached parquet + geojson.")
    except Exception as e:
        print(f"  Cache write failed: {e}")

    cities_json = build_json_data(merged)

    print("Writing HTML...")
    html_out = HTML_TEMPLATE.replace("%%CITIES_JSON%%", json.dumps(cities_json, separators=(",",":")))
    html_out = html_out.replace("%%VORONOI_JSON%%", json.dumps(vor_gj, separators=(",",":")))

    out_dir = os.path.dirname(OUTPUT_HTML or CITIES_CSV)
    out_path = OUTPUT_HTML or os.path.join(out_dir, "climate_map.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  Generated: {out_path}")
    print(f"  Size: {size_mb:.1f} MB  |  Cities: {len(cities_json)}  |  Cells: {len(vor_gj['features'])}")
    print(f"\n  Double-click the HTML file to open in your browser.")
