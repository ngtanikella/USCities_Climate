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
CLIMATE_DIR   = r"C:\Users\nag55\Downloads\Climate\Multi"
CITIES_CSV    = r"C:\Users\nag55\Downloads\Climate\uscities.csv"
CITY_SPACING  = 0.75        # ±degrees for deduplication (0.75→~896 cities)
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

# Short keys for compact JSON
KEY_MAP = {
    "MLY-TAVG-NORMAL":"tavg","MLY-TMAX-NORMAL":"tmax","MLY-TMIN-NORMAL":"tmin",
    "MLY-DUTR-NORMAL":"dutr","MLY-PRCP-NORMAL":"prcp","MLY-SNOW-NORMAL":"snow",
    "MLY-HTDD-NORMAL":"htdd","MLY-CLDD-NORMAL":"cldd",
    "MLY-TMIN-AVGNDS-LSTH032":"frost","MLY-TMAX-AVGNDS-GRTH090":"hot90",
    "MLY-PRCP-AVGNDS-GE010HI":"pdays","MLY-SNOW-AVGNDS-GE010TI":"sdays",
    "MLY-SNWD-AVGNDS-GE001WI":"scov",
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA PROCESSING (reused from v8)
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
    # Recompute annual
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
    f = merged["MLY-TMIN-AVGNDS-LSTH032"].fillna(0)
    h = merged["MLY-TMAX-AVGNDS-GRTH090"].fillna(0)
    mm = merged["month"] != 0
    ci = pd.Series(index=merged.index, dtype=float)
    ci[mm] = np.clip(100 - np.maximum(0,50-t[mm])*3 - np.maximum(0,t[mm]-80)*3 - p[mm]*5 - (f[mm]+h[mm])*2, 0, 100)
    merged["COMFORT-INDEX"] = ci
    for (c,s), grp in merged[mm].groupby(["city","state"]):
        idx = merged[(merged["city"]==c)&(merged["state"]==s)&(~mm)].index
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
            gdf = gpd.read_file(url); print(f"    OK")
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
# CONVERT TO COMPACT JSON
# ═══════════════════════════════════════════════════════════════════════════

def rnd(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)): return None
    return round(v, d)

def build_json_data(merged):
    """Convert merged DataFrame into compact JS-friendly structure."""
    print("Building compact JSON...")
    cities_json = []
    annual = merged[merged["month"]==0].reset_index(drop=True)

    for i, (_, ar) in enumerate(annual.iterrows()):
        city_obj = {
            "c": ar["city"], "s": ar["state"],
            "la": round(ar["lat"],4), "lo": round(ar["lng"],4),
            "p": int(ar["population"]),
            "el": rnd(ar["elevation_ft"],0),
            "m": {},  # monthly data
            "a": {},  # annual data
        }
        # Annual
        for col, key in KEY_MAP.items():
            city_obj["a"][key] = rnd(ar[col])
        city_obj["a"]["ci"] = rnd(ar.get("COMFORT-INDEX", np.nan))

        # Monthly (1-12)
        monthly = merged[(merged["city"]==ar["city"])&(merged["state"]==ar["state"])&
                         (merged["month"]>=1)&(merged["month"]<=12)].sort_values("month")
        for _, mr in monthly.iterrows():
            mo = int(mr["month"])
            md = {}
            for col, key in KEY_MAP.items():
                md[key] = rnd(mr[col])
            md["ci"] = rnd(mr.get("COMFORT-INDEX", np.nan))
            city_obj["m"][str(mo)] = md

        cities_json.append(city_obj)

    print(f"  {len(cities_json)} cities serialized.")
    return cities_json

# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>U.S. Climate Normals Map (1991–2020)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f4f5f7;height:100vh;overflow:hidden;display:flex;flex-direction:column}
#header{text-align:center;padding:3px 0 2px;background:#fff;border-bottom:1px solid #e0e0e0}
#header h1{font-size:20px;color:#2c3e50;margin:0}
#header .sub{font-size:11px;color:#888;margin:2px 0 0}
#controls{display:flex;justify-content:center;align-items:flex-end;gap:14px;padding:4px 16px;
  background:#fff;border-bottom:1px solid #e0e0e0;flex-wrap:wrap}
.ctrl{display:flex;flex-direction:column}
.ctrl label{font-size:11px;font-weight:600;color:#555;margin-bottom:2px}
.ctrl select,.ctrl input{font-size:13px;padding:4px 8px;border:1px solid #ccc;border-radius:4px;background:#fff}
.ctrl select{cursor:pointer} .ctrl input{width:180px}
#city-btns{display:flex;align-items:center;gap:4px}
#city-btns button{background:none;border:none;cursor:pointer;font-size:18px;padding:2px 4px;border-radius:4px}
#city-btns button:hover{background:#eee}
#main{display:flex;flex:1;overflow:hidden}
#map-wrap{flex:3;position:relative}
#map{height:100%;width:100%}
#panel{flex:1;min-width:320px;max-width:420px;display:flex;flex-direction:column;
  background:#fafafa;border-left:1px solid #e0e0e0;overflow-y:auto}
#panel-title{text-align:center;font-weight:700;font-size:13px;color:#2c3e50;
  padding:8px 8px 4px;min-height:24px;line-height:1.4}
#chart{flex:1;min-height:320px}
#histogram{height:160px;margin:4px 8px 8px;background:#f0f1f3;border-radius:6px}
#footer{text-align:center;font-size:10px;color:#999;padding:3px 0;background:#fff;border-top:1px solid #e0e0e0}
#footer a{color:#777}
#stats{text-align:center;font-size:11px;color:#666;padding:2px 0;background:#fff;border-top:1px solid #eee}
.legend{position:absolute;bottom:30px;left:10px;background:rgba(255,255,255,0.92);
  padding:8px 12px;border-radius:6px;font-size:11px;box-shadow:0 1px 4px rgba(0,0,0,0.2);z-index:1000}
.legend .grad{height:12px;width:180px;border-radius:2px;margin:4px 0}
.legend .labels{display:flex;justify-content:space-between;font-size:10px;color:#555}

/* Datalist/search styling */
#searchA,#searchB{width:170px}
.city-a{color:#d32f2f} .city-b{color:#7b1fa2}
</style>
</head>
<body>

<div id="header">
  <h1>U.S. Climate Normals Map (1991–2020)</h1>
</div>

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
    <div id="city-btns">
      <input id="searchA" list="cityList" placeholder="Click map or type...">
      <button id="clearA" title="Clear A" class="city-a">×</button>
      <button id="swapBtn" title="Swap A↔B">⇄</button>
    </div>
  </div>
  <div class="ctrl">
    <label class="city-b">City B (compare)</label>
    <div id="city-btns">
      <input id="searchB" list="cityList" placeholder="Optional...">
      <button id="clearB" title="Clear B" class="city-b">×</button>
    </div>
  </div>
  <div class="ctrl">
    <label>Click assigns to</label>
    <select id="clickMode"><option value="A">A</option><option value="B">B</option></select>
  </div>
  <datalist id="cityList"></datalist>
</div>

<div id="main">
  <div id="map-wrap">
    <div id="map"></div>
    <div class="legend" id="legend"></div>
  </div>
  <div id="panel">
    <div id="panel-title">Click a region or search above</div>
    <div id="chart"></div>
    <div id="histogram"></div>
  </div>
</div>

<div id="stats"></div>
<div id="footer">
  Data: <a href="https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals" target="_blank">NOAA 1991–2020 Climate Normals</a>
  · City database: <a href="https://simplemaps.com/data/us-cities" target="_blank">SimpleMaps</a>
</div>

<script>
// ═══ EMBEDDED DATA (injected by build script) ═══
const CITIES = %%CITIES_JSON%%;
const VORONOI = %%VORONOI_JSON%%;

// ═══ VARIABLE DEFINITIONS ═══
const VARS = [
  {k:"tavg",  label:"Mean Temp (°F)",         labelM:"Mean Temp (°C)",         scale:"RdBu_r", convF:v=>(v-32)*5/9},
  {k:"tmax",  label:"Mean Max Temp (°F)",      labelM:"Mean Max Temp (°C)",      scale:"RdBu_r", convF:v=>(v-32)*5/9},
  {k:"tmin",  label:"Mean Min Temp (°F)",      labelM:"Mean Min Temp (°C)",      scale:"RdBu_r", convF:v=>(v-32)*5/9},
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
const SMONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

// ═══ COLOR SCALES ═══
// Pre-built color arrays for each named scale
const SCALES = {
  "RdBu_r":   [[0,"#053061"],[0.1,"#2166ac"],[0.2,"#4393c3"],[0.3,"#92c5de"],[0.4,"#d1e5f0"],
               [0.5,"#f7f7f7"],[0.6,"#fddbc7"],[0.7,"#f4a582"],[0.8,"#d6604d"],[0.9,"#b2182b"],[1,"#67001f"]],
  "Blues":    [[0,"#f7fbff"],[0.25,"#c6dbef"],[0.5,"#6baed6"],[0.75,"#2171b5"],[1,"#08306b"]],
  "Reds":     [[0,"#fff5f0"],[0.25,"#fcbba1"],[0.5,"#fb6a4a"],[0.75,"#cb181d"],[1,"#67000d"]],
  "Oranges":  [[0,"#fff5eb"],[0.25,"#fdd49e"],[0.5,"#fdae6b"],[0.75,"#e6550d"],[1,"#8c2d04"]],
  "YlOrRd":   [[0,"#ffffcc"],[0.25,"#feb24c"],[0.5,"#fd8d3c"],[0.75,"#e31a1c"],[1,"#800026"]],
  "YlOrBr":   [[0,"#ffffe5"],[0.25,"#fee391"],[0.5,"#fe9929"],[0.75,"#cc4c02"],[1,"#662506"]],
  "ice_r":    [[0,"#08306b"],[0.25,"#2171b5"],[0.5,"#6baed6"],[0.75,"#c6dbef"],[1,"#f7fbff"]],
  "RdYlGn":   [[0,"#a50026"],[0.25,"#f46d43"],[0.5,"#ffffbf"],[0.75,"#66bd63"],[1,"#006837"]],
  "Viridis":  [[0,"#440154"],[0.25,"#31688e"],[0.5,"#35b779"],[0.75,"#90d743"],[1,"#fde725"]],
};

function interpolateColor(scale, t) {
  t = Math.max(0, Math.min(1, t));
  let stops = SCALES[scale] || SCALES["Viridis"];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i+1][0]) {
      let f = (t - stops[i][0]) / (stops[i+1][0] - stops[i][0]);
      let c1 = stops[i][1], c2 = stops[i+1][1];
      let r1=parseInt(c1.slice(1,3),16), g1=parseInt(c1.slice(3,5),16), b1=parseInt(c1.slice(5,7),16);
      let r2=parseInt(c2.slice(1,3),16), g2=parseInt(c2.slice(3,5),16), b2=parseInt(c2.slice(5,7),16);
      let r=Math.round(r1+f*(r2-r1)), g=Math.round(g1+f*(g2-g1)), b=Math.round(b1+f*(b2-b1));
      return `rgb(${r},${g},${b})`;
    }
  }
  return stops[stops.length-1][1];
}

// ═══ STATE ═══
let map, geoLayer, cityA=null, cityB=null;

// ═══ HELPERS ═══
function getVar() { return VARS[document.getElementById("varSel").selectedIndex]; }
function getMo()  { return parseInt(document.getElementById("moSel").value); }
function isMetric(){ return document.getElementById("unitSel").value==="met"; }
function getVal(city, varDef, mo) {
  let v;
  if (varDef.isElev) { v = city.el; }
  else if (mo === 0) { v = city.a[varDef.k]; }
  else { v = city.m[String(mo)] ? city.m[String(mo)][varDef.k] : null; }
  if (v == null) return null;
  if (isMetric() && varDef.convF) v = varDef.convF(v);
  return v;
}
function getLabel(varDef) { return isMetric() ? varDef.labelM : varDef.label; }
function fmtElev(city) {
  if (city.el == null) return "N/A";
  return isMetric() ? `${Math.round(city.el*0.3048).toLocaleString()} m` : `${Math.round(city.el).toLocaleString()} ft`;
}
function cityKey(city) { return `${city.c}, ${city.s}`; }

// ═══ INIT ═══
function init() {
  // Populate dropdowns
  let varSel = document.getElementById("varSel");
  VARS.forEach((v,i) => { let o=document.createElement("option"); o.value=i; o.textContent=v.label; varSel.appendChild(o); });
  let moSel = document.getElementById("moSel");
  MONTHS.forEach((m,i) => { let o=document.createElement("option"); o.value=i; o.textContent=m; moSel.appendChild(o); });

  // City datalist
  let dl = document.getElementById("cityList");
  CITIES.forEach(c => { let o=document.createElement("option"); o.value=cityKey(c); dl.appendChild(o); });

  // Map
  map = L.map("map",{zoomControl:true}).setView([39.5,-98.5],4);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",{
    attribution:'',maxZoom:18,subdomains:'abcd'
  }).addTo(map);

  // GeoJSON layer
  geoLayer = L.geoJSON(VORONOI, {
    style: () => ({fillColor:"#ccc",fillOpacity:0.8,weight:0.5,color:"rgba(255,255,255,0.5)"}),
    onEachFeature: (feature, layer) => {
      let idx = feature.properties.i;
      if (idx == null || idx >= CITIES.length) return;
      layer.on("click", () => {
        let mode = document.getElementById("clickMode").value;
        if (mode === "A") { cityA = CITIES[idx]; document.getElementById("searchA").value = cityKey(CITIES[idx]); }
        else              { cityB = CITIES[idx]; document.getElementById("searchB").value = cityKey(CITIES[idx]); }
        updateAll();
      });
      layer.on("mouseover", e => { e.target.setStyle({weight:2,color:"#333"}); });
      layer.on("mouseout",  e => { geoLayer.resetStyle(e.target); updateMapColors(); });
    }
  }).addTo(map);

  // Event listeners
  ["varSel","moSel","unitSel"].forEach(id => document.getElementById(id).addEventListener("change", updateAll));
  document.getElementById("searchA").addEventListener("change", e => { cityA=findCity(e.target.value); updateAll(); });
  document.getElementById("searchB").addEventListener("change", e => { cityB=findCity(e.target.value); updateAll(); });
  document.getElementById("clearA").addEventListener("click", () => { cityA=null; document.getElementById("searchA").value=""; updateAll(); });
  document.getElementById("clearB").addEventListener("click", () => { cityB=null; document.getElementById("searchB").value=""; updateAll(); });
  document.getElementById("swapBtn").addEventListener("click", () => {
    [cityA,cityB]=[cityB,cityA];
    document.getElementById("searchA").value=cityA?cityKey(cityA):"";
    document.getElementById("searchB").value=cityB?cityKey(cityB):"";
    updateAll();
  });

  updateAll();
}

function findCity(key) {
  if (!key) return null;
  return CITIES.find(c => cityKey(c) === key) || null;
}

// ═══ UPDATE ALL ═══
function updateAll() {
  updateMapColors();
  updateChart();
  updateHistogram();
}

// ═══ MAP COLORING ═══
function updateMapColors() {
  let varDef = getVar(), mo = getMo();
  let vals = CITIES.map(c => getVal(c, varDef, mo)).filter(v => v != null);
  if (!vals.length) return;
  vals.sort((a,b) => a-b);
  let vmin = vals[Math.floor(vals.length*0.02)];
  let vmax = vals[Math.ceil(vals.length*0.98)-1];
  if (vmin === vmax) { vmin -= 1; vmax += 1; }
  let range = vmax - vmin;

  geoLayer.eachLayer(layer => {
    let idx = layer.feature.properties.i;
    if (idx == null || idx >= CITIES.length) return;
    let v = getVal(CITIES[idx], varDef, mo);
    let color = v != null ? interpolateColor(varDef.scale, (v-vmin)/range) : "#ddd";

    // Highlight selected cities
    let isA = cityA && idx === CITIES.indexOf(cityA);
    let isB = cityB && idx === CITIES.indexOf(cityB);
    let w = (isA||isB) ? 3 : 0.5;
    let bc = isA ? "#d32f2f" : isB ? "#7b1fa2" : "rgba(255,255,255,0.5)";

    layer.setStyle({fillColor:color, fillOpacity:0.82, weight:w, color:bc});

    // Tooltip
    let city = CITIES[idx];
    let vs = v != null ? v.toFixed(1) : "N/A";
    layer.bindTooltip(`<b>${city.c}, ${city.s}</b><br>${getLabel(varDef)}: ${vs}<br>Elev: ${fmtElev(city)}<br>Pop: ${city.p.toLocaleString()}`,
      {sticky:true, className:"leaflet-tooltip"});
  });

  // Legend
  let lg = document.getElementById("legend");
  let stops = SCALES[varDef.scale] || SCALES["Viridis"];
  let gradColors = stops.map(s => s[1]).join(",");
  lg.innerHTML = `<div style="font-weight:600;margin-bottom:2px">${getLabel(varDef)} — ${MONTHS[mo]}</div>
    <div class="grad" style="background:linear-gradient(to right,${gradColors})"></div>
    <div class="labels"><span>${vmin.toFixed(1)}</span><span>${vmax.toFixed(1)}</span></div>`;

  // Stats
  let mean = vals.reduce((a,b)=>a+b,0)/vals.length;
  let median = vals[Math.floor(vals.length/2)];
  document.getElementById("stats").textContent =
    `${vals.length} cities · Min: ${vals[0].toFixed(1)} · Mean: ${mean.toFixed(1)} · Median: ${median.toFixed(1)} · Max: ${vals[vals.length-1].toFixed(1)}`;
}

// ═══ CHART ═══
function updateChart() {
  let unit = isMetric();
  let mo = getMo();

  if (!cityA && !cityB) {
    Plotly.react("chart",[],{
      paper_bgcolor:"#fafafa",plot_bgcolor:"#fafafa",
      xaxis:{visible:false},yaxis:{visible:false},
      annotations:[{text:"Click a region or search",xref:"paper",yref:"paper",x:0.5,y:0.5,showarrow:false,font:{size:14,color:"#aaa"}}],
      margin:{l:10,r:10,t:10,b:10}
    });
    document.getElementById("panel-title").textContent = "Click a region or search above";
    return;
  }

  let traces = [];
  let titleParts = [];
  let colorsA = ["#d32f2f","#f57c00","#1976d2"];
  let colorsB = ["#7b1fa2","#388e3c","#00acc1"];
  let comparing = cityA && cityB;

  function addTraces(city, colors, pfx, dash) {
    let tmax=[],tavg=[],tmin=[],frost=[],hot=[];
    for (let m=1;m<=12;m++) {
      let d = city.m[String(m)] || {};
      let mx=d.tmax, av=d.tavg, mn=d.tmin;
      if (unit && mx!=null) mx=(mx-32)*5/9;
      if (unit && av!=null) av=(av-32)*5/9;
      if (unit && mn!=null) mn=(mn-32)*5/9;
      tmax.push(mx); tavg.push(av); tmin.push(mn);
      frost.push(d.frost); hot.push(d.hot90);
    }
    let tu = unit?"°C":"°F";
    let nm = pfx ? pfx+" " : "";
    traces.push({x:SMONTHS,y:tmax,name:`${nm}Max (${tu})`,type:"scatter",mode:"lines+markers",
      line:{color:colors[0],width:2.5,dash:dash||"solid"},marker:{size:4},xaxis:"x",yaxis:"y",legendgroup:pfx});
    traces.push({x:SMONTHS,y:tavg,name:`${nm}Mean (${tu})`,type:"scatter",mode:"lines+markers",
      line:{color:colors[1],width:2.5,dash:"dot"},marker:{size:4},xaxis:"x",yaxis:"y",legendgroup:pfx});
    traces.push({x:SMONTHS,y:tmin,name:`${nm}Min (${tu})`,type:"scatter",mode:"lines+markers",
      line:{color:colors[2],width:2.5,dash:dash||"solid"},marker:{size:4},xaxis:"x",yaxis:"y",legendgroup:pfx});
    traces.push({x:SMONTHS,y:frost,name:`${nm}Freeze`,type:"bar",
      marker:{color:colors[2],opacity:0.7},xaxis:"x2",yaxis:"y2",legendgroup:pfx});
    traces.push({x:SMONTHS,y:hot,name:`${nm}90°F+`,type:"bar",
      marker:{color:colors[0],opacity:0.6},xaxis:"x2",yaxis:"y2",legendgroup:pfx});

    let el = fmtElev(city);
    titleParts.push(`${city.c}, ${city.s} (${el})`);
  }

  if (cityA) addTraces(cityA, colorsA, comparing?cityA.c:"", null);
  if (cityB) addTraces(cityB, colorsB, comparing?cityB.c:"", comparing?"dash":null);

  // Month highlight line
  let shapes = [];
  if (mo >= 1) {
    let xp = SMONTHS[mo-1];
    shapes.push({type:"line",x0:xp,x1:xp,y0:0,y1:1,yref:"paper",line:{color:"rgba(0,0,0,0.4)",width:2,dash:"dot"}});
  }

  let yRange = unit ? [-29,49] : [-20,120];
  Plotly.react("chart", traces, {
    grid:{rows:2,columns:1,pattern:"independent",roworder:"top to bottom",ygap:0.1},
    yaxis:{title:unit?"°C":"°F",range:yRange,anchor:"x",domain:[0.45,0.75]},
    xaxis:{anchor:"y"},yaxis:{title:unit?"°C":"°F",range:yRange,anchor:"x"},
    xaxis2:{anchor:"y2"},yaxis2:{title:"Days",anchor:"x2"},
    paper_bgcolor:"#fafafa",plot_bgcolor:"#fff",
    margin:{l:50,r:10,t:30,b:30},
    legend:{font:{size:9},orientation:"h",y:1.2,x:0.5,xanchor:"center"},
    barmode:"group",font:{size:10},shapes:shapes,
  },{responsive:true});

  document.getElementById("panel-title").textContent = comparing ? titleParts.join("  vs  ") : titleParts[0]||"";
}

// ═══ HISTOGRAM ═══
function updateHistogram() {
  let varDef = getVar(), mo = getMo();
  let vals = CITIES.map(c => getVal(c, varDef, mo)).filter(v => v != null);

  let traces = [{x:vals,type:"histogram",nbinsx:50,marker:{color:"#90caf9",line:{width:1,color:"white"}},showlegend:false}];
  let shapes = [];

  [[cityA,"#d32f2f"],[cityB,"#7b1fa2"]].forEach(([city,col]) => {
    if (!city) return;
    let v = getVal(city, varDef, mo);
    if (v != null) shapes.push({type:"line",x0:v,x1:v,y0:0,y1:1,yref:"paper",line:{color:col,width:3,dash:"dash"}});
  });

  Plotly.react("histogram", traces, {
    title:{text:`National Distribution: ${getLabel(varDef)}`,font:{size:11,color:"#2c3e50"}},
    margin:{l:10,r:10,t:28,b:20},paper_bgcolor:"#f0f1f3",plot_bgcolor:"#f0f1f3",
    xaxis:{title:getLabel(varDef),titlefont:{size:10},tickfont:{size:9}},
    yaxis:{visible:false},shapes:shapes,
  },{responsive:true});
}

// ═══ START ═══
document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════════════════════
# MAIN — BUILD
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

    cities_json = build_json_data(merged)

    # Build HTML
    print("Writing HTML...")
    html_out = HTML_TEMPLATE.replace("%%CITIES_JSON%%", json.dumps(cities_json, separators=(",",":")))
    html_out = html_out.replace("%%VORONOI_JSON%%", json.dumps(vor_gj, separators=(",",":")))

    out_dir = os.path.dirname(OUTPUT_HTML or CITIES_CSV)
    out_path = OUTPUT_HTML or os.path.join(out_dir, "climate_map.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n✓ Generated: {out_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Cities: {len(cities_json)}")
    print(f"  Voronoi cells: {len(vor_gj['features'])}")
    print(f"\n  Double-click the HTML file to open in your browser.")
    print(f"  Or run:  python launch.pyw")
