"""
Interactive U.S. Climate Map: NOAA 1991-2020 Monthly Normals (v8)
==================================================================
DATA SOURCES & ATTRIBUTIONS:
    Climate data: NOAA 1991-2020 U.S. Climate Normals
        https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals
        Palecki et al., 2021. NOAA National Centers for Environmental Information.
    City database: SimpleMaps U.S. Cities Database (free tier)
        https://simplemaps.com/data/us-cities
        (Required attribution per license terms)

SETUP:
    pip install pandas numpy dash plotly scipy geopandas shapely

USAGE:
    1. Edit CLIMATE_DIR and CITIES_CSV below.
    2. python climate_map.py
    3. Open http://127.0.0.1:8050
"""

import os, glob, json
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import MultiPoint, Point, mapping, shape
from shapely.ops import voronoi_diagram, unary_union
import geopandas as gpd

# [ USER CONFIG ]
CLIMATE_DIR = r"C:\Users\nag55\Downloads\Climate\Multi"
CITIES_CSV  = r"C:\Users\nag55\Downloads\Climate\uscities.csv"
# [ =========== ]

MONTH_NAMES = {0:"Annual",1:"January",2:"February",3:"March",4:"April",5:"May",
               6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
SHORT_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

CLIMATE_COLS = [
    "MLY-TAVG-NORMAL","MLY-TMAX-NORMAL","MLY-TMIN-NORMAL","MLY-DUTR-NORMAL",
    "MLY-PRCP-NORMAL","MLY-SNOW-NORMAL","MLY-HTDD-NORMAL","MLY-CLDD-NORMAL",
    "MLY-TMIN-AVGNDS-LSTH032","MLY-TMAX-AVGNDS-GRTH090",
    "MLY-PRCP-AVGNDS-GE010HI","MLY-SNOW-AVGNDS-GE010TI","MLY-SNWD-AVGNDS-GE001WI",
]
DISPLAY_NAMES = {
    "MLY-TAVG-NORMAL":"Mean Temp (°F)","MLY-TMAX-NORMAL":"Mean Max Temp (°F)",
    "MLY-TMIN-NORMAL":"Mean Min Temp (°F)","MLY-DUTR-NORMAL":"Diurnal Temp Range (°F)",
    "MLY-PRCP-NORMAL":"Precipitation (in)","MLY-SNOW-NORMAL":"Snowfall (in)",
    "MLY-HTDD-NORMAL":"Heating Degree Days","MLY-CLDD-NORMAL":"Cooling Degree Days",
    "MLY-TMIN-AVGNDS-LSTH032":"Freezing Days (min ≤ 32°F)",
    "MLY-TMAX-AVGNDS-GRTH090":"Days ≥ 90°F",
    "MLY-PRCP-AVGNDS-GE010HI":"Precip Days (≥ 0.10 in)",
    "MLY-SNOW-AVGNDS-GE010TI":"Snow Days (≥ 1.0 in)",
    "MLY-SNWD-AVGNDS-GE001WI":"Snow Cover Days (≥ 1 in depth)",
    "ELEVATION":"Elevation (ft)",
    "COMFORT-INDEX": "Comfort Index (0-100)"
}
COLOR_SCALES = {
    "MLY-TAVG-NORMAL":"RdBu_r","MLY-TMAX-NORMAL":"RdBu_r","MLY-TMIN-NORMAL":"RdBu_r",
    "MLY-DUTR-NORMAL":"Oranges","MLY-PRCP-NORMAL":"Blues","MLY-SNOW-NORMAL":"ice_r",
    "MLY-HTDD-NORMAL":"Blues","MLY-CLDD-NORMAL":"Reds",
    "MLY-TMIN-AVGNDS-LSTH032":"ice_r","MLY-TMAX-AVGNDS-GRTH090":"YlOrRd",
    "MLY-PRCP-AVGNDS-GE010HI":"Blues","MLY-SNOW-AVGNDS-GE010TI":"ice_r",
    "MLY-SNWD-AVGNDS-GE001WI":"ice_r","ELEVATION":"YlOrBr",
    "COMFORT-INDEX": "RdYlGn"
}
ANNUAL_SUM = {"MLY-PRCP-NORMAL","MLY-SNOW-NORMAL","MLY-HTDD-NORMAL","MLY-CLDD-NORMAL",
              "MLY-TMIN-AVGNDS-LSTH032","MLY-TMAX-AVGNDS-GRTH090",
              "MLY-PRCP-AVGNDS-GE010HI","MLY-SNOW-AVGNDS-GE010TI","MLY-SNWD-AVGNDS-GE001WI"}

# ===========================================================================
# US BORDER + VORONOI + DATA LOADING
# ===========================================================================

def get_us_border():
    cache = os.path.join(os.path.dirname(CITIES_CSV), "us_border_cache.geojson")
    if os.path.exists(cache):
        print("  Loading cached US border...")
        return unary_union(gpd.read_file(cache).geometry)

    print("  Downloading US states boundary (one-time)...")
    urls = [
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
        "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json",
    ]
    gdf = None
    for url in urls:
        try:
            gdf = gpd.read_file(url)
            print(f"    OK from {url.split('/')[2]}")
            break
        except Exception as e:
            print(f"    Failed: {e}")

    if gdf is None:
        print("  WARNING: download failed; using bounding-box fallback.")
        from shapely.geometry import box
        return unary_union([box(-125,24.5,-66.5,49.5), box(-180,51,-130,72), box(-161,18.5,-154,22.5)])

    us_geom = unary_union(gdf.geometry)
    try:
        gpd.GeoDataFrame(geometry=[us_geom], crs=gdf.crs).to_file(cache, driver="GeoJSON")
        print(f"    Cached -> {cache}")
    except Exception:
        pass
    return us_geom

def build_voronoi_geojson(lats, lngs, us_border):
    print("Building Voronoi tessellation...")
    points = MultiPoint([(lng, lat) for lng, lat in zip(lngs, lats)])
    envelope = us_border.buffer(2).envelope
    vor = voronoi_diagram(points, envelope=envelope)

    point_list = list(zip(lngs, lats))
    features = []
    matched = set()

    for poly in vor.geoms:
        clipped = poly.intersection(us_border)
        if clipped.is_empty:
            continue
        for i, (lng, lat) in enumerate(point_list):
            if i in matched:
                continue
            if poly.contains(Point(lng, lat)):
                clipped = clipped.simplify(0.01, preserve_topology=True)
                geom = mapping(clipped)
                if geom["type"] not in ("Polygon", "MultiPolygon"):
                    if geom["type"] == "GeometryCollection":
                        parts = [g for g in shape(geom).geoms if g.geom_type in ("Polygon","MultiPolygon")]
                        if parts:
                            geom = mapping(unary_union(parts))
                        else:
                            continue
                    else:
                        continue
                features.append({"type":"Feature","id":str(i),"geometry":geom,"properties":{"idx":i}})
                matched.add(i)
                break

    print(f"  {len(features)} Voronoi cells created ({len(point_list)} cities).")
    return {"type":"FeatureCollection","features":features}

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
    print(f"  {sdf['station'].nunique()} stations loaded ({sdf[sdf['has_temp']]['station'].nunique()} w/ temp).")
    return sdf

def load_and_filter_cities(csv, target=2000):
    print("Filtering cities...")
    df = pd.read_csv(csv, dtype={"population":float,"lat":float,"lng":float})
    df = df.dropna(subset=["lat","lng","population"]).sort_values("population",ascending=False).reset_index(drop=True)
    sel = []
    for _, r in df.iterrows():
        if len(sel) >= target: break
        la, lo = r["lat"], r["lng"]
        if any(abs(s["lat"]-la)<1 and abs(s["lng"]-lo)<1 for s in sel): continue
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
    m = pd.DataFrame(recs)
    print(f"  {len(m)} rows.")
    return m

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

def calculate_comfort_index(merged):
    print("Calculating Comfort Index...")
    t = merged["MLY-TAVG-NORMAL"].fillna(65)
    p = merged["MLY-PRCP-NORMAL"].fillna(0)
    f = merged["MLY-TMIN-AVGNDS-LSTH032"].fillna(0)
    h = merged["MLY-TMAX-AVGNDS-GRTH090"].fillna(0)

    monthly_mask = merged["month"] != 0
    c_idx = pd.Series(index=merged.index, dtype=float)
    
    # Calculate for months 1-12
    tm = t[monthly_mask]
    pm = p[monthly_mask]
    fm = f[monthly_mask]
    hm = h[monthly_mask]
    
    t_pen_m = np.maximum(0, 50 - tm) * 3 + np.maximum(0, tm - 80) * 3
    p_pen_m = pm * 5
    e_pen_m = (fm + hm) * 2
    
    c_idx[monthly_mask] = np.clip(100 - t_pen_m - p_pen_m - e_pen_m, 0, 100)
    merged["COMFORT-INDEX"] = c_idx
    
    # Calculate annual score as the mean of the 12 monthly scores
    ann_mask = merged["month"] == 0
    for city_state, group in merged[monthly_mask].groupby(["city", "state"]):
        mean_comfort = group["COMFORT-INDEX"].mean()
        idx = merged[(merged["city"] == city_state[0]) & (merged["state"] == city_state[1]) & ann_mask].index
        if len(idx) > 0:
            merged.loc[idx, "COMFORT-INDEX"] = mean_comfort
            
    return merged

def cv(val, col, unit):
    if pd.isna(val): return np.nan
    if col == "COMFORT-INDEX": return val # No unit conversion for index
    if unit=="metric":
        if col in ("MLY-TAVG-NORMAL","MLY-TMAX-NORMAL","MLY-TMIN-NORMAL"): return (val-32)*5/9
        if col=="MLY-DUTR-NORMAL": return val*5/9
        if col in ("MLY-PRCP-NORMAL","MLY-SNOW-NORMAL"): return val*25.4
        if col=="ELEVATION": return val*0.3048
    return val

def dn(col, unit):
    n = DISPLAY_NAMES.get(col, col)
    if col == "COMFORT-INDEX": return n
    if unit=="metric": n = n.replace("°F","°C").replace("(in)","(mm)").replace("(ft)","(m)")
    return n

def _get_city_monthly(merged, city_key):
    if not city_key:
        return None, None, None
    parts = city_key.split(", ")
    if len(parts) != 2:
        return None, None, None
    cn, sn = parts
    d = merged[(merged["city"]==cn)&(merged["state"]==sn)&
               (merged["month"]>=1)&(merged["month"]<=12)].sort_values("month")
    if d.empty:
        return None, None, None
    return d, cn, sn

# === CHART TRACES ==========================================================
def _add_city_traces(fig, cd, unit, label, colors, dash_style=None):
    x = [SHORT_MONTHS[m-1] for m in cd["month"].values]
    tmax = cd["MLY-TMAX-NORMAL"].values.astype(float)
    tavg = cd["MLY-TAVG-NORMAL"].values.astype(float)
    tmin = cd["MLY-TMIN-NORMAL"].values.astype(float)
    tu = "°F"
    if unit=="metric":
        tmax,tavg,tmin = (tmax-32)*5/9,(tavg-32)*5/9,(tmin-32)*5/9
        tu="°C"

    ls = dash_style or "solid"
    pfx = f"{label} " if label else ""

    fig.add_trace(go.Scatter(x=x,y=tmax,name=f"{pfx}Max ({tu})",
        line=dict(color=colors[0],width=3.5,dash=ls),
        mode="lines+markers",marker=dict(size=5),legendgroup=label),row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=tavg,name=f"{pfx}Mean ({tu})",
        line=dict(color=colors[1],width=3.5,dash="dot" if not dash_style else ls),
        mode="lines+markers",marker=dict(size=5),legendgroup=label),row=1,col=1)
    fig.add_trace(go.Scatter(x=x,y=tmin,name=f"{pfx}Min ({tu})",
        line=dict(color=colors[2],width=3.5,dash=ls),
        mode="lines+markers",marker=dict(size=5),legendgroup=label),row=1,col=1)

    fig.add_trace(go.Scatter(x=x+x[::-1],y=list(tmax)+list(tmin[::-1]),fill="toself",
        fillcolor=colors[3],line=dict(width=0),showlegend=False,hoverinfo="skip"),row=1,col=1)

    frost = cd["MLY-TMIN-AVGNDS-LSTH032"].values.astype(float)
    hot   = cd["MLY-TMAX-AVGNDS-GRTH090"].values.astype(float)

    fig.add_trace(go.Bar(x=x,y=frost,name=f"{pfx}Freeze Days",
        marker_color=colors[4],opacity=0.85,legendgroup=label),row=2,col=1)
    fig.add_trace(go.Bar(x=x,y=hot,name=f"{pfx}≥90°F Days",
        marker_color=colors[0],opacity=0.7,legendgroup=label),row=2,col=1)


COLORS_A = ["#d32f2f","#f57c00","#1976d2","rgba(211,47,47,0.12)","#81d4fa","#ff9800"]
COLORS_B = ["#7b1fa2","#388e3c","#00acc1","rgba(123,31,162,0.10)","#a5d6a7","#00acc1"]

# ===========================================================================
# DASH APP
# ===========================================================================

def build_app(merged, vor_gj):
    app = dash.Dash(__name__)
    
    options_cols = [c for c in CLIMATE_COLS if c in merged.columns and merged[c].notna().any()]
    options_cols.append("COMFORT-INDEX")
    
    vopts = [{"label":DISPLAY_NAMES[c],"value":c} for c in options_cols]
    vopts.append({"label":"Elevation (ft)","value":"ELEVATION"})
    
    mopts = [{"label":MONTH_NAMES[m],"value":m} for m in range(13)]
    ann = merged[merged["month"]==0].reset_index(drop=True)
    clabs = sorted(ann.apply(lambda r: f"{r['city']}, {r['state']}", axis=1).unique())
    city_opts = [{"label":c,"value":c} for c in clabs]

    app.layout = html.Div(style={"fontFamily":"Segoe UI, Arial","backgroundColor":"#f8f9fa",
                                  "height":"100vh","overflow":"hidden"}, children=[
        dcc.Store(id="selected-city", data=None),
        html.Div(style={"padding":"6px 20px 0"}, children=[
            html.H1("U.S. Climate Normals Map (1991 - 2020)",
                     style={"textAlign":"center","color":"#2c3e50","marginBottom":"2px","fontSize":"22px"}),
            html.P("NOAA 30-year normals · Click any region for 12-month profile",
                    style={"textAlign":"center","color":"#7f8c8d","marginTop":"0","marginBottom":"6px","fontSize":"12px"}),
        ]),

        # Controls row
        html.Div(style={"display":"flex","justifyContent":"center","gap":"12px",
                         "padding":"0 20px 6px","flexWrap":"wrap","alignItems":"flex-end"}, children=[
            html.Div([html.Label("Variable:",style={"fontWeight":"bold","fontSize":"12px"}),
                       dcc.Dropdown(id="var",options=vopts,value="MLY-TAVG-NORMAL",
                                    style={"width":"250px","fontSize":"13px"},clearable=False)]),
            html.Div([html.Label("Period:",style={"fontWeight":"bold","fontSize":"12px"}),
                       dcc.Dropdown(id="mo",options=mopts,value=0,
                                    style={"width":"130px","fontSize":"13px"},clearable=False)]),
            html.Div([html.Label("Units:",style={"fontWeight":"bold","fontSize":"12px"}),
                       dcc.RadioItems(id="unit",options=[{"label":" °F/in/ft","value":"imperial"},
                                                         {"label":" °C/mm/m","value":"metric"}],
                                      value="imperial",inline=True,style={"marginTop":"6px","fontSize":"12px"})]),

            # City A
            html.Div(style={"display":"flex","alignItems":"center","gap":"4px"}, children=[
                html.Div([html.Label("City A:",style={"fontWeight":"bold","fontSize":"12px","color":"#d32f2f"}),
                           dcc.Dropdown(id="search",options=city_opts,value=None,
                                        placeholder="Click map or search...",
                                        style={"width":"200px","fontSize":"13px"},clearable=True)]),
                html.Button("×", id="clear-a", style={"fontSize":"22px","color":"#d32f2f","background":"none","border":"none","cursor":"pointer","padding":"0"}),
                html.Button("⇄", id="swap", style={"fontSize":"20px","color":"#555","background":"none","border":"none","cursor":"pointer","padding":"0"}),
            ]),

            # City B
            html.Div(style={"display":"flex","alignItems":"center","gap":"4px"}, children=[
                html.Div([html.Label("City B:",style={"fontWeight":"bold","fontSize":"12px","color":"#7b1fa2"}),
                           dcc.Dropdown(id="compare",options=city_opts,value=None,
                                        placeholder="Optional...",
                                        style={"width":"200px","fontSize":"13px"},clearable=True)]),
                html.Button("×", id="clear-b", style={"fontSize":"22px","color":"#7b1fa2","background":"none","border":"none","cursor":"pointer","padding":"0"}),
            ]),

            html.Div([html.Label("Click map assigns to:", style={"fontWeight":"bold","fontSize":"12px"}),
                       dcc.RadioItems(id="click-mode",
                                      options=[{"label": " A ", "value": "A"},
                                               {"label": " B ", "value": "B"}],
                                      value="A", inline=True, style={"fontSize":"13px"})]),
        ]),

        # Map + chart panel
        html.Div(style={"display":"flex","padding":"0 8px","height":"calc(100vh - 130px)"}, children=[
            html.Div(style={"flex":"3"}, children=[
                dcc.Graph(id="map",style={"height":"100%"},config={"scrollZoom":True})]),

            html.Div(style={"flex":"1","minWidth":"340px","maxWidth":"480px",
                             "display":"flex","flexDirection":"column"}, children=[
                html.Div(id="ctitle",style={"textAlign":"center","fontWeight":"bold",
                                             "fontSize":"13px","padding":"6px 4px 0","color":"#2c3e50",
                                             "minHeight":"22px","lineHeight":"1.3"}),
                dcc.Graph(id="chart",style={"flex":"1","minHeight":"350px"},config={"displayModeBar":False}),
                
                # Distribution Histogram
                html.Div(style={"marginTop": "8px", "backgroundColor": "#f1f3f5", "borderRadius": "8px", "padding":"4px"}, children=[
                     dcc.Graph(id="histogram", style={"height": "180px"}, config={"displayModeBar": False})
                ])
            ]),
        ]),
        html.Div(id="stats",style={"textAlign":"center","color":"#555","fontSize":"12px","padding":"2px 0 1px"}),
        html.Div(style={"textAlign":"center","fontSize":"10px","color":"#999","padding":"0 0 4px"},
                 children=[
                     "Data: ",
                     html.A("NOAA 1991–2020 Climate Normals", href="https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals",
                            target="_blank", style={"color":"#888"}),
                     "  ·  City database: ",
                     html.A("SimpleMaps", href="https://simplemaps.com/data/us-cities",
                            target="_blank", style={"color":"#888"}),
                 ]),
    ])

    # === Callbacks ===========================================================

    # SINGLE unified callback for all city selection actions
    # (Fixes the duplicate-Output crash that froze controls)
    @app.callback(
        [Output("selected-city","data"), Output("compare","value"), Output("search","value")],
        [Input("map","clickData"), Input("search","value"),
         Input("clear-a","n_clicks"), Input("clear-b","n_clicks"), Input("swap","n_clicks")],
        [State("selected-city","data"), State("compare","value"), State("click-mode","value")]
    )
    def unified_selection(click, search_val, n_clear_a, n_clear_b, n_swap,
                          cur_a, cur_b, click_mode):
        ctx = dash.callback_context
        if not ctx.triggered:
            return cur_a, cur_b, no_update
        trig_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trig_id == "clear-a":
            return None, cur_b, None
        elif trig_id == "clear-b":
            return cur_a, None, no_update
        elif trig_id == "swap":
            return cur_b, cur_a, cur_b  # update search box to show new A
        elif trig_id == "search":
            return search_val, cur_b, no_update
        elif trig_id == "map" and click and click.get("points"):
            cd = click["points"][0].get("customdata")
            if cd and len(cd) == 2:
                city_str = f"{cd[0]}, {cd[1]}"
                if click_mode == "B":
                    return cur_a, city_str, no_update
                else:
                    return city_str, cur_b, city_str
        return cur_a, cur_b, no_update

    @app.callback([Output("map","figure"),Output("stats","children")],
                  [Input("var","value"),Input("mo","value"),Input("unit","value"),
                   Input("selected-city","data"),Input("compare","value")],
                  prevent_initial_call=False) # Changed to False so it loads immediately on startup
    def update_map(variable, month, unit, city_a, city_b):
        sub = merged[merged["month"]==month].reset_index(drop=True)
        col = variable
        dname = dn(col, unit)
        period = MONTH_NAMES[month]
        raw = sub["elevation_ft"] if col=="ELEVATION" else sub[col]
        vals = raw.apply(lambda v: cv(v, col, unit))
        sub["dv"] = vals
        eu = "m" if unit=="metric" else "ft"

        hover, locs, zv = [], [], []
        for i, (_, r) in enumerate(sub.iterrows()):
            v = r["dv"]; vs = f"{v:.1f}" if pd.notna(v) else "N/A"
            e = cv(r["elevation_ft"],"ELEVATION",unit); es = f"{e:,.0f}" if pd.notna(e) else "N/A"
            if col == "COMFORT-INDEX":
                hover.append(f"<b>{r['city']}, {r['state']}</b><br>Comfort Score: {vs}/100<br>Elevation: {es} {eu}<br>Pop: {r['population']:,.0f}")
            else:
                hover.append(f"<b>{r['city']}, {r['state']}</b><br>{dname}: {vs}<br>Elevation: {es} {eu}<br>Pop: {r['population']:,.0f}")
            locs.append(str(i)); zv.append(v if pd.notna(v) else None)

        valid = pd.Series(zv).dropna()
        cmin = valid.quantile(0.02) if len(valid) else 0
        cmax = valid.quantile(0.98) if len(valid) else 1

        fig = go.Figure()
        fig.add_trace(go.Choroplethmapbox(
            geojson=vor_gj, locations=locs, z=zv,
            colorscale=COLOR_SCALES.get(col,"Viridis"), zmin=cmin, zmax=cmax,
            marker_opacity=0.92, marker_line_width=0.8, marker_line_color="rgba(255,255,255,0.4)",
            hovertext=hover, hoverinfo="text",
            colorbar=dict(title=dict(text=dname,font=dict(size=11)),thickness=12,len=0.7),
            customdata=sub[["city","state"]].values.tolist(),
        ))

        # Restored original Trace rendering logic exactly as you provided it
        for ckey, color, symbol in [(city_a,"#d32f2f","A"),(city_b,"#7b1fa2","B")]:
            if not ckey: continue
            parts = ckey.split(", ")
            if len(parts)!=2: continue
            row = sub[(sub["city"]==parts[0])&(sub["state"]==parts[1])]
            if row.empty: continue
            r = row.iloc[0]
            fig.add_trace(go.Scattermapbox(
                lat=[r["lat"]],lon=[r["lng"]],mode="markers+text",
                marker=dict(size=20,color="rgba(0,0,0,0)",line=dict(width=3,color=color)),
                text=[symbol],textfont=dict(size=11,color=color),textposition="top center",
                hoverinfo="skip",showlegend=False))

        # Restored original Layout rendering logic exactly as you provided it
        fig.update_layout(
            mapbox_style="carto-positron",mapbox_center={"lat":39.5,"lon":-98.5},mapbox_zoom=3.3,
            title=dict(text=f"{dname} — {period}",x=0.5,font=dict(size=15,color="#2c3e50")),
            margin=dict(l=0,r=0,t=32,b=0),paper_bgcolor="#f8f9fa",clickmode="event")

        st = ""
        if len(valid):
            st = f"Showing {len(valid)} cities  ·  Min: {valid.min():.1f}  ·  Mean: {valid.mean():.1f}  ·  Median: {valid.median():.1f}  ·  Max: {valid.max():.1f}"
        return fig, st

    @app.callback(Output("histogram", "figure"),
                  [Input("var", "value"), Input("mo", "value"), Input("unit", "value"),
                   Input("selected-city", "data"), Input("compare", "value")])
    def update_histogram(variable, month, unit, city_a, city_b):
        sub = merged[merged["month"] == month].dropna(subset=[variable])
        if sub.empty:
            return go.Figure()

        col = variable
        dname = dn(col, unit)
        vals = sub[col].apply(lambda v: cv(v, col, unit))

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=50, marker_color="#90caf9",
                                   marker_line_width=1, marker_line_color="white",
                                   showlegend=False))

        for ckey, color in [(city_a, "#d32f2f"), (city_b, "#7b1fa2")]:
            if ckey:
                parts = ckey.split(", ")
                if len(parts) == 2:
                    city_row = sub[(sub["city"]==parts[0])&(sub["state"]==parts[1])]
                    if not city_row.empty:
                        city_val = cv(city_row[col].iloc[0], col, unit)
                        if pd.notna(city_val):
                            fig.add_vline(x=city_val, line_width=3, line_color=color, line_dash="dash")

        fig.update_layout(
            title=dict(text=f"National Distribution: {dname}", font=dict(size=11, color="#2c3e50")),
            margin=dict(l=10, r=10, t=25, b=20),
            paper_bgcolor="#f1f3f5", plot_bgcolor="#f1f3f5",
            xaxis=dict(title=dname, title_font=dict(size=10), tickfont=dict(size=9)),
            yaxis=dict(visible=False)
        )
        return fig

    @app.callback([Output("chart","figure"),Output("ctitle","children")],
                  [Input("selected-city","data"),Input("compare","value"),
                   Input("unit","value"),Input("mo","value")],
                  prevent_initial_call=False)
    def update_chart(city_a, city_b, unit, month):
        cd_a, cn_a, sn_a = _get_city_monthly(merged, city_a)
        cd_b, cn_b, sn_b = _get_city_monthly(merged, city_b)

        if cd_a is None and cd_b is None:
            fig = go.Figure()
            fig.update_layout(paper_bgcolor="#f8f9fa",plot_bgcolor="#f8f9fa",
                              xaxis=dict(visible=False),yaxis=dict(visible=False),
                              annotations=[dict(text="Click a region on the map<br>or search above",
                                                xref="paper",yref="paper",x=0.5,y=0.5,
                                                showarrow=False,font=dict(size=14,color="#aaa"))],
                              margin=dict(l=10,r=10,t=10,b=10))
            return fig, ""

        comparing = cd_a is not None and cd_b is not None

        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.12,
                            row_heights=[0.65,0.35],
                            subplot_titles=("Temperature", "Freeze & Hot Days"))

        if cd_a is not None:
            _add_city_traces(fig, cd_a, unit, f"{cn_a}" if comparing else "", COLORS_A)

        if cd_b is not None:
            _add_city_traces(fig, cd_b, unit, f"{cn_b}" if comparing else "", COLORS_B,
                             dash_style="dash" if comparing else None)

        y_min, y_max = (-20, 120) if unit == "imperial" else (-29, 49)
        fig.update_yaxes(title_text="Temperature (°F)" if unit=="imperial" else "Temperature (°C)",
                         range=[y_min, y_max], row=1, col=1)

        fig.update_layout(paper_bgcolor="#f8f9fa",plot_bgcolor="#ffffff",
                          margin=dict(l=50,r=10,t=30,b=30),
                          legend=dict(font=dict(size=9),orientation="h",
                                      yanchor="bottom",y=1.02,xanchor="center",x=0.5),
                          barmode="group",font=dict(size=10))

        if month >= 1:
            x_highlight = SHORT_MONTHS[month-1]
            fig.add_vline(x=x_highlight, line_width=2.5, line_dash="dot",
                          line_color="rgba(0,0,0,0.5)", row=1, col=1)
            fig.add_vline(x=x_highlight, line_width=2.5, line_dash="dot",
                          line_color="rgba(0,0,0,0.5)", row=2, col=1)

        parts = []
        for cd, cn, sn in [(cd_a,cn_a,sn_a),(cd_b,cn_b,sn_b)]:
            if cd is None: continue
            e = cv(cd["elevation_ft"].iloc[0],"ELEVATION",unit)
            eu = "m" if unit=="metric" else "ft"
            es = f"{e:,.0f}{eu}" if pd.notna(e) else "N/A"
            parts.append(f"{cn}, {sn} ({es})")
        title = " vs ".join(parts) if comparing else parts[0] if parts else ""

        return fig, title

    return app

# ===========================================================================
if __name__ == "__main__":
    stations = load_stations(CLIMATE_DIR)
    cities = load_and_filter_cities(CITIES_CSV, target=2000)
    cities = match_cities(cities, stations)
    merged = build_merged(cities, stations)
    merged = interpolate_missing(merged, stations)
    merged = calculate_comfort_index(merged) # Generate the new index

    us_border = get_us_border()
    ann = merged[merged["month"]==0].reset_index(drop=True)
    vor_gj = build_voronoi_geojson(ann["lat"].values, ann["lng"].values, us_border)

    base = os.path.dirname(CITIES_CSV)
    try:
        with open(os.path.join(base,"voronoi_cells.geojson"),"w") as f: json.dump(vor_gj, f)
        merged.to_parquet(os.path.join(base,"climate_merged_v7.parquet"))
        print("  Cached.")
    except: pass

    print("\n-> http://127.0.0.1:8050\n")
    app = build_app(merged, vor_gj)
    app.run(debug=False, port=8050)