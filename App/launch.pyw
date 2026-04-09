"""
launch.pyw — Open climate_map.html in a native desktop window.
Package as .exe with:   pyinstaller --onefile --noconsole launch.pyw

Requires:  pip install pywebview
"""
import os, sys, webview

html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "USclimate_map.html")

if not os.path.exists(html_path):
    print(f"ERROR: {html_path} not found. Run build_USclimate_app.py first.")
    sys.exit(1)

webview.create_window(
    "U.S. Climate Normals (1991-2020)",
    url=html_path,
    width=1400,
    height=900,
    resizable=True,
    min_size=(1000, 600),
)
webview.start()
