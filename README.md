# Chicago South Side Inequality Dashboard

This dashboard explores **address density** (a proxy for housing concentration), **income**, and **uninsured rates** at the **Census Tract** level in Chicago's South Side.

## Usage

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Build tract-level dataset (generates `derived-data/merged_tract.geojson`)
```bash
python preprocessing.py
```

3) Run the dashboard
```bash
streamlit run app.py
```

## Notes
- Address density is computed as: addresses per square kilometer.
- Address density is a proxy (not official housing units).
- Findings are descriptive and correlational.
