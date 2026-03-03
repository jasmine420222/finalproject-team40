# Housing Density, Income, and Uninsured Rates in Chicago’s South Side

This project explores how housing density and neighborhood income relate to uninsured rates at the Census Tract level. It provides descriptive, tract-level evidence for policy discussion.

## Usage

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Build tract-level dataset (generates `derived-data/merged_tract.geojson`)
```bash
python code/preprocessing.py
```

3) Run the dashboard
```bash
streamlit run streamlit-app/app.py
```

## Notes
- Address density is computed as: addresses per square kilometer.
- Address density is a proxy (not official housing units).
- Findings are descriptive and correlational.
