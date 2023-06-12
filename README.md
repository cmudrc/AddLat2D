---
title: AddLat2D
emoji: 😻
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---

Use the following code to load items in the 2D_Lattice.csv file
import pandas as pd
import json
df = pd.read_csv('2D_Lattice.csv')
row = 0
box = df.iloc[row,1]
array = np.array(json.loads(box))

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
