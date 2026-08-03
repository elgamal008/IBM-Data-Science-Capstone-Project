# SpaceX Falcon 9 Landing Prediction — IBM Data Science Capstone

Predicting whether a SpaceX Falcon 9 first-stage booster will land successfully, using
public launch data, SQL/visual exploratory analysis, and classification models.

## Project Overview

SpaceX advertises Falcon 9 launches at roughly $62 million, compared to $165 million+
from other providers — largely because SpaceX recovers and reuses the first-stage
booster instead of discarding it after every flight. If a booster fails to land, that
cost advantage disappears for that launch.

This project asks: **can we predict, from data available before or at launch, whether
a given Falcon 9 booster will land successfully?** A reliable answer to that question
would let a competing launch provider estimate SpaceX's real cost per launch — useful
information when bidding for the same contracts.

## Pipeline

1. **Data Collection** — pulled historical launch records from the SpaceX v4 REST API
   and cross-checked/supplemented them with a scraped Wikipedia table of Falcon 9 and
   Falcon Heavy launches.
2. **Data Wrangling** — merged both sources, filled missing payload values, and
   engineered a binary `Class` label (1 = landed, 0 = did not) from the outcome text.
3. **Exploratory Data Analysis** — used Seaborn/Matplotlib visualizations and SQL
   queries (SQLite) to look for patterns between launch site, orbit, payload mass, and
   landing outcome.
4. **Interactive Visual Analytics** — built an interactive Folium map of launch sites
   and a Plotly Dash dashboard for exploring success rate by site and payload range.
5. **Predictive Analysis** — trained and tuned four classification models (Logistic
   Regression, SVM, Decision Tree, KNN) with `GridSearchCV` and 10-fold cross-validation
   to predict landing outcome.

## Key Findings

- Landing success rate improved steadily from 2013 onward as SpaceX iterated on
  booster and landing technique.
- Orbit type and launch site are stronger predictors of landing outcome than payload
  mass on its own — payload mainly matters in combination with a demanding orbit like
  GTO.
- CCAFS LC-40 and KSC LC-39A (Florida) handle the large majority of launches; VAFB
  SLC-4E (California) is used more narrowly, mainly for polar/sun-synchronous orbits.
- Logistic Regression, SVM, and KNN tied at 83.3% test accuracy. Decision Tree scored
  highest on cross-validation (88.9%) but dropped to 77.8% on the held-out test set —
  a sign of mild overfitting, so it isn't the outright winner it first appears to be.

## Repository Contents

| Notebook | Purpose |
|---|---|
| `capstone1_jupyter-labs-spacex-data-collection-api-v2.ipynb` | Collects launch data from the SpaceX API |
| `capstone2_jupyter-labs-webscraping.ipynb` | Scrapes launch data from Wikipedia |
| `capstone3_labs-jupyter-spacex-Data_wrangling.ipynb` | Cleans and merges the collected data, engineers the `Class` label |
| `capstone4_jupyter-labs-eda-sql-coursera_sqllite.ipynb` | Exploratory analysis using SQL queries |
| `capstone5_jupyter-labs-eda-dataviz-v2.ipynb` | Exploratory analysis using visualizations |
| `capstone6_lab-jupyter-launch-site-location-v2.ipynb` | Interactive Folium map of launch sites |
| `capstone7_SpaceX-Machine-Learning-Prediction-Part-5-v1.ipynb` | Trains and tunes the classification models |
| `spacex_dash_app.py` | Interactive Plotly Dash dashboard |

## Tools & Libraries

Python, pandas, NumPy, requests, BeautifulSoup, SQLite (`%sql` magic), Seaborn,
Matplotlib, Folium, Plotly Dash, scikit-learn (`GridSearchCV`, Logistic Regression,
SVM, Decision Tree, KNN).

## How to Run

1. Clone the repo and install dependencies (`pandas`, `numpy`, `beautifulsoup4`,
   `requests`, `seaborn`, `matplotlib`, `folium`, `dash`, `scikit-learn`, `ipython-sql`).
2. Run the notebooks in order (1 → 7) — each one builds on the data produced by the
   previous step.
3. Run `spacex_dash_app.py` locally to launch the interactive dashboard.

## Author

[Your name] — IBM Data Science Professional Certificate, Applied Data Science Capstone
