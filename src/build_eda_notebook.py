"""Builds and executes notebooks/03_exploratory_data_analysis.ipynb (outputs embedded).
All numbers come from eda.run(); the notebook only displays them.
Usage:  cd src && python build_eda_notebook.py"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "03_exploratory_data_analysis.ipynb"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Stage 03 — Exploratory Data Analysis\n**Georgia Income Analytics — Gender & Income Study**\n\n"
       "Descriptive analysis of net monthly earnings of employees (Geostat LFS 2017–2025) by sex. "
       "Two methodological periods (2017–2019, 2020–2025) are kept separate. Earnings are published in "
       "bands: means use band midpoints (Pareto mean for the open top band), quantiles are interpolated "
       "within bands. All statistics weighted unless stated. **Descriptive only — no causal claims.**"),
    code("import sys, pandas as pd\nfrom IPython.display import Image, display\nsys.path.insert(0, '../src')\n"
         "import eda\npd.set_option('display.width', 200, 'display.max_columns', 30)\nr = eda.run()\n"
         "print(len(r['figures']), 'figures written to reports/figures/')"),
    md("## 1. Data used"),
    code("for p in eda.PERIODS:\n    d = r['data'][p]\n    print(p, 'employees with valid earnings band:', len(d),\n"
         "          '| women', (d.gender=='female').sum(), '| men', (d.gender=='male').sum())"),
    md("## 2. Descriptive statistics by sex (pooled period, real 2025 GEL)"),
    code("d = r['desc']\nd[d.scope.str.startswith('pooled')][['period','sex','weighting','n','mean','sd','min','p25',"
         "'median','p75','max']].round(0)"),
    md("## 3. Gender income comparison by year"),
    code("cols = ['period','year','female_mean_real_weighted','male_mean_real_weighted','gap_mean_pct_real_weighted',"
         "'gap_median_pct_real_weighted','gap_mean_pct_real_unweighted','gap_hourly_mean_pct',"
         "'official_enterprise_gap_pct','official_adjusted_gpg_monthly_pct']\nr['gc'][cols]"),
    code("display(Image('../reports/figures/01_male_vs_female_mean_income.png'))\n"
         "display(Image('../reports/figures/02_male_vs_female_median_income.png'))"),
    code("display(Image('../reports/figures/03_income_distribution_histogram.png'))\n"
         "display(Image('../reports/figures/04_gender_income_boxplot.png'))\nr['groups']"),
    md("## 4. Age"), code("display(r['tables']['age'])\ndisplay(Image('../reports/figures/05_income_by_age_group.png'))"),
    md("## 5. Education"),
    code("display(r['tables']['education'])\ndisplay(Image('../reports/figures/06_income_by_education.png'))"),
    md("## 6. Regions"),
    code("display(r['tables']['region'])\ndisplay(Image('../reports/figures/07_regional_income_comparison.png'))\n"
         "display(Image('../reports/figures/09_gender_pay_gap_by_region.png'))"),
    md("## 7. Occupation and industry"),
    code("display(r['tables']['occupation'])\ndisplay(Image('../reports/figures/08_occupation_income_comparison.png'))\n"
         "display(Image('../reports/figures/11_gender_income_scatter.png'))"),
    md("## 8. Working hours: monthly vs hourly gap"),
    code("display(r['tables']['hours'])\ndisplay(r['tables']['part_time'])\n"
         "display(Image('../reports/figures/13_working_hours_and_hourly_gap.png'))"),
    md("## 9. Time series"), code("display(Image('../reports/figures/10_income_trends_by_year.png'))"),
    md("## 10. Correlations (Spearman, not causal)"),
    code("display(r['corr']['2020_2025'].loc[['log_real_income']].T)\n"
         "display(Image('../reports/figures/12_correlation_heatmap.png'))"),
    md("## 11. Household income (HIES) — separate, household-level"),
    code("display(r['hies'])\ndisplay(Image('../reports/figures/14_hies_household_income_by_head_sex.png'))"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
nbf.write(nb, NB)
print("written", NB)
