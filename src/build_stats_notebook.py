"""Builds and executes notebooks/04_statistical_analysis.ipynb (outputs embedded).
The notebook shows the saved Stage 04 results and re-checks key numbers independently.
Full recomputation:  cd src && python survey_analysis.py && python hypothesis_testing.py
                     && python regression_analysis.py && python decomposition.py
Usage:  cd src && python build_stats_notebook.py"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "04_statistical_analysis.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
F = "../reports/figures/stage_04/"
cells = [
    md("# Stage 04 — Statistical Analysis & Hypothesis Testing\nGeorgia Income Analytics — Gender & Income Study.\n\n"
       "Geostat LFS employees, real (2025 GEL) net monthly earnings; periods 2017–2019 and 2020–2025 separate. "
       "**Primary** tests are survey-adjusted (weights + household-clustered SE); classical tests are shown for "
       "reference. Strata/PSU are not published, so this is an approximation of the full design. "
       "Regression and decomposition describe associations, **not causal effects**."),
    code("import sys, numpy as np, pandas as pd\nfrom IPython.display import Image, display\nsys.path.insert(0,'../src')\n"
         "pd.set_option('display.width',200,'display.max_colwidth',80,'display.max_columns',30)\nR='../reports/'"),
    md("## 1. Survey design"), code("pd.read_csv(R+'survey_design_summary.csv').T"),
    md("## 2. Hypotheses and tests (BH-adjusted p-values)"),
    code("h = pd.read_csv(R+'hypothesis_results.csv')\nh[['period','hypothesis','H0','test','statistic','p_value',"
         "'p_value_bh','estimate','ci_low','ci_high','estimate_meaning']]"),
    md("## 3. Gender pay gap with 95% CI"), code("pd.read_csv(R+'gap_confidence_intervals.csv')"),
    code(f"display(Image('{F}01_confidence_interval_comparison.png'))"),
    md("## 4. Gaps by subgroup"),
    code("pd.read_csv(R+'subgroup_gaps.csv')[['period','dimension','group','n_female','n_male','reliability',"
         "'gap_mean_pct','ci_low','ci_high','p_value_bh']]"),
    code(f"for f in ['02_gender_pay_gap_by_age','03_gender_pay_gap_by_education','04_gender_pay_gap_by_region']:\n"
         f"    display(Image('{F}'+f+'.png'))"),
    md("## 5. Effect sizes"), code(f"display(pd.read_csv(R+'effect_sizes.csv'))\ndisplay(Image('{F}06_effect_size_comparison.png'))"),
    md("## 6. Regression: female coefficient as % (exp(b)-1)"),
    code("r = pd.read_csv(R+'regression_results.csv')\nr[r.term=='female'][['period','model','coef','pct_effect',"
         "'pct_ci_low','pct_ci_high','p_value','n','r2','adj_r2']]"),
    code("display(pd.read_csv(R+'vif_results.csv'))\ndisplay(pd.read_csv(R+'regression_diagnostics.csv'))"),
    code(f"display(Image('{F}05_regression_coefficient_plot.png'))"),
    md("## 7. Oaxaca–Blinder decomposition"),
    code(f"display(pd.read_csv(R+'oaxaca_results.csv'))\ndisplay(Image('{F}07_oaxaca_blinder_decomposition.png'))"),
    md("## 8. Independent re-check of headline numbers"),
    code("import survey_analysis as sa, regression_analysis as ra\nd = sa.load_period('2020_2025')\n"
         "f, m = d[d.female==1], d[d.female==0]\nwm, wf = np.average(m.y, weights=m.w), np.average(f.y, weights=f.w)\n"
         "print('weighted mean gap % (recomputed):', round((wm-wf)/wm*100, 2))\n"
         "s = ra.estimation_sample(d)\nb = sa.wls_cluster('log_y ~ female', s).params['female']\n"
         "print('M1 log gap -> % :', round((np.exp(b)-1)*100, 2))"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
nbf.write(nb, NB)
print("written", NB)
