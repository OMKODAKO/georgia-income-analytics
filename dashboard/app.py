"""
Stage 06 — Georgia Income Analytics dashboard (Streamlit).

Run:   .venv/bin/streamlit run dashboard/app.py
Data:  database/income_data.db + reports/*.csv + models/ (all produced by Stages 01-05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D

st.set_page_config(page_title="Georgia Income Analytics", page_icon="📊", layout="wide")
F, M = "#C44E52", "#4C72B0"
COLORS = {"ქალი": F, "კაცი": M}


def gel(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f} ₾"


def pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}%"


def note(text):
    st.caption(text)


# ------------------------------------------------------------------ sidebar ---
st.sidebar.title("ფილტრები")
period = st.sidebar.radio("პერიოდი (მეთოდოლოგია)", list(D.PERIODS), format_func=D.PERIODS.get, index=1,
                          help="2020 წელს სამუშაო ძალის კვლევის მეთოდოლოგია შეიცვალა — პერიოდები არ ერთიანდება.")
opt = D.options(period)
years = st.sidebar.multiselect("წლები", opt["years"], default=opt["years"])
ages = st.sidebar.multiselect("ასაკობრივი ჯგუფი", opt["ages"], default=[], placeholder="ყველა")
regions = st.sidebar.multiselect("რეგიონი", opt["regions"], default=[], placeholder="ყველა",
                                 format_func=lambda r: D.REGION_KA.get(r, r))
education = st.sidebar.multiselect("განათლება", opt["education"], default=[], placeholder="ყველა",
                                   format_func=lambda e: D.EDU_KA.get(e, e))
occupation = st.sidebar.multiselect("პროფესია", opt["occupation"], default=[], placeholder="ყველა",
                                    format_func=lambda o: D.OCC_KA.get(o, o))
st.sidebar.caption("ცარიელი ფილტრი = ყველა. ყველა თანხა — რეალური, 2025 წლის ფასებში.")


@st.cache_data(show_spinner="ვითვლი…")
def filtered(period, years, ages, regions, education, occupation):
    return D.apply_filters(period, list(years), list(ages), list(regions), list(education), list(occupation))


d = filtered(period, tuple(years), tuple(ages), tuple(regions), tuple(education), tuple(occupation))

st.title("📊 საქართველო: შემოსავლები და სქესი")
st.markdown(f"**{D.PERIODS[period]}** · დაქირავებულები · ჩანაწერები: **{len(d):,}** · "
            f"ხელზე ასაღები თვიური ხელფასი, ₾ 2025 წლის ფასებში")

tabs = st.tabs(["მიმოხილვა", "ჯგუფები", "დინამიკა", "სტატისტიკური შედეგები", "Machine Learning", "მეთოდოლოგია"])

# ---------------------------------------------------------------- overview ---
with tabs[0]:
    if len(d) == 0:
        st.warning("არჩეულ ფილტრებში მონაცემი არ არის.")
    else:
        s = D.gender_summary(d)
        if s["reliability"] == "suppressed":
            st.error(f"ძალიან მცირე შერჩევა (ქალი {s['female']['n']}, კაცი {s['male']['n']}) — შედეგი არ ნაჩვენებია.")
        else:
            if s["reliability"] == "low_n":
                st.warning("100-ზე ნაკლები დაკვირვება ერთ-ერთ სქესში — შედეგი არასანდოა, მყარი დასკვნა არ გააკეთოთ.")
            c = st.columns(4)
            c[0].metric("ქალი — საშუალო", gel(s["female"]["mean"]))
            c[0].caption(f"მედიანა: {gel(s['female']['median'])}")
            c[1].metric("კაცი — საშუალო", gel(s["male"]["mean"]))
            c[1].caption(f"მედიანა: {gel(s['male']['median'])}")
            c[2].metric("სხვაობა (საშუალო)", pct(s["gap_mean"]), help="(კაცი − ქალი) / კაცი")
            c[3].metric("სხვაობა (მედიანა)", pct(s["gap_median"]))
            left, right = st.columns([1, 1])
            with left:
                t = pd.DataFrame({"სქესი": ["ქალი", "კაცი"] * 2,
                                  "საზომი": ["საშუალო"] * 2 + ["მედიანა"] * 2,
                                  "₾": [s["female"]["mean"], s["male"]["mean"], s["female"]["median"], s["male"]["median"]]})
                fig = px.bar(t, x="საზომი", y="₾", color="სქესი", barmode="group", color_discrete_map=COLORS,
                             title="საშუალო და მედიანური ხელფასი სქესის მიხედვით", text_auto=".0f")
                fig.update_layout(yaxis_title="₾ თვეში (2025 ფასები)", xaxis_title="", legend_title="")
                st.plotly_chart(fig, width="stretch")
            with right:
                b = D.band_shares(d).reset_index().melt(id_vars="index", var_name="სქესი", value_name="წილი")
                fig = px.bar(b, x="index", y="წილი", color="სქესი", barmode="group", color_discrete_map=COLORS,
                             title="განაწილება ხელფასის ინტერვალებში (ნომინალური ₾)")
                fig.update_layout(yaxis_tickformat=".0%", xaxis_title="ინტერვალი, ₾", legend_title="")
                st.plotly_chart(fig, width="stretch")
            t = pd.DataFrame({"": ["დაკვირვება", "შეწონილი რაოდენობა (საშ. წელი)", "საშუალო", "P25", "მედიანა", "P75"],
                              "ქალი": [s["female"]["n"], round(s["female"]["population"]), *[round(s["female"][k])
                                                                                              for k in ("mean", "p25", "median", "p75")]],
                              "კაცი": [s["male"]["n"], round(s["male"]["population"]), *[round(s["male"][k])
                                                                                         for k in ("mean", "p25", "median", "p75")]]})
            st.dataframe(t, hide_index=True, width="stretch")
    note(D.SOURCE + " მედიანა და კვარტილები გამოთვლილია ინტერვალის შიგნით ინტერპოლაციით.")

# ------------------------------------------------------------------ groups ---
with tabs[1]:
    dims = {"ასაკი": ("age_group", ["15-17"] + D.eda.AGE_LABELS, {}),
            "განათლება": ("education_eda", D.eda.EDU_ORDER, D.EDU_KA),
            "რეგიონი": ("region_eda", None, D.REGION_KA),
            "პროფესია": ("occupation_label", list(D.eda.ISCO1.values()), D.OCC_KA)}
    dim = st.radio("დაჯგუფება", list(dims), horizontal=True)
    col, order, ka = dims[dim]
    g = D.by_group(d, col, order)
    if g.empty:
        st.warning("მონაცემი არ არის.")
    else:
        g["label"] = g.group.map(lambda x: ka.get(x, x)) + np.where(g.reliability == "low_n", " *", "")
        shown = g[g.reliability != "suppressed"]
        long = shown.melt(id_vars="label", value_vars=["female_mean", "male_mean"], var_name="სქესი", value_name="₾")
        long["სქესი"] = long["სქესი"].map({"female_mean": "ქალი", "male_mean": "კაცი"})
        fig = px.bar(long, x="₾", y="label", color="სქესი", orientation="h", barmode="group",
                     color_discrete_map=COLORS, title=f"საშუალო ხელფასი — {dim}", height=120 + 45 * len(shown))
        fig.update_layout(yaxis_title="", xaxis_title="₾ თვეში (2025 ფასები)", legend_title="",
                          yaxis={"categoryorder": "array", "categoryarray": list(shown.label)[::-1]})
        st.plotly_chart(fig, width="stretch")
        fig = px.bar(shown, x="gap_mean", y="label", orientation="h", title=f"სქესებს შორის სხვაობა (%) — {dim}",
                     color_discrete_sequence=["#55A868"], height=120 + 35 * len(shown), text_auto=".1f")
        fig.update_layout(yaxis_title="", xaxis_title="(კაცი − ქალი) / კაცი, %",
                          yaxis={"categoryorder": "array", "categoryarray": list(shown.label)[::-1]})
        st.plotly_chart(fig, width="stretch")
        tab = g.assign(group=g.group.map(lambda x: ka.get(x, x)))[
            ["group", "n_female", "n_male", "female_mean", "male_mean", "female_median", "male_median",
             "gap_mean", "gap_median", "reliability"]].round(1)
        tab.columns = ["ჯგუფი", "n ქალი", "n კაცი", "ქალი საშ.", "კაცი საშ.", "ქალი მედ.", "კაცი მედ.",
                       "სხვაობა საშ. %", "სხვაობა მედ. %", "სანდოობა"]
        st.dataframe(tab, hide_index=True, width="stretch")
        note("* = 100-ზე ნაკლები დაკვირვება ერთ სქესში (დასკვნა არ კეთდება); 30-ზე ნაკლები — დამალულია. " + D.SOURCE)

# ------------------------------------------------------------------ trends ---
with tabs[2]:
    st.markdown("წლიური მონაცემები ფილტრების მიხედვით (წლის ფილტრის გარდა). **2017–19 და 2020–25 ცალ-ცალკეა** — "
                "მეთოդოლოგიური წყვეტა.")
    show_both = st.checkbox("ორივე პერიოდის ჩვენება (ცალ-ცალკე ხაზებად)", value=True)
    per = list(D.PERIODS) if show_both else [period]
    frames = []
    for p in per:
        o = D.options(p)
        dd = filtered(p, tuple(o["years"]), tuple(a for a in ages if a in o["ages"]),
                      tuple(r for r in regions if r in o["regions"]), tuple(education), tuple(occupation))
        y = D.by_year(dd)
        if not y.empty:
            frames.append(y.assign(period=D.PERIODS[p]))
    if frames:
        ty = pd.concat(frames)
        c1, c2 = st.columns(2)
        fig = go.Figure()
        for p_, q in ty.groupby("period"):
            fig.add_scatter(x=q.year, y=q.female_mean, name=f"ქალი, {p_}", line=dict(color=F), mode="lines+markers")
            fig.add_scatter(x=q.year, y=q.male_mean, name=f"კაცი, {p_}", line=dict(color=M), mode="lines+markers")
        fig.add_vline(x=2019.5, line_dash="dash", line_color="grey")
        fig.update_layout(title="საშუალო რეალური ხელფასი", yaxis_title="₾ (2025 ფასები)", xaxis_title="წელი",
                          legend=dict(orientation="h", y=-0.25))
        c1.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        ref = D.report_csv("gender_income_comparison")
        ref = ref[ref.year != "pooled"].assign(year=lambda x: x.year.astype(int))
        for p_, q in ty.groupby("period"):
            fig.add_scatter(x=q.year, y=q.gap_mean, name=f"საშუალო, {p_}", line=dict(color="#55A868"), mode="lines+markers")
            fig.add_scatter(x=q.year, y=q.gap_median, name=f"მედიანა, {p_}", line=dict(color="#8172B2"), mode="lines+markers")
        fig.add_scatter(x=ref.year, y=ref.official_adjusted_gpg_monthly_pct, name="საქსტატი, კორექტირებული (ყველა)",
                        mode="markers", marker=dict(symbol="x", size=10, color="black"))
        fig.add_vline(x=2019.5, line_dash="dash", line_color="grey")
        fig.update_layout(title="სხვაობა სქესებს შორის, %", yaxis_title="%", xaxis_title="წელი",
                          legend=dict(orientation="h", y=-0.25))
        c2.plotly_chart(fig, width="stretch")
        ty_show = ty.round(1).rename(columns={"year": "წელი", "female_mean": "ქალი საშ.", "male_mean": "კაცი საშ.",
                                              "female_median": "ქალი მედ.", "male_median": "კაცი მედ.",
                                              "gap_mean": "სხვაობა საშ. %", "gap_median": "სხვაობა მედ. %",
                                              "reliability": "სანდოობა", "period": "პერიოდი"})
        st.dataframe(ty_show, hide_index=True, width="stretch")
    note("ყოველი წელი — თავისი წლიური წონებით. საქსტატის კორექტირებული სხვაობა ეხება ყველა დაქირავებულს (ფილტრების გარეშე). "
         + D.SOURCE)

# ------------------------------------------------------------- statistics ---
with tabs[3]:
    st.markdown("Stage 04-ის შედეგები (მთელ მონაცემზე; ფილტრები აქ არ მოქმედებს). შეწონილი, ოჯახებზე კლასტერიზებული "
                "სტანდარტული შეცდომებით.")
    ci = D.report_csv("gap_confidence_intervals")
    ci = ci[ci.measure.isin(["mean gap %", "median gap %", "hourly median gap %"])]
    ci["label"] = ci.measure.map({"mean gap %": "საშუალო", "median gap %": "მედიანა",
                                  "hourly median gap %": "საათობრივი მედიანა"}) + " · " + ci.period.map(D.PERIODS)
    fig = go.Figure(go.Scatter(x=ci.estimate, y=ci.label, mode="markers",
                               error_x=dict(type="data", array=ci.ci_high - ci.estimate,
                                            arrayminus=ci.estimate - ci.ci_low), marker=dict(size=10, color="#55A868")))
    fig.update_layout(title="დაუკორექტირებელი სხვაობა, 95% ნდობის ინტერვალი", xaxis_title="%", height=350)
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    reg = D.report_csv("regression_results")
    lad = reg[(reg.term == "female") & reg.model.str.match(r"(M\d|H7|IR7)")].copy()
    lad["period"] = lad.period.map(D.PERIODS)
    fig = px.scatter(lad, x="pct_effect", y="model", color="period", error_x=lad.pct_ci_high - lad.pct_effect,
                     error_x_minus=lad.pct_effect - lad.pct_ci_low, color_discrete_sequence=["#999999", "#55A868"],
                     title="„ქალის“ კოეფიციენტი მოდელებში (%)")
    fig.update_layout(yaxis={"autorange": "reversed"}, xaxis_title="% (exp(b) − 1)", yaxis_title="", legend_title="")
    c1.plotly_chart(fig, width="stretch")
    oax = D.report_csv("oaxaca_results")
    oax = oax[(oax.reference == "pooled (Neumark)") & oax.component.isin(["explained", "unexplained"])]
    oax = oax.assign(component=oax.component.map({"explained": "ახსნილი", "unexplained": "აუხსნელი"}),
                     period=oax.period.map(D.PERIODS))
    fig = px.bar(oax, x="period", y="share_of_gap_pct", color="component", text_auto=".0f",
                 color_discrete_sequence=["#4C72B0", "#C44E52"], title="Oaxaca-Blinder: სხვაობის წილები (%)")
    fig.update_layout(yaxis_title="% სხვაობიდან", xaxis_title="", legend_title="")
    c2.plotly_chart(fig, width="stretch")
    st.info("ყველა დაკვირვებადი ფაქტორის (განათლება, პროფესია, დარგი, საათები, რეგიონი…) გათვალისწინებით სხვაობა "
            "~17% (2020–25) / ~20% (2017–19) რჩება. აუხსნელი ნაწილი **არ ნიშნავს** დისკრიმინაციას — მასში შედის "
            "ყველაფერი, რაც მონაცემებში არ არის (სტაჟი, საწარმო, თანამდებობა…).")
    hyp = D.report_csv("hypothesis_results")
    hyp = hyp[hyp.test.str.contains("PRIMARY")][["period", "hypothesis", "H0", "statistic", "p_value_bh", "reject_h0_bh"]]
    st.dataframe(hyp, hide_index=True, width="stretch")

# ---------------------------------------------------------------------- ML ---
with tabs[4]:
    EXP_KA = {"A_gender_included": "A: სქესით", "B_gender_excluded": "B: სქესის გარეშე"}
    comp = D.report_csv("ml_model_comparison").assign(experiment=lambda x: x.experiment.map(EXP_KA))
    t = comp[comp.split == "test"]
    c1, c2 = st.columns(2)
    fig = px.bar(t, x="MAE", y="model", color="experiment", barmode="group", orientation="h",
                 color_discrete_sequence=[M, "#BBBBBB"], title="მოდელების შეცდომა (MAE, ₾) — test set")
    fig.update_layout(xaxis_range=[t.MAE.min() * 0.9, t.MAE.max() * 1.02], yaxis_title="", legend_title="")
    c1.plotly_chart(fig, width="stretch")
    imp = D.report_csv("ml_permutation_importance")
    imp = imp[imp.experiment == "A_gender_included"].sort_values("mae_increase_gel")
    fig = px.bar(imp, x="mae_increase_gel", y="feature", orientation="h", title="ფაქტორების მნიშვნელობა (permutation)",
                 color_discrete_sequence=[M])
    fig.update_layout(xaxis_title="MAE-ის ზრდა ცვლადის არევისას, ₾", yaxis_title="")
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(t[["experiment", "model", "MAE", "RMSE", "R2", "MedAE"]].round(3)
                 .rename(columns={"experiment": "ექსპერიმენტი", "model": "მოდელი", "MedAE": "მედიანური შეცდომა"}),
                 hide_index=True,
                 width="stretch")
    fair = D.report_csv("ml_fairness_by_gender").assign(experiment=lambda x: x.experiment.map(EXP_KA))
    fair = fair[(fair.model == "XGBoost") & fair.gender.isin(["female", "male"])]
    fair = fair.assign(gender=fair.gender.map({"female": "ქალი", "male": "კაცი"}))
    fig = px.bar(fair, x="gender", y="mean_error", color="experiment", barmode="group",
                 color_discrete_sequence=[M, "#BBBBBB"], title="საშუალო შეცდომა სქესის მიხედვით (პროგნოზი − რეალური)")
    fig.update_layout(yaxis_title="₾", xaxis_title="", legend_title="")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**სქესის გარეშე მოდელი (B)** ქალებს აჭარბებს, კაცებს ამცირებს — სქესის ამოღება სამართლიან პროგნოზს არ "
                "იძლევა; სქესის ინფორმაცია სხვა ცვლადებშიც არის (ROC AUC 0.89).")

    st.subheader("პროგნოზი პროფილისთვის")
    st.caption("სადემონსტრაციო: შეიყვანეთ **გამოგონილი** პროფილი. შედეგი — მოდელის შეფასება (XGBoost, LFS 2020–25), "
               "არა რეალური ადამიანის ხელფასი; არ გამოიყენება გადაწყვეტილებებისთვის.")
    meta = D.model_metadata()
    cat = meta["categories"]
    with st.form("predict"):
        a, b, c = st.columns(3)
        prof = dict(
            age=a.number_input("ასაკი", 18, 75, 35), working_hours=a.number_input("კვირის საათები", 5, 84, 40),
            household_size=a.number_input("ოჯახის ზომა", 1, 12, 3),
            education=b.selectbox("განათლება", cat["education"], index=cat["education"].index("Bachelor"),
                                  format_func=lambda e: D.EDU_KA.get(e, e)),
            occupation=b.selectbox("პროფესია", cat["occupation"], index=cat["occupation"].index("Professionals"),
                                   format_func=lambda o: D.OCC_KA.get(o, o)),
            industry=b.selectbox("დარგი", cat["industry"], index=cat["industry"].index("Education")),
            region=c.selectbox("რეგიონი", cat["region"], index=cat["region"].index("Tbilisi"),
                               format_func=lambda r: D.REGION_KA.get(r, r)),
            sector=c.selectbox("სექტორი", cat["sector"], index=cat["sector"].index("private"),
                               format_func={"private": "კერძო", "state": "სახელმწიფო", "other": "სხვა"}.get),
            contract=c.selectbox("კონტრაქტი", cat["contract"], index=cat["contract"].index("written")),
        )
        go_ = st.form_submit_button("პროგნოზი")
    if go_:
        import predict_income as pi
        base = dict(prof, urban_rural="urban", informality="formal", survey_year=2025,
                    part_time="part-time" if prof["working_hours"] < 35 else "full-time",
                    potential_experience=max(prof["age"] - 6 - {"Below secondary": 9, "Secondary general": 12,
                                                                 "Vocational": 12, "Higher professional": 14,
                                                                 "Bachelor": 16, "Master": 18, "Doctor": 21}[prof["education"]], 0))
        rows = pd.DataFrame([dict(base, gender="female"), dict(base, gender="male")])
        out = pi.predict(rows)
        f_, m_ = out.predicted_net_monthly_earnings_gel_2025
        k = st.columns(3)
        k[0].metric("ქალი", gel(f_))
        k[1].metric("კაცი", gel(m_))
        k[2].metric("სხვაობა", pct((m_ - f_) / m_ * 100))
        st.caption("იგივე პროფილი, მხოლოდ სქესი განსხვავდება — ეს მოდელის თვისებაა (ასახავს მონაცემებში არსებულ "
                   "კავშირს), არა მიზეზშედეგობრივი ეფექტი.")

# ------------------------------------------------------------- methodology ---
with tabs[5]:
    st.markdown("""
### მონაცემები
- **საქსტატი — სამუშაო ძალის კვლევა (LFS) 2017–2025**, საჯარო ანონიმიზებული მიკრომონაცემები; დაქირავებულები
  ვალიდური ხელფასის ინტერვალით (2017–19: 40,141; 2020–25: 81,166 ჩანაწერი).
- ხელფასი: **ხელზე ასაღები, ძირითადი სამუშაო, ბოლო ნამუშევარი თვე**, ₾; ქვეყნდება **ინტერვალებით** →
  თანხა = ინტერვალის შუა წერტილი (ზედა ღია ინტერვალში Pareto), რეალური — CPI-ით 2025 წლის ფასებში.
- წონები: პირის წონა; რამდენიმე წლის გაერთიანებისას წონა იყოფა წლების რაოდენობაზე.

### მეთოდი
- სხვაობა = (კაცი − ქალი) / კაცი × 100. მედიანა — ინტერვალის შიგნით ინტერპოლაციით.
- **2017–19 და 2020–25 არ ერთიანდება** (2020 წლის მეთოდოლოგიური ცვლილება, ILO-ს ახალი სტანდარტი).
- 30-ზე ნაკლები დაკვირვება ერთ სქესში — შედეგი დამალულია; 100-ზე ნაკლები — „*", დასკვნა არ კეთდება.

### შეზღუდვები
- მხოლოდ დაქირავებულები; ხელფასის კითხვაზე უარი იზრდება (2025 — 23%); სამუშაო სტაჟი, საწარმო და თანამდებობა არ არის.
- შერჩევის სრული დიზაინი (strata/PSU) არ ქვეყნდება.
- აღწერითი და კორელაციური შედეგები — **არა მიზეზშედეგობრივი**.

ანგარიშები: `reports/stage00_feasibility_report.md` … `reports/stage_05_report.md`.
""")
