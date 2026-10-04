import numpy as np, streamlit as st, pandas as pd
from sim import day, evaluate, SCEN, H
st.set_page_config(page_title="UrjaSetu", layout="wide")
st.title("UrjaSetu: neighbourhood flexibility simulator")
st.caption("Synthetic feeder: critical / flexible / deferrable loads, variable supply, shared second-life battery.")
c = st.sidebar
n = c.slider("Households", 50, 300, 150, 10); kwh = c.slider("Battery (kWh)", 0, 120, 60, 10)
kw = c.slider("Battery power (kW)", 5, 60, 20, 5); fx = c.slider("Flexible load share", 0.0, 2.0, 1.0, 0.25)
sc = c.selectbox("Scenario", list(SCEN))
b = day(sc, n_hh=n, urja=False); u = day(sc, n_hh=n, batt_kwh=kwh, batt_kw=kw, flex_share=fx)
t1, t2, t3 = st.tabs(["Day view", "60-day evaluation", "DISCOM view"])
with t1:
    m = st.columns(3)
    m[0].metric("Critical-outage hours (baseline)", int(b["crit_out"].sum()))
    m[1].metric("Critical-outage hours (UrjaSetu)", int(u["crit_out"].sum()))
    m[2].metric("Critical energy unserved (kWh)", f"{u['crit_un'].sum():.1f}", f"{u['crit_un'].sum()-b['crit_un'].sum():.1f}")
    st.line_chart(pd.DataFrame({"Supply cap": u["S"], "Baseline demand": b["demand"], "UrjaSetu grid draw": u["grid"]}, index=H))
    st.line_chart(pd.DataFrame({"Battery state (kWh)": u["soc"], "Shifted flexible load (kW)": u["shifted"]}, index=H))
with t2:
    r, _ = evaluate(n_hh=n, batt_kwh=kwh, batt_kw=kw, flex_share=fx)
    st.dataframe(pd.DataFrame(r).round(2))
    st.line_chart(pd.DataFrame({k: [evaluate(n_hh=n, batt_kwh=x, batt_kw=max(1, x/3), flex_share=fx)[0]["urjasetu"]["crit_unserved_kwh"] for x in range(0, 121, 20)] for k in ["Critical unserved kWh vs battery size (0-120 kWh)"]}))
with t3:
    stress = (u["S"] < u["demand"]); st.subheader("Predicted feeder stress by hour")
    st.bar_chart(pd.DataFrame({"Shortfall vs supply (kW)": np.clip(u["demand"]-u["S"], 0, None)}, index=H))
    hrs = [int(h) for h in H[stress]]
    st.info(f"Recommended DR action: pre-shift flexible loads before hour {hrs[0]}:00 and reserve battery for hours {hrs}." if hrs else "No stress predicted.")
