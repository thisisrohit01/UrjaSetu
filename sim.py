"""UrjaSetu feeder simulation: baseline vs forecast-driven flexibility + community battery."""
import numpy as np
H = np.arange(24)
def _bell(c, w): return np.exp(-0.5*((H-c)/w)**2)
def profiles(n_hh=150):
    crit = 0.20*(0.35+0.5*_bell(20,2.2)+0.25*_bell(7,1.5))      # kW per household: lights, fans, fridge
    flex = 0.08*(0.2+_bell(19,2.5)+0.6*_bell(7,1.5)); flex *= (H>=6)  # pumps, mills, EV charging (natural, evening-heavy)
    defr = 0.04*(0.1+_bell(20,2.0))                                  # iron, geyser
    return n_hh*crit, n_hh*flex, n_hh*defr
SCEN = {"Clear day": (1.0, 0.0), "Partly cloudy": (0.8, 0.0), "Monsoon day": (0.6, 0.0), "Evening grid stress": (1.0, 0.25)}
def supply(cap, solar_cut, eve_cut):
    sol = _bell(12.5, 3.2)*(H>=7)*(H<=18)
    a = 0.62 + 0.38*(sol*solar_cut)                 # availability of feeder capacity (renewable-dependent)
    a = a*(1-eve_cut*((H>=18)&(H<=22)))
    return cap*a
def day(scen, n_hh=150, cap=None, batt_kwh=60, batt_kw=20, flex_share=1.0, noise=0.08, urja=True, rng=None, soc0=0.5):
    rng = rng or np.random.default_rng(0)
    C, F, D = profiles(n_hh); F = F*flex_share
    cap = cap or 0.9*(C+F+D).max()
    S = supply(cap, *SCEN[scen])
    if not urja:
        dem = C+F+D; srv = np.minimum(dem, S)
        short = np.maximum(0, dem-S); frac = short/dem
        return dict(S=S, demand=dem, served=srv, unserved=short, crit_un=C*frac, soc=np.full(24, np.nan), outage=(short>1e-6), crit_out=(C*frac>0.05*C), grid=srv, peak=srv.max())
    Sh = S*(1+rng.normal(0, noise, 24))              # forecast with error
    # shift flexible energy into forecast-headroom hours
    Ft = np.zeros(24); e = F.sum(); head = np.maximum(0, Sh-C-D*0.5)*((H>=8)&(H<=22))
    order = np.argsort(-head)
    for h in order:
        if e <= 0: break
        x = min(e, head[h], F.max()*1.6); Ft[h] += x; e -= x
    Ft[(H>=8)&(H<=22)] += e/15 if e>0 else 0
    bdis = np.zeros(24); grid = np.zeros(24); soc = soc0*batt_kwh; served = np.zeros(24); un = np.zeros(24); cun = np.zeros(24); socs = np.zeros(24)
    for h in H:
        crit, flx, df = C[h], Ft[h], D[h]
        avail = S[h]; use_c = min(crit, avail); rem = avail-use_c
        gap_c = crit-use_c
        if gap_c > 0:                                  # battery covers critical first
            d = min(gap_c, batt_kw, soc*0.9); soc -= d/0.9; use_c += d; gap_c -= d
        use_f = min(flx, rem); rem -= use_f
        gap_f = flx-use_f
        if gap_f > 0 and gap_c <= 1e-9:
            d = min(gap_f, batt_kw*0.5, max(0, soc-0.3*batt_kwh)*0.9); soc -= d/0.9; use_f += d; gap_f -= d
        use_d = min(df, rem); rem -= use_d
        gch = 0
        if rem > 0 and soc < batt_kwh:                 # charge from headroom
            ch = min(rem, batt_kw, (batt_kwh-soc)/0.9); soc += ch*0.9; gch = ch
        served[h] = use_c+use_f+use_d
        bdis[h] = max(0, served[h]-min(S[h], served[h])) + (use_c+use_f - min(use_c+use_f, avail))*0
        grid[h] = min(S[h], use_c+use_f+use_d) + gch
        un[h] = max(0, (crit+flx+df)-(use_c+use_f+use_d)); cun[h] = max(0, crit-use_c); socs[h] = soc
    return dict(S=S, demand=C+F+D, served=served, unserved=un, crit_un=cun, soc=socs, outage=(un>1e-6), crit_out=(cun>0.05*C), grid=grid, peak=grid.max(), shifted=Ft)
def evaluate(n_days=60, seed=1, **kw):
    rng = np.random.default_rng(seed); names = list(SCEN); out = {}
    picks = [names[i] for i in rng.integers(0, 4, n_days)]
    for mode in ("baseline", "urjasetu"):
        o = []
        for sc in picks:
            r = day(sc, urja=(mode=="urjasetu"), rng=np.random.default_rng(rng.integers(1e9)), **kw)
            win = (H>=17)&(H<=23)
            o.append((r["crit_out"][win].sum(), r["crit_out"].sum(), r["unserved"].sum(), r["crit_un"].sum(), r["grid"][win].max()))
        a = np.array(o); out[mode] = dict(eve_crit_outage_h=a[:,0].mean(), crit_outage_h=a[:,1].mean(), unserved_kwh=a[:,2].mean(), crit_unserved_kwh=a[:,3].mean(), eve_peak_kw=a[:,4].mean())
    return out, picks
if __name__ == "__main__":
    import json
    res, _ = evaluate()
    b, u = res["baseline"], res["urjasetu"]
    print(json.dumps(res, indent=1))
    for k in b: print(f"{k}: {b[k]:.2f} -> {u[k]:.2f}  ({100*(1-u[k]/b[k]) if b[k] else 0:.0f}% reduction)")
    for sc in SCEN:
        rb = day(sc, urja=False); ru = day(sc, urja=True)
        print(f"{sc}: evening critical-outage hrs {rb['crit_out'][17:24].sum()} -> {ru['crit_out'][17:24].sum()}, crit unserved {rb['crit_un'].sum():.1f} -> {ru['crit_un'].sum():.1f} kWh")
