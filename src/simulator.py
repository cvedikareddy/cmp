"""Objective 2: circular-economy profitability simulator.

Per tonne traded (all money in INR, CO2 in kg):
    transport      = distance * freight_rate
    consumer saves = virgin_price - price - consumer_freight_share * transport - pretreatment
    producer gains = price - (1 - consumer_freight_share) * transport + storage_avoided + disposal_avoided
    joint surplus  = virgin_price - transport - pretreatment + storage_avoided + disposal_avoided   (price cancels)
A deal is viable when the pair is spec-feasible, both sides gain, and price <= the consumer's ceiling.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd

TRADE_PARAMS = ["price", "virgin_price", "freight_rate", "distance_km", "pretreatment_cost", "volume_tpy", "capex_per_tpy"]


@dataclass
class Scenario:
    producer_id: str
    consumer_id: str
    material_id: str
    volume_tpy: float             # tonnes traded per year
    price: float                  # INR/t paid by the consumer
    virgin_price: float           # INR/t of the virgin material replaced
    distance_km: float
    freight_rate: float           # INR per tonne-km
    pretreatment_cost: float      # INR/t, consumer side (meets the consumer's spec)
    co2_saving: float             # kg CO2e/t avoided vs virgin
    co2_transport: float = 0.062  # kg CO2e per tonne-km
    max_price: float = np.inf     # consumer's price ceiling
    storage_avoided: float = 0.   # INR/t the producer no longer spends holding the material
    disposal_avoided: float = 0.  # INR/t of disposal the producer avoids (not in the data: set it yourself)
    consumer_freight_share: float = 1.0
    capex_per_tpy: float = 0.     # INR per annual tonne of consumer-side set-up
    years: int = 5
    discount: float = 0.10
    carbon_price: float = 0.      # INR per tCO2e (0 = ignore)
    feasible: bool = True         # consumer's moisture / calorific / heavy-metal specs are met


def _calc(v, price, virgin, dist, rate, pre, co2s, co2t, stor, disp, share, capex, years, disc, cp, maxp, feas):
    """Vectorised economics: every argument may be a scalar or a numpy array (used by the Monte Carlo)."""
    tr = dist * rate
    cons = virgin - price - share * tr - pre
    prod = price - (1 - share) * tr + stor + disp
    joint = virgin - tr - pre + stor + disp
    co2 = co2s - dist * co2t
    cap = capex * v
    ann = np.where(disc > 0, (1 - (1 + disc) ** -years) / np.maximum(disc, 1e-12), years)
    carbon = co2 / 1000 * cp * v
    cnpv, pnpv = cons * v * ann - cap, prod * v * ann
    return dict(transport_per_t=tr, consumer_saving_per_t=cons, producer_gain_per_t=prod, joint_surplus_per_t=joint,
                co2_net_kg_per_t=co2, annual_consumer_saving=cons * v, annual_producer_gain=prod * v,
                annual_co2_saved_t=co2 * v / 1000, capex=cap, consumer_npv=cnpv, producer_npv=pnpv,
                joint_npv=cnpv + pnpv + carbon * ann,
                roi_pct=np.where(cap > 0, cons * v / np.where(cap > 0, cap, 1) * 100, np.nan),
                payback_years=np.where((cap > 0) & (cons > 0), cap / np.where(cons * v > 0, cons * v, 1), np.nan),
                viable=np.asarray(feas, bool) & (cons > 0) & (prod > 0) & (price <= maxp))


def _args(s: Scenario):
    return (s.volume_tpy, s.price, s.virgin_price, s.distance_km, s.freight_rate, s.pretreatment_cost, s.co2_saving,
            s.co2_transport, s.storage_avoided, s.disposal_avoided, s.consumer_freight_share, s.capex_per_tpy, s.years,
            s.discount, s.carbon_price, s.max_price, s.feasible)


def evaluate(s: Scenario) -> dict:
    out = {k: (bool(x) if k == "viable" else float(x)) for k, x in _calc(*_args(s)).items()}
    # prices at which each side just breaks even, and the farthest viable distance
    out["max_price_consumer_accepts"] = min(s.virgin_price - s.consumer_freight_share * out["transport_per_t"]
                                            - s.pretreatment_cost, s.max_price)
    out["min_price_producer_accepts"] = (1 - s.consumer_freight_share) * out["transport_per_t"] - s.storage_avoided - s.disposal_avoided
    out["max_viable_distance_km"] = ((s.virgin_price - s.pretreatment_cost + s.storage_avoided + s.disposal_avoided) / s.freight_rate
                                     if s.freight_rate > 0 else np.inf)
    return out


def fair_price(s: Scenario, producer_share=0.5) -> float:
    """Price that gives the producer `producer_share` of the joint surplus (0.5 = even split)."""
    e = evaluate(s)
    p = producer_share * e["joint_surplus_per_t"] + (1 - s.consumer_freight_share) * e["transport_per_t"] - s.storage_avoided - s.disposal_avoided
    return float(min(p, s.max_price))


def sweep(s: Scenario, param: str, values) -> pd.DataFrame:
    """Re-run the scenario while one parameter takes each value (for line charts)."""
    return pd.DataFrame([{param: v, **evaluate(replace(s, **{param: v}))} for v in values])


def sensitivity(s: Scenario, pct=0.2, metric="joint_npv", params=TRADE_PARAMS) -> pd.DataFrame:
    """Tornado table: metric when each parameter is moved -pct / +pct on its own, biggest swing first."""
    base, rows = evaluate(s)[metric], []
    for p in params:
        x = getattr(s, p)
        if not x:
            continue
        lo, hi = (evaluate(replace(s, **{p: x * m}))[metric] for m in (1 - pct, 1 + pct))
        rows.append(dict(param=p, base_value=x, metric_low=lo, metric_high=hi, swing=abs(hi - lo),
                         delta_low=lo - base, delta_high=hi - base))
    return pd.DataFrame(rows).sort_values("swing", ascending=False, ignore_index=True)


def monte_carlo(s: Scenario, n=5000, seed=42, price_sigma=.10, virgin_sigma=.10, freight_sigma=.10, volume_sigma=.10,
                pretreat_sigma=.15) -> tuple[dict, pd.DataFrame]:
    """Sample uncertain inputs (lognormal / normal) and return (summary, samples). Default sigmas are annual."""
    rng = np.random.default_rng(seed)
    ln = lambda base, sg: base * np.exp(rng.normal(-sg ** 2 / 2, sg, n))  # mean-preserving lognormal
    price, virgin = ln(s.price, price_sigma), ln(s.virgin_price, virgin_sigma)
    rate, pre = ln(s.freight_rate, freight_sigma), ln(s.pretreatment_cost, pretreat_sigma)
    vol = np.clip(rng.normal(s.volume_tpy, volume_sigma * s.volume_tpy, n), 0, None)
    a = list(_args(s)); a[0], a[1], a[2], a[4], a[5] = vol, price, virgin, rate, pre
    r = _calc(*a)
    df = pd.DataFrame({"price": price, "virgin_price": virgin, "freight_rate": rate, "volume_tpy": vol,
                       **{k: r[k] for k in ("consumer_saving_per_t", "producer_gain_per_t", "consumer_npv", "producer_npv",
                                            "joint_npv", "viable")}})
    q = lambda c: {f"p{p}": float(np.percentile(df[c], p)) for p in (5, 50, 95)}
    summary = dict(p_viable=float(df.viable.mean()), consumer_npv=q("consumer_npv"), producer_npv=q("producer_npv"),
                   joint_npv=q("joint_npv"), expected_joint_npv=float(df.joint_npv.mean()),
                   value_at_risk_5pct=float(df.joint_npv.mean() - np.percentile(df.joint_npv, 5)))
    return summary, df


# ----------------------------------------------------------------- data-backed helpers
def load_data(root="data") -> dict:
    """Read the generated CSVs once; everything below takes this dict."""
    pf = pd.read_csv(Path(root, "processed", "pair_features.csv"))
    rt = pd.read_csv(Path(root, "raw", "transport_routes.csv")).set_index(["origin_industry_id", "dest_industry_id"])
    pr = pd.read_csv(Path(root, "raw", "production.csv")).groupby(["industry_id", "material_id"]).storage_cost_per_tonne.mean()
    ts = pd.read_csv(Path(root, "processed", "market_timeseries.csv"))
    ret = ts.sort_values("month").groupby("material_id").avg_price.apply(lambda x: np.log(x).diff().std())
    return dict(pf=pf, routes=rt, storage=pr, price_vol_monthly=ret)


def scenario(data, producer_id, consumer_id, material_id=None, volume_share=.5, **overrides) -> Scenario:
    """Build a Scenario from one producer-consumer pair in the dataset; any field can be overridden."""
    g = data["pf"][(data["pf"].producer_id == producer_id) & (data["pf"].consumer_id == consumer_id)]
    if material_id:
        g = g[g.material_id == material_id]
    if g.empty:
        raise KeyError(f"no candidate pair {producer_id} -> {consumer_id} ({material_id})")
    r, rt = g.iloc[0], data["routes"].loc[(producer_id, consumer_id)]
    f = float
    base = dict(producer_id=producer_id, consumer_id=consumer_id, material_id=str(r.material_id),
                volume_tpy=f(min(r.avg_monthly_tonnes * 12, r.required_tonnes) * volume_share), price=f(r.market_price),
                virgin_price=f(r.virgin_price), distance_km=f(r.distance_km), freight_rate=f(r.transport_cost_per_tonne / r.distance_km),
                pretreatment_cost=f(r.pretreatment_cost_per_tonne), co2_saving=f(r.co2_saving_per_tonne),
                co2_transport=f(rt.co2_per_tonne_km), max_price=f(r.max_price_per_tonne),
                storage_avoided=f(data["storage"].get((producer_id, r.material_id), 0)), feasible=bool(r.feasible))
    return Scenario(**{**base, **overrides})


def compare(data, pairs, **overrides) -> pd.DataFrame:
    """Side-by-side economics for several (producer, consumer[, material]) pairs, best joint NPV first."""
    rows = []
    for p in pairs:
        s = scenario(data, *p, **overrides)
        rows.append(dict(producer_id=s.producer_id, consumer_id=s.consumer_id, material_id=s.material_id,
                         distance_km=s.distance_km, volume_tpy=s.volume_tpy, price=s.price, **evaluate(s)))
    return pd.DataFrame(rows).sort_values("joint_npv", ascending=False, ignore_index=True)


def marketable_value(data, producer_id, price_mult=1., freight_mult=1., virgin_mult=1., **overrides):
    """What a producer's by-products can fetch: fill each material's output into the best viable buyers.

    Buyers are ranked by producer gain per tonne; each takes up to its annual requirement until supply runs out.
    Returns (per-buyer table, per-material summary).
    """
    rows, mat = [], []
    for m, grp in data["pf"][data["pf"].producer_id == producer_id].groupby("material_id"):
        supply, left, sold = grp.avg_monthly_tonnes.iloc[0] * 12, grp.avg_monthly_tonnes.iloc[0] * 12, 0.
        opts = []
        for c in grp.consumer_id.unique():
            s = scenario(data, producer_id, c, m, **overrides)
            s = replace(s, price=s.price * price_mult, freight_rate=s.freight_rate * freight_mult,
                        virgin_price=s.virgin_price * virgin_mult)
            e = evaluate(replace(s, volume_tpy=1.))
            opts.append((e["producer_gain_per_t"], c, s, e["viable"]))
        for gain, c, s, ok in sorted(opts, key=lambda t: -t[0]):
            cap = float(grp[grp.consumer_id == c].required_tonnes.iloc[0])
            vol = min(left, cap) if ok else 0.
            e = evaluate(replace(s, volume_tpy=vol)) if vol else evaluate(replace(s, volume_tpy=0.))
            left -= vol; sold += vol
            rows.append(dict(material_id=m, consumer_id=c, viable=ok, distance_km=s.distance_km, price=s.price,
                             tonnes_placed=vol, producer_gain_per_t=gain, annual_producer_gain=e["annual_producer_gain"],
                             annual_co2_saved_t=e["annual_co2_saved_t"]))
        t = pd.DataFrame([r for r in rows if r["material_id"] == m])
        mat.append(dict(material_id=m, supply_tpy=supply, tonnes_placed=sold, share_placed=sold / supply if supply else 0.,
                        n_viable_buyers=int(t.viable.sum()), annual_value=t.annual_producer_gain.sum(),
                        annual_co2_saved_t=t.annual_co2_saved_t.sum()))
    return pd.DataFrame(rows), pd.DataFrame(mat)


def shock_grid(data, producer_id, price_mults=(.7, .85, 1., 1.15, 1.3), freight_mults=(.8, 1., 1.25, 1.5, 2.), **kw) -> pd.DataFrame:
    """Total marketable value of a producer's by-products under combined price and freight shocks."""
    g = pd.DataFrame({f"freight x{f:g}": [marketable_value(data, producer_id, p, f, **kw)[1].annual_value.sum()
                                          for p in price_mults] for f in freight_mults}, index=[f"price x{p:g}" for p in price_mults])
    return g.round(0)


if __name__ == "__main__":
    d = load_data()
    t = d["pf"][d["pf"].label == 1].iloc[0]
    s = scenario(d, t.producer_id, t.consumer_id, t.material_id, capex_per_tpy=50)
    print("SCENARIO", s, "\n")
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in evaluate(s).items()}, "\n")
    print("fair price:", round(fair_price(s), 1), "\n")
    print(sensitivity(s).round(1).to_string(index=False), "\n")
    print({k: v for k, v in monte_carlo(s)[0].items()})