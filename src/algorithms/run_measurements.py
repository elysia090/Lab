
# run_measurements.py
# - Methodology-fixed bench (cold vs warm)
# - Small parameter sweep (grid & rgg)
# Produces CSVs and PNGs into /mnt/data

import os, sys, math, time, random, gc, statistics, importlib.util
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/mnt/data"
mod_path = os.path.join(BASE, "LARCH_deltaRAG_general.py")
spec = importlib.util.spec_from_file_location("larch_deltarag_general", mod_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["larch_deltarag_general"] = mod
spec.loader.exec_module(mod)

def build_graph(family, n, seed, oracle=False):
    if family=="grid":
        side = int(round(math.sqrt(n))); n = side*side
        wmin = 1.0 if oracle else 1.0
        wmax = 1.0 if oracle else 3.0
        g = mod.gen_grid(side, side, wmin, wmax, seed)
        scales = (3.0, 6.0, 10.0, float(max(10, side)))
        grid_exact=True; grid_oracle=oracle
    elif family=="rgg":
        g = mod.gen_rgg(n, radius=0.09, wmin=1.0, wmax=3.0, seed=seed)
        scales = (2.0, 4.0, 8.0); grid_exact=False; grid_oracle=False
    elif family=="ba":
        g = mod.gen_ba(n, m=3, wmin=1.0, wmax=3.0, seed=seed)
        scales = (3.0, 6.0, 12.0); grid_exact=False; grid_oracle=False
    else:
        raise ValueError("family")
    return g, scales, grid_exact, grid_oracle

def make_cfg(scales, seed, epsilon, cap, kappa_cap, grid_exact, grid_oracle, urn_topk):
    return mod.Config(scales=scales, seed=seed, pairs=0, epsilon=epsilon,
                      candidate_cap=cap, urn_topk=urn_topk, portals_per_intersection=1,
                      kappa_cap=kappa_cap, grid_exact=grid_exact, grid_oracle=grid_oracle)

def measure(g, cfg, pairs, repeats=4, warm=False):
    ev  = mod.Evaluator(cfg, g)
    eng = mod.QueryEngine(cfg, g, ev.cover, ev.H_small, ev.H_fp, ev.DVH, ev.urn)
    rng = random.Random(cfg.seed)
    Q = [(rng.randrange(g.n), rng.randrange(g.n)) for _ in range(pairs)]
    if warm:
        for u,v in Q: eng.query(u,v)

    def run_once():
        gc.disable()
        t0 = time.perf_counter()
        for u,v in Q: eng.query(u,v)
        t1 = time.perf_counter()
        for u,v in Q: g.dijkstra(u, targets={v})
        t2 = time.perf_counter()
        gc.enable()
        return (t1-t0)*1e6/len(Q), (t2-t1)*1e6/len(Q)

    dR, DJ = [], []
    for _ in range(repeats):
        td, tj = run_once(); dR.append(td); DJ.append(tj)

    # accuracy snapshot
    eq=0; st=[]; bounds=[]
    for u,v in Q:
        ub, lb, _ = eng.query_bounds(u,v)
        tru = g.dijkstra(u, targets={v})[v]
        if math.isfinite(ub) and math.isfinite(tru):
            if abs(ub - tru) < 1e-9: eq += 1
            if tru>0: st.append(ub/tru)
        if lb>0: bounds.append(ub/max(lb,1e-9))

    return {
        "us_per_query_dRAG_mean": statistics.mean(dR),
        "us_per_query_dRAG_std": statistics.pstdev(dR) if len(dR)>1 else 0.0,
        "us_per_query_dijkstra_mean": statistics.mean(DJ),
        "us_per_query_dijkstra_std": statistics.pstdev(DJ) if len(DJ)>1 else 0.0,
        "speedup_x_mean": statistics.mean([DJ[i]/max(dR[i],1e-9) for i in range(len(DJ))]),
        "equal_rate": eq/len(Q) if Q else 1.0,
        "mean_stretch": sum(st)/len(st) if st else float("nan"),
        "p99_stretch": (sorted(st)[int(0.99*len(st))] if st else float("nan")),
        "median_UB_over_LB": (statistics.median(bounds) if bounds else float("nan"))
    }

def bench_suite():
    out=[]
    suites = [
        ("grid", [16*16, 32*32], True),
        ("grid_orc", [16*16, 32*32], False),
        ("rgg", [300, 500], False),
        ("ba", [300, 600], False),
    ]
    for fam, ns, non_oracle in suites:
        for n in ns:
            if fam=="grid_orc":
                g, scales, grid_exact, grid_oracle = build_graph("grid", n, seed=777+n, oracle=True)
                epsilon=0.0
            elif fam=="grid":
                g, scales, grid_exact, grid_oracle = build_graph("grid", n, seed=123+n, oracle=False)
                epsilon=0.20
            else:
                g, scales, grid_exact, grid_oracle = build_graph(fam, n, seed=456+n, oracle=False)
                epsilon=0.20

            # cold
            cfg_cold = make_cfg(scales, seed=100+n, epsilon=epsilon, cap=12, kappa_cap=128,
                                 grid_exact=grid_exact, grid_oracle=grid_oracle, urn_topk=0)
            res_cold = measure(g, cfg_cold, pairs=300, repeats=3, warm=False)
            res_cold.update({"family":fam, "n":g.n, "mode":"cold"})

            # warm
            cfg_warm = make_cfg(scales, seed=200+n, epsilon=epsilon, cap=12, kappa_cap=128,
                                 grid_exact=grid_exact, grid_oracle=grid_oracle, urn_topk=500)
            res_warm = measure(g, cfg_warm, pairs=300, repeats=3, warm=True)
            res_warm.update({"family":fam, "n":g.n, "mode":"warm"})

            out += [res_cold, res_warm]

    df = pd.DataFrame(out)
    df.to_csv(os.path.join(BASE,"meas_methodology_fixed.csv"), index=False)
    return df

def sweep_small():
    epsilons = [0.30, 0.20, 0.10]
    caps = [12, 48]
    rows=[]
    for fam, ns in [("grid", [16*16, 32*32, 40*40]), ("rgg", [300, 500])]:
        for n in ns:
            for eps in epsilons:
                for cap in caps:
                    g, scales, grid_exact, grid_oracle = build_graph(fam, n, seed=900+int(eps*1000)+cap, oracle=False)
                    cfg = make_cfg(scales, seed=901+cap+n, epsilon=eps, cap=cap, kappa_cap=max(128,cap),
                                   grid_exact=grid_exact, grid_oracle=grid_oracle, urn_topk=0)
                    r = measure(g, cfg, pairs=200, repeats=3, warm=False)
                    r.update({"family":fam, "n":g.n, "epsilon":eps, "cap":cap})
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE,"sweep_small.csv"), index=False)
    return df

def plot_metric(df, family, metric, title, out_png):
    d = df[df["family"]==family].copy()
    d.sort_values(["n","epsilon","cap"], inplace=True)
    labels = sorted(d[["epsilon","cap"]].drop_duplicates().itertuples(index=False, name=None))
    plt.figure()
    for (eps, cap) in labels:
        sub = d[(d["epsilon"]==eps) & (d["cap"]==cap)]
        plt.plot(sub["n"], sub[metric], marker="o", label=f"ε={eps}, cap={cap}")
    plt.title(title); plt.xlabel("n (nodes)"); plt.ylabel(metric)
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.show()

if __name__=="__main__":
    df_bench = bench_suite()
    df_sweep = sweep_small()
    # plots for sweep
    for fam in ["grid","rgg"]:
        for metric in ["us_per_query_dRAG_mean","speedup_x_mean","equal_rate","mean_stretch","p99_stretch","median_UB_over_LB"]:
            out = os.path.join(BASE, f"sweep_small_{fam}_{metric}.png")
            plot_metric(df_sweep, fam, metric, f"{fam}: {metric} vs n (ε, cap sweep)", out)
    # dump quick summary json
    summary = {
        "bench_rows": len(df_bench),
        "sweep_rows": len(df_sweep),
    }
    with open(os.path.join(BASE,"run_summary.json"),"w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
