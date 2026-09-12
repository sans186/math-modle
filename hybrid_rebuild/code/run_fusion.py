#!/usr/bin/env python3
"""融合重算版：严格时序预测、跨日 SOC、发布时刻组合与波动价格。"""
from __future__ import annotations

import itertools
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_hybrid as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code" / "outputs"
FIG = ROOT / "figures"
REPORT = ROOT / "reports" / "RESULTS_REPORT.md"
T = core.T
DT = core.DT


def forecast_asof(x: np.ndarray, target: int, cutoff: int) -> np.ndarray:
    """Forecast target-day profile using rows strictly before cutoff."""
    if cutoff <= 0:
        return np.zeros(T)
    recent = np.median(x[max(0, cutoff - 7):cutoff], axis=0)
    idx = np.array([j for j in range(max(0, cutoff - 35), cutoff) if j % 7 == target % 7], dtype=int)
    same = np.median(x[idx[-4:]], axis=0) if len(idx) else recent
    return 0.65 * same + 0.35 * recent


def prediction_matrices(x: np.ndarray) -> np.ndarray:
    return np.vstack([forecast_asof(x, i, i) for i in range(len(x))])


def margin_asof(actual_net: np.ndarray, pred_net: np.ndarray, cutoff: int, q: float = 0.8) -> np.ndarray:
    lo = max(1, cutoff - 28)
    if cutoff - lo < 7:
        return np.zeros(T)
    residual = actual_net[lo:cutoff] - pred_net[lo:cutoff]
    residual = residual[np.isfinite(residual).all(axis=1)]
    return np.quantile(residual, q, axis=0) if len(residual) else np.zeros(T)


def profile48(x: np.ndarray, i: int, pred: np.ndarray) -> np.ndarray:
    nxt = forecast_asof(x, i + 1, i) if i + 1 < len(x) else pred[i]
    return np.r_[pred[i], nxt]


def slice_schedule(s: core.Schedule, n: int = T) -> core.Schedule:
    g = s.g[:n].copy(); c = s.charge[:n].copy(); d = s.discharge[:n].copy(); soc = s.soc[:n + 1].copy()
    return core.Schedule(g, c, d, soc, s.objective, s.status, s.nit, s.runtime,
                         s.balance_slack_min, s.soc_residual_max, s.simultaneous_max)


def zero_schedule(s0: float) -> core.Schedule:
    z = np.zeros(T); soc = np.full(T + 1, s0)
    return core.Schedule(z.copy(), z.copy(), z.copy(), soc, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0)


def simulate_q2(load, pv, price_actual, price_pred, load_pred, pv_pred, start_output=core.TEST_START):
    net_actual = (load - pv) * DT
    pred_net = (load_pred - pv_pred) * DT
    s0 = core.SOC_SAFE
    rows = []; metrics = []; checks = []; iterations = []
    for i in range(len(load)):
        margin = margin_asof(net_actual, pred_net, i)
        if i == 0:
            day = zero_schedule(s0)
        else:
            target = np.r_[pred_net[i] + margin,
                           (forecast_asof(load, i + 1, i) - forecast_asof(pv, i + 1, i)) * DT + margin] if i + 1 < len(load) else pred_net[i] + margin
            pnext = forecast_asof(price_actual, i + 1, i) if i + 1 < len(load) else price_pred[i]
            ph = np.r_[price_pred[i], pnext] if i + 1 < len(load) else price_pred[i]
            full = core.optimize_schedule(target, np.maximum(ph, 1e-6), s0, terminal_min=core.SOC_SAFE)
            day = slice_schedule(full)
        settle = core.actual_settlement(day, net_actual[i], price_actual[i])
        s0 = day.soc[-1]
        checks.append(check_schedule(day, net_actual[i], settle["emergency"], f"Q2@{i}"))
        iterations.append([i, day.nit, day.runtime])
        if i >= start_output:
            date = str(pd.Timestamp(core_dates.iloc[i]).date())
            rows.append({"date": date, "plan": day.g, "charge": day.charge, "discharge": day.discharge,
                         "soc": day.soc, "emergency": settle["emergency"], "total_cost": settle["total_cost"]})
            metrics.append([date, settle["total_cost"], settle["plan_cost"], settle["emergency_cost"],
                            settle["emergency_kwh"], settle["waste_kwh"], day.soc[0], day.soc[-1]])
    return rows, pd.DataFrame(metrics, columns=["date","total_cost","plan_cost","emergency_cost","emergency_kwh","waste_kwh","start_soc","end_soc"]), checks, iterations


def pv_forecast_24(forecast, date: str, issue: int, current_kw: float) -> np.ndarray:
    vals = forecast[(date, f"{issue}:00")]
    return np.interp(np.arange(1, T + 1), np.arange(0, T + 1, 6), np.r_[current_kw, vals])


def horizon_profile(x, pred, i, issue, bias=0.0):
    cur = pred[i].copy()
    nxt = forecast_asof(x, i + 1, i) if i + 1 < len(x) else cur.copy()
    cur = np.maximum(cur + bias, 0)
    nxt = np.maximum(nxt + bias, 0)
    st = issue * 6
    return np.r_[cur[st:], nxt[:st]]


def q3_target(i, issue, load, pv, forecast, load_pred, pred_net, net_actual):
    date = str(pd.Timestamp(core_dates.iloc[i]).date())
    st = issue * 6
    lbias = float(np.median(load[i, :st] - load_pred[i, :st])) if st else 0.0
    lh = horizon_profile(load, load_pred, i, issue, lbias)
    current_pv = float(pv[i, st - 1]) if st else 0.0
    pvh = pv_forecast_24(forecast, date, issue, current_pv)
    margin = margin_asof(net_actual, pred_net, i)
    mh = np.r_[margin[st:], margin[:st]]
    return (lh - pvh) * DT + mh


def price_horizon(i, issue, price_actual, price_pred, update):
    st = issue * 6
    bias = float(np.median(price_actual[i, :st] - price_pred[i, :st])) if update and st else 0.0
    return np.maximum(horizon_profile(price_actual, price_pred, i, issue, bias), 1e-6)


def check_schedule(s, net_actual, emergency, tag):
    bal = s.g + s.discharge - s.charge + emergency - net_actual
    soc_calc = s.soc[:-1] + core.ETA_C * s.charge - s.discharge / core.ETA_D
    return {"scope": tag, "balance_min": float(bal.min()), "soc_residual_max": float(np.max(np.abs(s.soc[1:] - soc_calc))),
            "soc_min": float(s.soc.min()), "soc_max": float(s.soc.max()),
            "simultaneous_max": float(np.minimum(s.charge, s.discharge).max()),
            "passed": bool(bal.min() >= -1e-7 and np.max(np.abs(s.soc[1:] - soc_calc)) <= 1e-7 and s.soc.min() >= core.SOC_MIN - 1e-7 and s.soc.max() <= core.SOC_MAX + 1e-7)}


def simulate_q3_combo(combo, load, pv, forecast, price_actual, price_pred, load_pred, pv_pred, update_price=False):
    net_actual = (load - pv) * DT
    pred_net = (load_pred - pv_pred) * DT
    s0 = core.SOC_SAFE
    rows=[]; metrics=[]; checks=[]; iterations=[]
    issues = sorted(combo)
    for i in range(len(load)):
        date = str(pd.Timestamp(core_dates.iloc[i]).date())
        target0 = q3_target(i, 0, load, pv, forecast, load_pred, pred_net, net_actual)
        p0 = price_horizon(i, 0, price_actual, price_pred, update_price)
        base = core.optimize_schedule(target0, p0, s0, terminal_min=core.SOC_SAFE)
        plan = base.g.copy()
        fg=np.zeros(T); fc=np.zeros(T); fd=np.zeros(T); fs=np.zeros(T+1); fs[0]=s0
        starts=[0] + [h*6 for h in issues]; ends=[h*6 for h in issues] + [T]
        current=base
        for st,en in zip(starts,ends):
            if st:
                issue=st//6
                target=q3_target(i, issue, load, pv, forecast, load_pred, pred_net, net_actual)
                ph=price_horizon(i, issue, price_actual, price_pred, update_price)
                pref=np.r_[plan[st:], np.full(st, plan[-1])]
                current=core.optimize_schedule(target, ph, fs[st], terminal_min=core.SOC_SAFE, plan_ref=pref)
            n=en-st
            fg[st:en]=current.g[:n]; fc[st:en]=current.charge[:n]; fd[st:en]=current.discharge[:n]
            for t in range(st,en): fs[t+1]=fs[t]+core.ETA_C*fc[t]-fd[t]/core.ETA_D
            iterations.append([date, "+".join(map(str,issues)) or "none", st//6, current.nit, current.runtime])
        sched=core.Schedule(fg,fc,fd,fs,0,0,0,0,0,0,float(np.minimum(fc,fd).max()))
        settle=core.actual_settlement(sched,net_actual[i],price_actual[i],plan=plan,adjusted=bool(issues))
        s0=sched.soc[-1]
        checks.append(check_schedule(sched,net_actual[i],settle["emergency"],f"Q3-{issues}@{date}"))
        if i >= core.TEST_START:
            rows.append({"date":date,"plan":plan,"adjust":fg,"charge":fc,"discharge":fd,"soc":fs,"emergency":settle["emergency"],"total_cost":settle["total_cost"]})
            metrics.append([date,"+".join(map(str,issues)) or "none",settle["total_cost"],settle["plan_cost"],settle["up_cost"],settle["down_cost"],settle["emergency_cost"],settle["emergency_kwh"],settle["waste_kwh"],fs[0],fs[-1]])
    cols=["date","combo","total_cost","plan_cost","up_cost","down_cost","emergency_cost","emergency_kwh","waste_kwh","start_soc","end_soc"]
    return rows,pd.DataFrame(metrics,columns=cols),checks,iterations


def metrics_forecast(actual, pred):
    mask=np.isfinite(pred)
    err=np.abs(actual[mask]-pred[mask])
    return {"MAE":float(err.mean()),"RMSE":float(np.sqrt(np.mean(err**2))),"WAPE":float(err.sum()/np.abs(actual[mask]).sum())}


def q1_sensitivity(net, price):
    rows=[]
    for eta in [0.80,0.85,0.90,0.95,1.00]:
        oldc,oldd=core.ETA_C,core.ETA_D; core.ETA_C=core.ETA_D=eta
        s=core.optimize_schedule(net,price,core.SOC_SAFE,terminal_eq=core.SOC_SAFE)
        rows.append(["效率",eta,float(np.sum(price*s.g))])
        core.ETA_C,core.ETA_D=oldc,oldd
    return pd.DataFrame(rows,columns=["parameter","value","cost"])


def make_plots(q1, load1, pv1, price1, pred, q2m, q3all, q43all, sensitivity):
    x=np.arange(T)/6
    plotdir=OUT/"records"/"plot_data"; plotdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"hour":x,"load_kw":load1,"pv_kw":pv1,"price_yuan_per_kwh":price1}).to_csv(plotdir/"fig01_input_profiles.csv",index=False)
    fig,ax=plt.subplots(2,1,figsize=(7.2,5.4),sharex=True); ax[0].plot(x,load1,label="负荷"); ax[0].plot(x,pv1,label="光伏"); ax[0].set_ylabel("功率/kW"); ax[0].legend(); ax[0].grid(alpha=.2); ax[1].plot(x,price1,color="#d97706"); ax[1].set(xlabel="时刻/h",ylabel="电价/(元/kWh)"); ax[1].grid(alpha=.2); core.savefig("fig01_input_profiles.pdf")
    fig,ax=plt.subplots(2,1,figsize=(7.2,5.4),sharex=True); ax[0].plot(x,q1.g,label="购电"); ax[0].plot(x,q1.charge,label="充电"); ax[0].plot(x,q1.discharge,label="放电"); ax[0].legend(ncol=3); ax[0].set_ylabel("电量/kWh"); ax[0].grid(alpha=.2); ax[1].plot(np.arange(T+1)/6,q1.soc,color="#059669"); ax[1].set(xlabel="时刻/h",ylabel="SOC/kWh"); ax[1].grid(alpha=.2); core.savefig("fig02_q1_dispatch.pdf")
    pd.DataFrame({"hour":x,"grid_kwh":q1.g,"charge_kwh":q1.charge,"discharge_kwh":q1.discharge}).to_csv(plotdir/"fig02_q1_dispatch.csv",index=False)
    c3=q3all.groupby("combo").total_cost.sum().sort_values(); c43=q43all.groupby("combo").total_cost.sum().sort_values(); fig,ax=plt.subplots(1,2,figsize=(9,3.8)); ax[0].bar(c3.index,c3.values/1e6,color="#3b82f6"); ax[0].set(ylabel="费用/百万元",xlabel="启用发布时刻"); ax[0].tick_params(axis='x',rotation=35); ax[1].bar(c43.index,c43.values/1e6,color="#f59e0b"); ax[1].set(xlabel="波动价下启用发布时刻"); ax[1].tick_params(axis='x',rotation=35); core.savefig("fig03_combo_comparison.pdf")
    pd.DataFrame({"combo":c3.index,"fixed_price_cost":c3.values,"variable_price_cost":[c43.get(k,np.nan) for k in c3.index]}).to_csv(plotdir/"fig03_combo_comparison.csv",index=False)
    i=79; fig,ax=plt.subplots(2,1,figsize=(7.2,5.2),sharex=True); ax[0].plot(x,core_load[i],label="实际"); ax[0].plot(x,pred['load'][i],label="预测"); ax[0].legend(); ax[0].set_ylabel("负荷/kW"); ax[1].plot(x,core_pv[i],label="实际"); ax[1].plot(x,pred['pv'][i],label="预测"); ax[1].set(xlabel="时刻/h",ylabel="光伏/kW"); ax[1].legend(); core.savefig("fig04_prediction_example.pdf")
    pd.DataFrame({"hour":x,"load_actual":core_load[i],"load_forecast":pred['load'][i],"pv_actual":core_pv[i],"pv_forecast":pred['pv'][i]}).to_csv(plotdir/"fig04_prediction_example.csv",index=False)
    best=next(k for k in c3.index if k!="none"); z=q3all[q3all.combo=="none"].set_index("date").total_cost-q3all[q3all.combo==best].set_index("date").total_cost; fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.hist(z,bins=35,color="#3b82f6",alpha=.85); ax.axvline(0,color="black",ls="--"); ax.set(xlabel=f"不调整费用－最佳非空组合({best})费用/元",ylabel="天数"); core.savefig("fig05_q3_daily_savings.pdf")
    z.rename("daily_saving").to_csv(plotdir/"fig05_q3_daily_savings.csv")
    sample=np.arange(core.TEST_START,len(core_dates),7); fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.plot(core_dates.iloc[sample],core_price4[sample].mean(1),label="实际日均电价"); ax.plot(core_dates.iloc[sample],pred['price'][sample].mean(1),label="历史滚动预测"); ax.set(ylabel="元/kWh"); ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate(); core.savefig("fig06_price_forecast.pdf")
    pd.DataFrame({"date":core_dates.iloc[sample].astype(str).values,"actual_mean_price":core_price4[sample].mean(1),"forecast_mean_price":pred['price'][sample].mean(1)}).to_csv(plotdir/"fig06_price_forecast.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.plot(pd.to_datetime(q2m.date),q2m.end_soc,color="#059669"); ax.axhline(core.SOC_SAFE,color="black",ls="--",lw=.8); ax.set(xlabel="日期",ylabel="日末SOC/kWh"); ax.grid(alpha=.2); fig.autofmt_xdate(); core.savefig("fig07_crossday_soc.pdf")
    q2m[["date","start_soc","end_soc"]].to_csv(plotdir/"fig07_crossday_soc.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.6,3.8)); ax.plot(sensitivity.value,sensitivity.cost,marker="o"); ax.set(xlabel="充/放电效率",ylabel="问题一费用/元"); ax.grid(alpha=.2); core.savefig("fig08_efficiency_sensitivity.pdf")
    sensitivity.to_csv(plotdir/"fig08_efficiency_sensitivity.csv",index=False)


def main():
    global core_dates, core_load, core_pv, core_price4
    core.mkdirs(); core.setup_plot(); np.random.seed(core.SEED)
    dates,labels,price1,load1,pv1,load,pv,forecast,price4=core.read_inputs()
    core_dates,core_load,core_pv,core_price4=dates,load,pv,price4
    checks_data,dq=core.validate_data(dates,price1,load1,pv1,load,pv,forecast,price4); core.save_cleaned(dates,load,pv,price4)
    pred={"load":prediction_matrices(load),"pv":prediction_matrices(pv),"price":prediction_matrices(price4)}
    pmetrics={k:metrics_forecast(v,pred[k]) for k,v in {"load":load,"pv":pv,"price":price4}.items()}
    (OUT/"records"/"prediction_metrics.json").write_text(json.dumps(pmetrics,ensure_ascii=False,indent=2),encoding="utf-8")

    net1=(load1-pv1)*DT; q1=core.optimize_schedule(net1,price1,core.SOC_SAFE,terminal_eq=core.SOC_SAFE)
    q1cost=float(np.sum(price1*q1.g)); q1base=float(np.sum(price1*np.maximum(net1,0)))
    wb=core.load_workbook(core.ATT/"附件5"/"result1.xlsx"); ws=wb["计划购电量"]
    for i,v in enumerate(q1.g,start=2): ws.cell(i,2,float(v))
    ws=wb["充放电量"]
    for j in range(6): ws.cell(j+2,2,float(q1.charge[j*24:(j+1)*24].sum())); ws.cell(j+2,3,float(q1.discharge[j*24:(j+1)*24].sum()))
    ws.cell(2,5,float(q1.soc[0])); ws.cell(3,5,float(q1.soc[-1])); wb.save(OUT/"results"/"result1.xlsx")

    q2rows,q2m,q2checks,q2it=simulate_q2(load,pv,np.tile(price1,(len(load),1)),np.tile(price1,(len(load),1)),pred["load"],pred["pv"])
    q42rows,q42m,q42checks,q42it=simulate_q2(load,pv,price4,pred["price"],pred["load"],pred["pv"])
    combos=[tuple(c) for r in range(4) for c in itertools.combinations([6,12,18],r)]
    q3runs=[]; q43runs=[]; allchecks=q2checks+q42checks; allit=[]; rows3={}; rows43={}
    for combo in combos:
        r,m,ch,it=simulate_q3_combo(combo,load,pv,forecast,np.tile(price1,(len(load),1)),np.tile(price1,(len(load),1)),pred["load"],pred["pv"],False); rows3[m.combo.iloc[0]]=r; q3runs.append(m); allchecks+=ch; allit+=it
        r,m,ch,it=simulate_q3_combo(combo,load,pv,forecast,price4,pred["price"],pred["load"],pred["pv"],True); rows43[m.combo.iloc[0]]=r; q43runs.append(m); allchecks+=ch; allit+=it
    q3all=pd.concat(q3runs,ignore_index=True); q43all=pd.concat(q43runs,ignore_index=True)
    best3=q3all.groupby("combo").total_cost.sum().idxmin(); best43=q43all.groupby("combo").total_cost.sum().idxmin()
    core.fill_output("result2.xlsx","result2.xlsx",dates.iloc[core.TEST_START:],q2rows,False)
    core.fill_output("result3.xlsx","result3.xlsx",dates.iloc[core.TEST_START:],rows3[best3],True)
    core.fill_output("result4-2.xlsx","result4-2.xlsx",dates.iloc[core.TEST_START:],q42rows,False)
    core.fill_output("result4-3.xlsx","result4-3.xlsx",dates.iloc[core.TEST_START:],rows43[best43],True)

    q2m.to_csv(OUT/"records"/"q2_daily.csv",index=False); q42m.to_csv(OUT/"records"/"q42_daily.csv",index=False); q3all.to_csv(OUT/"records"/"q3_combos.csv",index=False); q43all.to_csv(OUT/"records"/"q43_combos.csv",index=False)
    pd.DataFrame(allchecks).to_csv(OUT/"records"/"constraint_checks.csv",index=False); pd.DataFrame(allit,columns=["date","combo","issue","iterations","runtime_s"]).to_csv(OUT/"records"/"iteration_log.csv",index=False)
    sensitivity=q1_sensitivity(net1,price1); sensitivity.to_csv(OUT/"records"/"q1_efficiency_sensitivity.csv",index=False)
    params={"seed":core.SEED,"dt_hours":DT,"eta_c":core.ETA_C,"eta_d":core.ETA_D,"soc_bounds":[core.SOC_MIN,core.SOC_MAX],"initial_soc":"2025-01-01 00:00 = 6000 kWh","official_output_start":"2025-02-01","forecast":"strict expanding/rolling history","margin_quantile":0.8,"q3_combos":[list(x) for x in combos],"selected_q3_combo":best3,"selected_q43_combo":best43,"settlement":"plan + positive 0.5 down deviation + 1.5 up deviation + 5 emergency"}
    (OUT/"records"/"model_parameters.json").write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding="utf-8")
    make_plots(q1,load1,pv1,price1,pred,q2m,q3all,q43all,sensitivity)

    c3=q3all.groupby("combo").total_cost.sum().sort_values(); c43=q43all.groupby("combo").total_cost.sum().sort_values()
    passrate=float(pd.DataFrame(allchecks).passed.mean())
    report=f"""# 融合重算版计算结果报告

## 运行环境

- Python {platform.python_version()}，NumPy {np.__version__}，pandas {pd.__version__}，SciPy {core.scipy.__version__}。
- 固定随机种子 {core.SEED}；HiGHS 线性规划；10 分钟时步。

## 数据与信息集

365 日、每日 144 点数据门禁全部通过。所有正式策略只读取决策时刻以前的数据；2025 年 1 月用于预测与 SOC 状态热身，正式输出为 2 月 1 日至 12 月 31 日共 334 日。

## 问题一

费用 **{q1cost:,.2f} 元**，无储能基线 **{q1base:,.2f} 元**，节省 **{core.pct(q1base,q1cost):.2f}%**；购电量 **{q1.g.sum():,.2f} kWh**。

## 问题二

负荷预测 MAE **{pmetrics['load']['MAE']:.2f} kW**、WAPE **{pmetrics['load']['WAPE']:.2%}**；光伏预测 MAE **{pmetrics['pv']['MAE']:.2f} kW**、WAPE **{pmetrics['pv']['WAPE']:.2%}**。融合策略费用 **{q2m.total_cost.sum():,.2f} 元**，紧急购电 **{q2m.emergency_kwh.sum():,.2f} kWh**。A 版完美信息结果只作不可实现下界，不进入正式工作簿。

## 问题三

对 6/12/18 三个更新时刻的 8 种组合逐一回测。最低费用组合为 **{best3}**，费用 **{c3.iloc[0]:,.2f} 元**；完全不调整费用 **{c3['none']:,.2f} 元**。正式 result3 使用最低费用组合。各组合完整记录见 `q3_combos.csv`。

## 问题四

价格预测 MAE **{pmetrics['price']['MAE']:.4f} 元/kWh**、WAPE **{pmetrics['price']['WAPE']:.2%}**。问题 4-2 费用 **{q42m.total_cost.sum():,.2f} 元**；问题 4-3 最低费用组合为 **{best43}**，费用 **{c43.iloc[0]:,.2f} 元**，不调整费用 **{c43['none']:,.2f} 元**。

## 灵敏度与验证

问题一效率 0.80--1.00 的费用响应已记录。所有 **{len(allchecks):,}** 项逐日/逐组合约束回代通过率为 **{passrate:.2%}**。费用分量、跨日 SOC、供需和储能边界均保存为机器可读记录。

## 图表

`fig01` 输入曲线；`fig02` 问题一调度；`fig03` 更新组合费用；`fig04` 预测示例；`fig05` 逐日调整收益；`fig06` 价格预测；`fig07` 跨日 SOC；`fig08` 效率敏感性。所有图均为矢量 PDF。

## 从零复现

```bash
MPLCONFIGDIR=.mplconfig python3 code/run_fusion.py
```
"""
    REPORT.write_text(report,encoding="utf-8")
    print(json.dumps({"status":"ok","q1_cost":q1cost,"q2_cost":q2m.total_cost.sum(),"q3_best":best3,"q3_cost":float(c3.iloc[0]),"q42_cost":q42m.total_cost.sum(),"q43_best":best43,"q43_cost":float(c43.iloc[0]),"checks":len(allchecks),"pass_rate":passrate},ensure_ascii=False))


if __name__ == "__main__":
    main()
