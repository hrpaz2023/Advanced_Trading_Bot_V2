# -*- coding: utf-8 -*-
"""
Backtesting de salidas (PMI) + Monte Carlo + Calibración por símbolo.

- Lee trades ejecutados de logs/signals_history.jsonl
- Simula reglas de salida: targets USD, breakeven por debilidad, límites por tiempo, ladder R opcional
- Calibra thresholds por símbolo (grid/random) y estima IC por Monte Carlo
- Genera reportes en 'reports/'

Uso típico:
    python src/tools/exit_backtest.py --symbols EURUSD,GBPUSD,AUDUSD,USDJPY --timeframe M5 \
        --n-mc 2000 --search random --seed 42 --out reports/exit_20250809

Requisitos:
    pandas, numpy, pyarrow (para parquet)
"""

import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ----------------------------
# Utilidades de mercado
# ----------------------------

def pip_size(symbol: str) -> float:
    s = symbol.upper()
    if s.endswith("JPY"):
        return 0.01
    return 0.0001

def usd_per_pip_per_lot(symbol: str, price: float) -> float:
    """
    Valor de 1 pip por 1.0 lot en USD, aproximado.
    - Para XXXUSD: ~10 USD/pip/lot (EURUSD, GBPUSD, AUDUSD).
    - Para USDJPY: 1000/price USD/pip/lot (p. ej. price=150 => ~6.67 USD/pip/lot).
    """
    s = symbol.upper()
    if s.endswith("USD"):  # EURUSD, GBPUSD, AUDUSD
        return 10.0
    if s == "USDJPY":
        if price <= 0:
            price = 150.0
        return 1000.0 / price
    # Fallback genérico
    if price <= 0:
        price = 1.0
    pipsz = pip_size(symbol)
    # Supongamos contract_size=100000 (standard lot) y quote en USD (aprox)
    return (pipsz / price) * 100000.0 * price / pipsz * 0.0001  # simplificado (mantener 10 para majors)
    # Nota: arriba dejamos majors con fórmulas específicas.

def to_utc(ts: str) -> datetime:
    try:
        dt = pd.to_datetime(ts, utc=True).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        # fallback naive a utc
        return datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)

# ----------------------------
# Carga de datos
# ----------------------------

def load_executed_trades(signals_path: Path) -> pd.DataFrame:
    """
    Carga trades ejecutados del log de señales. Espera __status__ == 'EXECUTED'.
    Campos mínimos: timestamp_utc, symbol, side, entry_price, position_size, ticket
    """
    df = pd.read_json(signals_path, lines=True)
    if "__status__" in df.columns:
        df = df[df["__status__"] == "EXECUTED"].copy()
    elif "status" in df.columns:
        df = df[df["status"].str.upper() == "EXECUTED"].copy()

    # Normalizar
    req = ["timestamp_utc", "symbol", "side", "entry_price", "position_size"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en {signals_path}")
    df["timestamp_utc"] = df["timestamp_utc"].astype(str).map(to_utc)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.upper()
    if "ticket" not in df.columns:
        df["ticket"] = 0
    # Ordenar por tiempo
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df

def load_candles(symbol: str, timeframe: str, root: Path) -> pd.DataFrame:
    """
    Lee velas desde data/candles/{SYMBOL}_{TF}.parquet
    Columnas: timestamp, open, high, low, close, atr (opcional)
    """
    f = root / f"{symbol.upper()}_{timeframe.upper()}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"No encontré velas para {symbol} en {f}")
    df = pd.read_parquet(f)
    # Normalizar timestamp a UTC
    tcol = "timestamp" if "timestamp" in df.columns else "time"
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.rename(columns={tcol: "timestamp"})
    # Orden y tipos
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Asegurar ATR si no está: usaremos Wilder(14) si hace falta
    if "atr" not in df.columns:
        # ATR simple (no Wilder exacto) para no traer dependencias
        try:
            tr1 = (df["high"] - df["low"]).abs()
            tr2 = (df["high"] - df["close"].shift(1)).abs()
            tr3 = (df["low"] - df["close"].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14, min_periods=1).mean()
        except Exception:
            df["atr"] = np.nan
    return df

# ----------------------------
# Señales de debilidad (Fase 1)
# ----------------------------

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def weakness_flag(side: str, close: pd.Series) -> pd.Series:
    """
    Heurística de debilidad:
      - BUY: EMA9 slope < 0  y close < EMA21
      - SELL: EMA9 slope > 0 y close > EMA21
    Devuelve serie booleana por barra.
    """
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    slope = e9.diff()
    if side == "BUY":
        return (slope < 0) & (close < e21)
    else:  # SELL
        return (slope > 0) & (close > e21)

# ----------------------------
# Simulador de salidas
# ----------------------------

@dataclass
class ExitParams:
    targets_usd: List[float]         # p.ej. [150, 350, 600]
    fractions: List[float]           # p.ej. [0.5, 0.5, 1.0]
    breakeven_after_min: int         # p.ej. 60
    breakeven_weak_threshold: float  # p.ej. 0.55 (en Fase 1 solo usamos bandera debilidad; el umbral activa/desactiva)
    commission_per_lot: float        # 3 USD / lot
    commission_buffer_usd: float     # buffer extra sobre comisión
    grace_minutes: int               # 90
    max_hours: int                   # 12
    ladder_r_partial: Optional[float] = None  # p.ej. 1.0 -> parcial al alcanzar 1R
    ladder_r_close: Optional[float] = None    # p.ej. 2.0 -> cerrar todo al alcanzar 2R

def simulate_exit_on_candles(
    symbol: str,
    entry_time: datetime,
    entry_price: float,
    side: str,
    lots: float,
    df: pd.DataFrame,
    params: ExitParams,
) -> Dict:
    """
    Simula salidas a partir de la barra de entrada (siguiente barra inclusiva).
    Retorna dict con pnl_usd, bars, minutes, actions_detalle
    """
    # Cortar ventana desde entrada
    df_fwd = df[df["timestamp"] >= entry_time].copy()
    if df_fwd.empty or len(df_fwd) < 2:
        return {"pnl_usd": 0.0, "minutes": 0, "actions": ["no_data"]}

    # Valores base
    price_series = df_fwd["close"]
    high = df_fwd["high"]
    low = df_fwd["low"]
    atr_series = df_fwd["atr"].replace(0, np.nan).ffill()
    atr_entry = float(df[df["timestamp"] <= entry_time]["atr"].tail(1).fillna(method="ffill").values[0]) if not df[df["timestamp"] <= entry_time].empty else float(atr_series.iloc[0])
    atr_entry = abs(atr_entry) if not math.isnan(atr_entry) else max(1e-6, float(price_series.iloc[0]) * 0.0005)

    pip = pip_size(symbol)
    uppl = usd_per_pip_per_lot(symbol, float(price_series.iloc[0]))
    comm_usd = max(0.0, params.commission_per_lot * lots)
    be_buffer = params.commission_buffer_usd

    # Debilidad
    weak = weakness_flag(side, price_series)

    # Estado de posición (para ladder/partials)
    remaining_lots = lots
    realized_pnl = 0.0
    actions = []
    entry_idx = df_fwd.index[0]
    entry_px = entry_price

    # Helpers
    def floating_pnl_usd(px: float, lots_now: float) -> float:
        delta = (px - entry_px) if side == "BUY" else (entry_px - px)
        pips = delta / pip
        return pips * uppl * lots_now

    def r_multiple(px: float) -> float:
        delta = abs(px - entry_px)
        return delta / max(1e-9, atr_entry)

    # Tiempos
    first_time = df_fwd["timestamp"].iloc[0]
    for i in range(1, len(df_fwd)):
        ts = df_fwd["timestamp"].iloc[i]
        px_close = price_series.iloc[i]
        bar_minutes = int((ts - first_time).total_seconds() // 60)

        # 1) Ladder por R (opcional) - parcial/cierre
        if params.ladder_r_partial:
            if r_multiple(px_close) >= params.ladder_r_partial and remaining_lots > lots * 0.25:
                part = 0.5 * remaining_lots
                pnl = floating_pnl_usd(px_close, part)
                realized_pnl += pnl - (params.commission_per_lot * part)  # comisión proporcional
                remaining_lots -= part
                actions.append(f"ladder_partial@{params.ladder_r_partial}R")

        if params.ladder_r_close:
            if r_multiple(px_close) >= params.ladder_r_close and remaining_lots > 0:
                pnl = floating_pnl_usd(px_close, remaining_lots)
                realized_pnl += pnl - (params.commission_per_lot * remaining_lots)
                actions.append(f"ladder_close@{params.ladder_r_close}R")
                return {"pnl_usd": realized_pnl, "minutes": bar_minutes, "actions": actions}

        # 2) Targets USD por peldaños
        for t_usd, frac in zip(params.targets_usd, params.fractions):
            if remaining_lots <= 0:
                break
            # Chequeo intrabar (optimista): si BUY mira high; si SELL mira low
            intrabar_px = float(high.iloc[i]) if side == "BUY" else float(low.iloc[i])
            pnl_intrabar = floating_pnl_usd(intrabar_px, remaining_lots)
            if pnl_intrabar >= (t_usd + comm_usd):
                close_vol = max(0.01, round(remaining_lots * frac, 2))
                pnl = floating_pnl_usd(intrabar_px, close_vol)
                realized_pnl += pnl - (params.commission_per_lot * close_vol)
                remaining_lots = round(remaining_lots - close_vol, 2)
                actions.append(f"target_{t_usd:g}@{frac:.2f}")
                # seguir por si hay más targets en la misma barra

        # 3) Breakeven por debilidad (tras min)
        minutes_open = bar_minutes
        if minutes_open >= params.breakeven_after_min and remaining_lots > 0:
            if weak.iloc[i] and params.breakeven_weak_threshold > 0:
                # ¿tocó breakeven + buffer + comisión?
                be_level = comm_usd + be_buffer
                intrabar_px = float(high.iloc[i]) if side == "BUY" else float(low.iloc[i])
                pnl_intrabar = floating_pnl_usd(intrabar_px, remaining_lots)
                if pnl_intrabar >= be_level:
                    # cerrar todo a intrabar_px
                    pnl = floating_pnl_usd(intrabar_px, remaining_lots)
                    realized_pnl += pnl - (params.commission_per_lot * remaining_lots)
                    actions.append("breakeven_weak_close")
                    return {"pnl_usd": realized_pnl, "minutes": bar_minutes, "actions": actions}

        # 4) Límite temporal
        if minutes_open >= (params.max_hours * 60):
            # cerrar al close actual
            pnl = floating_pnl_usd(px_close, remaining_lots)
            realized_pnl += pnl - (params.commission_per_lot * remaining_lots)
            actions.append("time_max_close")
            return {"pnl_usd": realized_pnl, "minutes": bar_minutes, "actions": actions}

    # Si no cerró, cerrar al último close
    last_px = float(price_series.iloc[-1])
    pnl = ((last_px - entry_px) if side == "BUY" else (entry_px - last_px)) / pip * uppl * remaining_lots
    realized_pnl += pnl - (params.commission_per_lot * remaining_lots)
    actions.append("end_of_sample_close")
    total_minutes = int((df_fwd["timestamp"].iloc[-1] - first_time).total_seconds() // 60)
    return {"pnl_usd": realized_pnl, "minutes": total_minutes, "actions": actions}

# ----------------------------
# Calibración + Monte Carlo
# ----------------------------

def evaluate_symbol(
    symbol: str,
    trades: pd.DataFrame,
    candles: pd.DataFrame,
    param_grid: List[ExitParams],
    n_mc: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evalúa un conjunto de parámetros y realiza Monte Carlo sobre el conjunto de trades.
    Retorna:
      - df_results: métrica por combinación
      - best: dict con mejor combinación + IC MC
    """
    rng = random.Random(seed)
    rows = []
    # Evaluación determinista (sin MC) por combinación
    for p in param_grid:
        pnls = []
        durations = []
        n = 0
        for _, r in trades.iterrows():
            res = simulate_exit_on_candles(
                symbol=symbol,
                entry_time=r["timestamp_utc"],
                entry_price=float(r["entry_price"]),
                side=str(r["side"]).upper(),
                lots=float(r["position_size"]),
                df=candles,
                params=p,
            )
            pnls.append(res["pnl_usd"])
            durations.append(res["minutes"])
            n += 1

        pnls = np.array(pnls, dtype=float)
        durations = np.array(durations, dtype=float)
        expectancy = pnls.mean() if len(pnls) else 0.0
        hit = float((pnls > 0).mean()) if len(pnls) else 0.0
        std = pnls.std(ddof=1) if len(pnls) > 1 else 0.0
        sharpe = expectancy / std if std > 1e-9 else 0.0
        max_dd = _max_drawdown(pnls)

        rows.append({
            "symbol": symbol,
            "n_trades": n,
            "expectancy": expectancy,
            "hit_rate": hit,
            "sharpe_like": sharpe,
            "max_drawdown": max_dd,
            "params": p.__dict__,
        })

    df_res = pd.DataFrame(rows).sort_values(["expectancy", "sharpe_like"], ascending=[False, False]).reset_index(drop=True)

    if df_res.empty:
        return df_res, {}

    # Seleccion base
    best_row = df_res.iloc[0]
    best_params = ExitParams(**best_row["params"])

    # Monte Carlo bootstrap con los pnls de la mejor combinación
    base_pnls = []
    for _, r in trades.iterrows():
        resb = simulate_exit_on_candles(
            symbol=symbol,
            entry_time=r["timestamp_utc"],
            entry_price=float(r["entry_price"]),
            side=str(r["side"]).upper(),
            lots=float(r["position_size"]),
            df=candles,
            params=best_params,
        )
        base_pnls.append(float(resb["pnl_usd"]))

    base_pnls = np.array(base_pnls, dtype=float)

    rng.seed(seed)
    mc_totals = []
    for _ in range(n_mc):
        sample = rng.choices(list(base_pnls), k=len(base_pnls)) if len(base_pnls) else [0.0]
        mc_totals.append(float(np.sum(sample)))
    mc_arr = np.array(mc_totals, dtype=float)
    ci_low, ci_high = np.percentile(mc_arr, [5, 95]) if len(mc_arr) else (0.0, 0.0)

    best = {
        "symbol": symbol,
        "best_params": best_params.__dict__,
        "deterministic": {
            "expectancy": float(best_row["expectancy"]),
            "hit_rate": float(best_row["hit_rate"]),
            "sharpe_like": float(best_row["sharpe_like"]),
            "max_drawdown": float(best_row["max_drawdown"]),
            "n_trades": int(best_row["n_trades"]),
        },
        "mc": {
            "n": int(n_mc),
            "sum_mean": float(mc_arr.mean()) if len(mc_arr) else 0.0,
            "sum_ci5": float(ci_low),
            "sum_ci95": float(ci_high),
        }
    }
    return df_res, best

def _max_drawdown(pnls: np.ndarray) -> float:
    """
    Max drawdown sobre la curva acumulada de PnL.
    """
    if pnls.size == 0:
        return 0.0
    curve = np.cumsum(pnls)
    peaks = np.maximum.accumulate(curve)
    dd = (curve - peaks).min()
    return float(dd)

# ----------------------------
# Búsqueda de parámetros
# ----------------------------

def build_param_grid(mode: str, n_samples: int) -> List[ExitParams]:
    """
    Crea un grid (o random) de parámetros razonable.
    Puedes ajustar rangos según tu universo.
    """
    # Rangos razonables para majors con lotes 2.0 (ajusta si hace falta)
    target_sets = [
        ([120, 300, 550], [0.5, 0.5, 1.0]),
        ([150, 350, 600], [0.5, 0.5, 1.0]),
        ([200, 400, 700], [0.5, 0.5, 1.0]),
    ]
    be_minutes = [45, 60, 75]
    be_weak = [0.50, 0.55, 0.60]
    comm_per_lot = [3.0]
    comm_buf = [0.0, 2.0, 5.0]
    grace = [60, 90, 120]
    maxh = [8, 12]
    ladd_part = [None, 1.0]
    ladd_close = [None, 2.0]

    grid = []
    if mode == "grid":
        for ts, fr in target_sets:
            for bem in be_minutes:
                for bew in be_weak:
                    for cpl in comm_per_lot:
                        for cb in comm_buf:
                            for g in grace:
                                for mh in maxh:
                                    for lp in ladd_part:
                                        for lc in ladd_close:
                                            grid.append(ExitParams(
                                                targets_usd=ts,
                                                fractions=fr,
                                                breakeven_after_min=bem,
                                                breakeven_weak_threshold=bew,
                                                commission_per_lot=cpl,
                                                commission_buffer_usd=cb,
                                                grace_minutes=g,
                                                max_hours=mh,
                                                ladder_r_partial=lp,
                                                ladder_r_close=lc,
                                            ))
    else:
        # random: muestreamos combinaciones
        rng = random.Random(123)
        for _ in range(n_samples):
            ts, fr = rng.choice(target_sets)
            bem = rng.choice(be_minutes)
            bew = rng.choice(be_weak)
            cpl = rng.choice(comm_per_lot)
            cb = rng.choice(comm_buf)
            g = rng.choice(grace)
            mh = rng.choice(maxh)
            lp = rng.choice(ladd_part)
            lc = rng.choice(ladd_close)
            grid.append(ExitParams(
                targets_usd=ts, fractions=fr,
                breakeven_after_min=bem, breakeven_weak_threshold=bew,
                commission_per_lot=cpl, commission_buffer_usd=cb,
                grace_minutes=g, max_hours=mh,
                ladder_r_partial=lp, ladder_r_close=lc,
            ))
    return grid

# ----------------------------
# Main CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Backtesting de salidas (PMI) + Monte Carlo + calibración por símbolo")
    ap.add_argument("--symbols", type=str, default="EURUSD,GBPUSD,AUDUSD,USDJPY")
    ap.add_argument("--timeframe", type=str, default="M5")
    ap.add_argument("--signals", type=str, default="logs/signals_history.jsonl")
    ap.add_argument("--candles-root", type=str, default="data/candles")
    ap.add_argument("--out", type=str, default="reports/exit_backtest")
    ap.add_argument("--search", type=str, choices=["grid", "random"], default="random")
    ap.add_argument("--samples", type=int, default=60, help="n combinaciones (si random)")
    ap.add_argument("--n-mc", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Cargar trades ejecutados
    trades_all = load_executed_trades(Path(args.signals))
    if trades_all.empty:
        print(f"[WARN] No hay trades ejecutados en {args.signals}")
        return

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    grid = build_param_grid(args.search, args.samples)

    summary = {}
    for sym in symbols:
        t_sym = trades_all[trades_all["symbol"] == sym].copy()
        if t_sym.empty:
            print(f"[INFO] Sin trades ejecutados para {sym}, salto.")
            continue

        # Cargar velas símbolo
        cnd = load_candles(sym, args.timeframe, Path(args.candles_root))

        df_res, best = evaluate_symbol(
            symbol=sym,
            trades=t_sym,
            candles=cnd,
            param_grid=grid,
            n_mc=args.n_mc,
            seed=args.seed,
        )

        # Guardar resultados por símbolo
        csv_all = outdir / f"exit_mc_results_{sym}.csv"
        df_res.to_json(outdir / f"exit_mc_results_{sym}.json", orient="records", lines=True, force_ascii=False)
        df_res.to_csv(csv_all, index=False)
        with open(outdir / f"exit_mc_best_{sym}.json", "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)

        summary[sym] = best
        print(f"[OK] {sym} -> expectancy={best.get('deterministic',{}).get('expectancy',0):.2f} | MC mean={best.get('mc',{}).get('sum_mean',0):.2f} | 90%CI=({best.get('mc',{}).get('sum_ci5',0):.2f}, {best.get('mc',{}).get('sum_ci95',0):.2f})")

    with open(outdir / "exit_mc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
