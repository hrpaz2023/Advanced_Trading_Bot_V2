# Función corregida para simulate_exit_on_candles
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
    df_fwd = df[df["timestamp"] >= entry_time].copy().reset_index(drop=True)
    if df_fwd.empty or len(df_fwd) < 2:
        return {"pnl_usd": 0.0, "minutes": 0, "actions": ["no_data"]}

    # Valores base
    price_series = df_fwd["close"]
    high = df_fwd["high"]
    low = df_fwd["low"]
    atr_series = df_fwd["atr"].replace(0, np.nan).ffill()
    
    # FIX: Usar ffill() en lugar de fillna(method="ffill")
    entry_atr_data = df[df["timestamp"] <= entry_time]["atr"].tail(1)
    if not entry_atr_data.empty:
        atr_entry = float(entry_atr_data.ffill().values[0])
    else:
        atr_entry = float(atr_series.iloc[0]) if not atr_series.empty else 1e-6
    
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
    
    # FIX: Usar range basado en len() para evitar problemas de indexing
    for i in range(1, len(df_fwd)):
        ts = df_fwd["timestamp"].iloc[i]
        px_close = price_series.iloc[i]
        bar_minutes = int((ts - first_time).total_seconds() // 60)

        # Verificar que tenemos datos válidos
        if pd.isna(px_close):
            continue

        # 1) Ladder por R (opcional) - parcial/cierre
        if params.ladder_r_partial:
            if r_multiple(px_close) >= params.ladder_r_partial and remaining_lots > lots * 0.25:
                part = 0.5 * remaining_lots
                pnl = floating_pnl_usd(px_close, part)
                realized_pnl += pnl - (params.commission_per_lot * part)
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
            
            # Verificar que el precio intrabar es válido
            if pd.isna(intrabar_px):
                continue
                
            pnl_intrabar = floating_pnl_usd(intrabar_px, remaining_lots)
            if pnl_intrabar >= (t_usd + comm_usd):
                close_vol = max(0.01, round(remaining_lots * frac, 2))
                pnl = floating_pnl_usd(intrabar_px, close_vol)
                realized_pnl += pnl - (params.commission_per_lot * close_vol)
                remaining_lots = round(remaining_lots - close_vol, 2)
                actions.append(f"target_{t_usd:g}@{frac:.2f}")

        # 3) Breakeven por debilidad (tras min)
        minutes_open = bar_minutes
        if minutes_open >= params.breakeven_after_min and remaining_lots > 0:
            if i < len(weak) and weak.iloc[i] and params.breakeven_weak_threshold > 0:
                be_level = comm_usd + be_buffer
                intrabar_px = float(high.iloc[i]) if side == "BUY" else float(low.iloc[i])
                
                if not pd.isna(intrabar_px):
                    pnl_intrabar = floating_pnl_usd(intrabar_px, remaining_lots)
                    if pnl_intrabar >= be_level:
                        pnl = floating_pnl_usd(intrabar_px, remaining_lots)
                        realized_pnl += pnl - (params.commission_per_lot * remaining_lots)
                        actions.append("breakeven_weak_close")
                        return {"pnl_usd": realized_pnl, "minutes": bar_minutes, "actions": actions}

        # 4) Límite temporal
        if minutes_open >= (params.max_hours * 60):
            pnl = floating_pnl_usd(px_close, remaining_lots)
            realized_pnl += pnl - (params.commission_per_lot * remaining_lots)
            actions.append("time_max_close")
            return {"pnl_usd": realized_pnl, "minutes": bar_minutes, "actions": actions}

    # Si no cerró, cerrar al último close
    last_px = float(price_series.iloc[-1])
    if not pd.isna(last_px) and remaining_lots > 0:
        pnl = floating_pnl_usd(last_px, remaining_lots)
        realized_pnl += pnl - (params.commission_per_lot * remaining_lots)
        actions.append("end_of_sample_close")
    
    total_minutes = int((df_fwd["timestamp"].iloc[-1] - first_time).total_seconds() // 60)
    return {"pnl_usd": realized_pnl, "minutes": total_minutes, "actions": actions}