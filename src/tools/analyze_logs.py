# src/tools/analyze_logs.py
import os, json, re, glob
from collections import Counter
import pandas as pd

pd.set_option("display.max_columns", 100)
os.makedirs("reports", exist_ok=True)

# -------------------- Helpers -------------------- #
def read_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # línea corrupta -> ignorar
                pass
    return pd.DataFrame(rows)

def parse_ts_col(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

def normalize_action(val):
    # soporta Enum-like "DecisionAction.HOLD" y strings
    s = str(val) if val is not None else ""
    return s.split(".")[-1].upper()

def normalize_status(val):
    s = str(val) if val is not None else ""
    return s.upper()

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def print_section(title):
    print("\n" + "="*5 + f" {title} " + "="*5)

# -------------------- Carga de fuentes -------------------- #
pmi_paths = ["logs/pmi_decisions.jsonl", "pmi_decisions.jsonl"]
pmi_path = next((p for p in pmi_paths if os.path.exists(p)), None)
df_pmi = read_jsonl(pmi_path)

sig_paths = ["logs/signals_history.jsonl", "signals_history.jsonl"]
sig_path = next((p for p in sig_paths if os.path.exists(p)), None)
df_sig = read_jsonl(sig_path)

# -------------------- Normalizaciones -------------------- #
if not df_pmi.empty:
    if "action" in df_pmi.columns:
        df_pmi["action"] = df_pmi["action"].map(normalize_action)
    if "reason" not in df_pmi.columns:
        df_pmi["reason"] = ""
    for col in ["timestamp", "timestamp_utc"]:
        parse_ts_col(df_pmi, col)
    to_numeric(df_pmi, ["ml_confidence", "historical_prob", "historical_prob_lb90", "tcd_prob", "divergence_score"])

if not df_sig.empty:
    # status y rejection_reason pueden venir con distintos nombres
    for cand in ["final_status", "status", "decision"]:
        if cand in df_sig.columns:
            df_sig["__status__"] = df_sig[cand].map(normalize_status)
            break
    else:
        df_sig["__status__"] = ""

    for cand in ["rejection_reason", "reason"]:
        if cand in df_sig.columns:
            df_sig["__reject_reason__"] = df_sig[cand].astype(str)
            break
    else:
        df_sig["__reject_reason__"] = ""

    for col in ["timestamp", "timestamp_utc", "signal_time"]:
        parse_ts_col(df_sig, col)

    to_numeric(df_sig, ["ml_confidence", "historical_prob", "historical_prob_lb90", "tcd_prob"])

# -------------------- PMI DECISIONS -------------------- #
print_section("PMI DECISIONS")
print(f"Total decisiones: {len(df_pmi)}  | Archivo: {pmi_path or '-'}")

if not df_pmi.empty and "action" in df_pmi.columns:
    counts = df_pmi["action"].value_counts(dropna=False)
    print("\nPor acción:")
    print(counts)
    counts.to_csv("reports/pmi_by_action.csv", header=["count"])

    if "symbol" in df_pmi.columns:
        pvt = df_pmi.pivot_table(index="symbol", columns="action", aggfunc="size", fill_value=0)
        print("\nPor símbolo x acción:")
        print(pvt)
        pvt.to_csv("reports/pmi_by_symbol_action.csv")

    reasons = Counter(df_pmi.get("reason", pd.Series([], dtype=str)).dropna().astype(str))
    if reasons:
        print("\nTop 10 razones:")
        for r, c in reasons.most_common(10):
            print(f"  - {r}: {c}")
        pd.Series(dict(reasons)).to_csv("reports/pmi_top_reasons.csv", header=["count"])

    # métricas promedio si existen
    metrics = [c for c in ["ml_confidence","historical_prob","historical_prob_lb90","tcd_prob","divergence_score"] if c in df_pmi.columns]
    if metrics:
        means = df_pmi[metrics].mean(numeric_only=True).to_frame("mean").round(3)
        print("\nPromedio de métricas (PMI):")
        print(means)
        means.to_csv("reports/pmi_metrics_mean.csv")

# -------------------- SIGNALS HISTORY -------------------- #
print_section("SIGNALS HISTORY")
print(f"Total señales: {len(df_sig)}  | Archivo: {sig_path or '-'}")

if not df_sig.empty:
    if "__status__" in df_sig.columns:
        by_status = df_sig["__status__"].value_counts(dropna=False)
        print("\nPor estado:")
        print(by_status)
        by_status.to_csv("reports/signals_by_status.csv", header=["count"])

    # motivos de rechazo más frecuentes
    rej = df_sig[df_sig["__status__"].str.contains("REJECT", case=False, na=False)]["__reject_reason__"]
    if not rej.empty:
        top_rej = rej.value_counts().head(10)
        print("\nTop motivos de rechazo:")
        print(top_rej)
        top_rej.to_csv("reports/signals_top_rejections.csv", header=["count"])

    # métricas promedio de señales
    smetrics = [c for c in ["ml_confidence","historical_prob","historical_prob_lb90","tcd_prob"] if c in df_sig.columns]
    if smetrics:
        means_sig = df_sig[smetrics].mean(numeric_only=True).to_frame("mean").round(3)
        print("\nPromedio de métricas de señales:")
        print(means_sig)
        means_sig.to_csv("reports/signals_metrics_mean.csv")

    # Embudo (generated -> approved -> executed)
    funnel = {}
    for name in ["GENERATED","APPROVED","EXECUTED","BLOCKED_POSITION","REJECTED","REJECTED_BY_CONTROLLER"]:
        funnel[name] = int((df_sig["__status__"] == name).sum())
    print("\nEmbudo señales (conteo):")
    print(pd.Series(funnel))
    pd.Series(funnel).to_csv("reports/signals_funnel.csv", header=["count"])

# -------------------- LOTES EFECTIVOS DESDE LOGS -------------------- #
rows = []
for logpath in glob.glob(os.path.join("logs","*.log")):
    try:
        with open(logpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Extrae: símbolo + lots solicitados/normalizados + volume en request
        pattern = (
            r"Ejecutando orden.+?Símbolo:\s*(?P<sym>\S+).+?"
            r"Lotes solicitados(?: \(post-override\))?:\s*(?P<req>[\d\.]+).+?"
            r"Lotes normalizados:\s*(?P<norm>[\d\.]+).+?"
            r"Request:\s*\{[^}]*\"volume\":\s*(?P<vol>[\d\.]+)"
        )
        for m in re.finditer(pattern, text, re.S):
            rows.append({
                "symbol": m.group("sym"),
                "lots_requested": float(m.group("req")),
                "lots_normalized": float(m.group("norm")),
                "volume_request": float(m.group("vol")),
                "logfile": os.path.basename(logpath),
            })
    except Exception:
        pass

print_section("LOT SIZE (desde logs)")
if rows:
    df_lots = pd.DataFrame(rows)
    last_by_sym = df_lots.groupby("symbol").tail(1).set_index("symbol")
    print(last_by_sym[["lots_requested","lots_normalized","volume_request","logfile"]])
    last_by_sym.to_csv("reports/lot_size_last_seen.csv")
else:
    print("(No encontré líneas con 'Lotes solicitados' en logs/*)")

print("\nListo. Reportes en carpeta 'reports/'.")
