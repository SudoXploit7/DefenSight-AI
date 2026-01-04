import os
import json
import xmltodict
import pandas as pd
from datetime import datetime

RAW_DIR = "raw_data"
OUT_DIR = "normalized"
SUPPORTED_EXTENSIONS = [".xml", ".json", ".csv", ".log", ".txt"]
IDS_MIN_SIZE_MB = 10  # CSVs ≥ 5MB will be shortened automatically

def ensure_output_dir():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

def detect_type(filepath):
    basename = os.path.basename(filepath).lower()
    if "_config" in basename:
        return "config"
    elif "_rules" in basename or "_compliance" in basename:
        return "compliance"
    elif "_ids" in basename:
        return "ids"
    elif "_nmap" in basename:
        return "nmap"
    elif "_flog" in basename:
        return "log"
    elif "_cert" in basename:
        return "cert"
    elif "_gateway" in basename:
        return "gateway"
    elif "_filter" in basename:
        return "filter"
    elif "_ipsec" in basename:
        return "ipsec"
    elif "_traffic" in basename:
        return "traffic"
    return "unknown"

def flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out

def parse_xml(filepath):
    with open(filepath, "r") as f:
        data = xmltodict.parse(f.read())
    flat = flatten(data)
    return [{"description": f"{k}: {v}", "raw": f"{k}: {v}"} for k, v in flat.items()]

def parse_csv(filepath, shorten=True):
    df = pd.read_csv(filepath).fillna("")
    df.columns = [col.lower().strip() for col in df.columns]

    # === Column alias mapping ===
    column_map = {
        "srcip": ["srcip", "src_ip", "source_ip", "sip"],
        "dstip": ["dstip", "dst_ip", "destination_ip", "dip"],
        "proto": ["proto", "protocol"],
        "attack_cat": ["attack_cat", "category"],
        "severity": ["severity", "threat_level", "risk", "level"],
        "msg": ["msg", "message", "alert", "description"],
        "timestamp": ["timestamp", "time", "ts"]
    }

    def find_column(possibilities):
        for col in possibilities:
            if col in df.columns:
                return col
        return None

    # === Dynamically detect real columns ===
    cols = {key: find_column(possibilities) for key, possibilities in column_map.items()}

    # === Apply severity filter ===
    if cols["severity"]:
        df = df[df[cols["severity"]].astype(str).str.isnumeric()]
        df[cols["severity"]] = df[cols["severity"]].astype(int)
        df = df[df[cols["severity"]] >= 3]

    # === Optional: attack category filter ===
    if cols["attack_cat"]:
        keep = {"DoS", "Recon", "Shellcode", "Exploit"}
        df = df[df[cols["attack_cat"]].isin(keep)]

    # === Build clean description ===
    def summarize(row):
        parts = []
        if cols["timestamp"] and pd.notna(row[cols["timestamp"]]):
            parts.append(f"[{row[cols['timestamp']]}]")
        if cols["srcip"] and cols["dstip"]:
            parts.append(f"{row[cols['srcip']]} → {row[cols['dstip']]}")
        if cols["proto"]:
            parts.append(f"Proto: {row[cols['proto']]}")
        if cols["attack_cat"]:
            parts.append(f"Cat: {row[cols['attack_cat']]}")
        if cols["severity"]:
            parts.append(f"Severity: {row[cols['severity']]}")
        if cols["msg"]:
            parts.append(f"Msg: {row[cols['msg']]}")
        return " | ".join([p for p in parts if p])

    parsed = []
    for _, row in df.iterrows():
        desc = summarize(row)
        parsed.append({
            "description": desc,
            "raw": desc,
            "timestamp": row[cols["timestamp"]] if cols["timestamp"] else datetime.now().isoformat(),
            "type": "ids",
            "source_file": os.path.basename(filepath)
        })

    print(f"📉 IDS CSV: Reduced to {len(parsed)} rows from {len(df)} after filtering")
    return parsed


def parse_log(filepath):
    results = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                results.append({
                    "description": line.strip(),
                    "raw": line.strip(),
                    "line_number": i + 1
                })
    return results

def parse_txt(filepath):
    return parse_log(filepath)  # treat .txt same as .log

def parse_json_passthrough(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def convert_file(filepath):
    # make sure output folder exists for any usage context
    ensure_output_dir()

    ext = os.path.splitext(filepath)[1].lower()
    basename = os.path.basename(filepath)
    detected_type = detect_type(filepath)
    handler = None
    shorten = False

    if ext not in SUPPORTED_EXTENSIONS:
        print(f"❌ Unsupported file type: {basename}")
        return

    if ext == ".xml":
        handler = parse_xml
    elif ext == ".csv":
        shorten = detected_type == "ids" and os.path.getsize(filepath) >= IDS_MIN_SIZE_MB * 1024 * 1024
        handler = lambda fp: parse_csv(fp, shorten=shorten)
    elif ext == ".log":
        handler = parse_log
    elif ext == ".txt":
        handler = parse_txt
    elif ext == ".json":
        handler = parse_json_passthrough

    if not handler:
        print(f"❌ No handler for {basename}")
        return

    try:
        parsed = handler(filepath)
        for entry in parsed:
            entry.setdefault("type", detected_type)
            entry.setdefault("source_file", basename)
            entry.setdefault("timestamp", datetime.now().isoformat())

        out_path = os.path.join(OUT_DIR, basename.replace(ext, ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)

        size_note = "🔻 shortened" if shorten else ""
        print(f"✅ Converted {basename} → {os.path.basename(out_path)} {size_note}")

    except Exception as e:
        print(f"❌ Failed to convert {basename}: {e}")

def main():
    ensure_output_dir()
    for file in os.listdir(RAW_DIR):
        full_path = os.path.join(RAW_DIR, file)
        if os.path.isfile(full_path) and os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
            convert_file(full_path)

if __name__ == "__main__":
    main()

# === Real-time normalization wrapper ===
def normalize_file(filepath):
    """
    Normalizes ANY supported log file (XML, CSV, JSON, LOG, TXT)
    and returns the normalized output path.
    """
    # Ensure output folder exists
    ensure_output_dir()

    # Convert using existing function
    convert_file(filepath)

    # Produce output JSON path
    ext = os.path.splitext(filepath)[1].lower()
    basename = os.path.basename(filepath)
    out_path = os.path.join(OUT_DIR, basename.replace(ext, ".json"))

    return out_path
