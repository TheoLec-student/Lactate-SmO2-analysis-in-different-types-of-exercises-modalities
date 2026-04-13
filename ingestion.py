import pandas as pd
import duckdb
from io import StringIO

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_trainred_csv(filepath, subject_name):
    """Parse a Train.Red CSV file and return (meta_dict, sensor_dict, timeseries_df)."""
    with open(filepath, encoding="utf-8-sig") as f:
        raw = f.read()

    lines = raw.replace("\r\n", "\n").split("\n")

    meta = {"subject_name": subject_name}
    muscle_states = {}
    sensor_meta = {"subject_name": subject_name}
    in_sensor_section = False

    for line in lines:
        kv = line.strip().split(",", 1)
        if len(kv) != 2:
            continue
        key, val = kv[0].strip(), kv[1].strip()

        # --- Session metadata (Table 1) ---
        if key == "Train.Red Export for":
            meta["export_for"] = val
        elif key == "Measurement Date":
            meta["measurement_date"] = val
        elif key == "Type":
            meta["type"] = val
        elif key == "Sport":
            meta["sport"] = val
        elif key == "Focus":
            meta["focus"] = val
        elif key == "Number of Sensors":
            meta["number_of_sensors"] = int(val) if val else None
        elif key == "Samples":
            meta["samples"] = int(val) if val else None
        elif key == "Measurement Duration":
            meta["measurement_duration"] = val
        elif key == "Distance":
            meta["distance"] = val
        elif key == "MSS Avg" and not in_sensor_section:
            meta["mss_avg"] = float(val) if val else None
        elif key == "TSI Min" and "tsi_min" not in meta:
            meta["tsi_min"] = float(val) if val else None
        elif key == "TSI Max" and "tsi_max" not in meta:
            meta["tsi_max"] = float(val) if val else None
        elif key == "TSI Avg" and "tsi_avg" not in meta:
            meta["tsi_avg"] = float(val) if val else None
        # Muscle states
        elif key == "Rest" and not in_sensor_section:
            muscle_states["rest_duration"] = val
        elif key == "Easy":
            muscle_states["easy_duration"] = val
        elif key == "Medi":
            muscle_states["medi_duration"] = val
        elif key == "Hard":
            muscle_states["hard_duration"] = val
        elif key == "Load":
            muscle_states["load_duration"] = val

        # --- Sensor metadata (Table 2) ---
        elif key == "Sensor 1":
            in_sensor_section = True
            sensor_meta["sensor_name"] = val
        elif key == "Sensor DataType":
            sensor_meta["sensor_data_type"] = val
        elif key == "Sensor ID":
            sensor_meta["sensor_id"] = val
        elif key == "Mac Address":
            sensor_meta["mac_address"] = val
        elif key == "Firmware Version":
            sensor_meta["firmware_version"] = val if val else None
        elif key == "Sensor Position":
            sensor_meta["sensor_position"] = val
        elif key == "Initial TSI":
            sensor_meta["initial_tsi"] = float(val) if val else None
        elif key == "Initial HBDiff":
            sensor_meta["initial_hbdiff"] = float(val) if val else None
        elif key == "Initial o2Hb":
            sensor_meta["initial_o2hb"] = float(val) if val else None
        elif key == "Initial HHb":
            sensor_meta["initial_hhb"] = float(val) if val else None
        elif key == "Tsi Var":
            sensor_meta["tsi_var"] = float(val) if val else None
        elif key == "Filter Id":
            sensor_meta["filter_id"] = val

    meta.update(muscle_states)

    # --- Timeseries (Table 2 - timeseries part) ---
    ts_start = next((i for i, l in enumerate(lines) if l.startswith("Timestamp (seconds passed)")), None)
    ts_df = None
    if ts_start is not None:
        csv_text = "\n".join(lines[ts_start:])
        ts_df = pd.read_csv(StringIO(csv_text), on_bad_lines="skip")
        ts_df.columns = [c.strip() for c in ts_df.columns]
        ts_df = ts_df.dropna(axis=1, how="all")
        ts_df = ts_df.rename(columns={
            "Timestamp (seconds passed)": "timestamp_s",
            "Lap/Event": "lap_event",
            "SmO2": "smo2",
            "HBDiff": "hbdiff",
            "Muscle state": "muscle_state",
            "Muscle trend": "muscle_trend",
            "SmO2 unfiltered": "smo2_unfiltered",
            "O2HB unfiltered": "o2hb_unfiltered",
            "HHb unfiltered": "hhb_unfiltered",
            "THb unfiltered": "thb_unfiltered",
            "HBDiff unfiltered": "hbdiff_unfiltered",
        })
        ts_df["subject_name"] = subject_name
        ts_df = ts_df[pd.to_numeric(ts_df["timestamp_s"], errors="coerce").notna()].copy()
        ts_df["timestamp_s"] = pd.to_numeric(ts_df["timestamp_s"])

    return meta, sensor_meta, ts_df


def parse_lactate_csv(filepath):
    """Parse the lactate CSV file (semicolon-separated, European decimals)."""
    with open(filepath, encoding="utf-8-sig") as f:
        raw = f.read()

    lines = raw.replace("\r\n", "\n").split("\n")
    # Header spans 2 lines due to quoted newline in "BW\n(self-reported)"
    header_line = (lines[0] + lines[1]).replace('"BW(self-reported)"', "BW_kg")
    csv_text = header_line + "\n" + "\n".join(lines[2:])

    df = pd.read_csv(StringIO(csv_text), sep=";", on_bad_lines="skip", decimal=",")
    df = df.dropna(how="all")
    df.columns = [c.strip().replace("[", "").replace("]", "") for c in df.columns]
    df = df.rename(columns={
        "Name": "subject_name",
        "BW_kg": "bw_kg",
        "Time": "time_min",
        "Condition": "condition",
        "La": "lactate_mmol_l",
        "SmO2": "smo2",
        "Synchro SmO2": "synchro_smo2",
    })
    keep = ["subject_name", "bw_kg", "time_min", "condition", "lactate_mmol_l", "smo2", "synchro_smo2"]
    df = df[[c for c in keep if c in df.columns]]
    df = df[df["subject_name"].notna() & (df["subject_name"].str.strip() != "")]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

alexis = parse_trainred_csv("data/Alexis_wingate.csv",  "Alexis")
antoine = parse_trainred_csv("data/Antoine_wingate.csv", "Antoine")
enzo   = parse_trainred_csv("data/Enzo_wingate.csv",   "Enzo")
jinwei = parse_trainred_csv("data/Jinwei_wingate.csv",  "Jinwei")
victor = parse_trainred_csv("data/Victor_wingate.csv",  "Victor")

lactate_df = parse_lactate_csv("data/TP2 - Wingate.csv")

# Each variable is a tuple: (meta_dict, sensor_dict, timeseries_df)
all_subjects = [alexis, antoine, enzo, jinwei, victor]

# ─────────────────────────────────────────────────────────────────────────────
# Create database
# ─────────────────────────────────────────────────────────────────────────────

con = duckdb.connect("lactate.duckdb")

# ─────────────────────────────────────────────────────────────────────────────
# Create Table 1 — Session metadata (one row per subject)
# ─────────────────────────────────────────────────────────────────────────────

# ✅ Renamed to session_metadata_df to avoid collision with the SQL table name
session_metadata_df = pd.DataFrame([meta for meta, _, _ in all_subjects])
session_metadata_df["measurement_date"] = pd.to_datetime(session_metadata_df["measurement_date"], errors="coerce")

con.execute("DROP TABLE IF EXISTS session_metadata")
con.execute("""
    CREATE TABLE session_metadata (
        subject_name        VARCHAR,
        export_for          VARCHAR,
        measurement_date    TIMESTAMP,
        type                VARCHAR,
        sport               VARCHAR,
        focus               VARCHAR,
        number_of_sensors   INTEGER,
        samples             INTEGER,
        measurement_duration VARCHAR,
        distance            VARCHAR,
        mss_avg             DOUBLE,
        tsi_min             DOUBLE,
        tsi_max             DOUBLE,
        tsi_avg             DOUBLE,
        rest_duration       VARCHAR,
        easy_duration       VARCHAR,
        medi_duration       VARCHAR,
        hard_duration       VARCHAR,
        load_duration       VARCHAR
    )
""")
# ✅ References the DataFrame variable, not the SQL table
con.execute("INSERT INTO session_metadata SELECT * FROM session_metadata_df")

# ─────────────────────────────────────────────────────────────────────────────
# Create Table 2 — Sensor info + timeseries
# ─────────────────────────────────────────────────────────────────────────────

# 2a — Sensor info
# ✅ Renamed to sensor_info_df to avoid collision with the SQL table name
sensor_info_df = pd.DataFrame([s for _, s, _ in all_subjects])

con.execute("DROP TABLE IF EXISTS sensor_info")
con.execute("""
    CREATE TABLE sensor_info (
        subject_name     VARCHAR,
        sensor_name      VARCHAR,
        sensor_data_type VARCHAR,
        sensor_id        VARCHAR,
        mac_address      VARCHAR,
        firmware_version VARCHAR,
        sensor_position  VARCHAR,
        initial_tsi      DOUBLE,
        initial_hbdiff   DOUBLE,
        initial_o2hb     DOUBLE,
        initial_hhb      DOUBLE,
        tsi_var          DOUBLE,
        filter_id        VARCHAR
    )
""")
# ✅ References the DataFrame variable, not the SQL table
con.execute("INSERT INTO sensor_info SELECT * FROM sensor_info_df")

# 2b — Sensor timeseries
ts_cols = ["subject_name", "timestamp_s", "lap_event", "smo2", "hbdiff",
           "muscle_state", "muscle_trend", "smo2_unfiltered", "o2hb_unfiltered",
           "hhb_unfiltered", "thb_unfiltered", "hbdiff_unfiltered"]

timeseries_frames = []
for _, _, ts in all_subjects:
    if ts is not None:
        timeseries_frames.append(ts[[c for c in ts_cols if c in ts.columns]])

# ✅ Renamed to sensor_timeseries_df to avoid collision with the SQL table name
sensor_timeseries_df = pd.concat(timeseries_frames, ignore_index=True)

con.execute("DROP TABLE IF EXISTS sensor_timeseries")
con.execute("""
    CREATE TABLE sensor_timeseries (
        subject_name      VARCHAR,
        timestamp_s       DOUBLE,
        lap_event         INTEGER,
        smo2              DOUBLE,
        hbdiff            DOUBLE,
        muscle_state      INTEGER,
        muscle_trend      INTEGER,
        smo2_unfiltered   DOUBLE,
        o2hb_unfiltered   DOUBLE,
        hhb_unfiltered    DOUBLE,
        thb_unfiltered    DOUBLE,
        hbdiff_unfiltered DOUBLE
    )
""")
# ✅ References the DataFrame variable, not the SQL table
con.execute("INSERT INTO sensor_timeseries SELECT * FROM sensor_timeseries_df")

# ─────────────────────────────────────────────────────────────────────────────
# Create Table 3 — Lactate (all subjects)
# ─────────────────────────────────────────────────────────────────────────────

con.execute("DROP TABLE IF EXISTS lactate_data")
con.execute("""
    CREATE TABLE lactate_data (
        subject_name    VARCHAR,
        bw_kg           DOUBLE,
        time_min        DOUBLE,
        condition       VARCHAR,
        lactate_mmol_l  DOUBLE,
        smo2            DOUBLE,
        synchro_smo2    VARCHAR
    )
""")
# ✅ References lactate_df (renamed from lactate), not the SQL table
con.execute("INSERT INTO lactate_data SELECT * FROM lactate_df")

# ─────────────────────────────────────────────────────────────────────────────
# Show head of Table 1, 2, 3
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("TABLE 1 — session_metadata")
print("=" * 60)
print(con.execute("SELECT * FROM session_metadata").df().to_string(index=False))

print("\n" + "=" * 60)
print("TABLE 2a — sensor_info")
print("=" * 60)
print(con.execute("SELECT * FROM sensor_info").df().to_string(index=False))

print("\n" + "=" * 60)
print("TABLE 2b — sensor_timeseries (head 5 per subject)")
print("=" * 60)
print(con.execute("""
    SELECT * FROM sensor_timeseries
    QUALIFY ROW_NUMBER() OVER (PARTITION BY subject_name ORDER BY timestamp_s) <= 5
    ORDER BY subject_name, timestamp_s
""").df().to_string(index=False))

print("\n" + "=" * 60)
print("TABLE 3 — lactate_data")
print("=" * 60)
print(con.execute("SELECT * FROM lactate_data").df().to_string(index=False))

con.close()
print("\n✅ lactate.duckdb créée avec succès.")