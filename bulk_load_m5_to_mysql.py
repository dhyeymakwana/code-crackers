# bulk_load_m5_to_mysql.py
"""
Robust chunked loader for M5 CSVs into MySQL.

Fix: reader-detection no longer consumes the first chunk (avoids 0it issue).
Tries multiple encodings/separators/engines and returns a fresh TextFileReader for processing.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------- CONFIG ----------------
here = Path(__file__).resolve().parent

# .env path as a Path object
dotenv_path = Path("C:/Users/Uday Kumar/Downloads/HCL/.env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print(f".env not found at {dotenv_path}, using environment variables if set.")
    load_dotenv()  # fallback to system env vars

DB_USER = os.getenv("DB_USER", "m5user")
DB_PASS = os.getenv("DB_PASS", "m5pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "m5db")

# folder with CSV files (adjust if needed)
CSV_DIR = Path("C:/Users/Uday Kumar/Downloads/m5-forecasting-accuracy")

# chunk sizes
SALES_CHUNKSIZE = 1000    # number of wide rows to read per chunk from wide sales CSV
TO_SQL_CHUNKSIZE = 20000  # rows per multi-insert when writing to DB

# SQLAlchemy engine creation
DB_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    engine = create_engine(DB_URI, pool_pre_ping=True)
except Exception as e:
    print("Error creating DB engine. Check DB_URI and installed DB drivers.")
    print("DB_URI (masked):", DB_URI.replace(DB_PASS, "****") if DB_PASS else DB_URI)
    raise

# ---------------- HELPERS ----------------
def list_csv_files(csv_dir: Path):
    if not csv_dir.exists():
        print("CSV_DIR not found:", csv_dir)
        sys.exit(1)
    return sorted(csv_dir.glob("*.csv"))


def create_sales_long_table(engine):
    """
    Create sales_long table if not exists, then create index only if missing.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS sales_long (
        id VARCHAR(128),
        item_id VARCHAR(64),
        dept_id VARCHAR(64),
        cat_id VARCHAR(64),
        store_id VARCHAR(64),
        state_id VARCHAR(64),
        date DATE,
        d_label VARCHAR(32),
        sales INT,
        wm_yr_wk INT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

    # Create index if missing (MySQL doesn't support CREATE INDEX IF NOT EXISTS)
    check_sql = text("""
    SELECT COUNT(*) AS cnt
    FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'sales_long'
      AND index_name = 'idx_sales_long_id_date';
    """)
    with engine.begin() as conn:
        cnt = conn.execute(check_sql).scalar()
        if cnt == 0:
            conn.execute(text("CREATE INDEX idx_sales_long_id_date ON sales_long (id, date);"))
            print("Created index idx_sales_long_id_date")
        else:
            print("Index idx_sales_long_id_date already exists")


def load_small_table(df: pd.DataFrame, table_name: str):
    """
    Load smaller CSVs directly using pandas.to_sql (replace mode).
    """
    print(f"Inserting {len(df):,} rows into `{table_name}` via to_sql ...")
    df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=TO_SQL_CHUNKSIZE, method="multi")
    print(" -> done.")


def get_d_columns(columns):
    return [c for c in columns if str(c).startswith("d_")]


def find_working_chunk_reader(path: Path, chunksize: int):
    """
    Try several read_csv parameter combinations and return a working TextFileReader + description.
    Do NOT consume chunks to test: use pd.read_csv(..., nrows=1) to validate params, then open
    a fresh iterator with the same params.
    """
    attempts = []
    encodings = ["utf-8", "latin1", "cp1252"]
    seps = [",", ";", "\t"]
    engines = ["c", "python"]
    for enc in encodings:
        for sep in seps:
            for eng in engines:
                desc = f"encoding={enc} sep={repr(sep)} engine={eng}"
                try:
                    # quick test read (one row) to validate parser will work
                    pd.read_csv(path, nrows=1, encoding=enc, sep=sep, engine=eng, compression="infer")
                    # if we reach here, params are workable - open a fresh iterator for full processing
                    reader = pd.read_csv(path, iterator=True, chunksize=chunksize, dtype=str,
                                         compression="infer", encoding=enc, sep=sep, engine=eng)
                    return reader, desc
                except Exception as e:
                    attempts.append((desc, repr(e)))
    # none worked
    msg_lines = ["No working chunked reader found. Attempts:"]
    for d, res in attempts:
        msg_lines.append(f" - {d} -> {res}")
    raise RuntimeError("\n".join(msg_lines))


def process_sales_wide_and_load(sales_csv_path: Path, calendar_df: pd.DataFrame):
    """
    Chunked melt of wide sales CSV into sales_long table.
    Requires calendar_df (with columns 'd' and 'date' at minimum).
    """
    print(f"Processing wide sales file (chunked melt) -> sales_long: {sales_csv_path.name}")
    create_sales_long_table(engine)

    # Attempt to get a working iterator (tries multiple encodings/seps/engines)
    try:
        reader, used_desc = find_working_chunk_reader(sales_csv_path, SALES_CHUNKSIZE)
    except Exception as e:
        print("ERROR: Could not open sales CSV in chunked mode.")
        print(e)
        print("You can try re-saving the CSV in UTF-8, ensure it's not empty, or run a debug script.")
        return

    total_inserted = 0
    chunk_i = 0
    # iterate and process
    # Use a manual counter to demonstrate progress if tqdm shows 0it unexpectedly
    for chunk in reader:
        chunk_i += 1
        if chunk_i % 10 == 0:
            print(f"Processing chunk #{chunk_i} (rows in this chunk: {len(chunk)})")
        d_cols = get_d_columns(chunk.columns)
        if not d_cols:
            print(f"Warning: no d_ columns detected in chunk {chunk_i} of {sales_csv_path.name}. Skipping chunk.")
            continue
        id_cols = [c for c in chunk.columns if c not in d_cols]

        # melt chunk to long format
        long = chunk.melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="sales")

        # merge calendar to get date and wm_yr_wk
        if 'd' not in calendar_df.columns:
            raise ValueError("calendar_df must have a 'd' column")
        long = long.merge(calendar_df[['d', 'date', 'wm_yr_wk']], on='d', how='left')

        # ensure expected id columns exist; fallback to case variations or placeholder
        expected_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        for c in expected_cols:
            if c not in long.columns:
                if c.upper() in long.columns:
                    long[c] = long[c.upper()]
                elif c.lower() in long.columns:
                    long[c] = long[c.lower()]
                else:
                    long[c] = ""

        # rename and select final columns
        long = long.rename(columns={'d': 'd_label'})
        long = long[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'date', 'd_label', 'sales', 'wm_yr_wk']]

        # convert types
        long['date'] = pd.to_datetime(long['date'], errors='coerce')
        long['sales'] = pd.to_numeric(long['sales'], errors='coerce').fillna(0).astype(int)

        # append to DB
        long.to_sql("sales_long", engine, if_exists="append", index=False, chunksize=TO_SQL_CHUNKSIZE, method="multi")
        total_inserted += len(long)

    print(f"Total chunks processed: {chunk_i:,}")
    print(f"Total rows appended to sales_long: {total_inserted:,}")


# ---------------- MAIN ----------------
def main():
    print("CSV_DIR =", CSV_DIR)
    print("DB_URI (masked) =", DB_URI.replace(DB_PASS, "****") if DB_PASS else DB_URI)

    csv_files = list_csv_files(CSV_DIR)
    if not csv_files:
        print("No CSV files found in", CSV_DIR)
        return

    print("Found CSV files:")
    for f in csv_files:
        print(" -", f.name)

    # Load calendar first (used to map d -> date)
    calendar_df = None
    for f in csv_files:
        if f.name.lower().startswith("calendar"):
            print("Loading calendar:", f.name)
            # try common encodings when reading calendar (it's small)
            for enc in ("utf-8", "latin1"):
                try:
                    calendar_df = pd.read_csv(f, parse_dates=['date'], encoding=enc)
                    break
                except Exception:
                    calendar_df = None
            if calendar_df is None:
                print("Failed to read calendar.csv with utf-8 and latin1 encodings. Please check the file.")
                return
            # ensure 'd' column exists with name 'd'
            if 'd' not in calendar_df.columns:
                calendar_df = calendar_df.rename(columns={calendar_df.columns[0]: 'd'})
            break

    if calendar_df is None:
        print("ERROR: calendar CSV not found in folder. The wide sales file requires calendar.csv to map 'd' labels to dates.")
        return

    # iterate through CSVs
    for f in csv_files:
        fname = f.name.lower()

        # calendar (already processed)
        if fname.startswith("calendar"):
            load_small_table(calendar_df, "calendar")
            continue

        # sell_prices
        if "sell_prices" in fname or "sell-prices" in fname or "sellprices" in fname:
            print("Loading sell_prices:", f.name)
            # read with robust encoding fallback
            for enc in ("utf-8", "latin1"):
                try:
                    df = pd.read_csv(f, encoding=enc)
                    break
                except Exception:
                    df = None
            if df is None:
                print(f"Failed to read {f.name} with utf-8 and latin1. Skipping.")
                continue
            load_small_table(df, "sell_prices")
            continue

        # Only process the real sales training file (validation). Skip evaluation file.
        if fname.startswith("sales_train_validation"):
            print("Processing REAL sales wide file (validation):", f.name)
            process_sales_wide_and_load(f, calendar_df)
            continue

        if fname.startswith("sales_train_evaluation"):
            print("Skipping evaluation file (no real training values):", f.name)
            continue

        # fallback: load other CSVs as tables named by file stem
        tblname = f.stem.replace('-', '_').lower()
        print(f"Loading generic table {tblname} from {f.name}")
        # try encoding fallbacks
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(f, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            print(f"Failed to read {f.name} with utf-8 and latin1. Skipping.")
            continue
        load_small_table(df, tblname)

    print("All files processed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", type(e).__name__, e)
        raise
