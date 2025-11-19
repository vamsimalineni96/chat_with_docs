import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from prune_src.utils.config import TEST_DATASET, DB_DOCS


# -----------------------------
# Helpers: name normalization & variants (kept for future use)
# -----------------------------
def split_camel_snake(name: str) -> List[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    toks = [t.lower() for t in s.split() if t.strip()]
    return toks


def singularize(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("sses") or word.endswith("xes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def simple_variants(name: str) -> List[str]:
    toks = split_camel_snake(name)
    phrase = " ".join(toks)
    sing = " ".join(singularize(t) for t in toks)
    variants = {phrase, sing}

    joined = "_".join(toks)
    if joined.endswith("_id"):
        variants.update(
            {
                "id",
                "identifier",
                phrase.replace("_", " "),
                joined.replace("_", " "),
                phrase,
            }
        )

    alias_map = {
        "first name": ["fname", "first"],
        "last name": ["lname", "last"],
        "middle initial": ["mname", "middle"],
        "phone number": ["phone", "telephone", "mobile"],
        "email address": ["email", "mail"],
        "user name": ["username", "login", "login name"],
        "order": ["orders"],
        "item": ["items", "line items"],
        "quantity": ["qty"],
        "wins count": ["wins"],
        "events number": ["events"],
        "status code": ["status"],
        "date": ["day", "month", "year", "datetime", "time"],
        "price": ["cost", "amount"],
        "city": ["town", "city name"],
        "club": ["clubs"],
        "player": ["players"],
    }
    phrase_lc = phrase
    for k, alts in alias_map.items():
        if k in phrase_lc:
            variants.update(alts)

    specials = {
        "first_name": ["first name", "fname"],
        "last_name": ["last name", "lname"],
        "middle_initial": ["middle initial", "mname"],
        "phone_number": ["phone", "telephone", "mobile"],
        "email_address": ["email", "mail"],
        "order_id": ["order id"],
        "player_id": ["player id"],
        "club_id": ["club id"],
        "invoice_number": ["invoice", "invoice number", "bill", "receipt"],
        "shipment_id": ["shipment", "delivery", "shipping"],
    }
    if joined in specials:
        variants.update(specials[joined])

    return sorted(v for v in variants if v)


# -----------------------------
# Schema parsing (names + types + PK + FK)
# -----------------------------
CREATE_RE = re.compile(
    r'CREATE\s+TABLE\s+([`"\[]?)(?P<table>[A-Za-z0-9_]+)\1\s*\((?P<body>.*?)\)\s*;',
    re.IGNORECASE | re.DOTALL,
)
COL_RE = re.compile(
    r'^\s*([`"\[]?)(?P<col>[A-Za-z0-9_]+)\1\s+(?P<type>[A-Za-z0-9_()]+)',
    re.IGNORECASE,
)
PK_RE = re.compile(
    r'PRIMARY\s+KEY\s*\((?P<cols>[^)]+)\)',
    re.IGNORECASE,
)
# Fixed FK regex: no broken backreference, handles optional quotes
FK_RE = re.compile(
    r'FOREIGN\s+KEY\s*\((?P<from_cols>[^)]+)\)\s+REFERENCES\s+[`"\[]?(?P<ref_table>[A-Za-z0-9_]+)[`"\]]?\s*\((?P<ref_cols>[^)]+)\)',
    re.IGNORECASE,
)

SQL_TYPE_CANON = {
    # integers
    "int": "int",
    "integer": "int",
    "bigint": "int",
    "smallint": "int",
    # floats / numeric
    "real": "real",
    "float": "real",
    "double": "real",
    "numeric": "real",
    "decimal": "real",
    # text-ish
    "text": "text",
    "varchar": "text",
    "char": "text",
    "string": "text",
    "nvarchar": "text",
    # temporal
    "datetime": "datetime",
    "timestamp": "datetime",
    "date": "date",
    "time": "time",
    # bool -> treat as int for simplicity
    "bool": "int",
    "boolean": "int",
}


def canon_type(sql_type: str) -> str:
    t = sql_type.lower()
    t = re.sub(r"\(.*?\)", "", t)  # strip size/precision
    t = t.strip()
    return SQL_TYPE_CANON.get(t, t or "text")


def _split_columns_sql(body: str) -> List[str]:
    """
    Split the body of a CREATE TABLE (...) into top-level comma-separated chunks,
    ignoring commas inside parentheses (e.g., PRIMARY KEY(a, b)).
    """
    chunks = []
    curr = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            chunk = "".join(curr).strip()
            if chunk:
                chunks.append(chunk)
            curr = []
        else:
            curr.append(ch)
    if curr:
        chunk = "".join(curr).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def parse_schema(schema_sql: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
    {
      table_name: {
        "columns":      {column_name: type_tag, ...},
        "primary_keys": [col1, col2, ...],
        "foreign_keys": [
          {"from_column": ..., "to_table": ..., "to_column": ...},
          ...
        ],
      },
      ...
    }
    """
    tables: Dict[str, Dict[str, Any]] = {}
    for m in CREATE_RE.finditer(schema_sql or ""):
        table = m.group("table")
        body = m.group("body")

        cols: Dict[str, str] = {}
        primary_keys: List[str] = []
        foreign_keys: List[Dict[str, str]] = []

        chunks = _split_columns_sql(body)

        for ch in chunks:
            # IMPORTANT: handle constraints BEFORE column regex,
            # so we don't mis-parse "PRIMARY KEY ..." or "FOREIGN KEY ..." as columns.
            pm = PK_RE.search(ch)
            if pm:
                pk_cols = [
                    c.strip().strip('`"[]')
                    for c in pm.group("cols").split(",")
                    if c.strip()
                ]
                primary_keys.extend(pk_cols)
                continue

            fm = FK_RE.search(ch)
            if fm:
                from_cols = [
                    c.strip().strip('`"[]')
                    for c in fm.group("from_cols").split(",")
                    if c.strip()
                ]
                ref_table = fm.group("ref_table")
                ref_cols = [
                    c.strip().strip('`"[]')
                    for c in fm.group("ref_cols").split(",")
                    if c.strip()
                ]
                for fc, rc in zip(from_cols, ref_cols):
                    foreign_keys.append(
                        {
                            "from_column": fc,
                            "to_table": ref_table,
                            "to_column": rc,
                        }
                    )
                continue

            cm = COL_RE.match(ch)
            if cm:
                col = cm.group("col")
                typ = canon_type(cm.group("type"))
                cols[col] = typ
                continue

        if cols:
            tables[table] = {
                "columns": cols,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            }

    return tables


# -----------------------------
# FK guessing from names only (optional, kept for completeness)
# -----------------------------
def guess_fk_edges(tables: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Uses only column names (ignores real FK metadata) to guess edges.
    Accepts both:
      {table: {col: type}}
    and
      {table: {"columns": {...}, "primary_keys": [...], "foreign_keys": [...]}}
    """
    edges = set()

    # normalize to {table: {col: type}}
    norm_tables: Dict[str, Dict[str, str]] = {}
    for t, info in tables.items():
        if isinstance(info, dict) and "columns" in info:
            norm_tables[t] = info["columns"]
        else:
            norm_tables[t] = info  # assume flat mapping

    lower_to_table = {t.lower(): t for t in norm_tables.keys()}

    id_cols = defaultdict(list)
    for t, cols in norm_tables.items():
        for c in cols:
            if c.lower().endswith("_id"):
                id_cols[t].append(c)

    # Rule 1: *_id → table with matching base (singular/plural tolerant)
    for t, idlist in id_cols.items():
        for c in idlist:
            base = c[:-3].lower()
            candidates = {base, singularize(base), base + "s", singularize(base) + "s"}
            for cand in candidates:
                if cand in lower_to_table and lower_to_table[cand] != t:
                    a, b = sorted([t, lower_to_table[cand]])
                    edges.add((a, b))

    # Rule 2: same *_id present in multiple tables
    col_to_tables = defaultdict(set)
    for t, cols in norm_tables.items():
        for c in cols:
            if c.lower().endswith("_id"):
                col_to_tables[c.lower()].add(t)
    for _, ts in col_to_tables.items():
        ts = sorted(ts)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                edges.add(tuple(sorted([a, b])))

    return sorted(edges)


# -----------------------------
# Doc builder (includes types + PK + FK, in your exact format)
# -----------------------------
def table_doc_str(table: str, info: Dict[str, Any]) -> str:
    """
    Desired format examples:

    club. columns: club_id (int), name (text), manager (text), captain (text),
                   manufacturer (text), sponsor (text), primary key (club_id)

    player. columns: player_id (real), name (text), country (text), earnings (real),
                     events_number (int), wins_count (int), club_id (int),
                     primary key (player_id), foreign key (club_id),
                     references club (club_id)
    """
    columns: Dict[str, str] = info.get("columns", {}) or {}
    primary_keys: List[str] = info.get("primary_keys", []) or []
    foreign_keys: List[Dict[str, str]] = info.get("foreign_keys", []) or []

    # columns part
    col_parts = [
        f"{col.lower()} ({typ})"
        for col, typ in columns.items()
    ]
    doc = f"{table.lower()}. columns: {', '.join(col_parts)}" if col_parts else f"{table.lower()}."

    # primary key part
    if primary_keys:
        pk_str = ", ".join(pk.lower() for pk in primary_keys)
        doc += f", primary key ({pk_str})"

    # foreign key(s) part
    for fk in foreign_keys:
        fc = fk.get("from_column")
        tt = fk.get("to_table")
        tc = fk.get("to_column")
        if not (fc and tt and tc):
            continue
        doc += (
            f", foreign key ({fc.lower()}), "
            f"references {tt.lower()} ({tc.lower()})"
        )

    return doc


# -----------------------------
# JSONL I/O
# -----------------------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON error on line {i}: {e}")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Main
# -----------------------------
def create_docs():
    ensure_dir(DB_DOCS)

    # Collect the schema per db_id
    db_to_schema: Dict[str, str] = {}
    for obj in read_jsonl(TEST_DATASET):
        db_id = obj.get("db_id")
        schema = obj.get("db_schema") or obj.get("schema") or ""
        if not db_id or not schema:
            continue
        if db_id not in db_to_schema:
            db_to_schema[db_id] = schema

    # Build per-db table docs (WITH TYPES + PK + FK) and write the merged JSONL
    merged_out_path = os.path.join(DB_DOCS, "tables_with_types_and_keys_by_db.jsonl")
    rows_out: List[Dict] = []

    for db_id, schema_sql in sorted(db_to_schema.items()):
        tables = parse_schema(schema_sql)
        table_docs = [table_doc_str(t, tables[t]) for t in tables.keys()]
        # Line shape: { "<db_id>": { "tables": [ ... ] } }
        rows_out.append({db_id: {"tables": table_docs}})

    write_jsonl(merged_out_path, rows_out)

    print(f"[OK] Wrote {len(rows_out)} records to {merged_out_path}")


if __name__ == "__main__":
    create_docs()
