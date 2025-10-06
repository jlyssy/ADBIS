#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DreamShop → LiveRamp transformer

Creates the initial inclusion file (wide, one column per segment) and weekly exclusion
files (hashed identifiers) for upload to LiveRamp.

Segments (fixed): HDSDSR, HDSDSE, HDSHDI, HDSHDS, HDSASR, HDSASE, HDSADI, HDSADS

Usage examples
--------------
# Build initial inclusions from a pipe file and write outputs into ./out
python dreamshop_liveramp.py inclusions \
  --input ./data/DreamShop_Audience_2025-10-10.txt \
  --sep '|' \
  --campaign DreamShop \
  --outdir ./out

# Build weekly email exclusions (sha256 only) from a csv
python dreamshop_liveramp.py exclusions \
  --input ./data/DreamShop_Exclusions_2025-10-17.csv \
  --sep ',' \
  --outdir ./out

# (Optional) Upload a file via SFTP with key-based auth
python dreamshop_liveramp.py upload \
  --local ./out/Aletheia_DreamShop_Inclusions_LR_initial_20251010.csv \
  --remote /uploads/aletheia_dreamshop/Aletheia_DreamShop_Inclusions_LR_initial_20251010.csv \
  --host files.liveramp.com \
  --username aletheia-dreamshop \
  --key ~/.ssh/id_ed25519 \
  --known-hosts ~/.ssh/known_hosts

Notes
-----
• No secrets in code. Use key-based auth and a known_hosts file for host key pinning.
• ZIP codes are handled as strings with leading zeros preserved.
• Emails are normalized (strip + lower) before hashing.
• If LANDING_PAGE_URL exists, a URL→segment QA table is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# ------------------------------
# Constants & helpers
# ------------------------------
SEGMENTS: List[str] = [
    "HDSDSR", "HDSDSE", "HDSHDI", "HDSHDS", "HDSASR", "HDSASE", "HDSADI", "HDSADS",
]

PII_COLS_BASE = [
    "FIRSTNAME", "LASTNAME", "ADDRESS1", "ADDRESS2", "CITY", "STATE", "ZIP",
    "EMAIL1", "EMAIL2", "EMAIL3", "EMAIL4", "PHONE1", "PHONE2",
]

CANDIDATE_SEGMENT_COLS = ["DM_OFFERCODE", "EM_OFFERCODE", "SEGMENT_NAME", "SEGMENTCODE"]


def _today_yyyymmdd() -> str:
    return dt.date.today().strftime("%Y%m%d")


def _normalize_email(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if s.strip() == "" or s.lower() in {"nan", "none"}:
        return None
    return s.strip().lower()


def _hash_value(value: str, algo: str) -> str:
    value_bytes = value.encode("utf-8")
    if algo == "sha256":
        return hashlib.sha256(value_bytes).hexdigest()
    if algo == "md5":
        return hashlib.md5(value_bytes).hexdigest()
    if algo == "sha1":
        return hashlib.sha1(value_bytes).hexdigest()
    raise ValueError(f"Unsupported hash algo: {algo}")


# ------------------------------
# Read & prepare input
# ------------------------------

def read_table(path: Path, sep: str) -> pd.DataFrame:
    """Read delimited text as *strings* to preserve leading zeros etc."""
    return pd.read_csv(path, sep=sep, dtype="string", keep_default_na=True, na_filter=True, low_memory=False)


def choose_segment_source(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in CANDIDATE_SEGMENT_COLS:
        if c in df.columns:
            return c
    # As a fallback, if a column named 'SEGMENT' exists with values matching our codes
    if "SEGMENT" in df.columns and df["SEGMENT"].astype("string").str.upper().isin(SEGMENTS).any():
        return "SEGMENT"
    raise KeyError(
        "Could not locate a segment column. Provide --segment-col or include one of: "
        f"{[preferred] + CANDIDATE_SEGMENT_COLS + ['SEGMENT']}"
    )


def standardize_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with standardized PII columns required by LiveRamp.

    Accepts variants (FNAME/LNAME, ADDRESS) and creates ADDRESS2/EMAIL2-4/PHONE1-2 if missing.
    ZIP is left as string and zero-padded to 5 when possible.
    """
    out = df.copy()

    # Name columns
    if "FIRSTNAME" not in out.columns and "FNAME" in out.columns:
        out = out.rename(columns={"FNAME": "FIRSTNAME"})
    if "LASTNAME" not in out.columns and "LNAME" in out.columns:
        out = out.rename(columns={"LNAME": "LASTNAME"})

    # Address columns
    if "ADDRESS1" not in out.columns and "ADDRESS" in out.columns:
        out = out.rename(columns={"ADDRESS": "ADDRESS1"})
    if "ADDRESS2" not in out.columns:
        out["ADDRESS2"] = pd.NA

    # City/State/ZIP
    for col in ["CITY", "STATE", "ZIP"]:
        if col not in out.columns:
            out[col] = pd.NA

    # ZIP as 5-char string where possible
    out["ZIP"] = (
        out["ZIP"].astype("string").str.extract(r"(\d+)", expand=False).fillna(out["ZIP"]).astype("string")
    )
    # Zero-pad if numeric-like and length < 5
    out["ZIP"] = out["ZIP"].where(out["ZIP"].str.len() >= 5, out["ZIP"].str.zfill(5))

    # Emails → EMAIL1..4
    if "EMAIL1" not in out.columns:
        if "EMAIL" in out.columns:
            out = out.rename(columns={"EMAIL": "EMAIL1"})
        else:
            out["EMAIL1"] = pd.NA
    for extra in ["EMAIL2", "EMAIL3", "EMAIL4"]:
        if extra not in out.columns:
            out[extra] = pd.NA

    # Phones
    for p in ["PHONE1", "PHONE2"]:
        if p not in out.columns:
            out[p] = pd.NA

    # Keep only columns we need + any others for later joins
    # We'll select final columns when writing inclusions.
    return out


# ------------------------------
# Inclusion building
# ------------------------------

def build_inclusion(df: pd.DataFrame, segment_source_col: str) -> pd.DataFrame:
    base = standardize_pii(df)

    # Normalize the segment source to uppercase strings
    seg_series = base[segment_source_col].astype("string").str.strip().str.upper()

    # Start with base PII columns
    inclusion = base.copy()

    # Ensure only the PII columns + segment columns are output
    # (But keep extra cols for possible QA before final selection.)

    # Create one column per segment with 1/blank membership
    for seg in SEGMENTS:
        inclusion[seg] = (seg_series == seg).map({True: "1", False: pd.NA})

    # Final column ordering for output
    cols = [c for c in PII_COLS_BASE] + SEGMENTS
    # Ensure we have FIRSTNAME/LASTNAME/ADDRESS1/CITY/STATE/ZIP/EMAIL1
    required_now = ["FIRSTNAME", "LASTNAME", "ADDRESS1", "CITY", "STATE", "ZIP", "EMAIL1"]
    missing_required = [c for c in required_now if c not in inclusion.columns]
    if missing_required:
        raise KeyError(f"Missing required PII columns: {missing_required}")

    inclusion = inclusion.reindex(columns=cols)
    return inclusion


def write_inclusion_files(
    inclusion: pd.DataFrame,
    raw: pd.DataFrame,
    outdir: Path,
    campaign: str,
    date_str: Optional[str] = None,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if not date_str:
        date_str = _today_yyyymmdd()

    inc_name = f"Aletheia_{campaign}_Inclusions_LR_initial_{date_str}.csv"
    inc_path = outdir / inc_name
    inclusion.to_csv(inc_path, index=False)

    # Optional URL→segment QA if LANDING_PAGE_URL exists
    url_col = None
    for c in ["LANDING_PAGE_URL", "LANDING_PAGE", "URL"]:
        if c in raw.columns:
            url_col = c
            break
    seg_src = None
    for c in CANDIDATE_SEGMENT_COLS + ["SEGMENT"]:
        if c in raw.columns:
            seg_src = c
            break
    if url_col and seg_src:
        qa = (
            raw[[url_col, seg_src]]
            .astype("string")
            .groupby([url_col, seg_src], dropna=False)
            .size()
            .reset_index(name="count")
        )
        qa_name = f"Aletheia_{campaign}_URLMATCH_{date_str}.csv"
        qa_path = outdir / qa_name
        qa.to_csv(qa_path, index=False)

    return inc_path


# ------------------------------
# Exclusion building
# ------------------------------

def build_exclusions(
    df: pd.DataFrame,
    id_col: str = "EMAIL",
    hashes: Iterable[str] = ("sha256",),
) -> pd.DataFrame:
    if id_col not in df.columns:
        # Try EMAIL1 as an alternative
        if id_col == "EMAIL" and "EMAIL1" in df.columns:
            id_col = "EMAIL1"
        else:
            raise KeyError(f"Could not find identifier column '{id_col}' in input")

    emails = df[id_col].astype("string").map(_normalize_email)
    emails = emails.dropna()

    out = pd.DataFrame(index=emails.index)
    for algo in hashes:
        out_col = "email_" + algo if algo != "md5_hash" else "md5_hash"
        # Maintain historical names if needed
        if algo == "md5":
            out_col = "md5_hash"
        elif algo == "sha1":
            out_col = "sha1_hash"
        elif algo == "sha256":
            out_col = "sha256_hash"
        out[out_col] = emails.map(lambda x: _hash_value(x, algo))

    out = out.reset_index(drop=True)
    return out


def write_exclusion_file(
    excl: pd.DataFrame,
    outdir: Path,
    campaign: str,
    label: str = "week",
    date_str: Optional[str] = None,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if not date_str:
        date_str = _today_yyyymmdd()
    name = f"Aletheia_{campaign}_Exclusions_LR_{label}_{date_str}.csv"
    path = outdir / name
    excl.to_csv(path, index=False)
    return path


# ------------------------------
# SFTP uploader (key-based, pinned host keys)
# ------------------------------

def sftp_upload(
    local_path: Path,
    remote_path: str,
    host: str,
    username: str,
    key_path: Path,
    known_hosts: Optional[Path] = None,
    port: int = 22,
) -> None:
    """Upload a file via SFTP using key-based auth and known_hosts pinning.

    Requires: paramiko
    """
    import paramiko  # lazy import to keep base runtime light

    client = paramiko.SSHClient()
    if known_hosts and known_hosts.exists():
        client.load_host_keys(str(known_hosts))
    else:
        # Fall back to system host keys
        client.load_system_host_keys()

    # Reject unknown hosts (do NOT AutoAddPolicy)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    pkey = None
    try:
        pkey = paramiko.Ed25519Key.from_private_key_file(str(key_path))
    except Exception:
        # Try RSA if ed25519 not used
        pkey = paramiko.RSAKey.from_private_key_file(str(key_path))

    client.connect(hostname=host, port=port, username=username, pkey=pkey, look_for_keys=False)
    try:
        with client.open_sftp() as sftp:
            sftp.put(str(local_path), remote_path)
    finally:
        client.close()


# ------------------------------
# CLI
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DreamShop → LiveRamp transformer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # inclusions
    p_inc = sub.add_parser("inclusions", help="Create initial inclusion file (wide)")
    p_inc.add_argument("--input", required=True, type=Path)
    p_inc.add_argument("--sep", default="|", help="Input delimiter (default: '|')")
    p_inc.add_argument("--outdir", required=True, type=Path)
    p_inc.add_argument("--campaign", default="DreamShop")
    p_inc.add_argument("--segment-col", default=None, help="Column to read segment codes from (default: auto-detect)")
    p_inc.add_argument("--date", default=None, help="Override yyyymmdd in filenames")

    # exclusions
    p_exc = sub.add_parser("exclusions", help="Create weekly exclusion (hashed emails)")
    p_exc.add_argument("--input", required=True, type=Path)
    p_exc.add_argument("--sep", default=",", help="Input delimiter (default: ',')")
    p_exc.add_argument("--outdir", required=True, type=Path)
    p_exc.add_argument("--campaign", default="DreamShop")
    p_exc.add_argument("--id-col", default="EMAIL", help="Identifier column (default: EMAIL; EMAIL1 also supported)")
    p_exc.add_argument(
        "--hashes",
        default="sha256",
        help="Comma-separated list of hash algos (sha256,md5,sha1). Default: sha256",
    )
    p_exc.add_argument("--label", default="week", help="Label used in filename, e.g., week19_2025")
    p_exc.add_argument("--date", default=None, help="Override yyyymmdd in filenames")

    # upload
    p_up = sub.add_parser("upload", help="Upload a file via SFTP (key-based)")
    p_up.add_argument("--local", required=True, type=Path)
    p_up.add_argument("--remote", required=True)
    p_up.add_argument("--host", required=True)
    p_up.add_argument("--username", required=True)
    p_up.add_argument("--key", required=True, type=Path)
    p_up.add_argument("--known-hosts", default=None, type=Path)
    p_up.add_argument("--port", default=22, type=int)

    args = parser.parse_args()

    if args.cmd == "inclusions":
        df = read_table(args.input, args.sep)
        seg_src = choose_segment_source(df, args.segment_col)
        inc = build_inclusion(df, seg_src)
        path = write_inclusion_files(inc, df, args.outdir, args.campaign, args.date)
        print(f"Inclusion file written: {path}")
    elif args.cmd == "exclusions":
        df = read_table(args.input, args.sep)
        algos = [a.strip().lower() for a in args.hashes.split(",") if a.strip()]
        excl = build_exclusions(df, id_col=args.id_col, hashes=algos)
        path = write_exclusion_file(excl, args.outdir, args.campaign, args.label, args.date)
        print(f"Exclusion file written: {path}")
    elif args.cmd == "upload":
        sftp_upload(args.local, args.remote, args.host, args.username, args.key, args.known_hosts, args.port)
        print(f"Uploaded {args.local} → {args.remote}")


if __name__ == "__main__":
    main()
