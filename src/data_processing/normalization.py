import ast
import json
import re

import numpy as np
import pandas as pd

# Trial Type Mappings
TRIAL_TYPE_MAP = {
    "lcmd": "left_command",
    "rcmd": "right_command",
    "lang": "language",
    "odd": "oddball",
    # 'beep': 'control',
    "loved_one": "loved_one_voice",
    "language_11": "language",
    # New CON010 (Mar 2026) stim_type values
    "familiar": "loved_one_voice",
    "unfamiliar": "control",
    "manual_sync_pulse": "manual_sync_pulse",  # preserved for EEG alignment
    "control": "control",
}


def normalize_trial_type(tt):
    """
    Normalizes a trial type string using standard mappings and rules.
    """
    if pd.isna(tt):
        return "unknown"

    tt_str = str(tt).lower().strip()

    # Remove '+p' suffix
    if tt_str.endswith("+p"):
        tt_str = tt_str[:-2]

    # Handle lang_XX patterns
    if tt_str.startswith("lang_") and tt_str[5:].isdigit():
        return "language"

    # Apply explict mapping
    if tt_str in TRIAL_TYPE_MAP:
        return TRIAL_TYPE_MAP[tt_str]

    return tt_str


def normalize_sentences(sample):
    """
    Converts inputs (strings, lists of strings, lists of ints, lists of dicts)
    into a consistent List[Dict] format: [{'event': ..., 'onset_time': ...}]
    """
    # 0. Handle types that break pd.isna or strict equality checks
    if isinstance(sample, (list, tuple, np.ndarray)):
        if len(sample) == 0:
            return []
    elif pd.isna(sample) or sample == "[]" or sample == "":
        return []

    parsed = None

    # 1. Parse stringified content if necessary
    if isinstance(sample, str):
        try:
            parsed = json.loads(sample)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(sample)
            except (ValueError, SyntaxError):
                # Treat as a single raw string event
                return [{"event": sample, "onset_time": None}]
    else:
        parsed = sample

    if not isinstance(parsed, list):
        # If it parsed to a dict/scalar, wrap it in a list
        parsed = [parsed]

    # 2. Normalize list items to Dicts
    normalized_list = []
    for item in parsed:
        if isinstance(item, dict):
            # Already a dict, preserve it.
            norm_item = item.copy()
            normalized_list.append(norm_item)

        elif isinstance(item, (int, float)):
            # Convert numeric event code -> Dict
            normalized_list.append({"event": str(item), "onset_time": None})

        elif isinstance(item, str):
            # Convert string event -> Dict
            normalized_list.append({"event": item, "onset_time": None})

        else:
            # Unknown type
            normalized_list.append({"event": str(item), "onset_time": None})

    return normalized_list


# ---------------------------------------------------------------------------
# New CON010 (Mar 2026) format support
# ---------------------------------------------------------------------------

_NEW_FORMAT_MARKER = "stim_type"  # column present only in the new format


def is_new_format(df: pd.DataFrame) -> bool:
    """Return True if *df* uses the new CON010 (Mar 2026) CSV schema.

    The new format has a ``stim_type`` column instead of ``trial_type``.
    """
    return _NEW_FORMAT_MARKER in df.columns and "trial_type" not in df.columns


def _parse_language_notes(notes: str) -> list:
    """Parse a language notes string like "Sentences: ['8', '16', ...]" into
    a list of event dicts ``[{'event': '8', 'onset_time': None}, ...]``.
    """
    match = re.search(r"\[([^\]]+)\]", str(notes))
    if not match:
        return []
    raw = match.group(1)
    # Extract individual quoted or bare numeric IDs
    ids = re.findall(r"'(\d+)'|\b(\d+)\b", raw)
    events = []
    for quoted, bare in ids:
        event_id = quoted or bare
        events.append({"event": event_id, "onset_time": None})
    return events


def _aggregate_oddball_rows(oddball_rows: pd.DataFrame) -> list[dict]:
    """Convert per-row oddball events from the new format into a list of
    event dicts, grouping them into contiguous blocks.

    Each row's ``notes`` is either ``'standard_tone'`` or ``'rare_tone'``.
    Returns a list of rows (as dicts) ready to concatenate back into the main
    DataFrame, one dict per contiguous block.
    """
    if oddball_rows.empty:
        return []

    # Assign a block ID: new block whenever there's a gap > 2 s between rows
    rows = oddball_rows.sort_values("start_time").copy()
    rows["_gap"] = rows["start_time"].diff().fillna(0)
    rows["_block"] = (rows["_gap"] > 2.0).cumsum()

    aggregated = []
    for _block_id, block in rows.groupby("_block"):
        sentences = [
            {
                "event": "standard" if "standard" in str(n) else "rare",
                "onset_time": float(t),
            }
            for n, t in zip(block["notes"], block["start_time"])
        ]
        aggregated.append(
            {
                "patient_id": block["patient_id"].iloc[0],
                "date": block["date"].iloc[0],
                "trial_type": "oddball",
                "sentences": sentences,
                "start_time": float(block["start_time"].iloc[0]),
                "end_time": float(block["end_time"].iloc[-1]),
                "duration": float(block["end_time"].iloc[-1] - block["start_time"].iloc[0]),
            }
        )
    return aggregated


def convert_new_format_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a new CON010 (Mar 2026) format DataFrame to the canonical
    ``trial_type`` / ``sentences`` schema consumed by the rest of the pipeline.

    Transformations applied:
    * ``stim_type`` → ``trial_type`` (with ``familiar``/``unfamiliar`` remapping)
    * ``manual_sync_pulse`` rows kept as ``trial_type='manual_sync_pulse'``
    * ``oddball+p`` individual rows aggregated into per-block grouped rows
    * ``language`` notes string parsed into ``[{event, onset_time}]`` lists
    * ``left_command`` / ``right_command`` get ``sentences=[]``
    * ``duration`` computed from ``end_time - start_time`` where absent
    """
    df = df.copy()

    # Rename stim_type → trial_type
    df = df.rename(columns={"stim_type": "trial_type"})

    # Map familiar/unfamiliar stim types
    _stim_map = {"familiar": "loved_one_voice", "unfamiliar": "control"}
    df["trial_type"] = df["trial_type"].replace(_stim_map)

    # Compute duration if missing
    if "duration" not in df.columns:
        df["duration"] = df["end_time"] - df["start_time"]
    else:
        missing = df["duration"].isna()
        df.loc[missing, "duration"] = df.loc[missing, "end_time"] - df.loc[missing, "start_time"]

    # --- Handle oddball+p rows (individual per tone → grouped per block) ------
    oddball_mask = df["trial_type"].str.lower().str.startswith("oddball")
    oddball_df = df[oddball_mask]
    non_oddball_df = df[~oddball_mask].copy()

    # Build sentences for non-oddball rows
    def _build_sentences(row) -> list:
        tt = str(row["trial_type"]).lower()
        notes = row.get("notes", "")
        if tt == "language":
            return _parse_language_notes(notes)
        # sync_pulse, loved_one_voice, control, left_command, right_command
        return []

    non_oddball_df["sentences"] = non_oddball_df.apply(_build_sentences, axis=1)

    # Aggregate oddball blocks
    oddball_records = _aggregate_oddball_rows(oddball_df)
    oddball_canonical = pd.DataFrame(oddball_records) if oddball_records else pd.DataFrame()

    # Concatenate and sort by start_time
    parts = [non_oddball_df]
    if not oddball_canonical.empty:
        parts.append(oddball_canonical)

    result = pd.concat(parts, ignore_index=True).sort_values("start_time").reset_index(drop=True)

    # Drop the now-redundant notes column (it becomes sentences)
    result = result.drop(columns=["notes"], errors="ignore")

    return result
