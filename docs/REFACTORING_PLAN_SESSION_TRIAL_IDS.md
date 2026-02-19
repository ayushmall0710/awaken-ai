# Refactoring Plan: session_id and trial_id Implementation

## Executive Summary

This document provides a detailed implementation plan for adding `session_id` and `trial_id` to the data pipeline, enabling robust entity tracking across the hierarchy: **patient → session → trial → events**.

**Timeline Estimate:** 3-4 weeks (includes testing and validation)

**Key Design Decision - Feature Granularity:**
The ERP pipeline currently aggregates all trials of the same type (e.g., multiple oddball trials) into ONE feature row per session. The deduplication key `["patient_id", "session_id", "trial_type", "processing_timestamp"]` supports this by:
- Allowing multiple trial types per session (oddball, language, command)
- Using `processing_timestamp` to track analysis reruns  
- Adding `rare_event_trial_ids` field to maintain trial-level traceability (e.g., "1,3,5,7")

Alternative per-trial features can be implemented later with deduplication key `["patient_id", "session_id", "trial_id"]` if needed.

---

## 1. Data Pipeline & Unification (`src/data_processing/pipeline.py`)

### 1.1 Current State
```python
REQUIRED_COLS = [
    "patient_id",
    "date",
    "trial_type",
    "sentences",
    "start_time",
    "end_time",
    "duration",
    "source_file",
]
```

### 1.2 Refactoring Tasks

#### Task 1.1: Update Schema Definition
**File:** `src/data_processing/pipeline.py`  
**Lines:** 8-17

**Changes:**
```python
REQUIRED_COLS = [
    "patient_id",
    "session_id",      # NEW: deterministic hash of (patient_id, date)
    "date",            # KEPT: for human readability
    "trial_id",        # NEW: sequential ID within session
    "trial_type",
    "sentences",
    "start_time",
    "end_time",
    "duration",
    "source_file",
]
```

**Implementation Details:**
1. Add session_id generation function:
```python
import hashlib

def generate_session_id(patient_id: str, date: str) -> str:
    """
    Generate deterministic session_id from patient_id and date.
    
    Args:
        patient_id: Patient identifier (e.g., 'CON008')
        date: Session date (YYYY-MM-DD format)
    
    Returns:
        8-character hex hash (e.g., 'a3f2c1d8')
    
    Example:
        >>> generate_session_id('CON008', '2025-02-14')
        'a3f2c1d8'
    """
    key = f"{patient_id}_{date}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]
```

2. Add trial_id generation in `process_stimulus_df()`:
```python
def process_stimulus_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    
    # Generate session_id for each row
    df["session_id"] = df.apply(
        lambda row: generate_session_id(row["patient_id"], row["date"]), 
        axis=1
    )
    
    # Generate trial_id: sequential within each session
    df = df.sort_values(["patient_id", "date", "start_time"])
    df["trial_id"] = df.groupby(["patient_id", "session_id"]).cumcount() + 1
    
    # ... rest of existing logic ...
    
    return df.reindex(columns=REQUIRED_COLS)
```

#### Task 1.2: Update Deduplication Logic
**File:** `src/data_processing/pipeline.py`  
**Lines:** 93-107

**Current:**
```python
unified_df = unified_df.drop_duplicates(
    subset=[
        "patient_id",
        "date",
        "trial_type",
        "_sentences_str",
        "start_time",
        "end_time",
        "duration",
    ],
    keep="first",
)
```

**New:**
```python
# Deduplication now uses session_id + trial_id for stronger integrity
unified_df = unified_df.drop_duplicates(
    subset=["patient_id", "session_id", "trial_id"],
    keep="first",
)
```

**Rationale:** With explicit IDs, we can rely on them for deduplication. This prevents:
- Multiple identical trials within a session being incorrectly deduplicated
- Complexity of 7-column composite keys

### 1.3 Validation Criteria

**Data Integrity Checks:**
1. **session_id uniqueness within patient+date:**
   ```python
   # Each (patient_id, date) should map to exactly one session_id
   check = df.groupby(["patient_id", "date"])["session_id"].nunique()
   assert (check == 1).all(), "Multiple session_ids for same patient+date"
   ```

2. **trial_id sequence validity:**
   ```python
   # trial_ids should be sequential starting from 1 within each session
   for (patient, session), group in df.groupby(["patient_id", "session_id"]):
       trial_ids = sorted(group["trial_id"].unique())
       expected = list(range(1, len(trial_ids) + 1))
       assert trial_ids == expected, f"Gap in trial_id sequence for {patient}/{session}"
   ```

3. **trial_id chronological ordering:**
   ```python
   # trial_ids should correspond to chronological order (start_time)
   for (patient, session), group in df.groupby(["patient_id", "session_id"]):
       sorted_by_time = group.sort_values("start_time")
       assert (sorted_by_time["trial_id"] == sorted(sorted_by_time["trial_id"])).all()
   ```

### 1.4 Potential Limitations

1. **Multi-day sessions:** If a session spans midnight, date might change. Need to handle:
   - Option A: Use session start date only
   - Option B: Create separate sessions (recommended for simplicity)

2. **Retroactive processing:** Existing data lacks session_id/trial_id. Need migration:
   - Regenerate unified parquet with new schema
   - Provide backward compatibility for old data (allow optional columns)

3. **Hash collisions:** SHA-256 truncated to 8 chars has ~1/4B collision rate
   - Acceptable for dataset size (<1000 patients)
   - Monitor for duplicates in validation

### 1.5 Testing Requirements

**Unit Tests (`tests/test_pipeline.py`):**
```python
def test_session_id_generation():
    """session_id should be deterministic and consistent."""
    df = pd.DataFrame({
        "patient_id": ["CON008", "CON008", "CON009"],
        "date": ["2025-02-14", "2025-02-14", "2025-02-14"],
        "trial_type": ["language", "oddball", "language"],
        # ... other columns
    })
    result = process_stimulus_df(df, "test.csv")
    
    # Same patient+date should have same session_id
    assert result.iloc[0]["session_id"] == result.iloc[1]["session_id"]
    assert result.iloc[0]["session_id"] != result.iloc[2]["session_id"]

def test_trial_id_sequence():
    """trial_id should be sequential within session."""
    df = pd.DataFrame({
        "patient_id": ["CON008"] * 3,
        "date": ["2025-02-14"] * 3,
        "start_time": [1.0, 2.0, 3.0],
        # ... other columns
    })
    result = process_stimulus_df(df, "test.csv")
    
    assert result["trial_id"].tolist() == [1, 2, 3]

def test_trial_id_respects_chronology():
    """trial_id should follow start_time order."""
    df = pd.DataFrame({
        "patient_id": ["CON008"] * 3,
        "date": ["2025-02-14"] * 3,
        "start_time": [3.0, 1.0, 2.0],  # Out of order
        # ... other columns
    })
    result = process_stimulus_df(df, "test.csv")
    
    # After sorting by start_time, trial_ids should be sequential
    sorted_result = result.sort_values("start_time")
    assert sorted_result["trial_id"].tolist() == [1, 2, 3]
```

---

## 2. Unified Data Loader (`src/data_loading/unified_data_loader.py`)

### 2.1 Current State
- `trials_df` loaded from parquet without session_id/trial_id
- Methods like `get_patient_sessions()` return date strings
- `get_trial_summary()` loses session dimension

### 2.2 Refactoring Tasks

#### Task 2.1: Update Schema Validation
**File:** `src/data_loading/unified_data_loader.py`  
**Lines:** 382-450

**Changes:**
```python
def validate_schema(self) -> Dict[str, bool]:
    required_columns = [
        "patient_id",
        "session_id",     # NEW
        "date",
        "trial_id",       # NEW
        "trial_type",
        "sentences",
        "start_time",
        "end_time",
        "duration",
        "source_file",
    ]
    # ... rest of validation
```

#### Task 2.2: Add Session-Aware Query Methods
**File:** `src/data_loading/unified_data_loader.py`  
**New methods to add:**

```python
def get_session_ids(self, patient_id: Optional[str] = None) -> List[str]:
    """
    Get list of session IDs, optionally filtered by patient.
    
    Args:
        patient_id: Optional patient filter
        
    Returns:
        List of session_id strings
    """
    if patient_id:
        filtered = self.trials_df[self.trials_df["patient_id"] == patient_id]
    else:
        filtered = self.trials_df
    
    return sorted(filtered["session_id"].unique().tolist())

def get_session_info(self, session_id: str) -> Dict[str, Any]:
    """
    Get metadata about a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict with patient_id, date, trial_count, trial_types
    """
    session_trials = self.trials_df[self.trials_df["session_id"] == session_id]
    
    if len(session_trials) == 0:
        raise UnifiedDataLoadingError(f"Session '{session_id}' not found")
    
    return {
        "session_id": session_id,
        "patient_id": session_trials.iloc[0]["patient_id"],
        "date": session_trials.iloc[0]["date"],
        "trial_count": len(session_trials),
        "trial_types": session_trials["trial_type"].value_counts().to_dict(),
        "trial_id_range": (session_trials["trial_id"].min(), session_trials["trial_id"].max()),
    }

def get_trial_by_id(self, patient_id: str, session_id: str, trial_id: int) -> pd.Series:
    """
    Get a specific trial by its full ID path.
    
    Args:
        patient_id: Patient identifier
        session_id: Session identifier
        trial_id: Trial ID within session
        
    Returns:
        Single trial as pd.Series
    """
    mask = (
        (self.trials_df["patient_id"] == patient_id) &
        (self.trials_df["session_id"] == session_id) &
        (self.trials_df["trial_id"] == trial_id)
    )
    
    trial = self.trials_df[mask]
    
    if len(trial) == 0:
        raise UnifiedDataLoadingError(
            f"Trial not found: {patient_id}/{session_id}/{trial_id}"
        )
    
    if len(trial) > 1:
        raise UnifiedDataLoadingError(
            f"Multiple trials found (data integrity issue): {patient_id}/{session_id}/{trial_id}"
        )
    
    return trial.iloc[0]
```

#### Task 2.3: Update get_trial_summary()
**File:** `src/data_loading/unified_data_loader.py`  
**Lines:** 122-139

**Current:**
```python
def get_trial_summary(self) -> pd.DataFrame:
    summary = self.trials_df.groupby(["patient_id", "trial_type"]).size().reset_index(name="count")
    return summary.sort_values(["patient_id", "trial_type"])
```

**New:**
```python
def get_trial_summary(self, group_by_session: bool = False) -> pd.DataFrame:
    """
    Get summary statistics of trials.
    
    Args:
        group_by_session: If True, includes session dimension in grouping
        
    Returns:
        DataFrame with trial counts
    """
    if group_by_session:
        summary = self.trials_df.groupby(
            ["patient_id", "session_id", "date", "trial_type"]
        ).size().reset_index(name="count")
        return summary.sort_values(["patient_id", "date", "trial_type"])
    else:
        # Original behavior: aggregate across all sessions
        summary = self.trials_df.groupby(["patient_id", "trial_type"]).size().reset_index(name="count")
        return summary.sort_values(["patient_id", "trial_type"])
```

### 2.3 Validation Criteria

**ID Integrity Checks:**
```python
def validate_id_integrity(self) -> Dict[str, bool]:
    """Validate session_id and trial_id integrity."""
    results = {
        "session_ids_valid": True,
        "trial_ids_valid": True,
        "no_orphaned_trials": True,
    }
    
    # Check 1: session_id consistency within patient+date
    for (patient, date), group in self.trials_df.groupby(["patient_id", "date"]):
        session_ids = group["session_id"].unique()
        if len(session_ids) != 1:
            logger.warning(f"{patient} on {date} has {len(session_ids)} session_ids")
            results["session_ids_valid"] = False
    
    # Check 2: trial_id sequence
    for (patient, session), group in self.trials_df.groupby(["patient_id", "session_id"]):
        trial_ids = sorted(group["trial_id"].unique())
        expected = list(range(1, len(trial_ids) + 1))
        if trial_ids != expected:
            logger.warning(f"{patient}/{session} has gaps in trial_id sequence")
            results["trial_ids_valid"] = False
    
    # Check 3: No trials without session_id
    null_sessions = self.trials_df["session_id"].isna().sum()
    null_trials = self.trials_df["trial_id"].isna().sum()
    if null_sessions > 0 or null_trials > 0:
        results["no_orphaned_trials"] = False
    
    return results
```

### 2.4 Potential Limitations

1. **Backward compatibility:** Old parquet files lack new columns
   - Solution: Add schema migration utility
   - Allow graceful degradation (optional columns with warnings)

2. **Performance:** Additional columns increase memory footprint
   - Impact: ~16 bytes per row (8 for session_id, 8 for trial_id)
   - For 10K trials: ~160KB increase (negligible)

### 2.5 Testing Requirements

```python
def test_get_session_ids():
    """Should return unique session IDs."""
    session_ids = loader.get_session_ids()
    assert len(session_ids) == len(set(session_ids))  # All unique

def test_get_session_info():
    """Should return complete session metadata."""
    info = loader.get_session_info("a3f2c1d8")
    assert "patient_id" in info
    assert "date" in info
    assert "trial_count" in info
    assert info["trial_count"] > 0

def test_get_trial_by_id():
    """Should retrieve exact trial."""
    trial = loader.get_trial_by_id("CON008", "a3f2c1d8", 5)
    assert trial["patient_id"] == "CON008"
    assert trial["session_id"] == "a3f2c1d8"
    assert trial["trial_id"] == 5

def test_get_trial_by_id_not_found():
    """Should raise error for non-existent trial."""
    with pytest.raises(UnifiedDataLoadingError):
        loader.get_trial_by_id("CON008", "invalid", 999)
```

---

## 3. Timestamp Alignment (`src/data_processing/timestamp_aligner.py`)

### 3.1 Current State
- Iterates trials using `iterrows()` without trial identifiers
- Event dictionaries lack parent trial reference
- Output saved without session_id/trial_id

### 3.2 Refactoring Tasks

#### Task 3.1: Propagate IDs in _align_session()
**File:** `src/data_processing/timestamp_aligner.py`  
**Lines:** 135-167

**Current:**
```python
def _align_session(self, raw: mne.io.Raw, trials_df: pd.DataFrame) -> pd.DataFrame:
    # ... setup code ...
    for _, trial in trials_df.iterrows():
        trial_type = trial["trial_type"].lower()
        # Process trial...
```

**New:**
```python
def _align_session(self, raw: mne.io.Raw, trials_df: pd.DataFrame) -> pd.DataFrame:
    # ... setup code ...
    for idx, trial in trials_df.iterrows():
        # Extract IDs from trial
        session_id = trial.get("session_id")
        trial_id = trial.get("trial_id")
        
        if session_id is None or trial_id is None:
            # Backward compatibility: generate temporary IDs
            logger.warning(f"Trial missing IDs at index {idx}. Generating temporary IDs.")
            session_id = generate_session_id(trial["patient_id"], trial["date"])
            trial_id = idx + 1
        
        trial_type = trial["trial_type"].lower()
        
        # Pass IDs to alignment methods
        if trial_type == "language":
            result = self._align_sentence_trials(trial, session_id, trial_id)
        elif trial_type in ["left_command", "right_command"]:
            result = self._align_commands(trial, session_id, trial_id)
        # ... etc
```

#### Task 3.2: Update _build_trial_result()
**File:** `src/data_processing/timestamp_aligner.py`  
**Lines:** 520-542

**Current:**
```python
def _build_trial_result(
    self,
    trial: pd.Series,
    enriched_events: list,
    method: str,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "patient_id": self.patient_id,
        "date": trial["date"],
        "trial_type": trial["trial_type"],
        # ... other fields
    }])
```

**New:**
```python
def _build_trial_result(
    self,
    trial: pd.Series,
    enriched_events: list,
    method: str,
    session_id: str,
    trial_id: int,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "patient_id": self.patient_id,
        "session_id": session_id,     # NEW
        "date": trial["date"],
        "trial_id": trial_id,          # NEW
        "trial_type": trial["trial_type"],
        "start_time": trial["start_time"],
        "end_time": trial["end_time"],
        "duration": trial["duration"],
        "sentences": enriched_events,
        "dc_channel": self.dc_channel,
        "alignment_method": method,
        "source_file": trial.get("source_file", None),
    }])
```

#### Task 3.3: Enrich Event Dictionaries
**File:** `src/data_processing/timestamp_aligner.py`  
**Lines:** Various alignment methods

**Add to each event dict:**
```python
def _align_sentence_trials(
    self, trial: pd.Series, session_id: str, trial_id: int
) -> pd.DataFrame:
    # ... existing logic ...
    
    for event in events:
        enriched_event = {
            "event": event["id"],
            "event_start": aligned_start,
            "event_end": aligned_end,
            "event_duration": duration,
            "correlation_score": score,
            "session_id": session_id,     # NEW
            "trial_id": trial_id,         # NEW
        }
        enriched_events.append(enriched_event)
```

### 3.3 Validation Criteria

**Alignment Integrity Checks:**
```python
def validate_aligned_events(aligned_df: pd.DataFrame) -> Dict[str, bool]:
    """Validate aligned events have proper ID tracking."""
    results = {
        "all_trials_have_ids": True,
        "all_events_have_ids": True,
        "id_consistency": True,
    }
    
    # Check 1: All trials have session_id and trial_id
    if aligned_df["session_id"].isna().any() or aligned_df["trial_id"].isna().any():
        results["all_trials_have_ids"] = False
    
    # Check 2: All events in sentences have session_id and trial_id
    for idx, row in aligned_df.iterrows():
        sentences = row["sentences"]
        if isinstance(sentences, list):
            for event in sentences:
                if isinstance(event, dict):
                    if "session_id" not in event or "trial_id" not in event:
                        results["all_events_have_ids"] = False
                        break
    
    # Check 3: Event IDs match trial IDs
    for idx, row in aligned_df.iterrows():
        trial_session_id = row["session_id"]
        trial_trial_id = row["trial_id"]
        
        sentences = row["sentences"]
        if isinstance(sentences, list):
            for event in sentences:
                if isinstance(event, dict):
                    if event.get("session_id") != trial_session_id:
                        results["id_consistency"] = False
                    if event.get("trial_id") != trial_trial_id:
                        results["id_consistency"] = False
    
    return results
```

### 3.4 Potential Limitations

1. **Event dict size:** Adding IDs increases memory per event
   - Impact: ~40 bytes per event (2 string fields)
   - For 100K events: ~4MB increase (acceptable)

2. **Legacy data:** Existing aligned parquet files lack IDs
   - Solution: Reprocess from unified data with new schema
   - Provide migration script

3. **Partial alignment failures:** If alignment fails mid-trial
   - Ensure IDs still written to partial results
   - Add "alignment_status" field for debugging

### 3.5 Testing Requirements

```python
def test_align_session_propagates_ids():
    """Alignment should preserve session_id and trial_id."""
    trials_df = pd.DataFrame([{
        "patient_id": "CON008",
        "session_id": "a3f2c1d8",
        "date": "2025-02-14",
        "trial_id": 1,
        "trial_type": "language",
        # ... other fields
    }])
    
    aligner = TimestampAligner("CON008", trials_df, ...)
    result = aligner._align_session(mock_raw, trials_df)
    
    assert result.iloc[0]["session_id"] == "a3f2c1d8"
    assert result.iloc[0]["trial_id"] == 1

def test_events_include_parent_ids():
    """Events should reference parent trial via IDs."""
    # ... setup ...
    result = aligner._align_sentence_trials(trial, "a3f2c1d8", 1)
    
    sentences = result.iloc[0]["sentences"]
    assert isinstance(sentences, list)
    assert len(sentences) > 0
    
    for event in sentences:
        assert "session_id" in event
        assert "trial_id" in event
        assert event["session_id"] == "a3f2c1d8"
        assert event["trial_id"] == 1
```

---

## 4. ERP Pipeline (`src/data_processing/erp_pipeline.py`)

### 4.1 Current State
- Loops sessions by date without session_id
- `_extract_rare_events()` uses fragile row index as `trial_idx`
- Features saved with only `(patient_id, date)` keys
- Master table deduplicates on `["patient_id", "date"]`

### 4.2 Refactoring Tasks

#### Task 4.1: Update _process_session() Signature
**File:** `src/data_processing/erp_pipeline.py`  
**Lines:** 88-156

**Current:**
```python
def process_patient(self, patient_id: str, date: Optional[str] = None, ...):
    # ... code ...
    for session_date in sessions:
        session_trials = aligned_trials[aligned_trials["date"] == session_date]
        session_result = self._process_session(
            patient_id,
            session_date,
            session_trials,
            ...
        )
```

**New:**
```python
def process_patient(self, patient_id: str, date: Optional[str] = None, ...):
    # ... code ...
    
    # Group by session_id instead of date
    session_groups = aligned_trials.groupby("session_id")
    
    for session_id, session_trials in session_groups:
        session_date = session_trials.iloc[0]["date"]
        
        session_result = self._process_session(
            patient_id,
            session_id,      # NEW: pass session_id
            session_date,
            session_trials,
            ...
        )
```

**Update _process_session() signature:**
```python
def _process_session(
    self,
    patient_id: str,
    session_id: str,     # NEW
    date: str,
    aligned_trials: pd.DataFrame,
    custom_electrodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger.info(f"Processing session: {patient_id} - {session_id} ({date})")
    # ... rest of method
```

#### Important Design Decision: Feature Granularity

**Current ERP Pipeline Design:**
The existing `_process_session()` aggregates ALL oddball trials in a session:
1. Extracts rare events from all oddball trials
2. Creates epochs from all rare events
3. Averages epochs into ONE ERP
4. Generates ONE feature row per session

**This means:**
- Multiple oddball trials in a session → ONE aggregated feature row
- `trial_type` = "oddball" identifies what was analyzed
- `rare_event_trial_ids` tracks which specific trials contributed (e.g., "1,3,5")

**Deduplication Key Considerations:**

**Option A (Recommended - Session-level aggregation):**
```python
Key: ["patient_id", "session_id", "trial_type", "processing_timestamp"]
```
- Supports current aggregated design
- `trial_type` distinguishes oddball vs. language vs. command features
- `processing_timestamp` allows tracking analysis reruns
- `rare_event_trial_ids` provides trial traceability

**Option B (Future - Per-trial features):**
```python
Key: ["patient_id", "session_id", "trial_id"]
```
- Would require redesigning `_process_session()` to loop over individual trials
- Each oddball trial gets its own feature row
- More flexible but requires more storage
- Example: Session with 5 oddball trials → 5 feature rows instead of 1

**Recommendation:**
- Use Option A for initial implementation (matches current design)
- Add `rare_event_trial_ids` to maintain traceability
- Option B can be implemented later if per-trial analysis is needed

The user's concern about multiple trials of the same type is valid. With Option A, we handle this by:
1. Aggregating them into one feature row (current behavior)
2. Tracking contributing trials via `rare_event_trial_ids = "1,3,5,7"`
3. Using `processing_timestamp` to distinguish analysis reruns

#### Task 4.2: Update _extract_rare_events()
**File:** `src/data_processing/erp_pipeline.py`  
**Lines:** 312-365

**Current:**
```python
def _extract_rare_events(self, trials_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rare_events = []
    for idx, trial in trials_df.iterrows():
        # ... logic ...
        rare_events.append({
            "timestamp_unix": event["event_start"],
            "trial_idx": idx,  # FRAGILE: row index
            "event_type": "rare",
        })
    return rare_events
```

**New:**
```python
def _extract_rare_events(self, trials_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extract rare events from oddball trials with full ID tracking.
    
    Returns:
        List of dicts with timestamp_unix, session_id, trial_id, event_type
    """
    rare_events = []
    
    for idx, trial in trials_df.iterrows():
        session_id = trial.get("session_id")
        trial_id = trial.get("trial_id")
        
        if session_id is None or trial_id is None:
            logger.warning(f"Trial at index {idx} missing IDs. Skipping.")
            continue
        
        sentences = trial["sentences"]
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        
        for event_idx, event in enumerate(sentences):
            if isinstance(event, dict) and event.get("event") == "rare":
                rare_events.append({
                    "timestamp_unix": event["event_start"],
                    "session_id": session_id,      # NEW
                    "trial_id": trial_id,          # NEW
                    "event_idx": event_idx,        # NEW: position within trial
                    "event_type": "rare",
                })
    
    return rare_events
```

#### Task 4.3: Update Feature Schema
**File:** `src/data_processing/erp_pipeline.py`  
**Lines:** 570-645

**Current:**
```python
features = {
    "patient_id": patient_id,
    "date": date,
    "n_epochs": n_epochs,
    # ... P300 metrics
}
```

**New:**
```python
# First, extract unique trial_ids from rare_events
unique_trial_ids = sorted(set(event["trial_id"] for event in rare_events))

features = {
    "patient_id": patient_id,
    "session_id": session_id,     # NEW
    "date": date,
    "trial_type": "oddball",      # NEW: explicit trial type
    "n_epochs": n_epochs,
    "processing_timestamp": datetime.now().isoformat(),
    # ... P300 metrics
    
    # NEW: Trial-level tracking for session-level aggregation
    "n_total_oddball_trials": len(aligned_trials),
    "n_trials_with_rare_events": len(unique_trial_ids),
    "n_rare_events_extracted": len(rare_events),
    "rare_event_trial_ids": ",".join(map(str, unique_trial_ids)),  # e.g., "1,3,5,7"
}
```

**Explanation:**
- `rare_event_trial_ids` contains comma-separated trial_ids that contributed rare events
- Example: If session has oddball trials [1, 2, 3, 4, 5] but only trials [1, 3, 5] had rare events, then `rare_event_trial_ids = "1,3,5"`
- This provides full traceability: which specific trials were used to create the aggregated ERP


#### Task 4.4: Update Master Feature Table Deduplication
**File:** `src/data_processing/erp_pipeline.py`  
**Lines:** 894-915

**Current:**
```python
def _update_master_feature_table(self, incoming_features: pd.DataFrame) -> pd.DataFrame:
    # ... load existing ...
    combined = pd.concat([master_df, incoming_features], ignore_index=True)
    
    # Keep latest row per patient/date
    combined = combined.drop_duplicates(subset=["patient_id", "date"], keep="last")
    
    combined.to_parquet(master_path)
    return combined
```

**New:**
```python
def _update_master_feature_table(self, incoming_features: pd.DataFrame) -> pd.DataFrame:
    """
    Upsert session features into master table.
    
    Deduplication key options:
    
    **Option A (Recommended): Session-level aggregation**
    Key: (patient_id, session_id, trial_type, processing_timestamp)
    - One feature row per session per trial type
    - Aggregates all trials of same type (e.g., all oddball trials → one P300 feature)
    - Uses processing_timestamp to track analysis reruns
    - Suitable for current ERP pipeline design
    
    **Option B: Trial-level granularity**
    Key: (patient_id, session_id, trial_id)
    - One feature row per individual trial
    - Requires redesigning _process_session to loop over trials
    - More flexible for per-trial analysis
    - Higher storage requirements
    
    Current implementation uses Option A with trial tracking via rare_event_trial_ids.
    """
    master_path = self.output_dir / "features" / "p300_features.parquet"
    
    if master_path.exists():
        master_df = pd.read_parquet(master_path)
        combined = pd.concat([master_df, incoming_features], ignore_index=True)
    else:
        combined = incoming_features.copy()
    
    # Option A: Deduplicate by session + trial_type + processing_timestamp
    # This allows multiple oddball trials in a session to be aggregated
    # Use processing_timestamp to track different analysis runs
    combined = combined.drop_duplicates(
        subset=["patient_id", "session_id", "trial_type", "processing_timestamp"], 
        keep="last"
    )
    
    # Alternative (Option B): For per-trial features, use trial_id
    # combined = combined.drop_duplicates(
    #     subset=["patient_id", "session_id", "trial_id"], 
    #     keep="last"
    # )
    
    combined.to_parquet(master_path)
    logger.info(f"Updated master feature table: {master_path} ({len(combined)} rows)")
    return combined
```

#### Task 4.5: Update File Naming Convention
**File:** `src/data_processing/erp_pipeline.py`  
**Lines:** 859-892

**Current:**
```python
epochs_file = self.output_dir / "epochs" / f"{patient_id}_{date}_oddball-epo.fif"
erp_file = self.output_dir / "erps" / f"{patient_id}_{date}_oddball-ave.fif"
features_file = self.output_dir / "features" / f"{patient_id}_{date}_p300_features.parquet"
plot_file = self.output_dir / "plots" / "erp" / f"{patient_id}_{date}_oddball_erp.png"
```

**New (Option A - Include session_id):**
```python
# More explicit, handles multiple sessions per day
epochs_file = self.output_dir / "epochs" / f"{patient_id}_{session_id}_{date}_oddball-epo.fif"
erp_file = self.output_dir / "erps" / f"{patient_id}_{session_id}_{date}_oddball-ave.fif"
features_file = self.output_dir / "features" / f"{patient_id}_{session_id}_{date}_p300_features.parquet"
plot_file = self.output_dir / "plots" / "erp" / f"{patient_id}_{session_id}_{date}_oddball_erp.png"
```

**New (Option B - Keep date, rely on metadata):**
```python
# Simpler filenames, session_id in file content
# Recommended: stick with date for human readability, metadata has full IDs
epochs_file = self.output_dir / "epochs" / f"{patient_id}_{date}_oddball-epo.fif"
# Add session_id to file metadata/description field
epochs.info["description"] = f"session_id={session_id}"
```

**Recommendation:** Use Option B to maintain backward compatibility with existing scripts.

### 4.3 Validation Criteria

**Feature Integrity Checks:**
```python
def validate_feature_table(features_df: pd.DataFrame, granularity: str = "session") -> Dict[str, bool]:
    """
    Validate P300 feature table integrity.
    
    Args:
        features_df: Feature table to validate
        granularity: Either "session" (aggregated) or "trial" (per-trial features)
    """
    results = {
        "no_duplicate_entries": True,
        "all_have_session_id": True,
        "trial_id_tracking": True,
        "timestamp_monotonic": True,
    }
    
    # Check 1: No duplicate entries based on chosen granularity
    if granularity == "session":
        # For session-level: patient + session + trial_type + processing_timestamp
        key_cols = ["patient_id", "session_id", "trial_type", "processing_timestamp"]
        if features_df.duplicated(subset=key_cols).any():
            duplicates = features_df[features_df.duplicated(subset=key_cols, keep=False)]
            logger.error(f"Found {len(duplicates)} duplicate session-level features")
            results["no_duplicate_entries"] = False
    elif granularity == "trial":
        # For trial-level: patient + session + trial_id
        key_cols = ["patient_id", "session_id", "trial_id"]
        if features_df.duplicated(subset=key_cols).any():
            duplicates = features_df[features_df.duplicated(subset=key_cols, keep=False)]
            logger.error(f"Found {len(duplicates)} duplicate trial-level features")
            results["no_duplicate_entries"] = False
    
    # Check 2: All rows have session_id
    if features_df["session_id"].isna().any():
        results["all_have_session_id"] = False
    
    # Check 3: rare_event_trial_ids field populated (for session-level aggregation)
    if granularity == "session" and "rare_event_trial_ids" in features_df.columns:
        null_tracking = features_df["rare_event_trial_ids"].isna().sum()
        if null_tracking > 0:
            logger.warning(f"{null_tracking} features lack trial_id tracking")
            results["trial_id_tracking"] = False
    
    # Check 4: processing_timestamp increases with time (no old data overwriting new)
    if "processing_timestamp" in features_df.columns:
        for (patient, session), group in features_df.groupby(["patient_id", "session_id"]):
            timestamps = pd.to_datetime(group["processing_timestamp"])
            if not timestamps.is_monotonic_increasing:
                results["timestamp_monotonic"] = False
                logger.warning(f"{patient}/{session} has non-monotonic processing timestamps")
    
    return results
```

### 4.4 Potential Limitations

1. **File naming conflicts:** If two sessions on same date
   - Option A filenames prevent conflicts
   - Option B requires unique date strings (unlikely in practice)
   - **Decision:** Use Option B with validation check

2. **MNE file format:** `.fif` files have limited metadata
   - Can't store arbitrary session_id in standard fields
   - Workaround: Use `epochs.info["description"]` field
   - Include session_id in accompanying parquet metadata

3. **Rare event extraction performance:** Nested loops over trials and events
   - Current complexity: O(trials × events_per_trial)
   - No change in complexity, but IDs add minor overhead
   - For 100 trials × 50 events: ~5K iterations (acceptable)

### 4.5 Testing Requirements

```python
def test_process_session_with_session_id():
    """Session processing should use session_id."""
    pipeline = OddballERPPipeline()
    
    aligned_trials = pd.DataFrame([{
        "patient_id": "CON008",
        "session_id": "a3f2c1d8",
        "date": "2025-02-14",
        "trial_id": 1,
        "trial_type": "oddball",
        # ... other fields
    }])
    
    result = pipeline._process_session(
        "CON008", "a3f2c1d8", "2025-02-14", aligned_trials
    )
    
    assert result["session_id"] == "a3f2c1d8"
    assert "features" in result

def test_extract_rare_events_includes_trial_ids():
    """Rare events should reference parent trial."""
    trials_df = pd.DataFrame([{
        "patient_id": "CON008",
        "session_id": "a3f2c1d8",
        "trial_id": 5,
        "trial_type": "oddball",
        "sentences": [{"event": "rare", "event_start": 123.4}],
    }])
    
    pipeline = OddballERPPipeline()
    rare_events = pipeline._extract_rare_events(trials_df)
    
    assert len(rare_events) == 1
    assert rare_events[0]["session_id"] == "a3f2c1d8"
    assert rare_events[0]["trial_id"] == 5

def test_feature_table_deduplication():
    """Features should deduplicate on patient+session+trial_type."""
    features1 = pd.DataFrame([{
        "patient_id": "CON008",
        "session_id": "a3f2c1d8",
        "trial_type": "oddball",
        "p300_amplitude_uV": 5.0,
    }])
    
    features2 = pd.DataFrame([{
        "patient_id": "CON008",
        "session_id": "a3f2c1d8",
        "trial_type": "oddball",
        "p300_amplitude_uV": 6.0,  # Updated value
    }])
    
    pipeline = OddballERPPipeline()
    pipeline._update_master_feature_table(features1)
    result = pipeline._update_master_feature_table(features2)
    
    # Should have only 1 row (kept latest)
    assert len(result) == 1
    assert result.iloc[0]["p300_amplitude_uV"] == 6.0
```

---

## 5. Verification Scripts (`scripts/verify_erp_results.py`)

### 5.1 Current State
- Loads aligned events by patient (all sessions combined)
- Compares epoch counts with rare events by date
- No trial-level validation

### 5.2 Refactoring Tasks

#### Task 5.1: Add Session-Level Verification
**File:** `scripts/verify_erp_results.py`  
**New function to add:**

```python
def verify_session(patient_id: str, session_id: str):
    """
    Verify ERP results for a specific session.
    
    Args:
        patient_id: Patient identifier
        session_id: Session identifier
    """
    print(f"\n{'=' * 70}")
    print(f"  ERP Verification: {patient_id} - Session {session_id}")
    print(f"{'=' * 70}")
    
    # Load aligned events
    aligned_path = config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet"
    if not aligned_path.exists():
        print(f"  ✗ Aligned events not found: {aligned_path}")
        return False
    
    df = pd.read_parquet(aligned_path)
    
    # Filter for this session
    session_df = df[df["session_id"] == session_id]
    if len(session_df) == 0:
        print(f"  ✗ No data found for session {session_id}")
        return False
    
    date = session_df.iloc[0]["date"]
    print(f"  Session Date: {date}")
    
    # Filter oddball trials
    oddball = session_df[session_df["trial_type"] == "oddball"]
    print(f"  ✓ Oddball trials in session: {len(oddball)}")
    
    # Count rare events with trial_id tracking
    rare_event_trials = set()
    total_rare = 0
    
    for _, trial in oddball.iterrows():
        trial_id = trial.get("trial_id")
        sentences = trial["sentences"]
        
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        
        for event in sentences:
            if isinstance(event, dict) and event.get("event") == "rare":
                total_rare += 1
                if trial_id is not None:
                    rare_event_trials.add(trial_id)
    
    print(f"  ✓ Total rare events: {total_rare}")
    print(f"  ✓ Trials with rare events: {len(rare_event_trials)}")
    
    # Load features for this session
    features_file = config.PROCESSED_DATA_DIR / "features" / "p300_features.parquet"
    if features_file.exists():
        features_df = pd.read_parquet(features_file)
        session_features = features_df[
            (features_df["patient_id"] == patient_id) &
            (features_df["session_id"] == session_id)
        ]
        
        if len(session_features) > 0:
            feature_row = session_features.iloc[0]
            print(f"\n[Features]")
            print(f"  N epochs: {feature_row.get('n_epochs', 'N/A')}")
            print(f"  Trials contributing: {feature_row.get('rare_event_trial_ids', 'N/A')}")
            
            # Validate trial_id consistency
            if 'rare_event_trial_ids' in feature_row:
                feature_trial_ids = set(map(int, str(feature_row['rare_event_trial_ids']).split(',')))
                if feature_trial_ids == rare_event_trials:
                    print(f"  ✓ Trial ID tracking consistent")
                else:
                    print(f"  ⚠ Trial ID mismatch:")
                    print(f"    Events from: {rare_event_trials}")
                    print(f"    Features claim: {feature_trial_ids}")
    
    return True
```

#### Task 5.2: Update Main Verification Function
**File:** `scripts/verify_erp_results.py`  
**Lines:** 25-268

**Add session_id parameter:**
```python
def verify_patient(patient_id: str, date: Optional[str] = None, session_id: Optional[str] = None):
    """
    Verify ERP results for a patient.
    
    Args:
        patient_id: Patient identifier
        date: Optional session date filter
        session_id: Optional session ID filter (takes precedence over date)
    """
    if session_id:
        # Verify specific session by ID
        return verify_session(patient_id, session_id)
    
    # ... existing date-based logic ...
```

### 5.3 Validation Criteria

**Cross-Pipeline Validation:**
```python
def validate_pipeline_consistency(patient_id: str, session_id: str) -> Dict[str, bool]:
    """
    Validate consistency across pipeline stages.
    
    Checks:
    1. Unified data has session_id
    2. Aligned events preserve session_id
    3. ERP features reference correct session_id
    4. Trial IDs match across stages
    """
    results = {
        "unified_has_session": False,
        "aligned_has_session": False,
        "features_have_session": False,
        "trial_ids_consistent": False,
    }
    
    # Check unified data
    unified_path = config.UNIFIED_PARQUET_PATH
    if unified_path.exists():
        unified_df = pd.read_parquet(unified_path)
        session_rows = unified_df[
            (unified_df["patient_id"] == patient_id) &
            (unified_df["session_id"] == session_id)
        ]
        if len(session_rows) > 0:
            results["unified_has_session"] = True
            unified_trial_ids = set(session_rows["trial_id"].unique())
    
    # Check aligned events
    aligned_path = config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet"
    if aligned_path.exists():
        aligned_df = pd.read_parquet(aligned_path)
        session_rows = aligned_df[aligned_df["session_id"] == session_id]
        if len(session_rows) > 0:
            results["aligned_has_session"] = True
            aligned_trial_ids = set(session_rows["trial_id"].unique())
    
    # Check features
    features_path = config.PROCESSED_DATA_DIR / "features" / "p300_features.parquet"
    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        session_features = features_df[
            (features_df["patient_id"] == patient_id) &
            (features_df["session_id"] == session_id)
        ]
        if len(session_features) > 0:
            results["features_have_session"] = True
    
    # Compare trial_ids across stages
    if results["unified_has_session"] and results["aligned_has_session"]:
        if unified_trial_ids == aligned_trial_ids:
            results["trial_ids_consistent"] = True
    
    return results
```

### 5.4 Testing Requirements

```python
def test_verify_session():
    """Session verification should work with session_id."""
    # Setup test data with session_id
    # ...
    
    result = verify_session("CON008", "a3f2c1d8")
    assert result is True

def test_validate_pipeline_consistency():
    """Should detect ID consistency across pipeline stages."""
    results = validate_pipeline_consistency("CON008", "a3f2c1d8")
    
    assert results["unified_has_session"] is True
    assert results["aligned_has_session"] is True
    assert results["features_have_session"] is True
    assert results["trial_ids_consistent"] is True
```

---

## 6. Feature Storage & Master Table

### 6.1 Current State
- Master table at `processed_data/features/p300_features.parquet`
- Deduplicates on `["patient_id", "date"]`
- Loses information when multiple trial types per session

### 6.2 Refactoring Tasks

#### Task 6.1: Add Session/Trial Columns to Schema
**Expected columns in master table:**
```python
FEATURE_TABLE_SCHEMA = [
    # Identity
    "patient_id",
    "session_id",      # NEW
    "date",
    "trial_type",      # NEW: explicit (e.g., "oddball")
    
    # Processing metadata
    "processing_timestamp",
    "n_epochs",
    "n_total_oddball_trials",      # NEW
    "n_rare_events_extracted",     # NEW
    "rare_event_trial_ids",        # NEW: comma-separated list
    
    # P300 metrics (per electrode)
    "p300_amplitude_Pz_uV",
    "p300_latency_Pz_ms",
    "p300_amplitude_Cz_uV",
    "p300_latency_Cz_ms",
    "p300_amplitude_Fz_uV",
    "p300_latency_Fz_ms",
    
    # Composite metrics
    "p300_composite_amplitude_uV",
    "p300_composite_latency_ms",
    "p300_best_electrode",
    "p300_n_valid_electrodes",
    "p300_n_flagged_electrodes",
    
    # QC
    "baseline_std_uV",
    "qc_notes",
    
    # Timezone diagnostics
    "timezone_offset_hours",
    "timezone_confidence",
    # ... etc
]
```

#### Task 6.2: Migration Script for Existing Data
**New file:** `scripts/migrate_features_to_session_ids.py`

```python
#!/usr/bin/env python3
"""
Migrate existing feature table to include session_id and trial_id.

This script:
1. Loads existing p300_features.parquet
2. Generates session_id from (patient_id, date)
3. Adds trial_type column (defaults to "oddball")
4. Saves migrated version
"""

import pandas as pd
from pathlib import Path
import hashlib

def generate_session_id(patient_id: str, date: str) -> str:
    key = f"{patient_id}_{date}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]

def migrate_features():
    features_path = Path("processed_data/features/p300_features.parquet")
    backup_path = features_path.with_suffix(".parquet.backup")
    
    print(f"Loading {features_path}")
    df = pd.read_parquet(features_path)
    
    # Backup original
    df.to_parquet(backup_path)
    print(f"Backup saved to {backup_path}")
    
    # Generate session_id
    df["session_id"] = df.apply(
        lambda row: generate_session_id(row["patient_id"], row["date"]),
        axis=1
    )
    
    # Add trial_type if missing (default to oddball)
    if "trial_type" not in df.columns:
        df["trial_type"] = "oddball"
    
    # Add new tracking columns with defaults
    if "n_total_oddball_trials" not in df.columns:
        df["n_total_oddball_trials"] = pd.NA
    if "n_rare_events_extracted" not in df.columns:
        df["n_rare_events_extracted"] = pd.NA
    if "rare_event_trial_ids" not in df.columns:
        df["rare_event_trial_ids"] = ""
    
    # Save migrated version
    df.to_parquet(features_path)
    print(f"Migrated {len(df)} rows")
    print(f"New columns: session_id, trial_type, trial tracking fields")
    
    return df

if __name__ == "__main__":
    migrate_features()
```

### 6.3 Validation Criteria

**Schema Validation:**
```python
def validate_feature_table_schema(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate feature table has required columns."""
    required_cols = [
        "patient_id", "session_id", "date", "trial_type",
        "n_epochs", "p300_composite_amplitude_uV",
    ]
    
    results = {
        "has_required_columns": all(col in df.columns for col in required_cols),
        "session_ids_valid": True,
        "trial_type_populated": True,
    }
    
    # Check session_id format (8-char hex)
    if "session_id" in df.columns:
        invalid_session_ids = df[~df["session_id"].str.match(r"^[a-f0-9]{8}$", na=False)]
        if len(invalid_session_ids) > 0:
            results["session_ids_valid"] = False
    
    # Check trial_type populated
    if "trial_type" in df.columns:
        null_trial_types = df["trial_type"].isna().sum()
        if null_trial_types > 0:
            results["trial_type_populated"] = False
    
    return results
```

---

## 7. Implementation Timeline

### Phase 1: Core Schema Changes (Week 1)
**Goal:** Update schemas and ID generation

- **Day 1-2:** Implement `generate_session_id()` in `pipeline.py`
- **Day 3-4:** Add session_id and trial_id columns to `process_stimulus_df()`
- **Day 5:** Write unit tests for ID generation
- **Deliverable:** Unified parquet with new schema

### Phase 2: Propagate IDs Through Alignment (Week 2)
**Goal:** Update timestamp aligner to preserve IDs

- **Day 1-2:** Modify `_align_session()` to accept and propagate IDs
- **Day 3-4:** Update `_build_trial_result()` and event dictionaries
- **Day 5:** Integration tests for alignment pipeline
- **Deliverable:** Aligned events with session_id/trial_id

### Phase 3: ERP Pipeline Updates (Week 3)
**Goal:** Update feature extraction and storage

- **Day 1-2:** Modify `_process_session()` signature and `_extract_rare_events()`
- **Day 3:** Update feature schema and master table deduplication
- **Day 4:** Update file naming and metadata
- **Day 5:** End-to-end testing
- **Deliverable:** Features with full ID tracking

### Phase 4: Verification & Migration (Week 4)
**Goal:** Update verification tools and migrate existing data

- **Day 1-2:** Update `verify_erp_results.py` with session-level checks
- **Day 3:** Write and test migration script
- **Day 4:** Migrate existing data
- **Day 5:** Final validation and documentation
- **Deliverable:** Complete refactored pipeline with migrated data

---

## 8. Potential Limitations & Mitigation Strategies

### 8.1 Data Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Multi-day sessions** | Session spans midnight, date changes | Use session start date only; document in metadata |
| **Hash collisions** | Two sessions map to same session_id | Monitor during validation; SHA-256 truncated to 8 chars has ~1/4B collision rate (acceptable) |
| **Legacy data** | Existing parquet files lack IDs | Provide migration script; allow backward compatibility mode |
| **Partial failures** | Alignment fails mid-session | Ensure IDs still written to partial results; add "status" field |

### 8.2 Performance Limitations

| Concern | Impact | Mitigation |
|---------|--------|------------|
| **Memory overhead** | session_id (8 bytes) + trial_id (8 bytes) per row | For 10K trials: ~160KB increase (negligible) |
| **Processing time** | ID generation adds overhead | Vectorized operations using pandas; ~1ms per 1000 rows |
| **Event dict size** | IDs increase memory per event | ~40 bytes per event; for 100K events: ~4MB (acceptable) |

### 8.3 Implementation Limitations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking changes** | Old code expects (patient_id, date) keys | Deprecation period; backward compatibility layer |
| **Test coverage gaps** | New functionality may have bugs | Comprehensive unit and integration tests |
| **Documentation lag** | Users may not know about IDs | Update README, add migration guide |

---

## 9. Validation Criteria Summary

### 9.1 Unit Test Coverage

**Minimum Requirements:**
- ID generation: 95% coverage
- Schema validation: 100% coverage
- Deduplication logic: 100% coverage
- Event enrichment: 90% coverage

### 9.2 Integration Test Coverage

**Required Tests:**
1. End-to-end pipeline with IDs: unified → aligned → features
2. Multi-session patient processing
3. Session with multiple trial types
4. Migration script on real data subset

### 9.3 Data Quality Checks

**Pre-deployment Validation:**
```python
def comprehensive_validation(unified_df, aligned_df, features_df):
    checks = {
        # Schema checks
        "unified_has_session_id": "session_id" in unified_df.columns,
        "aligned_has_session_id": "session_id" in aligned_df.columns,
        "features_have_session_id": "session_id" in features_df.columns,
        
        # ID integrity
        "session_ids_consistent": validate_session_id_consistency(unified_df, aligned_df),
        "trial_ids_sequential": validate_trial_id_sequence(unified_df),
        "no_id_collisions": validate_no_collisions(unified_df),
        
        # Cross-pipeline
        "trial_count_match": validate_trial_count_match(unified_df, aligned_df),
        "event_trial_link": validate_event_trial_links(aligned_df),
        "feature_traceability": validate_feature_traceability(features_df, aligned_df),
    }
    
    return all(checks.values()), checks
```

---

## 10. Rollback Plan

### 10.1 Rollback Triggers
- Data integrity checks fail (>5% inconsistency)
- Performance degradation (>50% slower)
- Critical bug in production

### 10.2 Rollback Procedure
1. Restore parquet files from backup (`.parquet.backup` files)
2. Revert code changes via git
3. Run validation suite on restored data
4. Document issues and plan fixes

### 10.3 Backup Strategy
- Automatic backups before migration
- Retention: 30 days
- Storage: `processed_data/backups/{timestamp}/`

---

## 11. Documentation Updates Required

### 11.1 README Updates
- Add section on entity hierarchy (patient → session → trial → events)
- Document session_id and trial_id generation
- Update data schema diagrams

### 11.2 API Documentation
- Update `UnifiedDataLoader` docstrings
- Document new query methods (`get_session_ids()`, `get_trial_by_id()`)
- Add examples for session-aware queries

### 11.3 Migration Guide
- Create `docs/MIGRATION_SESSION_IDS.md`
- Step-by-step instructions for users
- FAQ section

---

## 12. Success Metrics

### 12.1 Technical Metrics
- ✅ All pipelines use session_id and trial_id
- ✅ 100% of new data has valid IDs
- ✅ <1% ID collision rate
- ✅ No performance degradation (within 10% of baseline)

### 12.2 Quality Metrics
- ✅ Feature table has 1:1 mapping to sessions
- ✅ Trial-level traceability from events to features
- ✅ Zero data loss during migration

### 12.3 Usability Metrics
- ✅ Verification scripts report ID consistency
- ✅ Debugging efficiency improved (can trace specific trials)
- ✅ Cross-session analyses supported

---

## Appendix A: Example Data Flow

**Before (current):**
```
Unified Data (patient_id, date, trial_type, ...)
    ↓
Alignment (iterrows with fragile index)
    ↓
ERP Features (patient_id, date) → CONFLICT if multiple trial types
```

**After (refactored):**
```
Unified Data (patient_id, session_id, date, trial_id, trial_type, ...)
    ↓ (IDs propagated)
Alignment (session_id, trial_id in each row and event)
    ↓ (IDs preserved)
ERP Features (patient_id, session_id, trial_type, rare_event_trial_ids)
    ↓
Traceable: feature → specific trials → specific events
```

---

## Appendix B: Code Review Checklist

Before merging, ensure:
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks acceptable
- [ ] Documentation updated
- [ ] Migration script tested on real data
- [ ] Backward compatibility verified
- [ ] Code review approved by 2+ reviewers
- [ ] QC validation checks pass

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-14  
**Author:** @copilot  
**Review Status:** Draft - Pending Stakeholder Approval
