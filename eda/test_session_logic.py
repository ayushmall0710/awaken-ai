"""
Test to verify single-session trial filtering works correctly
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loading.unified_data_loader import UnifiedDataLoader


def test_single_session_logic():
    loader = UnifiedDataLoader()

    # Test CON008 (known single-session patient)
    patient_id = "CON008"

    # Get all trials
    trials = loader.get_patient_trials(patient_id)
    print(f"\n{patient_id} Trials:")
    print(f"  Total trials: {len(trials)}")
    print(f"  Unique dates: {trials['date'].unique()}")

    # Load EDF
    raw = loader.load_edf(patient_id)
    print(f"\nEDF type: {type(raw)}")

    if not isinstance(raw, dict):
        print("Single Raw object returned")
        edf_date = raw.info["meas_date"].strftime("%Y-%m-%d")
        print(f"  EDF date: {edf_date}")

        # Check if EDF date matches trial dates
        trial_dates = set(trials["date"].unique())
        print(f"  Trial dates: {trial_dates}")

        if edf_date in trial_dates:
            print(f"  ✅ EDF date matches trial date")
        else:
            print(f"  ❌  Mismatch! EDF: {edf_date}, Trials: {trial_dates}")

        # Check if ALL trials match this date
        if len(trial_dates) == 1 and edf_date in trial_dates:
            print(f"  ✅ All trials are from same session as EDF")
        else:
            print(f"  ⚠️  trials span multiple dates but EDF is single session!")
    else:
        print("Dict returned (multiple sessions)")
        for date, raw_obj in raw.items():
            print(f"  Session: {date}")


if __name__ == "__main__":
    test_single_session_logic()
