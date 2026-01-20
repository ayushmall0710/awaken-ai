# Tests

This directory contains unit tests for the Awaken AI EEG data pipeline.

## Running Tests

To run all tests:

```bash
python tests/test_timestamp_alignment.py
```

## Test Coverage

### test_timestamp_alignment.py

Tests for the timestamp alignment functionality (ENG-01 and ENG-02):

- **test_imports**: Verifies that modules can be imported correctly
- **test_timestamp_conversion**: Tests conversion from EDF time to Unix timestamps
- **test_peak_detection**: Tests stimulus onset detection with synthetic signals
- **test_alignment_validation**: Tests alignment metrics computation

## Test Data

These tests use synthetic/mock data and do not require actual EDF or CSV files.
For integration testing with real data, use the demo script:

```bash
python examples/timestamp_alignment_demo.py --edf /path/to/file.EDF --csv /path/to/file.csv
```
