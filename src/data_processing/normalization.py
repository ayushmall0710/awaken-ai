import pandas as pd
import json
import ast

# Trial Type Mappings
TRIAL_TYPE_MAP = {
    'lcmd': 'left_command',
    'rcmd': 'right_command',
    'lang': 'language',
    'odd': 'oddball',
    # 'beep': 'control',
    'loved_one': 'loved_one_voice',
    'language_11': 'language',
}

def normalize_trial_type(tt):
    """
    Normalizes a trial type string using standard mappings and rules.
    """
    if pd.isna(tt):
        return "unknown"
    
    tt_str = str(tt).lower().strip()
    
    # Remove '+p' suffix
    if tt_str.endswith('+p'):
        tt_str = tt_str[:-2]
        
    # Handle lang_XX patterns
    if tt_str.startswith('lang_') and tt_str[5:].isdigit():
        return 'language'

    # Apply explict mapping
    if tt_str in TRIAL_TYPE_MAP:
        return TRIAL_TYPE_MAP[tt_str]
        
    return tt_str

def normalize_sentences(sample):
    """
    Converts inputs (strings, lists of strings, lists of ints, lists of dicts)
    into a consistent List[Dict] format: [{'event': ..., 'onset_time': ...}]
    """
    if pd.isna(sample) or sample == "[]" or sample == "":
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
                return [{'event': sample, 'onset_time': None}]
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
            normalized_list.append({'event': str(item), 'onset_time': None})
            
        elif isinstance(item, str):
            # Convert string event -> Dict
            normalized_list.append({'event': item, 'onset_time': None})
            
        else:
            # Unknown type
            normalized_list.append({'event': str(item), 'onset_time': None})
            
    return normalized_list
