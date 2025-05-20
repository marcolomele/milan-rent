Task: concatenate all these lines into a sleek function.

```python

tr_augmented = tr_processed.copy()

all_ngram_columns = set(tr_features_presence_df.columns)

important_features = [
    'security_door', 'tv_system', 'centralized_tv_system', 'optic_fiber',
    'video_entryphone', 'electric_gate', 'kitchen_diner', 'open_kitchen',
    'half-day_concierge', 'full_day_concierge', 'shared_garden', 'alarm_system',
    'partially_furnished', 'kitchen_nook', 'semi-habitable_kitchen', 'disabled_people'
]

print("Processing important features...")
for feature_name in important_features:
    col_name_in_df = f'has_{feature_name}'
    if feature_name in all_ngram_columns:
        tr_augmented[col_name_in_df] = tr_features_presence_df[feature_name].astype(int)
    else:
        print(f"Warning: Important feature '{feature_name}' not found in tr_features_presence_df columns. Creating a column of 0s for '{col_name_in_df}'.")
        tr_augmented[col_name_in_df] = 0
print("Done processing important features.\n")

window_features = {
    'basic': 'window_frames',
    'material': ['glass-pvc', 'glass-wood', 'glass-metal'],
    'type': ['double', 'triple']
}

print("Processing window features...")
# Basic window frames
wf_basic_str = window_features['basic']
# Find all columns in tr_features_presence_df that start with 'window_frames'
window_frame_related_cols = [col for col in all_ngram_columns if col.startswith(wf_basic_str)]

if window_frame_related_cols:
    tr_augmented['has_window_frames'] = tr_features_presence_df[window_frame_related_cols].any(axis=1).astype(int)
else:
    print(f"Warning: No n-grams starting with '{wf_basic_str}' found in tr_features_presence_df. 'has_window_frames' will be all 0s.")
    tr_augmented['has_window_frames'] = 0

# Window materials
for material_suffix in window_features['material']:
    # e.g., material_suffix = 'glass-pvc'
    # We are looking for n-grams like 'window_frames_..._glass-pvc_...'
    material_specific_cols = [
        col for col in window_frame_related_cols if material_suffix in col
    ]
    clean_material_name = material_suffix.replace("glass-", "") # pvc, wood, metal
    col_name_in_df = f'has_window_material_{clean_material_name}'
    
    if material_specific_cols:
        tr_augmented[col_name_in_df] = tr_features_presence_df[material_specific_cols].any(axis=1).astype(int)
    else:
        print(f"Warning: No n-grams for window material '{material_suffix}' found. '{col_name_in_df}' will be all 0s.")
        tr_augmented[col_name_in_df] = 0

# Window types
for type_suffix in window_features['type']:
    # e.g., type_suffix = 'double'
    # We are looking for n-grams like 'window_frames_..._double_...'
    type_specific_cols = [
        col for col in window_frame_related_cols if type_suffix in col
    ]
    col_name_in_df = f'has_window_type_{type_suffix}'
    
    if type_specific_cols:
        tr_augmented[col_name_in_df] = tr_features_presence_df[type_specific_cols].any(axis=1).astype(int)
    else:
        print(f"Warning: No n-grams for window type '{type_suffix}' found. '{col_name_in_df}' will be all 0s.")
        tr_augmented[col_name_in_df] = 0
print("Done processing window features.\n")

exposure_features = {
    'type': ['internal', 'external', 'double'],
    'direction': ['north', 'south', 'east', 'west']
}

print("Processing exposure features...")
# Exposure types
# We assume n-grams like 'internal_exposure', 'external_exposure', 'double_exposure'
for exp_type in exposure_features['type']:
    ngram_to_check = f"{exp_type}_exposure" # e.g., 'internal_exposure', 'double_exposure'
    col_name_in_df = f'has_exposure_type_{exp_type}'
    
    if ngram_to_check in all_ngram_columns:
        tr_augmented[col_name_in_df] = tr_features_presence_df[ngram_to_check].astype(int)
    else:
        print(f"Warning: N-gram '{ngram_to_check}' not found for exposure type. '{col_name_in_df}' will be all 0s.")
        tr_augmented[col_name_in_df] = 0
        
# Exposure directions
# This will capture if a direction is mentioned in any exposure-related n-gram
# e.g., 'exposure_north' or 'exposure_north-east' will make 'has_exposure_direction_north' = 1
for direction in exposure_features['direction']:
    col_name_in_df = f'has_exposure_direction_{direction}'
    # Find all n-grams that relate to 'exposure' and contain the specific direction
    direction_related_ngrams = [
        col for col in all_ngram_columns if ('exposure' in col or col.startswith(direction)) and direction in col
    ] # This logic might need tuning based on exact n-gram formats for exposure directions.
      # A simpler approach if n-grams are consistent:
    direction_related_ngrams = [
        col for col in all_ngram_columns if col.startswith('exposure_') and direction in col
    ]
    # Example: if 'exposure_north-east' is an n-gram, it will be caught for 'north' and 'east'.
    
    if direction_related_ngrams:
        tr_augmented[col_name_in_df] = tr_features_presence_df[direction_related_ngrams].any(axis=1).astype(int)
    else:
        print(f"Warning: No n-grams found for exposure direction '{direction}'. '{col_name_in_df}' will be all 0s.")
        tr_augmented[col_name_in_df] = 0
print("Done processing exposure features.\n")



```