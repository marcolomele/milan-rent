import pandas as pd
import googlemaps
import os
import time
from tqdm import tqdm

def fetch_nearby_services_for_zones(
    df_path: str,
    services_list: list,
    output_dir: str,
    output_filename: str = "zones_with_services_counts.csv",
    radius_meters: int = 1000,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Augments a DataFrame with counts of nearby services for each zone using Google Places API,
    with checkpointing to resume progress.

    Args:
        df_path (str): Path to the input CSV file containing zone data. 
                       Expected columns: 'zone', 'latitude', 'longitude'.
        services_list (list): A list of strings, where each string is a service type
                              to search for (e.g., "school", "supermarket").
        output_dir (str): Directory where the augmented CSV file will be saved.
        output_filename (str, optional): Name for the output CSV file. 
                                         Defaults to "zones_with_services_counts.csv".
        radius_meters (int, optional): Search radius in meters from the zone's center.
                                       Defaults to 1000 (1km).
        verbose (bool, optional): If True, prints detailed progress for each service query.
                                  Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame augmented with service counts.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set.")

    gmaps = googlemaps.Client(key=api_key)
    os.makedirs(output_dir, exist_ok=True)

    base_output_filename, ext = os.path.splitext(output_filename)
    checkpoint_filename = f"{base_output_filename}_checkpoint{ext}"
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)

    radius_km_str = str(radius_meters / 1000).replace('.', '_')
    service_cols_to_check = []
    for service in services_list:
        safe_service_name = service.lower().replace(" ", "_")
        service_cols_to_check.append(f'{safe_service_name}_count_{radius_km_str}km_radius')

    if os.path.exists(checkpoint_path):
        if verbose:
            print(f"Found checkpoint file: {checkpoint_path}. Loading...")
        df = pd.read_csv(checkpoint_path)
        # Ensure all expected service columns exist, add if not (e.g. new services added to list)
        for col in service_cols_to_check:
            if col not in df.columns:
                df[col] = pd.NA
        if verbose:
            print("Checkpoint loaded successfully.")
    else:
        if verbose:
            print(f"No checkpoint file found at {checkpoint_path}. Starting fresh from {df_path}.")
        try:
            df = pd.read_csv(df_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input DataFrame not found at {df_path}")
        for col in service_cols_to_check:
            df[col] = pd.NA

    if not all(col in df.columns for col in ['zone', 'latitude', 'longitude']):
        raise ValueError("DataFrame (from source or checkpoint) must contain 'zone', 'latitude', and 'longitude' columns.")

    if not verbose:
        print(f"Processing {len(df)} zones and {len(services_list)} services...")
        print(f"Using a search radius of {radius_meters} meters.")
        if os.path.exists(checkpoint_path):
             print(f"Resuming from checkpoint: {checkpoint_path}")

    processed_zones_count = 0
    for index, row in df.iterrows():
        all_services_done_for_zone = True
        for service_col in service_cols_to_check:
            if pd.isna(row[service_col]):
                all_services_done_for_zone = False
                break
        if all_services_done_for_zone:
            processed_zones_count +=1
            continue # Skip to next zone if all services already processed
    
    with tqdm(total=len(df), desc="Processing Zones", initial=processed_zones_count, disable=verbose) as pbar_zones:
        for index, row in df.iterrows():
            # Re-check if this zone was completed by a previous iteration if resuming (pbar.update done outside)
            current_zone_all_services_done = True
            for service_col in service_cols_to_check:
                if pd.isna(row[service_col]):
                    current_zone_all_services_done = False
                    break
            if current_zone_all_services_done:
                if pbar_zones.n < processed_zones_count: # Only update if not already counted in initial
                    pbar_zones.update(1) # Reflect already processed zone in progress bar
                continue

            if verbose:
                print(f"\nProcessing zone: {row['zone']} ({index + 1}/{len(df)})")
            
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                if verbose:
                    print(f"  Skipping zone {row['zone']} due to missing coordinates.")
                # Mark all services as 0 or NA for this zone if coords are missing, so it's skipped next time
                for service_col in service_cols_to_check:
                    if pd.isna(df.loc[index, service_col]): # Only fill if not already filled by checkpoint
                        df.loc[index, service_col] = 0 # Or pd.NA if preferred for un-queryable
                pbar_zones.update(1)
                df.to_csv(checkpoint_path, index=False) # Save checkpoint
                continue

            location = (row['latitude'], row['longitude'])

            service_iterator = tqdm(services_list, desc=f"Services for {row['zone']:<30}", leave=False, disable=verbose)
            for service in service_iterator:
                safe_service_name = service.lower().replace(" ", "_")
                col_name = f'{safe_service_name}_count_{radius_km_str}km_radius'

                if not pd.isna(df.loc[index, col_name]):
                    if verbose:
                        print(f"  Service '{service}' already processed for zone {row['zone']}. Count: {df.loc[index, col_name]}")
                    continue # Skip API call if data already exists

                if verbose:
                    print(f"  Querying for service: {service}...")
                
                total_results_for_service = 0
                try:
                    response = gmaps.places_nearby(
                        location=location,
                        radius=radius_meters,
                        keyword=service
                    )
                    results_on_page = response.get('results', [])
                    total_results_for_service += len(results_on_page)
                    next_page_token = response.get('next_page_token')
                    
                    page_count = 0
                    while next_page_token and page_count < 2: 
                        time.sleep(2) 
                        if verbose:
                            print(f"    Fetching next page for {service}...")
                        response = gmaps.places_nearby(page_token=next_page_token)
                        results_on_page = response.get('results', [])
                        total_results_for_service += len(results_on_page)
                        next_page_token = response.get('next_page_token')
                        page_count += 1
                    
                    df.loc[index, col_name] = total_results_for_service
                    if verbose:
                        print(f"    Found {total_results_for_service} instances of {service} for zone {row['zone']}.")

                except Exception as e:
                    print(f"\n  Error querying Google Maps API for zone {row['zone']}, service '{service}': {e}")
                    df.loc[index, col_name] = 0 # Mark as 0 on error to avoid re-querying this specific one
                
                time.sleep(0.1) 
            
            # Save checkpoint after all services for the current zone are processed
            df.to_csv(checkpoint_path, index=False)
            if verbose:
                print(f"Checkpoint saved for zone {row['zone']} at {checkpoint_path}")
            pbar_zones.update(1)
            time.sleep(0.5) 

    final_output_path = os.path.join(output_dir, output_filename)
    df.to_csv(final_output_path, index=False)
    print(f"\nProcessing complete. Final DataFrame saved to {final_output_path}")
    if verbose and os.path.exists(checkpoint_path):
        print(f"Checkpoint file kept at: {checkpoint_path}")

    return df

if __name__ == '__main__':
    dummy_df_path = "zones_coordinates.csv"
    if not os.path.exists(dummy_df_path):
        print(f"Creating dummy {dummy_df_path} for example run...")
        # Make a more comprehensive dummy for testing checkpoints
        dummy_data = {
            'zone': ['zone_a_test', 'zone_b_test', 'zone_c_test_missing_coords', 'zone_d_test'],
            'latitude': [45.4642, 45.4779, pd.NA, 45.4800],
            'longitude': [9.1900, 9.1234, pd.NA, 9.2000]
        }
        pd.DataFrame(dummy_data).to_csv(dummy_df_path, index=False)

    services_to_check = ["supermarket", "pharmacy", "park", "cafe"]
    input_dataframe_path = dummy_df_path 
    output_directory = "output_data_checkpoint_test"
    
    # Clean up old checkpoint and output for a clean test run if needed
    # test_checkpoint_path = os.path.join(output_directory, "milan_zones_services_ckpt_verbose_checkpoint.csv")
    # test_output_path = os.path.join(output_directory, "milan_zones_services_ckpt_verbose.csv")
    # if os.path.exists(test_checkpoint_path): os.remove(test_checkpoint_path)
    # if os.path.exists(test_output_path): os.remove(test_output_path)

    print("Starting example run with checkpointing (verbose=True). Make sure your GOOGLE_MAPS_API_KEY is set.")
    try:
        augmented_dataframe_verbose = fetch_nearby_services_for_zones(
            df_path=input_dataframe_path,
            services_list=services_to_check,
            output_dir=output_directory,
            output_filename="milan_zones_services_ckpt_verbose.csv",
            radius_meters=500, 
            verbose=True
        )
        print("\nVerbose example run with checkpointing finished. Augmented DataFrame head:")
        print(augmented_dataframe_verbose.head())
    except Exception as ex:
        print(f"An unexpected error occurred during verbose example run: {ex}")

    # print("\nStarting example run (verbose=False). Make sure your GOOGLE_MAPS_API_KEY is set.")
    # try:
    #     augmented_dataframe_quiet = fetch_nearby_services_for_zones(
    #         df_path=input_dataframe_path,
    #         services_list=services_to_check,
    #         output_dir=output_directory,
    #         output_filename="milan_zones_services_ckpt_quiet.csv",
    #         radius_meters=500, 
    #         verbose=False
    #     )
    #     print("\nQuiet example run finished. Augmented DataFrame head:")
    #     print(augmented_dataframe_quiet.head())
    # except Exception as ex:
    #     print(f"An unexpected error occurred during quiet example run: {ex}") 