#!/usr/bin/env python3
"""
Preprocessing script to create a unified training_metadata.csv from source CSVs.

This script merges prepared.csv (high-level metadata) with 
prepared_tile_view_w_annotations_fixed.csv (tile-level metadata) to create
a single validated CSV file for training.

Usage:
    python prepare_synthetic_metadata.py \
        --prepared_csv /path/to/prepared.csv \
        --tile_view_csv /path/to/prepared_tile_view_w_annotations_fixed.csv \
        --output_csv /clusterfs/nvme/segment_3d/databases/synthetic_data/training_metadata.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def load_and_merge_metadata(prepared_csv: str, tile_view_csv: str) -> pd.DataFrame:
    """
    Load and merge prepared.csv and tile-view CSV.
    
    Args:
        prepared_csv: Path to prepared.csv (high-level metadata)
        tile_view_csv: Path to prepared_tile_view_w_annotations_fixed.csv (tile-level metadata)
        
    Returns:
        Merged DataFrame with unified schema
    """
    print(f"Loading prepared.csv from {prepared_csv}")
    prepared_df = pd.read_csv(prepared_csv)
    print(f"  Loaded {len(prepared_df)} rows")
    
    print(f"Loading tile-view CSV from {tile_view_csv}")
    tile_df = pd.read_csv(tile_view_csv)
    print(f"  Loaded {len(tile_df)} rows")
    
    # Clean up any unnamed index columns
    tile_df = tile_df.loc[:, ~tile_df.columns.str.contains('^Unnamed')]
    
    # Join on prepared_id (tile-view) = id (prepared)
    print("Joining metadata...")
    merged_df = tile_df.merge(
        prepared_df,
        left_on="prepared_id",
        right_on="id",
        how="inner",
        suffixes=("_tile", "_prepared")
    )
    print(f"  Merged to {len(merged_df)} rows")
    
    # Validate that all prepared_ids in tile-view exist in prepared.csv
    missing_ids = set(tile_df["prepared_id"]) - set(prepared_df["id"])
    if missing_ids:
        print(f"  WARNING: {len(missing_ids)} prepared_ids in tile-view not found in prepared.csv")
        print(f"    Missing IDs: {sorted(list(missing_ids))[:10]}...")
    
    return merged_df


def create_unified_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified schema for training metadata.
    
    Uses tile-view columns as primary source, supplemented by prepared.csv fields.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        DataFrame with unified schema
    """
    # Start with tile-view columns as primary
    unified = pd.DataFrame()
    
    # ID fields
    unified["prepared_id"] = df["prepared_id"]
    
    # Paths (from tile-view, primary source)
    unified["server_folder"] = df["server_folder"]
    unified["output_folder"] = df["output_folder"]
    unified["tile_name"] = df["tile_name"]
    
    # Data location from prepared.csv (for real data paths)
    unified["data_location"] = df["data_location"]
    
    # Geometry - use tile_* fields as primary, fallback to regular fields
    unified["z_start"] = df.get("tile_z_start", df.get("z_start", 0))
    unified["y_start"] = df.get("tile_y_start", df.get("y_start", 0))
    unified["x_start"] = df.get("tile_x_start", df.get("x_start", 0))
    
    # Use tile_*_end if available, otherwise compute from start + size
    if "tile_z_end" in df.columns:
        unified["z_end"] = df["tile_z_end"]
    else:
        unified["z_end"] = unified["z_start"] + df.get("z_size", df.get("tile_z_size", 0))
    
    if "tile_y_end" in df.columns:
        unified["y_end"] = df["tile_y_end"]
    else:
        unified["y_end"] = unified["y_start"] + df.get("y_size", df.get("tile_y_size", 0))
    
    if "tile_x_end" in df.columns:
        unified["x_end"] = df["tile_x_end"]
    else:
        unified["x_end"] = unified["x_start"] + df.get("x_size", df.get("tile_x_size", 0))
    
    # Sizes - prefer tile_*_size, fallback to regular sizes
    unified["z_size"] = df.get("tile_z_size", df.get("z_size", unified["z_end"] - unified["z_start"]))
    unified["y_size"] = df.get("tile_y_size", df.get("y_size", unified["y_end"] - unified["y_start"]))
    unified["x_size"] = df.get("tile_x_size", df.get("x_size", unified["x_end"] - unified["x_start"]))
    
    # Time and channel sizes
    unified["time_size"] = df.get("tile_time_size", df.get("time_size", 1))
    unified["channel_size"] = df.get("tile_channel_size", df.get("channel_size", df.get("channel_size", 2)))
    
    # Cube size from prepared.csv (if available)
    unified["cube_size"] = df.get("cube_size", unified["z_size"])
    
    # Flags
    unified["is_synthetic"] = df["is_synthetic"].astype(bool)
    unified["exists"] = df.get("exists", True).astype(bool)
    
    # Construct paths
    unified["synthetic_cube_path"] = (
        unified["server_folder"].astype(str) + "/" +
        unified["output_folder"].astype(str) + "/" +
        unified["tile_name"].astype(str)
    )
    
    # Real cube path (only if data_location is not NaN)
    unified["real_cube_path"] = unified.apply(
        lambda row: (
            str(row["data_location"]) + "/" +
            str(row["output_folder"]) + "/" +
            str(row["tile_name"])
        ) if pd.notna(row["data_location"]) else None,
        axis=1
    )
    
    return unified


def validate_metadata(df: pd.DataFrame, check_paths: bool = False) -> None:
    """
    Validate metadata and print summary statistics.
    
    Args:
        df: DataFrame to validate
        check_paths: If True, check that paths actually exist
    """
    print("\n=== Metadata Validation ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique prepared_ids: {df['prepared_id'].nunique()}")
    print(f"Synthetic entries: {df['is_synthetic'].sum()}")
    print(f"Real entries: {(~df['is_synthetic']).sum()}")
    
    # Check geometry
    print("\nGeometry ranges:")
    print(f"  Z: {df['z_start'].min()}-{df['z_end'].max()} (size: {df['z_size'].min()}-{df['z_size'].max()})")
    print(f"  Y: {df['y_start'].min()}-{df['y_end'].max()} (size: {df['y_size'].min()}-{df['y_size'].max()})")
    print(f"  X: {df['x_start'].min()}-{df['x_end'].max()} (size: {df['x_size'].min()}-{df['x_size'].max()})")
    
    # Check for invalid geometry
    invalid_z = (df["z_end"] <= df["z_start"]).sum()
    invalid_y = (df["y_end"] <= df["y_start"]).sum()
    invalid_x = (df["x_end"] <= df["x_start"]).sum()
    if invalid_z > 0 or invalid_y > 0 or invalid_x > 0:
        print(f"\nWARNING: Found invalid geometry:")
        print(f"  Invalid Z ranges: {invalid_z}")
        print(f"  Invalid Y ranges: {invalid_y}")
        print(f"  Invalid X ranges: {invalid_x}")
    
    # Check paths if requested
    if check_paths:
        print("\nChecking paths (sample of 10)...")
        sample = df.head(10)
        for idx, row in sample.iterrows():
            synth_exists = os.path.exists(row["synthetic_cube_path"]) if pd.notna(row["synthetic_cube_path"]) else False
            real_exists = os.path.exists(row["real_cube_path"]) if pd.notna(row["real_cube_path"]) else False
            print(f"  Row {idx}: synthetic={synth_exists}, real={real_exists}")
    
    print("\n=== Validation Complete ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create unified training_metadata.csv from source CSVs"
    )
    parser.add_argument(
        "--prepared_csv",
        type=str,
        default="/clusterfs/vast/forsynthetic/benchmark_tests/data/synthetic_data_iteration_1/supabase_csvs/prepared.csv",
        help="Path to prepared.csv"
    )
    parser.add_argument(
        "--tile_view_csv",
        type=str,
        default="/clusterfs/nvme/segment_4d/databases/prepared_tile_view_w_annotations_fixed.csv",
        help="Path to prepared_tile_view_w_annotations_fixed.csv"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/clusterfs/nvme/segment_3d/databases/synthetic_data/training_metadata.csv",
        help="Output path for training_metadata.csv"
    )
    parser.add_argument(
        "--check_paths",
        action="store_true",
        help="Check that Zarr paths actually exist"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.prepared_csv):
        print(f"ERROR: prepared.csv not found at {args.prepared_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.tile_view_csv):
        print(f"ERROR: tile-view CSV not found at {args.tile_view_csv}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and merge
    merged_df = load_and_merge_metadata(args.prepared_csv, args.tile_view_csv)
    
    # Create unified schema
    unified_df = create_unified_schema(merged_df)
    
    # Validate
    validate_metadata(unified_df, check_paths=args.check_paths)
    
    # Save
    print(f"Saving unified metadata to {args.output_csv}")
    unified_df.to_csv(args.output_csv, index=False)
    print(f"  Saved {len(unified_df)} rows")
    
    # Print sample
    print("\nSample rows:")
    print(unified_df[["prepared_id", "tile_name", "is_synthetic", "synthetic_cube_path"]].head())


if __name__ == "__main__":
    main()

