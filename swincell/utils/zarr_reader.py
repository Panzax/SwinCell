"""
Zarr cube reader for 3D volumes.

This module provides utilities to read 3D volumes from petakit-style Zarr arrays.
"""

import os
from typing import Optional, Tuple

import numpy as np

try:
    import zarr
except ImportError:
    zarr = None

try:
    import tensorstore as ts
except ImportError:
    ts = None


def load_zarr_cube(
    zarr_path: str,
    t_index: int = 0,
    channel_indices: Optional[list] = None,
    image_channel: int = 0,
    label_channel: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D volume from a Zarr cube.

    Args:
        zarr_path: Path to the Zarr directory
        t_index: Time index to select (default 0)
        channel_indices: List of channel indices to load. If None, loads all channels.
        image_channel: Channel index for the image data (default 0)
        label_channel: Channel index for the label/mask data (default 1)

    Returns:
        Tuple of (image_volume, label_volume) as numpy arrays.
        Image shape: (Z, Y, X) or (C, Z, Y, X) if multiple channels
        Label shape: (Z, Y, X)
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    # Try to use zarr library first
    if zarr is not None:
        try:
            return _load_with_zarr(
                zarr_path, t_index, channel_indices, image_channel, label_channel
            )
        except Exception as e:
            # Fall back to tensorstore if zarr fails
            if ts is not None:
                return _load_with_tensorstore(
                    zarr_path, t_index, channel_indices, image_channel, label_channel
                )
            else:
                raise RuntimeError(
                    f"Failed to load with zarr: {e}. Tensorstore also not available."
                )

    # Try tensorstore if zarr is not available
    if ts is not None:
        return _load_with_tensorstore(
            zarr_path, t_index, channel_indices, image_channel, label_channel
        )

    raise RuntimeError(
        "Neither zarr nor tensorstore is available. Please install one of them."
    )


def _load_with_zarr(
    zarr_path: str,
    t_index: int,
    channel_indices: Optional[list],
    image_channel: int,
    label_channel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load using zarr library."""
    zarr_group = zarr.open(zarr_path, mode="r")

    # Petakit zarr format: data is typically in 'c' group with nested structure
    # Try to find the main array
    if "c" in zarr_group:
        # Navigate the nested structure: c/0/0/0/... (time/channel/z/y/x)
        # For petakit format, structure is: c/{time}/{channel}/{z}/{y}/{x}
        time_group = zarr_group["c"].get(str(t_index), None)
        if time_group is None:
            # Try integer key
            time_group = zarr_group["c"].get(t_index, None)

        if time_group is None:
            raise ValueError(
                f"Could not find time index {t_index} in zarr group {zarr_path}"
            )

        # Get image channel
        img_channel_group = time_group.get(str(image_channel), None)
        if img_channel_group is None:
            img_channel_group = time_group.get(image_channel, None)

        if img_channel_group is None:
            raise ValueError(
                f"Could not find image channel {image_channel} in zarr group"
            )

        # Get label channel
        label_channel_group = time_group.get(str(label_channel), None)
        if label_channel_group is None:
            label_channel_group = time_group.get(label_channel, None)

        if label_channel_group is None:
            raise ValueError(
                f"Could not find label channel {label_channel} in zarr group"
            )

        # Read the data - need to reconstruct the full array from chunks
        # Petakit stores data in nested z/y/x structure
        image_data = _read_nested_zarr_array(img_channel_group)
        label_data = _read_nested_zarr_array(label_channel_group)

        return image_data, label_data

    # Fallback: try to find any array directly
    if "0" in zarr_group:
        data = zarr_group["0"][:]
        return _extract_from_5d_array(
            data, t_index, image_channel, label_channel, channel_indices
        )

    # Try to find any array
    keys = list(zarr_group.keys())
    if keys:
        data = zarr_group[keys[0]][:]
        return _extract_from_5d_array(
            data, t_index, image_channel, label_channel, channel_indices
        )

    raise ValueError(f"Could not find data array in zarr file: {zarr_path}")


def _read_nested_zarr_array(group) -> np.ndarray:
    """
    Read a nested zarr array structure.

    Petakit stores data as nested groups: z/y/x where each contains chunks.
    We need to reconstruct the full array.
    """
    # Get all z indices
    z_indices = sorted([int(k) for k in group.keys() if k.isdigit()])
    if not z_indices:
        # Try string keys
        z_indices = sorted([int(k) for k in group.keys()])

    if not z_indices:
        raise ValueError("Could not find z indices in zarr group")

    # Read first z slice to determine shape
    z0_group = group[str(z_indices[0])]
    y_indices = sorted([int(k) for k in z0_group.keys() if k.isdigit()])
    if not y_indices:
        y_indices = sorted([int(k) for k in z0_group.keys()])

    y0_group = z0_group[str(y_indices[0])]
    x_indices = sorted([int(k) for k in y0_group.keys() if k.isdigit()])
    if not x_indices:
        x_indices = sorted([int(k) for k in y0_group.keys()])

    # Read a sample chunk to get dimensions
    sample_chunk = y0_group[str(x_indices[0])][:]
    chunk_shape = sample_chunk.shape

    # Determine full array shape
    # Assuming chunks are stored as (chunk_z, chunk_y, chunk_x)
    # We need to determine the full extent
    full_z = len(z_indices) * chunk_shape[0] if len(chunk_shape) >= 1 else len(
        z_indices
    )
    full_y = len(y_indices) * chunk_shape[1] if len(chunk_shape) >= 2 else len(
        y_indices
    )
    full_x = len(x_indices) * chunk_shape[2] if len(chunk_shape) >= 3 else len(
        x_indices
    )

    # Allocate output array
    if len(chunk_shape) == 3:
        output = np.zeros((full_z, full_y, full_x), dtype=sample_chunk.dtype)
    else:
        output = np.zeros((full_z, full_y, full_x), dtype=sample_chunk.dtype)

    # Read all chunks and assemble
    for z_idx in z_indices:
        z_group = group[str(z_idx)]
        for y_idx in y_indices:
            y_group = z_group[str(y_idx)]
            for x_idx in x_indices:
                chunk = y_group[str(x_idx)][:]
                # Determine position in output array
                z_start = z_idx * chunk_shape[0] if len(chunk_shape) >= 1 else z_idx
                y_start = y_idx * chunk_shape[1] if len(chunk_shape) >= 2 else y_idx
                x_start = x_idx * chunk_shape[2] if len(chunk_shape) >= 3 else x_idx

                z_end = z_start + chunk.shape[0]
                y_end = y_start + chunk.shape[1]
                x_end = x_start + chunk.shape[2]

                output[z_start:z_end, y_start:y_end, x_start:x_end] = chunk

    return output


def _extract_from_5d_array(
    data: np.ndarray,
    t_index: int,
    image_channel: int,
    label_channel: int,
    channel_indices: Optional[list],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract image and label volumes from a 5D array (T, C, Z, Y, X).

    Args:
        data: Input array, shape (T, C, Z, Y, X) or (C, Z, Y, X) or (Z, Y, X, C)
        t_index: Time index
        image_channel: Channel index for image
        label_channel: Channel index for label
        channel_indices: Optional list of channel indices to select

    Returns:
        Tuple of (image, label) arrays
    """
    # Handle different array shapes
    if data.ndim == 5:
        # (T, C, Z, Y, X)
        image = data[t_index, image_channel, ...]
        label = data[t_index, label_channel, ...]
    elif data.ndim == 4:
        # Could be (C, Z, Y, X) or (Z, Y, X, C)
        if data.shape[0] < data.shape[-1]:
            # Likely (Z, Y, X, C)
            image = data[..., image_channel]
            label = data[..., label_channel]
        else:
            # Likely (C, Z, Y, X)
            image = data[image_channel, ...]
            label = data[label_channel, ...]
    elif data.ndim == 3:
        # (Z, Y, X) - single channel, assume it's the image
        image = data
        label = np.zeros_like(image)  # No label available
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    return image, label


def _load_with_tensorstore(
    zarr_path: str,
    t_index: int,
    channel_indices: Optional[list],
    image_channel: int,
    label_channel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load using tensorstore library."""
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": zarr_path},
    }

    ds = ts.open(spec, read=True).result()

    # Read the data
    data = ds.read().result()

    return _extract_from_5d_array(
        data, t_index, image_channel, label_channel, channel_indices
    )

