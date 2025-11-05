import json
import shutil

from funlib.persistence import prepare_ds
import os
import shutil
import zarr
from funlib.geometry import Coordinate
from pathlib import Path
import numpy as np


def split_dataset_path(dataset_path, scale=None) -> tuple[str, str]:
    """Split the dataset path into the filename and dataset

    Args:
        dataset_path ('str'): Path to the dataset
        scale ('int'): Scale to use, if present

    Returns:
        Tuple of filename and dataset
    """

    # split at .zarr or .n5, whichever comes last
    splitter = (
        ".zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else ".n5"
    )

    filename, dataset = dataset_path.split(splitter)

    # include scale if present
    if scale is not None:
        dataset += f"/s{scale}"

    return filename + splitter, dataset


# From Yuri
def create_multiscale_metadata(multsc, levels):
    # store original array in a new .zarr file as an arr_name scale
    z_attrs = multsc
    base_scale = z_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"]
    base_trans = z_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        1
    ]["translation"]
    num_levels = levels
    for level in range(1, num_levels + 1):
        print(f"{level=}")

        # break the slices up into batches, to make things easier for the dask scheduler
        sn = [dim * pow(2, level) for dim in base_scale]
        trn = [
            (dim * (pow(2, level - 1) - 0.5)) + tr
            for (dim, tr) in zip(base_scale, base_trans)
        ]

        z_attrs["multiscales"][0]["datasets"].append(
            {
                "coordinateTransformations": [
                    {"type": "scale", "scale": sn},
                    {"type": "translation", "translation": trn},
                ],
                "path": f"s{level}",
            }
        )

    return z_attrs


def generate_multiscales_metadata(
    ds_name: str,
    voxel_size: list,
    translation: list,
    units: str,
    axes: list,
):
    z_attrs: dict = {"multiscales": [{}]}
    z_attrs["multiscales"][0]["axes"] = [
        {"name": axis, "type": "space", "unit": units} for axis in axes
    ]
    z_attrs["multiscales"][0]["coordinateTransformations"] = [
        {"scale": [1.0, 1.0, 1.0], "type": "scale"}
    ]
    z_attrs["multiscales"][0]["datasets"] = [
        {
            "coordinateTransformations": [
                {"scale": voxel_size, "type": "scale"},
                {"translation": translation, "type": "translation"},
            ],
            "path": ds_name,
        }
    ]

    z_attrs["multiscales"][0]["name"] = ""
    z_attrs["multiscales"][0]["version"] = "0.4"

    return z_attrs


def write_multiscales_metadata(
    base_ds_path, ds_name, voxel_Size, translation, units, axes
):
    multiscales_metadata = generate_multiscales_metadata(
        ds_name, voxel_Size, translation, units, axes
    )
    # write out metadata to .zattrs file
    with open(f"{base_ds_path}/.zattrs", "w") as f:
        json.dump(multiscales_metadata, f, indent=3)


def create_multiscale_dataset(
    output_path,
    dtype,
    voxel_size,
    total_roi,
    write_size,
    scale=0,
    mode="w",
    delete=True,
):

    filename, dataset = split_dataset_path(output_path, scale=scale)
    if delete:
        if ("zarr" in filename or "n5" in filename) and os.path.exists(output_path):
            # open zarr store
            shutil.rmtree(output_path)

    ds = prepare_ds(
        filename=filename,
        ds_name=dataset,
        dtype=dtype,
        voxel_size=voxel_size,
        total_roi=total_roi,
        write_size=write_size,
        force_exact_write_size=True,
        multiscales_metadata=True,
        delete=mode == "w",
    )

    write_multiscales_metadata(
        filename + "/" + dataset.rsplit(f"/s{scale}")[0],
        f"s{scale}",
        voxel_size,
        total_roi.get_begin(),
        "nanometer",
        ["z", "y", "x"],
    )
    return ds


def get_scale_info(zarr_grp):
    attrs = zarr_grp.attrs
    resolutions = {}
    offsets = {}
    shapes = {}
    # making a ton of assumptions here, hopefully triggering KeyErrors though if they don't apply
    for scale in attrs["multiscales"][0]["datasets"]:
        resolutions[scale["path"]] = scale["coordinateTransformations"][0]["scale"]
        offsets[scale["path"]] = scale["coordinateTransformations"][1]["translation"]
        shapes[scale["path"]] = zarr_grp[scale["path"]].shape
    # offset = min(offsets.values())
    return offsets, resolutions, shapes


def find_target_scale(data_path, target_resolution):
    if type(target_resolution) is int or type(target_resolution) is float:
        target_resolution = Coordinate(3 * [target_resolution])
    if type(data_path) is str:
        data_path = Path(data_path)
    try:
        zarr_grp = zarr.open_group(data_path, mode="r")
    except Exception as e:
        msg = f"Could not open zarr group at {data_path}, error: {e}"
        raise ValueError(msg)
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    target_scale = None
    min_difference = np.inf
    use_approximate = True
    for scale, res in resolutions.items():
        # get closest scale
        if Coordinate(res) == Coordinate(target_resolution):
            target_scale = scale
            use_approximate = False
            break
        else:
            difference = np.linalg.norm(Coordinate(res) - Coordinate(target_resolution))
            if difference < min_difference:
                min_difference = difference
                target_scale = scale

    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array with sampling {target_resolution}"
        raise ValueError(msg)
    if use_approximate:
        msg = f"Warning: Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array with exact sampling {target_resolution}, using closest scale {target_scale} with resolution {resolutions[target_scale]}"
        # print warning
        print(msg)
    target_path = data_path / target_scale
    return (
        str(target_path),
        target_scale,
        offsets[target_scale],
        shapes[target_scale],
    )


def find_highest_resolution_scale(data_path, min_resolution=None):
    """
    Find the highest resolution (smallest voxel size) scale available in a zarr dataset.

    Parameters:
    -----------
    data_path : str or Path
        Path to the zarr dataset
    min_resolution : int, float, or Coordinate, optional
        Minimum allowed resolution (maximum voxel size). If specified, will not use
        scales with resolution smaller than this value. Useful to avoid extremely
        high resolution scales that may be computationally prohibitive.

    Returns:
    --------
    tuple: (target_path, target_scale, offset, shape)
    """
    if type(data_path) is str:
        data_path = Path(data_path)
    try:
        zarr_grp = zarr.open_group(data_path, mode="r")
    except Exception as e:
        msg = f"Could not open zarr group at {data_path}, error: {e}"
        raise ValueError(msg)

    offsets, resolutions, shapes = get_scale_info(zarr_grp)

    # Convert min_resolution to Coordinate if provided
    if min_resolution is not None:
        if type(min_resolution) is int or type(min_resolution) is float:
            min_resolution = Coordinate(3 * [min_resolution])
        else:
            min_resolution = Coordinate(min_resolution)

    # Find the scale with the smallest resolution (highest resolution)
    # that is still >= min_resolution
    best_scale = None
    best_resolution = None

    for scale, res in resolutions.items():
        res_coord = Coordinate(res)

        # Skip scales that are finer than min_resolution (if specified)
        if min_resolution is not None:
            if any(r < min_r for r, min_r in zip(res_coord, min_resolution)):
                print(
                    f"Skipping scale {scale} with resolution {res} (finer than min_resolution {min_resolution})"
                )
                continue

        # Calculate total resolution (product of all dimensions)
        total_res = np.prod(res)

        if best_resolution is None or total_res < best_resolution:
            best_resolution = total_res
            best_scale = scale

    if best_scale is None:
        if min_resolution is not None:
            msg = f"No scales found in zarr {zarr_grp.store.path}, {zarr_grp.path} that meet min_resolution requirement {min_resolution}"
        else:
            msg = f"No scales found in zarr {zarr_grp.store.path}, {zarr_grp.path}"
        raise ValueError(msg)

    resolution_info = f"Using highest resolution scale {best_scale} with resolution {resolutions[best_scale]} for {data_path}"
    if min_resolution is not None:
        resolution_info += f" (min_resolution: {min_resolution})"
    print(resolution_info)

    target_path = data_path / best_scale
    return (
        str(target_path),
        best_scale,
        offsets[best_scale],
        shapes[best_scale],
    )
