import json
import shutil

from funlib.persistence import prepare_ds
import os
import shutil


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