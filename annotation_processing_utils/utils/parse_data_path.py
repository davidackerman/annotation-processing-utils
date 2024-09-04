from pathlib import Path


def parse_data_path(data_path: str):
    if not (".n5" in data_path or ".zarr" in data_path):
        raise Exception(f"Unrecognized file type for {data_path}")

    data_type = ".n5" if ".n5" in data_path else ".zarr"
    file_name, dataset_name = data_path.rsplit(data_type, 1)
    if dataset_name.startswith("/"):
        dataset_name = dataset_name[1:]

    if len(dataset_name) == 0:
        raise Exception(f"No dataset provided")

    return (
        file_name + data_type,
        dataset_name,
    )
