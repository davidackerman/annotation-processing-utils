import os
from dacapo.store.create_store import (
    create_config_store,
    create_weights_store,
)
from dacapo.experiments import Run
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    IntensitiesArrayConfig,
    IntensitiesArray,
)
from annotation_processing_utils.utils.parse_data_path import parse_data_path
from annotation_processing_utils.utils.predict_with_write_size import (
    predict_with_write_size,
)
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch
from pathlib import Path
import torch

from dacapo.experiments.model import Model
import logging
import shutil
from funlib.geometry import Roi

import getpass

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.NOTSET,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_model(architecture):
    head = torch.nn.Conv3d(72, 19, kernel_size=1)
    return Model(architecture, head, eval_activation=torch.nn.Sigmoid())


def inference(
    run: str,
    iteration: int,
    raw_path: str,
    inference_path: str,
    roi: Roi,
):
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run)
    run = Run(run_config)
    run.model = create_model(run.architecture)
    # create weights store and read weights
    weights_store = create_weights_store()
    weights = weights_store.retrieve_weights(run, iteration)
    run.model.load_state_dict(weights.model)
    torch.backends.cudnn.benchmark = True
    run.model.eval()

    raw_file_name, raw_dataset_name = parse_data_path(raw_path)
    inference_file_name, inference_dataset_name = parse_data_path(inference_path)

    tmp = ZarrArrayConfig(
        name="tmp",
        file_name=Path(inference_file_name),
        dataset=inference_dataset_name,
    )
    verified, _ = tmp.verify()
    if verified:
        shutil.rmtree(
            inference_path,
            ignore_errors=True,
        )

    prediction_array_identifier = LocalArrayIdentifier(
        Path(inference_file_name),
        inference_dataset_name,
    )

    raw_zarr_array_config = ZarrArrayConfig(
        name="raw",
        file_name=Path(raw_file_name),
        dataset=raw_dataset_name,
    )
    # We get an error without this, and will suggests having it as such https://cell-map.slack.com/archives/D02KBQ990ER/p1683762491204909
    raw_intensities_array_config = IntensitiesArrayConfig(
        name="raw", source_array_config=raw_zarr_array_config, min=0, max=255
    )
    raw_array = IntensitiesArray(raw_intensities_array_config)

    predict_with_write_size(
        run.model,
        raw_array,
        prediction_array_identifier,
        compute_context=LocalTorch(),
        output_roi=roi,
        write_size=[108 * 8, 108 * 8, 108 * 8, 9],
    )
