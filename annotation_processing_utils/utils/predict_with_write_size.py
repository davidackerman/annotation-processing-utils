from dacapo.gp import DaCapoArraySource
from dacapo.experiments.model import Model
from dacapo.experiments.datasplits.datasets.arrays import Array
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch, ComputeContext
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray

from funlib.geometry import Coordinate, Roi
import gunpowder as gp
import gunpowder.torch as gp_torch
import numpy as np
import zarr

from typing import Optional
import logging
import sys

from annotation_processing_utils.utils.zarr_write_select_channels import (
    ZarrWriteSelectChannels,
)

logger = logging.getLogger(__name__)


def predict_with_write_size(
    model: Model,
    raw_array: Array,
    prediction_array_identifier: LocalArrayIdentifier,
    num_cpu_workers: int = 4,
    compute_context: ComputeContext = LocalTorch(),
    output_roi: Optional[Roi] = None,
    write_size=None,
    input_shape_scaling_factor: int = 1,
):
    # get the model's input and output size
    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)

    # Get base shapes (without scaling)
    base_input_shape = Coordinate(model.eval_input_shape)
    base_input_size = input_voxel_size * base_input_shape
    base_output_size = output_voxel_size * model.compute_output_shape(base_input_shape)[1]

    # Calculate context (should remain constant regardless of scaling)
    context = (base_input_size - base_output_size) / 2

    # Scale the output size (without context), then add context back
    scaled_output_size = base_output_size * input_shape_scaling_factor
    input_size = scaled_output_size + context * 2
    output_size = scaled_output_size

    logger.info(
        "Predicting with input_shape_scaling_factor %d, context %s, input size %s, output size %s",
        input_shape_scaling_factor, context, input_size, output_size
    )
    sys.stdout.flush()
    sys.stderr.flush()

    # calculate input and output rois
    # (context is already calculated above)
    if output_roi is None:
        input_roi = raw_array.roi
        output_roi = input_roi.grow(-context, -context)
    else:
        input_roi = output_roi.grow(context, context)

    logger.info("Total input ROI: %s, output ROI: %s", input_roi, output_roi)
    sys.stdout.flush()
    sys.stderr.flush()

    # prepare prediction dataset
    axes = ["c"] + [axis for axis in raw_array.axes if axis != "c"]

    # Scale write_size if provided, otherwise use default
    scaled_write_size = None
    if write_size is not None:
        # Scale spatial dimensions (first 3 elements), keep channel dimension (last element) unchanged
        scaled_write_size = [ws * input_shape_scaling_factor for ws in write_size[:3]] + [write_size[3]]

    ZarrArray.create_from_array_identifier(
        prediction_array_identifier,
        axes,
        output_roi,
        9,
        output_voxel_size,
        np.uint8,
        write_size=scaled_write_size,
    )

    # create gunpowder keys

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICTION")

    # assemble prediction pipeline

    # prepare data source
    pipeline = DaCapoArraySource(raw_array, raw)
    # raw: (c, d, h, w)
    pipeline += gp.Pad(raw, Coordinate((None,) * input_voxel_size.dims))
    # raw: (c, d, h, w)
    pipeline += gp.Unsqueeze([raw])
    # raw: (1, c, d, h, w)

    gt_padding = (output_size - output_roi.shape) % output_size
    prediction_roi = output_roi.grow(gt_padding)

    # predict
    pipeline += gp_torch.Predict(
        model=model,
        inputs={"x": raw},
        outputs={0: prediction},
        array_specs={
            prediction: gp.ArraySpec(
                roi=prediction_roi, voxel_size=output_voxel_size, dtype=np.float32
            )
        },
        spawn_subprocess=False,
        device=str(compute_context.device),
    )
    # raw: (1, c, d, h, w)
    # prediction: (1, [c,] d, h, w)

    # prepare writing
    pipeline += gp.Squeeze([raw, prediction])
    # raw: (c, d, h, w)
    # prediction: (c, d, h, w)
    # raw: (c, d, h, w)
    # prediction: (c, d, h, w)

    pipeline += gp.IntensityScaleShift(prediction, 255, 0)
    pipeline += gp.AsType(prediction, np.uint8)

    # write to zarr
    # pipeline += gp.ZarrWrite(
    #     {prediction: prediction_array_identifier.dataset},
    #     prediction_array_identifier.container.parent,
    #     prediction_array_identifier.container.name,
    #     dataset_dtypes={prediction: np.uint8},
    # )

    pipeline += ZarrWriteSelectChannels(
        {prediction: prediction_array_identifier.dataset},
        prediction_array_identifier.container.parent,
        prediction_array_identifier.container.name,
        # dataset_dtypes={prediction: np.uint8},
        channel_slices=slice(0, 9, 1),
    )

    # create reference batch request
    ref_request = gp.BatchRequest()
    ref_request.add(raw, input_size)
    ref_request.add(prediction, output_size)
    pipeline += gp.Scan(ref_request)

    # build pipeline and predict in complete output ROI

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())

    container = zarr.open(prediction_array_identifier.container)
    dataset = container[prediction_array_identifier.dataset]
    dataset.attrs["axes"] = (
        raw_array.axes if "c" in raw_array.axes else ["c"] + raw_array.axes
    )
