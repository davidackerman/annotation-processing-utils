# annotation-processing-utils

End-to-end utilities for the CellMap annotation → training → inference → segmentation → metrics pipeline.

Annotations are drawn in neuroglancer (as line annotations), exported as CSVs, converted to cylindrical ground-truth and intersection masks, used to train DaCapo models, then run through inference, mutex watershed, and F1/IoU evaluation against a ground-truth segmentation.

See [docs/architecture_diagram.md](docs/architecture_diagram.md) for a full pipeline diagram.

## Layout

```
annotation_processing_utils/
├── process/                # annotation -> ROI split, cylindrical masks, training points
│   ├── training_validation_test_roi_calculator.py
│   └── cylindrical_annotations.py
├── postprocess/            # inference, MWS, metrics, best-config picker
│   ├── inference.py
│   ├── mws.py / rusty_mws_processor.py
│   ├── metrics.py
│   └── get_best.py
├── cli/                    # CLI entry points
│   ├── runners.py          # run-* (local execution)
│   └── cluster_submission.py  # submit-* (LSF bsub)
└── utils/                  # dacapo helpers, zarr helpers, bresenham3D, ...
```

## CLI entry points

Pip-installed as console scripts (see `pyproject.toml`).

| Command              | Purpose                                                 |
|----------------------|---------------------------------------------------------|
| `submit-dacapo-train`| Submit DaCapo training jobs to LSF                      |
| `submit-inference`   | Submit inference jobs                                   |
| `submit-mws`         | Submit mutex watershed segmentation                     |
| `submit-rusty-mws`   | Submit rusty_mws variant                                |
| `submit-metrics`     | Submit F1/IoU evaluation against ground truth           |
| `run-dacapo-train`   | Local execution (used inside the submitted job)         |
| `run-inference`      |                                                         |
| `run-mws`            |                                                         |
| `run-rusty-mws`      |                                                         |
| `run-metrics`        |                                                         |

Each `submit-*` parses a YAML config and fans out jobs over runs × iterations × ROIs × parameter sweeps, with automatic retry of failed array elements.

## Typical workflow

1. **Annotate** lines in neuroglancer; export CSVs of start/end coordinates.
2. **Prepare ROIs and masks** via `CylindricalAnnotations` — reads a `training_validation_test_roi_info.yaml`, splits ROIs into training / validation / test, writes:
   - `annotations_as_cylinders.zarr` (per-annotation cylinder fills)
   - `annotation_intersection_masks.zarr` (where multiple cylinders overlap → masked out)
   - `training_points.zarr` (per-voxel training point locations)
   - a neuroglancer URL to inspect kept/removed annotations and ROIs
3. **Train** via `submit-dacapo-train` (one DaCapo run per repetition × datasplit).
4. **Inference** via `submit-inference` (chunked predict per iteration).
5. **Segment** via `submit-mws` or `submit-rusty-mws`, sweeping postprocessing params.
6. **Score** via `submit-metrics` (blockwise overlap → linear-sum-assignment → F1/IoU).
7. **Pick best config** with `GetBest` (`postprocess/get_best.py`).

## Reproducing pre-Nov-2025 training-point selection

The Nov 2025 commits changed `get_all_training_points` in three ways that can perturb the training-point set:

- Added an `is_valid_center` check requiring each annotation center to lie inside exactly one training ROI.
- Widened the validation/test exclusion distance from `longest_box_diagonal` (~64 vx) to `ceil(sqrt(3·(36+shift)²)+1)` (~95 vx with shift=18).
- Flipped the random-shift jitter upper bound from exclusive to inclusive.
- Replaced pandas exact-equality CSV dedup with an `allclose` dedup (rtol=1e-5).

Set `use_legacy_jan2025_selection: true` in your ROI YAML to roll all four back for a Jan 2025 ↔ current comparison. See [training_validation_test_roi_calculator.py](annotation_processing_utils/process/training_validation_test_roi_calculator.py) for the flag's docstring.

## Environment notes

dacapo-ml 0.3.1.dev347+ge871a374.d20240904 requires funlib-persistence>=0.3.0, and we use funlib-persistence==0.3.0 (annotation_processing_utils) (submit-inference)
for mutex watershed, use annotation_processing_utils_mws (submit-mws)
for metrics, use annotation_processing_utils, submit-metrics
