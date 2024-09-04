import os
from datetime import datetime
from pathlib import Path
import yaml

# NOTE: need the following import torch here - even though unused - otherwise get the following error:
# Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
# 	Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
# This happens if import torch happens after importing TrainingValidationTestRoiCalculator
import torch
from annotation_processing_utils.processing.training_validation_test_roi_calculator import (
    TrainingValidationTestRoiCalculator,
)
import getpass
import argparse


def get_arguments_per_submission(submision_yaml_path, submission_type):
    username = getpass.getuser()
    with open(submision_yaml_path, "r") as stream:
        submission_info = yaml.safe_load(stream)
    yaml_name = Path(submision_yaml_path).stem

    runs = submission_info["runs"]
    analysis_info = submission_info["analysis_info"]
    iterations_start, iterations_end, iterations_step = submission_info["iterations"]
    postprocessing_suffixes = submission_info["postprocessing_suffixes"]
    arguments_per_submission = []

    timestamp = datetime.now().strftime(f"%Y%m%d/%H%M%S/")
    log_base_path = f"/nrs/cellmap/{username}/logs/{yaml_name}/{timestamp}"
    for current_analysis_info in analysis_info:
        analysis_info_name = current_analysis_info["name"]
        raw_path = current_analysis_info["raw_path"]
        gt_path = current_analysis_info["gt_path"]
        mask_path = current_analysis_info.get("mask_path", None)
        inference_base_path = Path(current_analysis_info["inference_base_path"])
        metrics_base_path = Path(current_analysis_info["metrics_base_path"])

        # get rois for this dataset
        roi_splitter = TrainingValidationTestRoiCalculator(
            current_analysis_info["training_validation_test_roi_yaml"]
        )
        roi_splitter.get_training_validation_test_rois()
        rois_dict = roi_splitter.rois_dict
        for validation_or_test in ["validation", "test"]:
            for roi_name, roi in rois_dict[validation_or_test].items():
                for run in runs:
                    for iteration in range(
                        iterations_start, iterations_end + 1, iterations_step
                    ):
                        log_path = Path(
                            f"{log_base_path}/{analysis_info_name}/{submission_type}/{run}/{validation_or_test}/{roi_name}/iteration_{iteration}"
                        )

                        affinities_path = (
                            inference_base_path
                            / "predictions"
                            / yaml_name
                            / validation_or_test
                            / run
                            / roi_name
                            / f"iteration_{iteration}"
                        )
                        if (
                            run
                            == "finetuned_3d_lsdaffs_weight_ratio_0.50_jrc_22ak351-leaf-2l_plasmodesmata_pseudorandom_training_centers_unet_default_v2_no_dataset_predictor_node_lr_5E-5__2"
                            and roi_name == "04"
                            and iteration == 35000
                        ) or (
                            run
                            == "finetuned_3d_lsdaffs_weight_ratio_0.50_jrc_22ak351-leaf-2l_plasmodesmata_pseudorandom_training_centers_unet_default_v2_no_dataset_predictor_node_lr_5E-5__2"
                            and roi_name == "01"
                            and iteration == 180000
                        ):
                            if submission_type == "inference":
                                roi_offset = ",".join(str(o) for o in roi.offset)
                                roi_shape = ",".join(str(s) for s in roi.shape)
                                arguments_per_submission.append(
                                    bsub_formatter(
                                        {
                                            "--run": run,
                                            "--raw_path": raw_path,
                                            "--inference_path": affinities_path,
                                            "--iteration": iteration,
                                            "--roi_offset": roi_offset,
                                            "--roi_shape": roi_shape,
                                            "bsub_args": "bsub -P cellmap -q gpu_tesla -n 4 -gpu num=1",
                                            "log_path": log_path,
                                        },
                                        submission_type,
                                    )
                                )

                            for postprocessing_suffix in postprocessing_suffixes:
                                base_segmentation_path = (
                                    inference_base_path
                                    / "processed"
                                    / yaml_name
                                    / validation_or_test
                                    / run
                                    / roi_name
                                    / f"iteration_{iteration}"
                                )

                                if submission_type == "mws":
                                    arguments_per_submission.append(
                                        bsub_formatter(
                                            {
                                                "--affinities_path": affinities_path,
                                                "--segmentation_path": base_segmentation_path,
                                                "bsub_args": "bsub -P cellmap -n 48",
                                                "log_path": log_path,
                                            },
                                            submission_type,
                                        )
                                    )
                                elif submission_type == "metrics":
                                    metrics_path = (
                                        metrics_base_path
                                        / yaml_name
                                        / validation_or_test
                                        / run
                                        / roi_name
                                        / f"iteration_{iteration}{postprocessing_suffix}_segs"
                                    )
                                    test_path = (
                                        base_segmentation_path.absolute().as_posix()
                                        + f"{postprocessing_suffix}_segs"
                                    )

                                    arguments_per_submission.append(
                                        bsub_formatter(
                                            {
                                                "--gt_path": gt_path,
                                                "--test_path": test_path,
                                                "--mask_path": mask_path,
                                                "--metrics_path": metrics_path,
                                                "--num_workers": 10,
                                                "bsub_args": "bsub -P cellmap -n 10",
                                                "log_path": log_path,
                                            },
                                            submission_type,
                                        )
                                    )

    return arguments_per_submission


def bsub_formatter(submission_arguments_dict, submission_type):
    bsub_args = submission_arguments_dict.pop("bsub_args")

    log_path = submission_arguments_dict.pop("log_path")
    os.makedirs(log_path.parents[0], exist_ok=True)
    log_path = log_path.as_posix()

    submission_args = " ".join(
        [f"{k} {v}" for k, v in submission_arguments_dict.items()]
    )
    submission_string = f"{bsub_args} -o {log_path}.o -e {log_path}.e run-{submission_type} {submission_args}"
    return submission_string


def generic_submitter(submission_type):
    parser = argparse.ArgumentParser(description=f"Submit {submission_type}")
    parser.add_argument(
        "submission_info_path", type=str, help="Path to the submission info yaml"
    )
    args = parser.parse_args()

    arguments_per_submission = get_arguments_per_submission(
        args.submission_info_path, submission_type
    )

    for submission_arguments in arguments_per_submission:
        os.system(submission_arguments)


def submit_inference():
    generic_submitter("inference")


def submit_mws():
    generic_submitter("mws")


def submit_metrics():
    generic_submitter("metrics")
