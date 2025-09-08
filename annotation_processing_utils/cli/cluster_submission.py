from datetime import datetime
from pathlib import Path
import yaml
import os
import subprocess

import json
import shlex
import time
# NOTE: need the following import torch here - even though unused - otherwise get the following error:
# Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
# 	Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
# This happens if import torch happens after importing TrainingValidationTestRoiCalculator
from annotation_processing_utils.process.training_validation_test_roi_calculator import (
    TrainingValidationTestRoiCalculator,
)
import getpass
import argparse


def get_arguments_per_submission(submision_yaml_path, submission_type):
    username = getpass.getuser()
    with open(submision_yaml_path, "r") as stream:
        submission_info = yaml.safe_load(stream)
    yaml_name = Path(submision_yaml_path).stem

    timestamp = datetime.now().strftime(f"%Y%m%d/%H%M%S/")
    arguments_per_submission = []

    runs = submission_info["runs"]

    log_base_path = f"/nrs/cellmap/{username}/logs/{yaml_name}"
    if submission_type == "dacapo-train":
        for run in runs:
            log_path = Path(f"{log_base_path}/{submission_type}/{run}/{timestamp}")
            arguments_per_submission.append(
                bsub_formatter(
                    {
                        "--run": run,
                        "bsub_args": "bsub -P cellmap -q gpu_short -n 12 -gpu num=1",
                        "log_path": log_path,
                    },
                    submission_type,
                )
            )
        return arguments_per_submission

    log_base_path = f"/nrs/cellmap/{username}/logs/{yaml_name}/{timestamp}"
    analysis_info = submission_info["analysis_info"]
    iterations_start, iterations_end, iterations_step = submission_info["iterations"]
    postprocessing = submission_info["postprocessing"]
    failed_info = [('finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3r_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__1', '07', 150000), ('finetuned_3d_lsdaffs_weight_ratio_0.5_combined_healthy_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__0', '03', 150000), ('finetuned_3d_lsdaffs_weight_ratio_0.5_combined_healthy_and_gall_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__1', '07', 375000), ('finetuned_3d_lsdaffs_weight_ratio_0.5_combined_healthy_and_gall_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__0', '03', 200000), ('finetuned_3d_lsdaffs_weight_ratio_0.5_combined_healthy_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__1', '05', 225000)]

    #     "validation/finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.0001_bs_4__0/11/iteration_125000",
    #     "test/finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.0001_bs_4__1/08/iteration_275000",
    #     "test/finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.0002_bs_4__0/11/iteration_325000",
    # ]
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
        for validation_or_test in ["test", "validation"]:
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
                        if len(failed_info)==0 or (run, roi_name, iteration) in failed_info:

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
                                            "bsub_args": "bsub -P cellmap -q gpu_short -n 4 -gpu num=1",
                                            "log_path": log_path,
                                        },
                                        submission_type,
                                    )
                                )

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
                                args = {
                                    "--affinities_path": affinities_path,
                                    "--segmentation_path": base_segmentation_path,
                                    "bsub_args": f"bsub -P cellmap -n 1 -J mws-{base_segmentation_path}",
                                    "log_path": log_path,
                                }

                                # Add conditionally if present in postprocessing
                                if "minimum_volume_nm_3" in postprocessing:
                                    args["--minimum_volume_nm_3"] = postprocessing[
                                        "minimum_volume_nm_3"
                                    ]

                                if "maximum_volume_nm_3" in postprocessing:
                                    args["--maximum_volume_nm_3"] = postprocessing[
                                        "maximum_volume_nm_3"
                                    ]

                                if "mask_config" in postprocessing:
                                    args["--mask_config"] = shlex.quote(
                                        json.dumps(postprocessing["mask_config"])
                                    )

                                arguments_per_submission.append(
                                    bsub_formatter(
                                        args,
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
                                    / f"iteration_{iteration}"
                                )

                                test_path = (
                                    base_segmentation_path.absolute().as_posix() + f"/s0"
                                )

                                metrics_submission_dict = {
                                    "--gt_path": gt_path,
                                    "--test_path": test_path,
                                    "--metrics_path": metrics_path,
                                    "--num_workers": 10,
                                    "bsub_args": "bsub -P cellmap -n 10",
                                    "log_path": log_path,
                                }

                                if mask_path:
                                    metrics_submission_dict["mask_path"] = mask_path
                                arguments_per_submission.append(
                                    bsub_formatter(
                                        metrics_submission_dict,
                                        submission_type,
                                    )
                                )

    return arguments_per_submission


def get_bjobs_matching(status, pattern=None):
    try:
        # Run bjobs with the -r option to get only running jobs
        result = subprocess.run(
            ["bjobs", "-w", status], capture_output=True, text=True, check=True
        )
        # Optionally filter lines that match a given pattern
        if pattern:
            jobs = [line for line in result.stdout.splitlines() if pattern in line]
        else:
            jobs = result.stdout.splitlines()
        return jobs
    except subprocess.CalledProcessError as e:
        print("Error running bjobs:", e)
        return []


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
    # if submission_type == "mws":
    #     submission_count = 0
    #     total_submissions = len(arguments_per_submission)
    #     while True:
    #         time.sleep(0.01)
    #         if len(arguments_per_submission) == 0:
    #             print("Finsihed mws submissions")
    #             break
    #         running_jobs = get_bjobs_matching("-r", "mws-mongo_")
    #         pending_jobs = get_bjobs_matching("-p", "mws-mongo_")
    #         if len(running_jobs) + len(pending_jobs) <= 20:
    #             submission_count += 1
    #             print(f"Submitting mws job {submission_count}/{total_submissions}")
    #             current_arguments = arguments_per_submission.pop(0)
    #             os.system(current_arguments)

    # else:
    for submission_arguments in arguments_per_submission:
        os.system(submission_arguments)


def submit_inference():
    generic_submitter("inference")


def submit_mws():
    generic_submitter("mws")


def submit_metrics():
    generic_submitter("metrics")


def submit_dacapo_train():
    generic_submitter("dacapo-train")
