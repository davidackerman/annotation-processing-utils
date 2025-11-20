from datetime import datetime
from pathlib import Path
import yaml
import os
import subprocess
import numpy as np

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


def dict_to_suffix(d: dict) -> str:
    parts = []
    for key, val in d.items():
        if isinstance(val, (list, tuple)):
            # join list elements as strings separated by underscores
            val_str = "_".join(str(v) for v in val)
        else:
            val_str = str(val)
        parts.append(f"{key}_{val_str}")
    return "_".join(parts)


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
                        "bsub_args": "bsub -P cellmap -q gpu_h100 -n 12 -gpu num=1",
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
    postprocessing_sweep = submission_info.get("postprocessing_sweep", None)
    failed_info = []
    #     "validation/finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.0001_bs_4__0/11/iteration_125000",
    #     "test/finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.0001_bs_4__1/08/iteration_275000",
    #     "test/finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.0002_bs_4__0/11/iteration_325000",
    # ]
    for current_analysis_info in analysis_info:
        analysis_info_name = current_analysis_info["name"]
        raw_path = current_analysis_info["raw_path"]
        gt_path = current_analysis_info["gt_path"]
        raw_min = current_analysis_info.get("raw_min", 0)
        raw_max = current_analysis_info.get("raw_max", 255)
        invert = current_analysis_info.get("invert", False)
        mask_path = current_analysis_info.get("mask_path", None)
        scale_coordinates = current_analysis_info.get(
            "scale_coordinates", np.array([1, 1, 1])
        )
        inference_base_path = Path(current_analysis_info["inference_base_path"])
        metrics_base_path = Path(current_analysis_info["metrics_base_path"])

        # get rois for this dataset
        roi_splitter = TrainingValidationTestRoiCalculator(
            current_analysis_info["training_validation_test_roi_yaml"],
            scale_coordinates=scale_coordinates,
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
                        if (
                            len(failed_info) == 0
                            or (run, roi_name, iteration) in failed_info
                        ):

                            if submission_type == "inference":
                                roi_offset = ",".join(str(o) for o in roi.offset)
                                roi_shape = ",".join(str(s) for s in roi.shape)
                                args = {
                                    "--run": run,
                                    "--raw_path": raw_path,
                                    "--inference_path": affinities_path,
                                    "--iteration": iteration,
                                    "--roi_offset": roi_offset,
                                    "--roi_shape": roi_shape,
                                    "--raw_min": raw_min,
                                    "--raw_max": raw_max,
                                    "bsub_args": f"bsub -P cellmap -q gpu_short -n 4 -gpu num=1 -J inference-{run}-{roi_name}-iter{iteration}",
                                    "log_path": log_path,
                                }

                                if invert:
                                    args["--invert"] = invert

                                arguments_per_submission.append(
                                    bsub_formatter(
                                        args,
                                        submission_type,
                                    )
                                )
                            for current_postprocessing_sweep in postprocessing_sweep:
                                current_log_path = Path(
                                    f"{log_path}_{dict_to_suffix(current_postprocessing_sweep)}"
                                )
                                base_segmentation_path = (
                                    inference_base_path
                                    / "processed"
                                    / yaml_name
                                    / validation_or_test
                                    / run
                                    / roi_name
                                    / f"iteration_{iteration}_{dict_to_suffix(current_postprocessing_sweep)}"
                                )
                                if submission_type == "mws":
                                    args = {
                                        "--affinities_path": affinities_path,
                                        "--segmentation_path": base_segmentation_path,
                                        "bsub_args": f"bsub -P cellmap -n 1 -J mws-{base_segmentation_path}",
                                        "log_path": current_log_path,
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

                                    if (
                                        "adjacent_edge_bias"
                                        in current_postprocessing_sweep
                                    ):
                                        args["--adjacent_edge_bias"] = (
                                            current_postprocessing_sweep[
                                                "adjacent_edge_bias"
                                            ]
                                        )
                                    if "lr_biases" in current_postprocessing_sweep:
                                        args["--lr_biases"] = " ".join(
                                            str(v)
                                            for v in current_postprocessing_sweep[
                                                "lr_biases"
                                            ]
                                        )
                                    if "filter_val" in current_postprocessing_sweep:
                                        args["--filter_val"] = (
                                            current_postprocessing_sweep["filter_val"]
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
                                        / f"iteration_{iteration}_{dict_to_suffix(current_postprocessing_sweep)}"
                                    )

                                    test_path = (
                                        base_segmentation_path.absolute().as_posix()
                                        + f"/s0"
                                    )

                                    metrics_submission_dict = {
                                        "--gt_path": gt_path,
                                        "--test_path": test_path,
                                        "--metrics_path": metrics_path,
                                        "--num_workers": 10,
                                        "bsub_args": f"bsub -P cellmap -n 10 -J metrics-{run}-{roi_name}-iter{iteration}",
                                        "log_path": current_log_path,
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


def parse_failure_logs(failure_log_paths):
    """Parse failure log files to extract failed job indices and commands"""
    failed_jobs = {}  # job_index -> command

    for log_path in failure_log_paths:
        log_path = Path(log_path)
        if not log_path.exists():
            print(f"Warning: Failure log {log_path} does not exist")
            continue

        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                # Handle new job array format: "Job X failed - reason: command"
                if line.startswith("Job ") and " failed - " in line and ": " in line:
                    # Extract job number and command from: "Job X failed - reason: command"
                    parts = line.split(" failed - ")
                    if len(parts) == 2:
                        job_num = parts[0].replace("Job ", "")
                        reason_command = parts[1]
                        if ": " in reason_command:
                            command = reason_command.split(": ", 1)[1]
                            try:
                                failed_jobs[int(job_num)] = command
                            except ValueError:
                                print(
                                    f"Warning: Could not parse job number from: {job_num}"
                                )
                # Handle old format: "Job X failed with exit code Y: command"
                elif line.startswith("Job ") and " failed with exit code " in line:
                    parts = line.split(" failed with exit code ")
                    if len(parts) == 2:
                        job_num = parts[0].replace("Job ", "")
                        command_part = parts[1].split(": ", 1)
                        if len(command_part) == 2:
                            command = command_part[1]
                            try:
                                failed_jobs[int(job_num)] = command
                            except ValueError:
                                print(
                                    f"Warning: Could not parse job number from: {job_num}"
                                )
                # Handle termination format: "Job X was killed/terminated: command"
                elif line.startswith("Job ") and " was killed/terminated: " in line:
                    parts = line.split(" was killed/terminated: ")
                    if len(parts) == 2:
                        job_num = parts[0].replace("Job ", "")
                        command = parts[1]
                        try:
                            failed_jobs[int(job_num)] = command
                        except ValueError:
                            print(
                                f"Warning: Could not parse job number from: {job_num}"
                            )

    return failed_jobs


def get_failed_jobs_from_array_dir(job_array_dir):
    """Extract failed jobs from a job array directory structure"""
    job_array_dir = Path(job_array_dir)
    commands_file = job_array_dir / "commands.txt"
    log_paths_file = job_array_dir / "log_paths.txt"
    failures_log = job_array_dir / "failures.log"

    if not commands_file.exists():
        raise ValueError(
            f"Commands file {commands_file} not found in job array directory"
        )

    if not failures_log.exists():
        print(
            f"Warning: No failures.log found in {job_array_dir}. No failed jobs to resubmit."
        )
        return []

    # Parse failed job indices from the failures log
    failed_jobs = parse_failure_logs([failures_log])

    if not failed_jobs:
        print("No failed jobs found in failures.log")
        return []

    # Read commands and log paths
    with open(commands_file, "r") as f:
        commands = f.readlines()

    log_paths = []
    if log_paths_file.exists():
        with open(log_paths_file, "r") as f:
            log_paths = f.readlines()

    failed_commands = []

    # Extract bsub options from the original script
    script_file = job_array_dir / "submit_array.sh"
    bsub_opts_str = ""

    if script_file.exists():
        with open(script_file, "r") as f:
            script_content = f.read()
            # Extract bsub options from the script (skip -J, -o, -e lines)
            lines = script_content.split("\n")
            bsub_opts = []
            for line in lines:
                line = line.strip()
                if (
                    line.startswith("#BSUB")
                    and not line.startswith("#BSUB -J")
                    and not line.startswith("#BSUB -o")
                    and not line.startswith("#BSUB -e")
                ):
                    opt = line.replace("#BSUB ", "").strip()
                    if opt:  # Only add non-empty options
                        bsub_opts.append(opt)

            bsub_opts_str = " ".join(bsub_opts)

    # Reconstruct original bsub commands for failed jobs
    for job_idx in sorted(failed_jobs.keys()):
        if 1 <= job_idx <= len(commands):
            command = commands[job_idx - 1].strip()  # Convert to 0-indexed

            # Get original log path if available
            log_path = ""
            if job_idx <= len(log_paths):
                log_path = log_paths[job_idx - 1].strip()

            # Reconstruct the original bsub command format
            if log_path:
                full_command = (
                    f"bsub {bsub_opts_str} -o {log_path}.o -e {log_path}.e {command}"
                )
            else:
                full_command = f"bsub {bsub_opts_str} {command}"

            failed_commands.append(full_command)
            print(f"Will resubmit job {job_idx}: {command}")
        else:
            print(f"Warning: Job index {job_idx} is out of range (1-{len(commands)})")

    return failed_commands


def generic_submitter(submission_type):
    parser = argparse.ArgumentParser(description=f"Submit {submission_type}")
    parser.add_argument(
        "--no-job-array",
        action="store_true",
        help="Disable job array submission (default: submit as job array)",
        default=False,
    )
    parser.add_argument(
        "submission_info_path", type=str, help="Path to the submission info yaml"
    )
    parser.add_argument(
        "--resubmit-failures",
        type=str,
        help="Comma-separated list of failure log files to resubmit failed jobs from",
    )
    parser.add_argument(
        "--resubmit-from-array",
        type=str,
        help="Path to job array directory to resubmit failed jobs from",
    )
    parser.add_argument(
        "--auto-resubmit",
        action="store_true",
        help="Automatically resubmit failed jobs until all succeed",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for auto-resubmit (default: 3)",
    )
    args = parser.parse_args()

    if args.resubmit_failures and args.resubmit_from_array:
        print(
            "Error: Cannot specify both --resubmit-failures and --resubmit-from-array"
        )
        return

    if args.resubmit_failures:
        # Parse failure logs to get failed commands
        failure_log_paths = [path.strip() for path in args.resubmit_failures.split(",")]
        failed_jobs = parse_failure_logs(failure_log_paths)

        if not failed_jobs:
            print("No failed jobs found in the provided failure logs")
            return

        print(f"Found {len(failed_jobs)} failed jobs to resubmit")
        # For failure logs, we need to reconstruct the commands from the log entries
        # This is more complex as we need to parse the actual command from the log
        arguments_per_submission = list(failed_jobs.values())

    elif args.resubmit_from_array:
        # Get failed jobs from job array directory
        try:
            arguments_per_submission = get_failed_jobs_from_array_dir(
                args.resubmit_from_array
            )
            if not arguments_per_submission:
                print("No failed jobs found in the job array directory")
                return
            print(
                f"Found {len(arguments_per_submission)} failed jobs to resubmit from {args.resubmit_from_array}"
            )
        except Exception as e:
            print(f"Error processing job array directory: {e}")
            return
    else:
        # Normal submission - get all jobs
        arguments_per_submission = get_arguments_per_submission(
            args.submission_info_path, submission_type
        )

    if not arguments_per_submission:
        print(f"No jobs to submit for {submission_type}")
        return

    if args.no_job_array:
        # Submit jobs individually
        for cmd in arguments_per_submission:
            print(f"Submitting job: {cmd}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"Error submitting job: {cmd}")
            time.sleep(1)  # brief pause between submissions
    else:
        # Create job array submission with auto-resubmit if requested
        submit_job_array_with_retries(
            arguments_per_submission,
            submission_type,
            args.submission_info_path,
            auto_resubmit=args.auto_resubmit,
            max_retries=args.max_retries,
        )


def submit_inference():
    generic_submitter("inference")


def submit_mws():
    generic_submitter("mws")


def submit_metrics():
    generic_submitter("metrics")


def submit_dacapo_train():
    generic_submitter("dacapo-train")


def submit_job_array_with_retries(
    arguments_per_submission,
    submission_type,
    submission_info_path,
    auto_resubmit=False,
    max_retries=3,
):
    """Submit job array with optional automatic resubmission of failed jobs"""
    retry_count = 0
    remaining_jobs = arguments_per_submission.copy()

    while retry_count <= max_retries and remaining_jobs:
        if retry_count == 0:
            print(f"Initial submission: {len(remaining_jobs)} jobs")
        else:
            print(
                f"Retry {retry_count}/{max_retries}: {len(remaining_jobs)} failed jobs"
            )

        # Submit the current batch of jobs
        job_array_dir = submit_job_array(
            remaining_jobs, submission_type, submission_info_path
        )

        if not job_array_dir:
            print("Job submission failed")
            break

        # Check if auto-resubmit is enabled
        if not auto_resubmit:
            print("Auto-resubmit disabled. Check results manually.")
            break

        # Check for failures and prepare for next retry if needed
        failures_log = job_array_dir / "failures.log"
        if not failures_log.exists():
            print("All jobs completed successfully!")
            break

        # Get failed jobs for resubmission
        try:
            failed_commands = get_failed_jobs_from_array_dir(job_array_dir)
            if not failed_commands:
                print("All jobs completed successfully!")
                break

            if retry_count >= max_retries:
                print(
                    f"Maximum retries ({max_retries}) reached. {len(failed_commands)} jobs still failing."
                )
                print(f"Check {job_array_dir}/failures.log for details")
                break

            remaining_jobs = failed_commands
            retry_count += 1

            # Add delay between retries
            if retry_count <= max_retries:
                print(f"Waiting 30 seconds before retry {retry_count}...")
                time.sleep(30)

        except Exception as e:
            print(f"Error processing failures for retry: {e}")
            break

    if retry_count > max_retries:
        print(
            f"Final result: {len(remaining_jobs)} jobs failed after {max_retries} retries"
        )
    elif not remaining_jobs:
        print("All jobs completed successfully!")


def submit_job_array(arguments_per_submission, submission_type, submission_info_path):
    """Submit jobs as an LSF job array with failure tracking"""
    username = getpass.getuser()
    yaml_name = Path(submission_info_path).stem
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S")

    # Create a directory for job array files
    job_array_dir = Path(
        f"/nrs/cellmap/{username}/job_arrays/{yaml_name}_{submission_type}_{timestamp}"
    )
    job_array_dir.mkdir(parents=True, exist_ok=True)

    # Write job commands and their original log paths to files
    commands_file = job_array_dir / "commands.txt"
    log_paths_file = job_array_dir / "log_paths.txt"

    with open(commands_file, "w") as f, open(log_paths_file, "w") as log_f:
        for i, cmd in enumerate(arguments_per_submission, 1):
            # Extract the original log path from the command
            parts = cmd.split(" -o ")
            original_log_path = ""
            if len(parts) == 2:
                log_part = parts[1].split(" -e ")[0]
                original_log_path = log_part.replace(".o", "")
            log_f.write(f"{original_log_path}\n")

            # Extract just the actual command part (everything after 'run-')
            if " run-" in cmd:
                actual_command = "run-" + cmd.split(" run-", 1)[1]
            else:
                actual_command = cmd
            f.write(f"{actual_command}\n")

    # Create the job array submission script
    script_file = job_array_dir / "submit_array.sh"
    num_jobs = len(arguments_per_submission)

    # Extract bsub options from the first command (assuming they're consistent)
    first_cmd = arguments_per_submission[0]
    bsub_parts = first_cmd.split(" run-")[0]

    # Parse bsub options
    bsub_options = []
    parts = bsub_parts.split()
    i = 1  # Skip 'bsub'

    while i < len(parts):
        if parts[i].startswith("-"):
            if parts[i] in ["-o", "-e", "-J"]:  # Skip original options
                i += 2
                continue
            elif parts[i] in [
                "-P",
                "-q",
                "-n",
                "-gpu",
            ]:
                if i + 1 < len(parts):
                    bsub_options.extend([parts[i], parts[i + 1]])
                    i += 2
                else:
                    bsub_options.append(parts[i])
                    i += 1
            else:
                bsub_options.append(parts[i])
                i += 1
        else:
            i += 1

    bsub_opts_str = " ".join(bsub_options)

    script_content = f"""#!/bin/bash
#BSUB -J {submission_type}_array[1-{num_jobs}]
#BSUB {bsub_opts_str}
#BSUB -o {job_array_dir}/job_%I.o
#BSUB -e {job_array_dir}/job_%I.e

# Get the command for this job
COMMAND=$(sed -n "${{LSB_JOBINDEX}}p" {commands_file})

# Execute the command - LSF will handle output to job_%I.o and job_%I.e
eval $COMMAND
"""

    with open(script_file, "w") as f:
        f.write(script_content)

    # Make the script executable
    os.chmod(script_file, 0o755)

    # Submit the job array
    submit_cmd = f"bsub -P cellmap < {script_file}"
    print(f"Submitting job array with {num_jobs} jobs for {submission_type}")
    print(f"Job array directory: {job_array_dir}")

    result = subprocess.run(submit_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split("<")[1].split(">")[0]
        print(f"Job array submitted successfully: {result.stdout.strip()}")
        print(f"Monitor with: bjobs {job_id}")
        print(f"Job array directory: {job_array_dir}")

        # Wait for completion and analyze results
        wait_for_completion_and_analyze(job_array_dir, num_jobs, commands_file, job_id)

        return job_array_dir
    else:
        print(f"Error submitting job array: {result.stderr.strip()}")
        return None


def wait_for_completion_and_analyze(job_array_dir, num_jobs, commands_file, job_id):
    """Wait for job array completion and analyze results"""
    job_array_dir = Path(job_array_dir)

    print(f"Waiting for job array {job_id} to complete...")

    # Wait for the job to finish using bjobs
    while True:
        try:
            result = subprocess.run(
                ["bjobs", job_id], capture_output=True, text=True, check=True
            )
            # If bjobs succeeds, the job is still in the system (running, pending, etc.)
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Header + at least one job line
                status_line = lines[1]  # First job line after header
                status = status_line.split()[2]  # Status is the 3rd column

                # Count completed jobs by checking individual log files
                completed_count = 0
                for i in range(1, num_jobs + 1):
                    job_log = job_array_dir / f"job_{i}.o"
                    if job_log.exists():
                        try:
                            with open(job_log, "r") as f:
                                content = f.read()
                            # Check if this job has completed (successfully or with error)
                            if (
                                "Successfully completed" in content
                                or "Exited" in content
                            ):
                                completed_count += 1
                        except:
                            # If we can't read the file, skip it
                            pass

                print(
                    f"Job {job_id} status: {status} | Progress: {completed_count}/{num_jobs} jobs completed ({completed_count/num_jobs*100:.1f}%)"
                )
            if completed_count >= num_jobs or "is not found" in result.stderr:
                print(f"All {num_jobs} jobs have completed.")
                break
            time.sleep(30)  # Check every 30 seconds

        except subprocess.CalledProcessError:
            # bjobs returns non-zero when job is not found (completed/failed)
            print(f"Job array {job_id} completed. Analyzing results...")
            break

    # Read commands for reference
    with open(commands_file, "r") as f:
        commands = f.readlines()

    # Create status tracking directory
    status_dir = job_array_dir / "status"
    status_dir.mkdir(exist_ok=True)

    # Check each individual job log
    successes = []
    failures = []

    for i in range(1, num_jobs + 1):
        job_log = job_array_dir / f"job_{i}.o"
        command = commands[i - 1].strip()

        if job_log.exists():
            with open(job_log, "r") as f:
                content = f.read()

            if "Exited" in content:
                failures.append(f"Job {i} failed - job exited with error: {command}")
                with open(status_dir / f"job_{i}.status", "w") as f:
                    f.write("FAILED - Job exited with error")
            elif "successfully completed" in content.lower():
                successes.append(f"Job {i} completed successfully")
                with open(status_dir / f"job_{i}.status", "w") as f:
                    f.write("SUCCESS - Found 'Successfully completed'")
            else:
                failures.append(
                    f"Job {i} failed - no completion status found: {command}"
                )
                with open(status_dir / f"job_{i}.status", "w") as f:
                    f.write("FAILED - No completion status found")
        else:
            failures.append(f"Job {i} failed - log file not found: {command}")
            with open(status_dir / f"job_{i}.status", "w") as f:
                f.write("FAILED - Log file not found")

    # Write results
    if successes:
        with open(job_array_dir / "success.log", "w") as f:
            f.write("\n".join(successes) + "\n")

    if failures:
        with open(job_array_dir / "failures.log", "w") as f:
            f.write("\n".join(failures) + "\n")

    # Summary
    summary = f"""Job array analysis complete:
  Total jobs: {num_jobs}
  Successful: {len(successes)}
  Failed: {len(failures)}
  Check failures: {job_array_dir}/failures.log
  Check successes: {job_array_dir}/success.log
"""

    print(summary)
    with open(status_dir / "summary.txt", "w") as f:
        f.write(summary)
