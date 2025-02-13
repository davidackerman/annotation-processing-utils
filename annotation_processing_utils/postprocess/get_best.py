import json
import yaml
import numpy as np
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from annotation_processing_utils.process.training_validation_test_roi_calculator import (
    TrainingValidationTestRoiCalculator,
)


class GetBest:
    def __init__(self, submission_yaml_path):
        with open(submission_yaml_path, "r") as stream:
            self.submission_info = yaml.safe_load(stream)
        self.yaml_name = Path(submission_yaml_path).stem
        self.runs = self.submission_info["runs"]
        self.analysis_info = self.submission_info["analysis_info"]
        (
            iterations_start,
            iterations_end,
            iterations_step,
        ) = self.submission_info["iterations"]
        self.iterations = range(iterations_start, iterations_end + 1, iterations_step)
        self.postprocessing_suffixes = self.submission_info["postprocessing_suffixes"]

    def get_combined_df(self):
        combined_df = pd.DataFrame(
            columns=[
                "run",
                "iteration",
                "crop",
                "validation_or_test",
                "f1_score",
                "tp",
                "fp",
                "fn",
                "iou",
            ]
        )
        for current_analysis_info in self.analysis_info:
            metrics_base_path = Path(current_analysis_info["metrics_base_path"])

            # get rois for this dataset
            roi_splitter = TrainingValidationTestRoiCalculator(
                current_analysis_info["training_validation_test_roi_yaml"]
            )
            roi_splitter.get_training_validation_test_rois()
            rois_dict = roi_splitter.rois_dict
            for (
                run,
                iteration,
                postprocessing_suffix,
                validation_or_test,
            ) in product(
                self.runs,
                self.iterations,
                self.postprocessing_suffixes,
                ["validation", "test"],
            ):
                for roi_name in rois_dict[validation_or_test].keys():
                    output_directory = f"{metrics_base_path}/{self.yaml_name}/{validation_or_test}/{run}/{roi_name}/iteration_{iteration}{postprocessing_suffix}_segs"
                    if not os.path.exists(f"{output_directory}/scores.json"):
                        raise Exception(f"Path {output_directory} does not exist")

                    with open(f"{output_directory}/scores.json") as f:
                        roi_dict = json.load(f)
                        new_row = {
                            "run": run,
                            "iteration": iteration,
                            "crop": roi_name,
                            "validation_or_test": validation_or_test,
                            "f1_score": roi_dict["f1_score_info"]["f1_score"],
                            "tp": roi_dict["f1_score_info"]["tp"],
                            "fp": roi_dict["f1_score_info"]["fp"],
                            "fn": roi_dict["f1_score_info"]["fn"],
                            "iou": roi_dict["f1_score_info"]["iou"],
                        }
                        new_row_df = pd.DataFrame([new_row])
                        combined_df = pd.concat(
                            [combined_df, new_row_df], ignore_index=True
                        )

        return combined_df

    def __update_combined_dict_f1_score(
        self, combined_dicts, output_directory, validation_or_test, roi_name
    ):
        combined_dict = combined_dicts[validation_or_test]
        with open(f"{output_directory}/scores.json") as f:
            roi_dict = json.load(f)
            for k in ["tp_gt_test_id_pairs", "fp_test_ids", "fn_gt_ids"]:
                del roi_dict["f1_score_info"][k]
            combined_dict[roi_name] = roi_dict
        for key in ["fp", "tp", "fn"]:
            combined_dict["combined"][key] += combined_dict[roi_name]["f1_score_info"][
                key
            ]
        combined_dict["combined"]["average_f1_score"].append(
            combined_dict[roi_name]["f1_score_info"]["f1_score"]
        )

    def f1_score(self):
        failed_paths = []
        for current_analysis_info in self.analysis_info:
            metrics_base_path = Path(current_analysis_info["metrics_base_path"])
            # get rois for this dataset
            roi_splitter = TrainingValidationTestRoiCalculator(
                current_analysis_info["training_validation_test_roi_yaml"]
            )
            roi_splitter.get_training_validation_test_rois()
            rois_dict = roi_splitter.rois_dict

            previous_best_f1_score = -1
            self.all_f1_scores_df = pd.DataFrame(
                columns=["run", "iteration", "validation_or_test", "f1_score"]
            )

            for run in self.runs:
                for iteration in self.iterations:
                    for postprocessing_suffix in self.postprocessing_suffixes:
                        combined_dicts = {
                            "validation": {
                                "combined": {
                                    "tp": 0,
                                    "fp": 0,
                                    "fn": 0,
                                    "f1_score": 0,
                                    "average_f1_score": [],
                                }
                            },
                            "test": {
                                "combined": {
                                    "tp": 0,
                                    "fp": 0,
                                    "fn": 0,
                                    "f1_score": 0,
                                    "average_f1_score": [],
                                }
                            },
                        }
                        for validation_or_test in ["validation", "test"]:
                            for roi_name in rois_dict[validation_or_test].keys():
                                output_directory = f"{metrics_base_path}/{self.yaml_name}/{validation_or_test}/{run}/{roi_name}/iteration_{iteration}{postprocessing_suffix}_segs"
                                if not os.path.exists(
                                    f"{output_directory}/scores.json"
                                ):
                                    failed_paths.append(output_directory)
                                    continue

                                self.__update_combined_dict_f1_score(
                                    combined_dicts,
                                    output_directory,
                                    validation_or_test,
                                    roi_name,
                                )

                            combined_dict = combined_dicts[validation_or_test]
                            if len(combined_dict["combined"]["average_f1_score"]) == 0:
                                continue

                            combined_dict["combined"]["f1_score"] = combined_dict[
                                "combined"
                            ]["tp"] / (
                                combined_dict["combined"]["tp"]
                                + 0.5
                                * (
                                    combined_dict["combined"]["fp"]
                                    + combined_dict["combined"]["fn"]
                                )
                            )

                            combined_dict["combined"]["average_f1_score"] = np.mean(
                                combined_dict["combined"]["average_f1_score"]
                            )

                            new_row = {
                                "run": run,
                                "iteration": iteration,
                                "validation_or_test": validation_or_test,
                                "f1_score": combined_dict["combined"]["f1_score"],
                            }

                            if self.all_f1_scores_df.empty:
                                self.all_f1_scores_df = pd.DataFrame([new_row])
                            else:
                                new_row_df = pd.DataFrame([new_row])
                                self.all_f1_scores_df = pd.concat(
                                    [self.all_f1_scores_df, new_row_df],
                                    ignore_index=True,
                                )

                        if (
                            combined_dicts["validation"]["combined"]["f1_score"]
                            >= previous_best_f1_score
                        ):
                            previous_best_f1_score = combined_dicts["validation"][
                                "combined"
                            ]["f1_score"]
                            print(
                                combined_dicts["validation"]["combined"],
                                combined_dicts["test"]["combined"],
                                combined_dicts["validation"],
                                combined_dicts["test"],
                                run,
                                iteration,
                            )
        return failed_paths

    def plot_f1_scores(
        self, validation_or_test, plot_type="line", merge_repetitions=False
    ):
        # Filter for validation rows
        df = self.all_f1_scores_df[
            self.all_f1_scores_df["validation_or_test"] == validation_or_test
        ]

        # Group by 'run' and plot each group
        if merge_repetitions:
            # Extract the grouping key by removing the last character
            df["group_key"] = df["run"].str[:-1]
        else:
            df["group_key"] = df["run"]

        plt.figure(figsize=(8, 6))
        for run, group in df.groupby("group_key"):
            if plot_type == "line":
                plt.plot(group["iteration"], group["f1_score"], label=f"Run {run}")
            elif plot_type == "histogram":
                # Compute histogram data
                # Define custom bin edges
                bin_edges = np.linspace(0, 1, 20)

                counts, _ = np.histogram(
                    group["f1_score"], bins=bin_edges, density=True
                )

                # Calculate bin centers
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Plot the histogram as lines
                plt.plot(bin_centers, counts, "-", label=f"Run {run}")

        # Add labels, legend, and title
        if plot_type == "line":
            plt.xlabel("Iteration")
            plt.ylabel("F1 Score")
        elif plot_type == "histogram":
            plt.xlabel("F1 Score")
            plt.ylabel("Frequency")

        plt.title(f"{validation_or_test.capitalize()} F1 Score per Run")

        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Runs")
        plt.tight_layout()  # Adjust layout to fit everything
        plt.grid(True)
        plt.show()
