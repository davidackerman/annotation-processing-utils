import json
import yaml
import numpy as np
from pathlib import Path

from annotation_processing_utils.processing.training_validation_test_roi_calculator import (
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
        for current_analysis_info in self.analysis_info:
            metrics_base_path = Path(current_analysis_info["metrics_base_path"])

            # get rois for this dataset
            roi_splitter = TrainingValidationTestRoiCalculator(
                current_analysis_info["training_validation_test_roi_yaml"]
            )
            roi_splitter.get_training_validation_test_rois()

            for current_analysis_info in self.analysis_info:
                # get rois for this dataset
                roi_splitter = TrainingValidationTestRoiCalculator(
                    current_analysis_info["training_validation_test_roi_yaml"]
                )
                roi_splitter.get_training_validation_test_rois()
                rois_dict = roi_splitter.rois_dict

                previous_best_f1_score = -1
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
                                    self.__update_combined_dict_f1_score(
                                        combined_dicts,
                                        output_directory,
                                        validation_or_test,
                                        roi_name,
                                    )

                                combined_dict = combined_dicts[validation_or_test]
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
                                    combined_dicts["test"],
                                    run,
                                    iteration,
                                )
