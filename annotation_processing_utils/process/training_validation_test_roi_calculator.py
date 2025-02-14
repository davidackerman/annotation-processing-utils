from collections import namedtuple
import os
import shutil
from funlib.geometry import Roi
import numpy as np
import pandas as pd

import neuroglancer
from neuroglancer.url_state import parse_url
import pandas as pd
import yaml
import os
from neuroglancer.write_annotations import AnnotationWriter
from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer import AnnotationPropertySpec
import yaml
import warnings


class RoiToSplitInVoxels:
    def __init__(
        self, roi_start, roi_end, resolution, split_dimension=None, keep_if_empty=False
    ):
        dims = ["z", "y", "x"]
        roi = Roi(roi_start, roi_end - roi_start)
        roi = roi.snap_to_grid((resolution, resolution, resolution), mode="shrink")
        roi /= resolution  # do this to keep things in voxels for splitting along exact voxel
        self.resolution = resolution
        self.roi = roi
        self.split_dimension = (
            dims.index(split_dimension) if split_dimension in dims else split_dimension
        )
        self.keep_if_empty = keep_if_empty


class TrainingValidationTestRoiCalculator:
    # currently predfined rois are only for validation/test
    def __init__(
        self,
        training_validation_test_roi_info_yaml,
    ):
        self.training_validation_test_roi_info_yaml = (
            training_validation_test_roi_info_yaml
        )
        self.rois_dict = {"training": {}, "validation": {}, "test": {}}
        self.__extract_training_validation_test_info(
            training_validation_test_roi_info_yaml
        )
        neuroglancer.set_server_bind_address("0.0.0.0")

    def __extract_training_validation_test_info(
        self, training_validation_test_info_yaml
    ):
        self.annotation_csvs = []
        self.annotations_to_split_automatically = {}
        self.rois_to_split_in_voxels = {}
        for roi_type in ["training_validation_test", "validation_test"]:
            self.annotations_to_split_automatically[roi_type] = []
            self.rois_to_split_in_voxels[roi_type] = {}

        with open(training_validation_test_info_yaml, "r") as stream:
            info = yaml.safe_load(stream)

        self.resolution = info["resolution"]

        # annotations
        for split_method, split_method_values in info["annotations"].items():
            if split_method == "split_by_rois":
                self.annotation_csvs.extend(split_method_values)
            elif split_method == "split_automatically":
                for roi_type, roi_infos in split_method_values.items():
                    for roi_info in roi_infos:
                        roi_name = roi_info["roi_name"]
                        annotation_csv = roi_info["annotation_csv"]
                        self.annotation_csvs.append(annotation_csv)
                        self.rois_to_split_in_voxels[roi_type][roi_name] = (
                            self.__get_roi_to_split_from_annotation_csv(annotation_csv)
                        )
        self.__get_all_annotation_centers_voxels()

        # rois
        if "rois_to_split" in info:
            for roi_type, roi_infos in info["rois_to_split"].items():
                if roi_type not in self.rois_to_split_in_voxels:
                    self.rois_to_split_in_voxels[roi_type] = {}
                for roi_info in roi_infos:
                    roi_name = roi_info["roi_name"]
                    roi_start = np.zeros((3, 1))
                    roi_end = np.zeros((3, 1))
                    split_dimension = (
                        roi_info["split_dimension"]
                        if "split_dimension" in roi_info
                        else None
                    )
                    keep_if_empty = (
                        roi_info["keep_if_empty"]
                        if "keep_if_empty" in roi_info
                        else False
                    )
                    for idx, dim in enumerate(["z", "y", "x"]):
                        dim_start, dim_end = roi_info[dim].split("-")
                        roi_start[idx] = int(dim_start)
                        roi_end[idx] = int(dim_end)
                    roi_to_split_in_voxels = RoiToSplitInVoxels(
                        roi_start,
                        roi_end,
                        self.resolution,
                        split_dimension,
                        keep_if_empty,
                    )
                    if self.__roi_is_not_empty(roi_to_split_in_voxels.roi):
                        self.rois_to_split_in_voxels[roi_type][
                            roi_name
                        ] = roi_to_split_in_voxels
                    elif roi_to_split_in_voxels.keep_if_empty:
                        self.rois_to_split_in_voxels[roi_type][
                            roi_name
                        ] = roi_to_split_in_voxels
                        warnings.warn(
                            f"Empty roi {roi_to_split_in_voxels.roi*self.resolution}, but keeping it because keep_if_empty is True"
                        )
                    else:
                        warnings.warn(
                            f"Empty roi {roi_to_split_in_voxels.roi*self.resolution}, skipping"
                        )

    def __get_all_annotation_centers_voxels(self):
        # get all centers
        dfs = []
        for annotation_csv in self.annotation_csvs:
            dfs.append(pd.read_csv(annotation_csv))
        df = pd.concat(dfs)
        (
            self.all_annotation_starts_voxels,
            self.all_annotation_ends_voxels,
            self.all_annotation_centers_voxels,
        ) = self.__get_annotation_start_end_center_voxels(df)

    def __get_annotation_start_end_center_voxels(self, df):
        starts = (
            np.array([df["start z (nm)"], df["start y (nm)"], df["start x (nm)"]]).T
            / self.resolution
        )
        ends = (
            np.array([df["end z (nm)"], df["end y (nm)"], df["end x (nm)"]]).T
            / self.resolution
        )
        centers = (starts + ends) / 2

        return starts, ends, centers

    def __roi_is_not_empty(self, roi):
        valid_annotations = self.__get_valid_annotations(
            roi,
            self.all_annotation_centers_voxels,
        )
        return np.sum(valid_annotations) > 0

    def __get_roi_to_split_from_annotation_csv(self, annotation_csv):
        df = pd.read_csv(annotation_csv)
        (
            annotation_starts,
            annotation_ends,
            _,
        ) = self.__get_annotation_start_end_center_voxels(df)

        annotation_endpoints = np.concatenate((annotation_starts, annotation_ends))
        roi_start = np.ceil(np.min(annotation_endpoints, axis=0)).astype(int)
        roi_end = np.floor(np.max(annotation_endpoints, axis=0)).astype(int)
        return RoiToSplitInVoxels(roi_start, roi_end, resolution=1)

    def __get_valid_annotations(self, roi: Roi, annotation_centers):
        # check annotations are within region

        valid_annotations = (
            (annotation_centers[:, 0] >= roi.begin[0])
            & (annotation_centers[:, 0] <= roi.end[0])
            & (annotation_centers[:, 1] >= roi.begin[1])
            & (annotation_centers[:, 1] <= roi.end[1])
            & (annotation_centers[:, 2] >= roi.begin[2])
            & (annotation_centers[:, 2] <= roi.end[2])
        )
        return valid_annotations

    def __get_minimal_bounding_roi(
        self,
        roi: Roi,
        annotation_starts,
        annotation_centers,
        annotation_ends,
    ):
        # find minimum box coordinates that keep the same centers
        valid_annotations = self.__get_valid_annotations(roi, annotation_centers)
        valid_starts = annotation_starts[valid_annotations]
        valid_ends = annotation_ends[valid_annotations]
        valid_endpoints = np.concatenate((valid_starts, valid_ends))

        roi_start = np.maximum(
            np.array(roi.begin),
            np.floor(np.min(valid_endpoints, axis=0)).astype(int),
        )
        roi_end = np.minimum(
            np.array(roi.end),
            np.ceil(np.max(valid_endpoints, axis=0)).astype(int),
        )
        return Roi(roi_start, roi_end - roi_start)

    def __roi_from_bounding_box(self, starts, ends):
        return Roi(starts, np.array(ends) - np.array(starts))

    def __split_roi_along_axis(
        self,
        roi: Roi,
        annotation_starts,
        annotation_centers,
        annotation_ends,
        split_dimension,
        desired_ratio=0.5,
    ):
        first_roi_end = np.array(roi.end)
        second_roi_start = np.array(roi.begin)
        best_score = np.inf

        valid_annotations = self.__get_valid_annotations(roi, annotation_centers)
        valid_centers = annotation_centers[valid_annotations]

        num_kept_annotations = 0
        longest_box_diagonal = np.ceil(np.sqrt(3 * 36**2) + 1)
        for roi_split in range(roi.begin[split_dimension], roi.end[split_dimension]):
            annotations_in_first_half = np.sum(
                valid_centers[:, split_dimension] < roi_split
            )
            annotations_in_second_half = np.sum(
                valid_centers[:, split_dimension] >= (roi_split + longest_box_diagonal)
            )
            if annotations_in_second_half > 0:
                ratio = annotations_in_first_half / annotations_in_second_half

                if np.abs(desired_ratio - ratio) < best_score:
                    best_score = np.abs(desired_ratio - ratio)
                    first_roi_end[split_dimension] = roi_split
                    second_roi_start[split_dimension] = roi_split + longest_box_diagonal
                    num_kept_annotations = (
                        annotations_in_first_half + annotations_in_second_half
                    )
        first_roi = self.__roi_from_bounding_box(roi.begin, first_roi_end)
        second_roi = self.__roi_from_bounding_box(second_roi_start, roi.end)

        # first_roi = self.__get_minimal_bounding_roi(
        #     self.__roi_from_bounding_box(roi.begin, first_roi_end),
        #     annotation_starts,
        #     annotation_centers,
        #     annotation_ends,
        # )
        # second_roi = self.__get_minimal_bounding_roi(
        #     self.__roi_from_bounding_box(second_roi_start, roi.end),
        #     annotation_starts,
        #     annotation_centers,
        #     annotation_ends,
        # )
        return (num_kept_annotations, first_roi, second_roi)

    def split_validation_test_roi(
        self,
        validation_test_roi: Roi,
        annotation_starts,
        annotation_centers,
        annotation_ends,
        split_dimension=None,
        validation_test_split_ratio=1,
        max_num_kept_annotations=0,
    ):
        if split_dimension is not None:
            split_dimensions_to_check = [split_dimension]
        else:
            # do it in x,y,z order [2,1,0] to be consistent with previous iteration of this code
            split_dimensions_to_check = [2, 1, 0]

        best_validation_roi = None
        best_test_roi = None
        for current_split_dimension in split_dimensions_to_check:
            (
                num_kept_annotations,
                validation_roi,
                test_roi,
            ) = self.__split_roi_along_axis(
                validation_test_roi,
                annotation_starts,
                annotation_centers,
                annotation_ends,
                split_dimension=current_split_dimension,
                desired_ratio=validation_test_split_ratio,
            )

            if num_kept_annotations > max_num_kept_annotations:
                max_num_kept_annotations = num_kept_annotations
                best_validation_roi = validation_roi
                best_test_roi = test_roi

        return max_num_kept_annotations, best_validation_roi, best_test_roi

    def split_training_validation_test_roi(
        self,
        training_validation_test_roi: Roi,
        annotation_starts,
        annotation_centers,
        annotation_ends,
        training_split_ratio=0.75 / 0.25,
        validation_test_split_ratio=1,
    ):
        max_num_kept_annotations = 0
        # do it in x,y,z order [2,1,0] to be consistent with previous iteration of this code
        for first_split_dimension in [2, 1, 0]:
            _, training_roi, validation_test_roi = self.__split_roi_along_axis(
                training_validation_test_roi,
                annotation_starts,
                annotation_centers,
                annotation_ends,
                first_split_dimension,
                desired_ratio=training_split_ratio,
            )

            (
                num_kept_annotations,
                validation_roi,
                test_roi,
            ) = self.split_validation_test_roi(
                validation_test_roi,
                annotation_starts,
                annotation_centers,
                annotation_ends,
                validation_test_split_ratio=validation_test_split_ratio,
                max_num_kept_annotations=max_num_kept_annotations,
            )

            if num_kept_annotations > max_num_kept_annotations:
                max_num_kept_annotations = num_kept_annotations
                best_training_roi = training_roi
                best_validation_roi = validation_roi
                best_test_roi = test_roi

        return best_training_roi, best_validation_roi, best_test_roi

    def get_training_validation_test_rois(
        self,
        training_split_ratio=0.75 / 0.25,
        validation_test_split_ratio=1,
    ):
        for roi_to_split_name, roi_to_split in self.rois_to_split_in_voxels[
            "training_validation_test"
        ].items():
            (
                best_training_roi,
                best_validation_roi,
                best_test_roi,
            ) = self.split_training_validation_test_roi(
                roi_to_split.roi,
                self.all_annotation_starts_voxels,
                self.all_annotation_centers_voxels,
                self.all_annotation_ends_voxels,
                training_split_ratio,
                validation_test_split_ratio,
            )
            self.rois_dict["training"][roi_to_split_name] = (
                best_training_roi * self.resolution
            )
            self.rois_dict["validation"][roi_to_split_name] = (
                best_validation_roi * self.resolution
            )
            self.rois_dict["test"][roi_to_split_name] = best_test_roi * self.resolution

        for roi_to_split_name, roi_to_split in self.rois_to_split_in_voxels[
            "validation_test"
        ].items():
            if roi_to_split.split_dimension == "do_not_split":
                # then it was deemed too small to split so we just use it as validation
                best_validation_roi = roi_to_split.roi
            else:
                _, best_validation_roi, best_test_roi = self.split_validation_test_roi(
                    roi_to_split.roi,
                    self.all_annotation_starts_voxels,
                    self.all_annotation_centers_voxels,
                    self.all_annotation_ends_voxels,
                    split_dimension=roi_to_split.split_dimension,
                    validation_test_split_ratio=validation_test_split_ratio,
                )
                self.rois_dict["test"][roi_to_split_name] = (
                    best_test_roi * self.resolution
                )
            self.rois_dict["validation"][roi_to_split_name] = (
                best_validation_roi * self.resolution
            )

    def write_roi_annotations(self, output_directory):
        output_directory = f"{output_directory}/bounding_boxes"
        if os.path.isdir(output_directory):
            shutil.rmtree(output_directory)
        annotation_writer = AnnotationWriter(
            CoordinateSpace(names=("z", "y", "x"), scales=(1, 1, 1), units="nm"),
            annotation_type="axis_aligned_bounding_box",
            properties=[
                AnnotationPropertySpec(id="identifier", type="uint16"),
                AnnotationPropertySpec(id="box_color", type="rgb"),
            ],
        )

        # since it is arbitrary to have endpoints for line segment in terms of fitting, will just fit a line and then truncate it

        roi_name_to_color_dict = {
            "training": (0, 255, 0),
            "validation": (0, 0, 255),
            "test": (255, 0, 0),
        }
        for id, (roi_name, roi_color) in enumerate(roi_name_to_color_dict.items()):
            rois = self.rois_dict[roi_name]
            for roi in rois.values():
                annotation_writer.add_axis_aligned_bounding_box(
                    point_a=roi.begin,
                    point_b=roi.end,
                    identifier=id,
                    box_color=roi_color,
                )
                annotation_writer.write(f"{output_directory}")

        precomputed_path = output_directory
        precomputed_path = precomputed_path.replace(
            "/nrs/cellmap", "precomputed://https://cellmap-vm1.int.janelia.org/nrs/"
        )
        precomputed_path = precomputed_path.replace(
            "/dm11/cellmap", "precomputed://https://cellmap-vm1.int.janelia.org/dm11/"
        )
        precomputed_path = precomputed_path.replace(
            "/prfs/cellmap", "precomputed://https://cellmap-vm1.int.janelia.org/prfs/"
        )

        print(f"rois as annotations: {precomputed_path}")
        self.rois_as_annotations = precomputed_path

    def get_neuroglancer_url(self):
        # load the roi yaml
        with open(
            self.training_validation_test_roi_info_yaml,
            "r",
        ) as f:
            roi_info = yaml.safe_load(f)

        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            is_first = True
            for split_type, csvs in roi_info["annotations"].items():
                for csv in csvs:
                    is_first = False
                    df = pd.read_csv(csv)
                    layer_name = (
                        split_type + "_" + os.path.basename(csv).split(".csv")[0]
                    )
                    neuroglancer_url = df["neuroglancer url"][0]
                    state = parse_url(neuroglancer_url)
                    for layer in state.layers:
                        if (
                            layer.name == "fibsem-uint8"
                            or layer.name == "raw"
                            and is_first
                        ):
                            layer.name = "raw"
                            s.layers["raw"] = layer
                        if layer.name == "saved_annotations":
                            layer.name = layer_name
                            s.layers[layer_name] = layer

            s.layers["rois"] = neuroglancer.AnnotationLayer(
                source=self.rois_as_annotations,
                shader="""
                void main() {
                    setColor(prop_box_color());
                }
                """,
            )
        self.neuroglancer_url = neuroglancer.to_url(viewer.state)
        print(f"Neuroglancer view:\n{self.neuroglancer_url}")

    def standard_processing(self, output_directory):
        self.get_training_validation_test_rois()
        self.write_roi_annotations(output_directory)
        self.get_neuroglancer_url()
