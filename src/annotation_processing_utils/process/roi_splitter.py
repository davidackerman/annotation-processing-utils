from collections import namedtuple
import os
import shutil
from funlib.geometry import Roi
import numpy as np
import pandas as pd

from neuroglancer.write_annotations import AnnotationWriter
from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer import AnnotationPropertySpec
import yaml
import warnings


class RoiToSplitInVoxels:
    def __init__(self, roi_start, roi_end, resolution, split_dimension=None):
        dims = ["z", "y", "x"]
        roi = Roi(roi_start, roi_end - roi_start)
        roi = roi.snap_to_grid((resolution, resolution, resolution), mode="shrink")
        roi /= resolution  # do this to keep things in voxels for splitting along exact voxel
        self.resolution = resolution
        self.roi = roi
        self.split_dimension = (
            dims.index(split_dimension) if split_dimension in dims else split_dimension
        )


class RoiSplitter:
    # currently predfined rois are only for validation/test
    def __init__(
        self,
        training_validation_test_rois_to_split_yml=None,
        annotations_to_split_automatically=[],
        validation_test_rois_to_split_yml=None,
        annotations_to_split_by_rois=[],
        resolution=8,
    ):
        if not (annotations_to_split_by_rois or annotations_to_split_automatically):
            raise Exception(
                "No annotations provided. Need to privde annotations_to_split_by_rois or annotations_to_split_automatically."
            )
        self.training_validation_test_rois_to_split_yml = (
            training_validation_test_rois_to_split_yml
        )
        self.validation_test_rois_to_split_yml = validation_test_rois_to_split_yml
        self.annotations_to_split_by_rois = annotations_to_split_by_rois
        self.annotations_to_split_automatically = annotations_to_split_automatically
        self.resolution = resolution
        self.rois_dict = {"training": [], "validation": [], "test": []}

        self.__get_all_annotation_centers_voxels()
        self.__get_rois_to_split()

    def __get_all_annotation_centers_voxels(self):
        # get all centers
        dfs = []
        for annotation_csv in (
            self.annotations_to_split_by_rois + self.annotations_to_split_automatically
        ):
            dfs.append(pd.read_csv(annotation_csv))
        df = pd.concat(dfs)
        (
            _,
            _,
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
        valid_centers = self.__get_valid_centers(
            roi, self.all_annotation_centers_voxels
        )
        return len(valid_centers) > 0

    def __get_roi_from_yml(self, yml_path):
        rois_to_split_in_voxels = []
        if yml_path:
            with open(yml_path, "r") as stream:
                yml = yaml.safe_load(stream)
            for roi in yml["rois"]:
                roi_start = np.zeros((3, 1))
                roi_end = np.zeros((3, 1))
                for idx, dim in enumerate(["z", "y", "x"]):
                    dim_start, dim_end = roi[dim].split("-")
                    roi_start[idx] = int(dim_start)
                    roi_end[idx] = int(dim_end)
                roi_to_split_in_voxels = RoiToSplitInVoxels(
                    roi_start,
                    roi_end,
                    self.resolution,
                    roi["split_dimension"] if "split_dimension" in roi else None,
                )

                if self.__roi_is_not_empty(roi_to_split_in_voxels.roi):
                    rois_to_split_in_voxels.append(roi_to_split_in_voxels)
                else:
                    warnings.warn(
                        f"Empty roi {rois_to_split_in_voxels.roi*self.resolution}"
                    )
        return rois_to_split_in_voxels

    def __get_rois_to_split(self):
        self.validation_test_rois_to_split_in_voxels = self.__get_roi_from_yml(
            self.validation_test_rois_to_split_yml
        )
        self.training_validation_test_rois_to_split_in_voxels = self.__get_roi_from_yml(
            self.training_validation_test_rois_to_split_yml
        )

        for annotation_csv in self.annotations_to_split_automatically:
            df = pd.read_csv(annotation_csv)
            (
                annotation_starts,
                annotation_ends,
                _,
            ) = self.__get_annotation_start_end_center_voxels(df)

            annotation_endpoints = np.concatenate((annotation_starts, annotation_ends))
            roi_start = np.ceil(np.min(annotation_endpoints, axis=0)).astype(int)
            roi_end = np.floor(np.max(annotation_endpoints, axis=0)).astype(int)
            self.training_validation_test_rois_to_split_in_voxels.append(
                RoiToSplitInVoxels(roi_start, roi_end, resolution=1)
            )

    def __get_valid_centers(self, roi: Roi, annotation_centers):
        # check annotation centers are within region
        valid_annotations = (
            (annotation_centers[:, 0] >= roi.begin[0])
            & (annotation_centers[:, 0] <= roi.end[0])
            & (annotation_centers[:, 1] >= roi.begin[1])
            & (annotation_centers[:, 1] <= roi.end[1])
            & (annotation_centers[:, 2] >= roi.begin[2])
            & (annotation_centers[:, 2] <= roi.end[2])
        )
        return annotation_centers[valid_annotations]

    def __get_minimal_bounding_roi(
        self,
        roi: Roi,
        annotation_centers,
    ):
        # find minimum box coordinates that keep the same centers
        valid_centers = self.__get_valid_centers(roi, annotation_centers)
        roi_start = np.maximum(
            np.array(roi.begin),
            np.floor(np.min(valid_centers, axis=0)).astype(int),
        )
        roi_end = np.minimum(
            np.array(roi.end),
            np.ceil(np.max(valid_centers, axis=0)).astype(int),
        )
        return Roi(roi_start, roi_end - roi_start)

    def __roi_from_bounding_box(self, starts, ends):
        return Roi(starts, np.array(ends) - np.array(starts))

    def __split_roi_along_axis(
        self,
        roi: Roi,
        annotation_centers,
        split_dimension,
        desired_ratio=0.5,
    ):
        first_roi_end = np.array(roi.end)
        second_roi_start = np.array(roi.begin)
        best_score = np.inf

        valid_centers = self.__get_valid_centers(roi, annotation_centers)

        num_kept_annotations = 0
        for roi_split in range(roi.begin[split_dimension], roi.end[split_dimension]):
            annotations_in_first_half = np.sum(
                valid_centers[:, split_dimension] < roi_split
            )
            annotations_in_second_half = np.sum(
                valid_centers[:, split_dimension] >= (roi_split + 145)
            )
            if annotations_in_second_half > 0:
                ratio = annotations_in_first_half / annotations_in_second_half

                if np.abs(desired_ratio - ratio) < best_score:
                    best_score = np.abs(desired_ratio - ratio)
                    first_roi_end[split_dimension] = roi_split
                    second_roi_start[split_dimension] = roi_split + 145
                    num_kept_annotations = (
                        annotations_in_first_half + annotations_in_second_half
                    )

        first_roi = self.__get_minimal_bounding_roi(
            self.__roi_from_bounding_box(roi.begin, first_roi_end), annotation_centers
        )
        second_roi = self.__get_minimal_bounding_roi(
            self.__roi_from_bounding_box(second_roi_start, roi.end), annotation_centers
        )
        return (num_kept_annotations, first_roi, second_roi)

    def split_validation_test_roi(
        self,
        validation_test_roi: Roi,
        annotation_centers,
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
                annotation_centers,
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
        annotation_centers,
        training_split_ratio=0.75 / 0.25,
        validation_test_split_ratio=1,
    ):
        max_num_kept_annotations = 0
        # do it in x,y,z order [2,1,0] to be consistent with previous iteration of this code
        for first_split_dimension in [2, 1, 0]:
            _, training_roi, validation_test_roi = self.__split_roi_along_axis(
                training_validation_test_roi,
                annotation_centers,
                first_split_dimension,
                desired_ratio=training_split_ratio,
            )

            (
                num_kept_annotations,
                validation_roi,
                test_roi,
            ) = self.split_validation_test_roi(
                validation_test_roi,
                annotation_centers,
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
        for roi_to_split in self.training_validation_test_rois_to_split_in_voxels:
            (
                best_training_roi,
                best_validation_roi,
                best_test_roi,
            ) = self.split_training_validation_test_roi(
                roi_to_split.roi,
                self.all_annotation_centers_voxels,
                training_split_ratio,
                validation_test_split_ratio,
            )
            self.rois_dict["training"].append(best_training_roi * self.resolution)
            self.rois_dict["validation"].append(best_validation_roi * self.resolution)
            self.rois_dict["test"].append(best_test_roi * self.resolution)

        for roi_to_split in self.validation_test_rois_to_split_in_voxels:
            if roi_to_split.split_dimension == "don't split":
                # then it was deemed too small to split so we just use it as validation
                best_validation_roi = roi_to_split.roi
            else:
                _, best_validation_roi, best_test_roi = self.split_validation_test_roi(
                    roi_to_split.roi,
                    self.all_annotation_centers_voxels,
                    split_dimension=roi_to_split.split_dimension,
                    validation_test_split_ratio=validation_test_split_ratio,
                )
                self.rois_dict["test"].append(best_test_roi * self.resolution)
            self.rois_dict["validation"].append(best_validation_roi * self.resolution)

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
            for roi in rois:
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

    def standard_processing(self, output_directory):
        self.get_training_validation_test_rois()
        self.write_roi_annotations(output_directory)
