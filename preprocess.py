import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from numcodecs.gzip import GZip
import os

import socket
import neuroglancer
import numpy as np

import neuroglancer
import neuroglancer.cli
import struct
import os
import struct
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import shutil
from funlib.geometry import Roi

from neuroglancer.write_annotations import AnnotationWriter
from neuroglancer.coordinate_space import CoordinateSpace
from neuroglancer import AnnotationPropertySpec

import warnings

class VoxelNmConverter:
    def __init__(
        self,
        resolution,
        voxel_coordinates: np.array = None,
        nm_coordinates: np.array = None,
    ):
        if voxel_coordinates is not None and nm_coordinates is not None:
            raise Exception(
                "Both voxel_coordinates and nm_coordinates were provided, but only one should be."
            )
        elif voxel_coordinates is not None:
            self.voxel = voxel_coordinates
            self.nm = voxel_coordinates * resolution
        else:
            self.nm = nm_coordinates
            self.voxel = nm_coordinates / resolution


class PreprocessCylindricalAnnotations:
    def __init__(
        self,
        username,
        annotation_name,
        radius,
        training_csvs: list[str] = [],
        training_validation_and_test_csvs: list[str] = [],
        rois_to_split_yml=None,
        rois_dict=None,
        mask_zarr=None,
        output_n5=None,
        dataset="jrc_22ak351-leaf-3m",
        raw_n5="/nrs/cellmap/data/",
        raw_dataset_name="em/fibsem-uint8",
    ):
        np.random.seed(0)  # set seed for consistency of locations
        self.training_csvs = training_csvs
        if type(self.training_csvs) is not list:
            self.training_csvs = [self.training_csvs]
        self.training_validation_and_test_csvs = training_validation_and_test_csvs
        if type(self.training_validation_and_test_csvs) is not list:
            self.training_validation_and_test_csvs = [
                self.training_validation_and_test_csvs
            ]

        if rois_dict:
            self.rois_dict = rois_dict

        self.username = username
        self.raw_dataset_name = raw_dataset_name
        self.annotation_name = annotation_name  # + "_as_cylinders"
        self.dataset = dataset
        zarr_file = zarr.open(f"{raw_n5}/{dataset}/{dataset}.n5", mode="r")
        self.raw_dataset = zarr_file[raw_dataset_name + "/s0"]
        if not mask_zarr:
            mask_zarr = f"/nrs/cellmap/{self.username}/cellmap/{self.annotation_name}/annotation_intersection_masks.zarr"
        if not output_n5:
            output_n5 = f"/nrs/cellmap/{self.username}/cellmap/{self.annotation_name}/annotations_as_cylinders.n5"
        self.mask_zarr = mask_zarr
        self.output_n5 = output_n5

        self.radius = radius
        self.rois_dict = {"training": [], "validation": [], "test": []}

        self.rois_to_split = []
        if rois_to_split_yml:
            self.get_rois_to_split(rois_to_split_yml)

        self.empty_annotations = []

    def get_rois_to_split(self, rois_to_split_yml):
        import yaml

        with open(rois_to_split_yml, "r") as stream:
            yml = yaml.safe_load(stream)
        for roi in yml["rois"]:
            starts = np.zeros((3,))
            extents = np.zeros((3,))
            for i, d in enumerate(["X", "Y", "Z"]):
                dim_start, dim_end = [int(s) for s in roi[d].split("-")]
                starts[i] = dim_start
                extents[i] = dim_end - dim_start
            self.rois_to_split.append(Roi(starts, extents))

    def in_cylinder(self, end_1, end_2, radius):
        # https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
        # normalized tangent vector
        d = np.divide(end_2 - end_1, np.linalg.norm(end_2 - end_1))

        # possible points
        mins = np.floor(np.minimum(end_1, end_2)).astype(int) - (
            np.ceil(radius).astype(int) + 1
        )  # 1s for padding
        maxs = np.ceil(np.maximum(end_1, end_2)).astype(int) + (
            np.ceil(radius).astype(int) + 1
        )

        x, y, z = [list(range(mins[i], maxs[i] + 1, 1)) for i in range(3)]
        p = np.array(np.meshgrid(x, y, z)).T.reshape((-1, 3))

        # signed parallel distance components
        s = np.dot(end_1 - p, d)
        t = np.dot(p - end_2, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros_like(s)])

        # perpendicular distance component
        c = np.linalg.norm(np.cross(p - end_1, d), axis=1)

        is_in_cylinder = (h == 0) & (c <= radius)
        return set(map(tuple, p[is_in_cylinder]))

    def extract_annotation_information(self):
        self.resolution = np.array(
            self.raw_dataset.attrs.asdict()["transform"]["scale"]
        )
        # https://cell-map.slack.com/archives/C04N9JUFQK1/p1683733456153269

        dfs = []
        for annotation_csv in (
            self.training_csvs + self.training_validation_and_test_csvs
        ):
            dfs.append(pd.read_csv(annotation_csv))
        df = pd.concat(dfs)

        self.annotation_starts = (
            np.array([df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]).T
            / self.resolution
        )
        self.annotation_ends = (
            np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T
            / self.resolution
        )

    def get_training_validation_and_test_rois(
        self,
        output_directory=None,
        voxel_size=8,
        training_split_ratio=0.75 / 0.25,
        validation_test_split_ratio=1,
    ):
        def get_annotation_start_end_center_nm(df):
            starts = np.array(
                [df["start x (nm)"], df["start y (nm)"], df["start z (nm)"]]
            ).T
            ends = np.array([df["end x (nm)"], df["end y (nm)"], df["end z (nm)"]]).T
            centers = (starts + ends) / 2

            return starts, ends, centers

        def get_valid_annotations(annotation_centers, box_start, box_end):
            rescaled_annotation_centers = annotation_centers / voxel_size
            # check annotation centers are within region
            valid_annotations = (
                (rescaled_annotation_centers[:, 0] >= box_start[0])
                & (rescaled_annotation_centers[:, 0] <= box_end[0])
                & (rescaled_annotation_centers[:, 1] >= box_start[1])
                & (rescaled_annotation_centers[:, 1] <= box_end[1])
                & (rescaled_annotation_centers[:, 2] >= box_start[2])
                & (rescaled_annotation_centers[:, 2] <= box_end[2])
            )
            # annotation_endpoints = annotation_endpoints[valid_annotations, :]
            annotation_centers = annotation_centers[valid_annotations, :]
            return annotation_centers  # , annotation_endpoints

        def get_minimal_bounding_box(annotation_centers, box_start, box_end):
            # find minimum box coordinates that keep the same centers
            valid_centers = get_valid_annotations(
                annotation_centers, box_start, box_end
            )
            box_start = np.maximum(
                box_start,
                np.floor(np.min(valid_centers / voxel_size, axis=0)).astype(int),
            )
            box_end = np.minimum(
                box_end, np.ceil(np.max(valid_centers / voxel_size, axis=0)).astype(int)
            )
            return box_start, box_end

        def split_box_along_axis(
            annotation_centers,
            whole_box_start,
            whole_box_end,
            split_dimension,
            desired_ratio=0.5,
        ):
            first_box_end = whole_box_end.copy()
            second_box_start = whole_box_start.copy()
            best_score = np.inf

            valid_centers = get_valid_annotations(
                annotation_centers, whole_box_start, whole_box_end
            )

            num_kept_annotations = 0
            for box_split in range(
                whole_box_start[split_dimension], whole_box_end[split_dimension]
            ):
                annotations_in_first_half = np.sum(
                    valid_centers[:, split_dimension] < box_split * voxel_size
                )
                annotations_in_second_half = np.sum(
                    valid_centers[:, split_dimension] >= (box_split + 145) * voxel_size
                )
                if annotations_in_second_half > 0:
                    ratio = annotations_in_first_half / annotations_in_second_half

                    if np.abs(desired_ratio - ratio) < best_score:
                        best_score = np.abs(desired_ratio - ratio)
                        first_box_end[split_dimension] = box_split
                        second_box_start[split_dimension] = box_split + 145
                        num_kept_annotations = (
                            annotations_in_first_half + annotations_in_second_half
                        )

            first_box_start, first_box_end = get_minimal_bounding_box(
                annotation_centers, whole_box_start, first_box_end
            )
            second_box_start, second_box_end = get_minimal_bounding_box(
                annotation_centers, second_box_start, whole_box_end
            )
            return num_kept_annotations, {
                "first_box": {"start": first_box_start, "end": first_box_end},
                "second_box": {"start": second_box_start, "end": second_box_end},
            }

        def write_roi_annotations():
            annotation_writer = AnnotationWriter(
                CoordinateSpace(names=("x", "y", "z"), scales=(1, 1, 1), units="nm"),
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
                        point_a=roi.begin[::-1],
                        point_b=roi.end[::-1],
                        identifier=id,
                        box_color=roi_color,
                    )
                    annotation_writer.write(
                        f"/groups/cellmap/cellmap/{self.username}/neuroglancer_annotations/{self.annotation_name}/splitting/{self.dataset}/bounding_boxes"
                    )

            precomputed_path = output_directory.replace(
                f"/groups/cellmap/cellmap/{self.username}/",
                f"precomputed://https://cellmap-vm1.int.janelia.org/dm11/{self.username}/",
            )

            print(f"rois as annotations: {precomputed_path}")

        for annotation_csv in self.training_validation_and_test_csvs:
            df = pd.read_csv(annotation_csv)
            annotation_starts, annotation_ends, _ = get_annotation_start_end_center_nm(
                df
            )

            annotation_endpoints = np.concatenate((annotation_starts, annotation_ends))
            box_start = np.ceil(
                np.min(annotation_endpoints, axis=0)  # / voxel_size
            ).astype(int)
            box_end = np.floor(
                np.max(annotation_endpoints, axis=0)  # / voxel_size
            ).astype(int)
            self.rois_to_split.append(Roi(box_start, box_end - box_start))

        # get all centers
        dfs = []
        for annotation_csv in (
            self.training_csvs + self.training_validation_and_test_csvs
        ):
            dfs.append(pd.read_csv(annotation_csv))
        df = pd.concat(dfs)
        _, _, annotation_centers = get_annotation_start_end_center_nm(df)

        for roi_to_split in self.rois_to_split:
            roi_to_split = roi_to_split.snap_to_grid(
                (voxel_size, voxel_size, voxel_size), mode="shrink"
            )
            roi_to_split /= voxel_size  # do this to keep things in voxels for splitting along exact voxel
            split_dimensions = np.argsort(roi_to_split.begin - roi_to_split.end)[
                :2
            ]  # use box_start - box_end so that argsort does it in "descending" order
            max_volume = np.inf
            max_num_kept_annotations = 0
            for first_split_dimension in range(3):
                _, split_boxes = split_box_along_axis(
                    annotation_centers,
                    np.array(roi_to_split.begin),
                    np.array(roi_to_split.end),
                    first_split_dimension,
                    desired_ratio=training_split_ratio,
                )

                training_box = split_boxes["first_box"]
                validation_and_testing_box = split_boxes["second_box"]
                for second_split_dimension in range(3):
                    current_num_kept_annotations, split_boxes = split_box_along_axis(
                        annotation_centers,
                        validation_and_testing_box["start"],
                        validation_and_testing_box["end"],
                        second_split_dimension,
                        desired_ratio=validation_test_split_ratio,
                    )
                    validation_box = split_boxes["first_box"]
                    test_box = split_boxes["second_box"]

                    training_box_volume = np.prod(
                        training_box["end"] - training_box["start"]
                    )
                    test_box_volume = np.prod(test_box["end"] - test_box["start"])

                    current_max_volume = np.max([training_box_volume, test_box_volume])

                    if current_num_kept_annotations > max_num_kept_annotations:
                        # try to get smallest volumes as possible so you dont have weirdly large ones with huge empty gaps
                        max_num_kept_annotations = current_num_kept_annotations
                        # swap axes to get in z,y,x
                        training_roi = Roi(
                            training_box["start"][::-1] * voxel_size,
                            (training_box["end"] - training_box["start"])[::-1]
                            * voxel_size,
                        )
                        validation_roi = Roi(
                            validation_box["start"][::-1] * voxel_size,
                            (validation_box["end"] - validation_box["start"])[::-1]
                            * voxel_size,
                        )
                        test_roi = Roi(
                            test_box["start"][::-1] * voxel_size,
                            (test_box["end"] - test_box["start"])[::-1] * voxel_size,
                        )

            self.rois_dict["training"].append(training_roi)
            self.rois_dict["validation"].append(validation_roi)
            self.rois_dict["test"].append(test_roi)

        if output_directory:
            write_roi_annotations()

    def get_negative_examples(filename="annotations_20230620_221638.csv"):
        negative_examples = pd.read_csv(filename)
        negative_example_centers = np.array(
            [
                negative_examples["x (nm)"],
                negative_examples["y (nm)"],
                negative_examples["z (nm)"],
            ]
        ).T
        negative_example_centers = list(
            map(tuple, np.round(negative_example_centers).astype(int))
        )
        # ensure no duplicate negative examples
        print(
            len(
                set(
                    (
                        zip(
                            list(negative_examples["x (nm)"]),
                            list(negative_examples["y (nm)"]),
                            list(negative_examples["z (nm)"]),
                        )
                    )
                )
            )
        )
        return negative_example_centers

    def write_annotations_as_cylinders_and_get_intersections(self, radius):
        # get all pd voxels and all overlapping/intersecting voxels between multiple pd
        all_annotation_voxels_set = set()
        self.intersection_voxels_set = set()
        for annotation_start, annotation_end in tqdm(
            zip(self.annotation_starts, self.annotation_ends),
            total=len(self.annotation_starts),
        ):
            voxels_in_cylinder = self.in_cylinder(
                annotation_start, annotation_end, radius=radius
            )
            self.intersection_voxels_set.update(
                all_annotation_voxels_set.intersection(voxels_in_cylinder)
            )
            all_annotation_voxels_set.update(voxels_in_cylinder)

        # repeat but now will write out the relevant voxels with appropriate id
        store = zarr.N5Store(self.output_n5)
        zarr_root = zarr.group(store=store)
        ds = zarr_root.create_dataset(
            name=self.dataset,
            dtype="u2",  # need to do this, right now it is doing it as floats
            shape=self.raw_dataset.shape,
            chunks=128,
            write_empty_chunks=False,
            compressor=GZip(level=6),
            overwrite=True,
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [8],
            "unit": "nm",
        }

        all_annotation_id = 1
        annotation_id = 1
        all_annotation_voxels_set -= self.intersection_voxels_set
        for annotation_start, annotation_end in tqdm(
            zip(self.annotation_starts, self.annotation_ends),
            total=len(self.annotation_starts),
        ):
            voxels_in_cylinder = (
                self.in_cylinder(annotation_start, annotation_end, radius=radius)
                - self.intersection_voxels_set
            )
            if len(voxels_in_cylinder) > 0:
                voxels_in_cylinder = np.array(list(voxels_in_cylinder))
                ds[
                    voxels_in_cylinder[:, 2],
                    voxels_in_cylinder[:, 1],
                    voxels_in_cylinder[:, 0],
                ] = annotation_id
                annotation_id += 1
                all_annotation_id += 1
            else:
                self.empty_annotations.append(all_annotation_id)
                all_annotation_id += 1
                warnings.warn(
                    f"Empty annotation #{all_annotation_id-1} ({annotation_start}-{annotation_end}) will be ignored"
                )

    def write_intersection_mask(self):
        zarr_path = self.mask_zarr
        if not os.path.exists(zarr_path):
            zarr_root = zarr.open(zarr_path, mode="w")
        else:
            zarr_root = zarr.open(zarr_path, mode="r+")
        ds = zarr_root.create_dataset(
            overwrite=True,
            name=self.dataset,
            dtype="u1",
            fill_value=1,
            shape=self.raw_dataset.shape,
            chunks=128,
            write_empty_chunks=False,
            compressor=GZip(level=6),
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [8],
            "unit": "nm",
        }
        intersection_voxels = np.array(list(self.intersection_voxels_set))
        if len(intersection_voxels) > 0:
            ds[
                intersection_voxels[:, 2],
                intersection_voxels[:, 1],
                intersection_voxels[:, 0],
            ] = 0

    def remove_validation_or_test_annotations_from_training(self):
        def point_is_valid_center_for_current_roi(pt, edge_length, offset, shape):
            # a point is considered a valid center if the input bounding box for it does not cross the validation crop
            if np.all((pt + edge_length) >= offset) and np.all(
                (pt - edge_length) <= (offset + shape)
            ):
                # then it overlaps validation
                return False
            return True

        def point_is_valid_center(pt, edge_length):
            for roi in self.rois_dict["validation"] + self.rois_dict["test"]:
                roi = roi.snap_to_grid(
                    (self.resolution[0], self.resolution[0], self.resolution[0]),
                    mode="shrink",
                )
                roi /= self.resolution[
                    0
                ]  # do this to keep things in voxels for splitting along exact voxel

                # at this point rois are in zyx, so need to flip the pt coordinates
                if not point_is_valid_center_for_current_roi(
                    pt[::-1], edge_length, roi.offset, roi.shape
                ):
                    return False
            return True

        def too_close_to_rois(annotation_start, annotation_end, edge_length):
            # either the start or end will be furthest from the box
            return not (
                point_is_valid_center(annotation_start, edge_length)
                or point_is_valid_center(annotation_end, edge_length)
            )

        self.pseudorandom_training_centers = []
        self.removed_ids = []
        for id, annotation_start, annotation_end in tqdm(
            zip(
                list(range(1, len(self.annotation_starts) + 1)),
                self.annotation_starts,
                self.annotation_ends,
            ),
            total=len(self.annotation_starts),
        ):
            # ultimately seems to predict on 36x36x36 region, so we need to make sure this doesn't overlap with validation
            # lets just shift by at most +/-10 in any dimension for the center to help ensure that a non-neglible part of the rasterization, and original annotation, are included in a box centered at that region
            max_shift = 18
            # first find a random coordinate along the annotation. this will be included within the box

            # now find a valid center
            # NB: since we want to make sure that we are far enough away from the validation to ensure that no validation voxels affect training voxels
            # we must make sure the distance is at least the run.model.eval_input_shape/2 = 288/2 = 144
            edge_length = 144 + 1  # add one for padding since we round later on
            annotation_length = np.linalg.norm(annotation_start - annotation_end)
            if not (
                too_close_to_rois(annotation_start, annotation_end, edge_length)
                or annotation_length == 0
            ):
                random_coordinate_along_annotation = (
                    annotation_start
                    + (annotation_end - annotation_start) * np.random.rand()
                )
                center = random_coordinate_along_annotation + np.random.randint(
                    low=-max_shift, high=max_shift, size=3
                )
                while not point_is_valid_center(center, edge_length):
                    random_coordinate_along_annotation = (
                        annotation_start
                        + (annotation_end - annotation_start) * np.random.rand()
                    )
                    center = random_coordinate_along_annotation + np.random.randint(
                        low=-max_shift, high=max_shift, size=3
                    )
                self.pseudorandom_training_centers.append(
                    tuple(np.round(center * self.resolution).astype(int))
                )
            else:
                if annotation_length == 0:
                    print(f"empty id {id} {annotation_start} {annotation_end}")
                # c = np.round(((annotation_start + annotation_end) * self.resolution / 2)).astype(int)
                self.removed_ids.append(id)
        print(
            f"number of original centers: {len(self.annotation_starts)}, number of training centers: {len(self.pseudorandom_training_centers)}"
        )
        # if self.use_negative_examples:
        #     pseudorandom_training_centers += negative_example_centers

    def write_out_annotations(self, output_directory, annotation_ids):
        annotation_type = "line"
        if os.path.isdir(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(f"{output_directory}/spatial0")
        if annotation_type == "line":
            coords_to_write = 6
        else:
            coords_to_write = 3

        annotations = (
            np.column_stack((self.annotation_starts, self.annotation_ends))
            * self.resolution[0]
        )
        annotations = np.array([annotations[id - 1, :] for id in annotation_ids])
        with open(f"{output_directory}/spatial0/0_0_0", "wb") as outfile:
            total_count = len(annotations)
            buf = struct.pack("<Q", total_count)
            for annotation in tqdm(annotations):
                annotation_buf = struct.pack(f"<{coords_to_write}f", *annotation)
                buf += annotation_buf
            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                f"<{total_count}Q", *range(1, len(annotations) + 1, 1)
            )  # so start at 1
            # id_buf = struct.pack('<%sQ' % len(coordinates), 3,1 )#s*range(len(coordinates)))
            buf += id_buf
            outfile.write(buf)

        max_extents = annotations.reshape((-1, 3)).max(axis=0) + 1
        max_extents = [int(max_extent) for max_extent in max_extents]
        info = {
            "@type": "neuroglancer_annotations_v1",
            "dimensions": {"x": [1, "nm"], "y": [1, "nm"], "z": [1, "nm"]},
            "by_id": {"key": "by_id"},
            "lower_bound": [0, 0, 0],
            "upper_bound": max_extents,
            "annotation_type": annotation_type,
            "properties": [],
            "relationships": [],
            "spatial": [
                {
                    "chunk_size": max_extents,
                    "grid_shape": [1, 1, 1],
                    "key": "spatial0",
                    "limit": 1,
                }
            ],
        }

        with open(f"{output_directory}/info", "w") as info_file:
            json.dump(info, info_file)

        precomputed_path = output_directory.replace(
            f"/groups/cellmap/cellmap/{self.username}/",
            f"precomputed://https://cellmap-vm1.int.janelia.org/dm11/{self.username}/",
        )
        print(f"annotations: {precomputed_path}")

    def visualize_removed_annotations(self, roi, radius):
        def add_segmentation_layer(state, data, name):
            dimensions = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"], units="nm", scales=[8, 8, 8]
            )
            state.dimensions = dimensions
            state.layers.append(
                name=name,
                segments=[str(i) for i in np.unique(data[data > 0])],
                layer=neuroglancer.LocalVolume(
                    data=data,
                    dimensions=neuroglancer.CoordinateSpace(
                        names=["z", "y", "x"],
                        units=["nm", "nm", "nm"],
                        scales=[8, 8, 8],
                        coordinate_arrays=[
                            None,
                            None,
                            None,
                        ],
                    ),
                    voxel_offset=(0, 0, 0),
                ),
            )

        # get data
        expand_by = 500
        expanded_offset = np.array(roi.offset) - expand_by
        expanded_dimension = np.array(roi.shape) + 2 * expand_by
        ds = np.zeros(expanded_dimension, dtype=np.uint64)
        for id, annotation_start, annotation_end, annotation_center in tqdm(
            zip(
                list(range(1, len(self.annotation_starts) + 1)),
                self.annotation_starts,
                self.annotation_ends,
                self.annotation_centers,
            ),
            total=len(self.annotation_starts),
        ):
            if id in self.removed_ids:
                voxels_in_cylinder = np.array(
                    list(
                        self.in_cylinder(
                            annotation_start, annotation_end, radius=radius
                        )
                    )
                )
                if np.any(
                    np.all(voxels_in_cylinder >= expanded_offset, axis=1)
                    & np.all(
                        voxels_in_cylinder <= expanded_offset + expanded_dimension,
                        axis=1,
                    )
                ):
                    ds[
                        voxels_in_cylinder[:, 2] - expanded_offset[2],
                        voxels_in_cylinder[:, 1] - expanded_offset[1],
                        voxels_in_cylinder[:, 0] - expanded_offset[0],
                    ] = id

        neuroglancer.set_server_bind_address(
            bind_address=socket.gethostbyname(socket.gethostname())
        )
        viewer = neuroglancer.Viewer()
        with viewer.txn() as state:
            add_segmentation_layer(state, ds, "removed")
        print(viewer)
        input("Press Enter to continue...")

    def get_neuroglancer_view(self):
        if self.training_validation_and_test_csvs:
            annotation_datetime = (
                self.training_validation_and_test_csvs[0]
                .split("annotations_")[-1]
                .split(".csv")[0]
            )
        else:
            annotation_datetime = (
                self.training_csvs[0].split("annotations_")[-1].split(".csv")[0]
            )
        url = f"https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B0.0%2C0.0%2C0.0%5D%2C%22crossSectionScale%22:1%2C%22projectionScale%22:16384%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22n5://https://cellmap-vm1.int.janelia.org/nrs/data/{self.dataset}/{self.dataset}.n5/{self.raw_dataset_name}/%22%2C%22tab%22:%22source%22%2C%22name%22:%22fibsem-uint8%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22precomputed://https://cellmap-vm1.int.janelia.org/dm11/{self.username}/neuroglancer_annotations/{self.annotation_name}/splitting/{self.dataset}/bounding_boxes%22%2C%22tab%22:%22rendering%22%2C%22shader%22:%22%5Cnvoid%20main%28%29%20%7B%5Cn%20%20setColor%28prop_box_color%28%29%5Cn%20%20%20%20%20%20%20%20%20%20%29%3B%5Cn%7D%5Cn%22%2C%22name%22:%22bounding_boxes%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22n5://https://cellmap-vm1.int.janelia.org/nrs/{self.username}/cellmap/{self.annotation_name}/{self.annotation_name}.n5/{self.dataset}/%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%5D%2C%22name%22:%22{self.annotation_name}%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22precomputed://https://cellmap-vm1.int.janelia.org/dm11/{self.username}/neuroglancer_annotations/{annotation_datetime}%22%2C%22tab%22:%22source%22%2C%22name%22:%22{annotation_datetime}%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%2220230830_155757%22%7D%2C%22layout%22:%224panel%22%7D"
        print(url)

    def standard_preprocessing(self):
        self.get_training_validation_and_test_rois(
            output_directory=f"/groups/cellmap/cellmap/{self.username}/neuroglancer_annotations/{self.annotation_name}/splitting/{self.dataset}/bounding_boxes",
        )
        self.extract_annotation_information()
        self.write_annotations_as_cylinders_and_get_intersections(radius=self.radius)
        self.write_intersection_mask()
        self.remove_validation_or_test_annotations_from_training()
        if self.removed_ids:
            self.write_out_annotations(
                output_directory=f"/groups/cellmap/cellmap/{self.username}/neuroglancer_annotations/{self.annotation_name}/removed_annotations/{self.dataset}/removed_annotations",
                annotation_ids=self.removed_ids,
            )
            self.write_out_annotations(
                output_directory=f"/groups/cellmap/cellmap/{self.username}/neuroglancer_annotations/{self.annotation_name}/removed_annotations/{self.dataset}/kept_annotations",
                annotation_ids=[
                    id
                    for id in range(1, len(self.annotation_starts) + 1)
                    if id not in self.removed_ids
                ],
            )
        # self.get_neuroglancer_view()
