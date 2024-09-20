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
from annotation_processing_utils.utils.bresenham3D import bresenham3DWithMask

import warnings
from .training_validation_test_roi_calculator import TrainingValidationTestRoiCalculator
from funlib.persistence import open_ds
import itertools
from neuroglancer.url_state import parse_url
from ..utils.zarr_util import create_multiscale_dataset
import pickle
from funlib.geometry import Coordinate
from ..utils.dacapo_util import DacapoRunBuilder
from dacapo.experiments.starts import StartConfig
import getpass
from dacapo.experiments.starts import CosemStartConfig


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
            self.nm = voxel_coordinates * resolution[0]
        else:
            self.nm = nm_coordinates
            self.voxel = nm_coordinates / resolution[0]


class CylindricalAnnotations:
    def __init__(
        self,
        organelle,
        radius,
        training_validation_test_roi_info_yaml,
        output_annotations_directory=None,
        output_mask_zarr=None,
        output_gt_zarr=None,
        output_training_points_zarr=None,
        dataset="jrc_22ak351-leaf-3m",
        training_point_selection_mode="all",
    ):
        np.random.seed(0)  # set seed for consistency of locations
        self.username = getpass.getuser()
        self.organelle = organelle
        self.dataset = dataset

        if not output_mask_zarr:
            output_mask_zarr = f"/nrs/cellmap/{self.username}/cellmap/{self.organelle}/annotation_intersection_masks.zarr"
        if not output_gt_zarr:
            output_gt_zarr = f"/nrs/cellmap/{self.username}/cellmap/{self.organelle}/annotations_as_cylinders.zarr"
        if not output_training_points_zarr:
            output_training_points_zarr = f"/nrs/cellmap/{self.username}/cellmap/{self.organelle}/training_points.zarr"
        if not output_annotations_directory:
            output_annotations_directory = f"/groups/cellmap/cellmap/{self.username}/neuroglancer_annotations/{self.organelle}/{self.dataset}"

        raw_zarr = f"/nrs/cellmap/data/{self.dataset}/{self.dataset}.zarr"
        if "em" in os.listdir(raw_zarr):
            raw_dataset_name = "/em/fibsem-uint8/s0"
        elif "volumes" in os.listdir(raw_zarr):
            raw_dataset_name = "/volumes/raw/s0"
        elif "recon-1":
            raw_dataset_name = "/recon-1/em/fibsem-uint8/s0"
        self.raw_dataset = open_ds(raw_zarr, raw_dataset_name)

        self.output_mask_zarr = output_mask_zarr
        self.output_gt_zarr = output_gt_zarr
        self.output_training_points_zarr = output_training_points_zarr
        self.output_annotations_directory = output_annotations_directory
        self.empty_annotations = []
        self.radius = radius

        self.roi_calculator = TrainingValidationTestRoiCalculator(
            training_validation_test_roi_info_yaml
        )
        self.roi_calculator.standard_processing(output_annotations_directory)
        self.training_point_selection_mode = training_point_selection_mode

        # 36x36x36 is shape of region used to caluclate loss,so we need to make sure that the center is at least the diagonal away from the validation/test rois
        self.longest_box_diagonal = int(np.ceil(np.sqrt(3 * (36**2)))) + 1

        neuroglancer.set_server_bind_address("0.0.0.0")

    def in_cylinder(self, end_1, end_2, radius):
        # subtract 0.5 so that an annotation centered at a voxel matches to eg 0,0,0
        end_2 = end_2 - 0.5
        end_1 = end_1 - 0.5
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

        z, y, x = [list(range(mins[i], maxs[i] + 1, 1)) for i in range(3)]
        p = np.array(np.meshgrid(z, y, x)).T.reshape((-1, 3))

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
        self.voxel_size = self.raw_dataset.voxel_size

        # https://cell-map.slack.com/archives/C04N9JUFQK1/p1683733456153269

        dfs = []
        for annotation_csv in self.roi_calculator.annotation_csvs:
            dfs.append(pd.read_csv(annotation_csv))
        df = pd.concat(dfs)
        # remove duplicate rows from dataframe
        df = df.drop_duplicates(
            subset=[
                "start z (nm)",
                "start y (nm)",
                "start x (nm)",
                "end z (nm)",
                "end y (nm)",
                "end x (nm)",
            ]
        )
        # # use only first 500 annotations
        # df = df.iloc[:100]
        self.annotation_starts = (
            np.array([df["start z (nm)"], df["start y (nm)"], df["start x (nm)"]]).T
            / self.voxel_size[0]
        )
        self.annotation_ends = (
            np.array([df["end z (nm)"], df["end y (nm)"], df["end x (nm)"]]).T
            / self.voxel_size[0]
        )

    def get_negative_examples(filename="annotations_20230620_221638.csv"):
        negative_examples = pd.read_csv(filename)
        negative_example_centers = np.array(
            [
                negative_examples["z (nm)"],
                negative_examples["y (nm)"],
                negative_examples["x (nm)"],
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
                            list(negative_examples["z (nm)"]),
                            list(negative_examples["y (nm)"]),
                            list(negative_examples["x (nm)"]),
                        )
                    )
                )
            )
        )
        return negative_example_centers

    def write_annotations_as_cylinders_and_get_intersections(self, radius):
        # get all pd voxels and all overlapping/intersecting voxels between multiple pd
        self.all_annotation_voxels = []
        self.all_annotation_voxels_set = set()
        self.intersection_voxels_set = set()
        for annotation_start, annotation_end in tqdm(
            zip(self.annotation_starts, self.annotation_ends),
            total=len(self.annotation_starts),
        ):
            voxels_in_cylinder = self.in_cylinder(
                annotation_start, annotation_end, radius=radius
            )
            self.all_annotation_voxels.append(list(voxels_in_cylinder))
            self.intersection_voxels_set.update(
                self.all_annotation_voxels_set.intersection(voxels_in_cylinder)
            )
            self.all_annotation_voxels_set.update(voxels_in_cylinder)

        # repeat but now will write out the relevant voxels with appropriate id
        ds = create_multiscale_dataset(
            output_path=f"{self.output_gt_zarr}/{self.dataset}",
            dtype="u2",
            voxel_size=self.voxel_size,
            total_roi=self.raw_dataset.roi,
            write_size=self.voxel_size * 128,
            delete=True,
        )

        all_annotation_id = 1
        annotation_id = 1
        # self.all_annotation_voxels_set -= self.intersection_voxels_set
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
                ds.data[
                    voxels_in_cylinder[:, 0],
                    voxels_in_cylinder[:, 1],
                    voxels_in_cylinder[:, 2],
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
        create_multiscale_dataset(
            output_path=f"{self.output_mask_zarr}/{self.dataset}",
            dtype="u1",
            voxel_size=self.voxel_size,
            total_roi=self.raw_dataset.roi,
            write_size=self.voxel_size * 128,
            delete=True,
        )
        zarray_metadata_path = f"{self.output_mask_zarr}/{self.dataset}/s0/.zarray"
        # Load the .zarray metadata
        with open(zarray_metadata_path, "r") as f:
            zarray_metadata = json.load(f)

        zarray_metadata["fill_value"] = 1

        # Save the updated metadata back to the .zarray file
        with open(zarray_metadata_path, "w") as f:
            json.dump(zarray_metadata, f, indent=4)

        ds = open_ds(self.output_mask_zarr, self.dataset + "/s0", mode="a")

        intersection_voxels = np.array(list(self.intersection_voxels_set))
        if len(intersection_voxels) > 0:
            ds.data[
                intersection_voxels[:, 0],
                intersection_voxels[:, 1],
                intersection_voxels[:, 2],
            ] = 0

    def write_training_points(self):
        ds = create_multiscale_dataset(
            output_path=f"{self.output_training_points_zarr}/{self.dataset}",
            dtype="u1",
            voxel_size=self.voxel_size,
            total_roi=self.raw_dataset.roi,
            write_size=self.voxel_size * 128,
            delete=True,
        )
        # convert list of training point tuples to numpy array
        for current_training_points in tqdm(self.training_points_by_object):
            current_training_points_voxels = (
                np.array(current_training_points) / self.voxel_size[0]
            )
            current_training_points_voxels = current_training_points_voxels.astype(int)
            ds.data[
                current_training_points_voxels[:, 0],
                current_training_points_voxels[:, 1],
                current_training_points_voxels[:, 2],
            ] = 1

        self.training_points = list(itertools.chain(*self.training_points_by_object))

    def get_pseudorandom_training_centers(self, random_shift_voxels=None):
        def point_is_valid_center_for_current_roi(pt, edge_length, offset, shape):
            # a point is considered a valid center if the input bounding box for it does not cross the validation crop
            if np.all((pt + edge_length) >= offset) and np.all(
                (pt - edge_length) <= (offset + shape)
            ):
                # then it overlaps validation
                return False
            return True

        def point_is_valid_center(pt, edge_length):
            for roi in list(
                self.roi_calculator.rois_dict["validation"].values()
            ) + list(self.roi_calculator.rois_dict["test"].values()):
                roi = roi.snap_to_grid(
                    3 * [self.voxel_size[0]],
                    mode="shrink",
                )

                # do this to keep things in voxels for splitting along exact voxel
                roi /= int(self.voxel_size[0])

                if not point_is_valid_center_for_current_roi(
                    pt, edge_length, roi.offset, roi.shape
                ):
                    return False
            return True

        def too_close_to_rois(annotation_start, annotation_end, edge_length):
            # either the start or end will be furthest from the box
            return not (
                point_is_valid_center(annotation_start, edge_length)
                or point_is_valid_center(annotation_end, edge_length)
            )

        self.training_points_by_object = []
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
                    if random_shift_voxels:
                        random_shift = random_shift_voxels * self.voxel_size[0]
                        center += np.random.randint(
                            -random_shift,
                            random_shift,
                            size=(3),
                        )
                self.training_points_by_object.append(
                    [Coordinate(np.round(center * self.voxel_size[0]).astype(int))]
                )
            else:
                if annotation_length == 0:
                    warnings.warn(f"empty id {id} {annotation_start} {annotation_end}")
                # c = np.round(((annotation_start + annotation_end) * self.resolution / 2)).astype(int)
                self.removed_ids.append(id)
        print(
            f"number of original centers: {len(self.annotation_starts)}, number of training centers: {len(self.training_points_by_object)}"
        )
        # if self.use_negative_examples:
        #     pseudorandom_training_centers += negative_example_centers

    def get_central_axis_points_as_training_points(self):
        def point_is_valid_for_current_roi(pt, min_distance, roi):
            # a point is considered a valid center if the loss bounding box for it does not cross the validation crop
            deltas = [
                np.max([roi.begin[d] - pt[d], 0, pt[d] - roi.end[d]]) for d in range(3)
            ]

            return np.linalg.norm(deltas) > min_distance

        def point_is_valid(pt, min_distance):
            if pt not in self.all_annotation_voxels_set:
                return False
            for roi in list(
                self.roi_calculator.rois_dict["validation"].values()
            ) + list(self.roi_calculator.rois_dict["test"].values()):
                roi = roi.snap_to_grid(
                    3 * [self.voxel_size[0]],
                    mode="shrink",
                )

                # do this to keep things in voxels for splitting along exact voxel
                roi /= int(self.voxel_size[0])

                if not point_is_valid_for_current_roi(pt, min_distance, roi):
                    return False
            return True

        self.training_points_by_object = []
        self.removed_ids = []
        for id, s, e in tqdm(
            zip(
                list(range(1, len(self.annotation_starts) + 1)),
                self.annotation_starts,
                self.annotation_ends,
            ),
            total=len(self.annotation_starts),
        ):
            s = np.floor(s)
            e = np.floor(e)
            annotation_length = np.linalg.norm(s - e)
            if annotation_length == 0:
                warnings.warn(f"empty id {id} {s} {e}")
                self.removed_ids.append(id)
                continue

            found_valid_point = False
            line = bresenham3DWithMask(s[0], s[1], s[2], e[0], e[1], e[2])
            for pt in line:
                if point_is_valid(pt, self.longest_box_diagonal):
                    self.training_points_by_object.append(
                        [
                            Coordinate(
                                np.round(np.array(pt) * self.voxel_size[0]).astype(int)
                            )
                        ]
                    )
                    found_valid_point = True

            if not found_valid_point:
                self.removed_ids.append(id)

    def get_all_training_points(self, random_shift_voxels=None):
        def valid_for_roi(pts, min_distance, roi):
            # a point is considered a valid center if the loss bounding box for it does not cross the validation crop
            # the loss bounding box is a cube with side length 2*min_distance
            deltas = [
                np.maximum(
                    0,
                    np.maximum(
                        roi.begin[d] - pts[:, d],
                        pts[:, d] - roi.end[d],
                    ),
                )
                for d in range(3)
            ]

            return np.linalg.norm(deltas, axis=0) > min_distance

        def get_valid_idxs(pts, min_distance):
            valid_idxs = np.ones(pts.shape[0], dtype=bool)
            for roi in list(
                self.roi_calculator.rois_dict["validation"].values()
            ) + list(self.roi_calculator.rois_dict["test"].values()):
                roi = roi.snap_to_grid(
                    3 * [self.voxel_size[0]],
                    mode="shrink",
                )

                # do this to keep things in voxels for splitting along exact voxel
                roi /= int(self.voxel_size[0])
                valid_idxs &= valid_for_roi(pts, min_distance, roi)

            return valid_idxs

        self.training_points_by_object = []
        self.removed_ids = []
        for id, cylinder_voxels in tqdm(
            enumerate(self.all_annotation_voxels), total=len(self.all_annotation_voxels)
        ):
            if len(cylinder_voxels) == 0:
                warnings.warn(f"empty id {id+1}")
                self.removed_ids.append(id + 1)
                continue

            found_valid_point = False
            cylinder_voxels = np.array(cylinder_voxels)
            valid_idxs = get_valid_idxs(cylinder_voxels, self.longest_box_diagonal)
            if np.sum(valid_idxs) > 0:
                valid_voxels = cylinder_voxels[valid_idxs]
                valid_voxels = np.round(valid_voxels * self.voxel_size[0]).astype(int)
                if random_shift_voxels:
                    random_shift = random_shift_voxels * self.voxel_size[0]
                    valid_voxels += np.random.randint(
                        -random_shift,
                        random_shift,
                        size=(len(valid_voxels), 3),
                    )
                # map cylinder voxels to tuple
                valid_voxels = list(map(Coordinate, valid_voxels))
                self.training_points_by_object.append(valid_voxels)
                found_valid_point = True

            if not found_valid_point:
                self.removed_ids.append(id + 1)

    def remove_validation_or_test_annotations_from_training(self, mode="all"):
        if mode == "deprecated_use_only_single_point":
            self.get_pseudorandom_training_centers()
        elif mode == "central_axis":
            self.get_central_axis_points_as_training_points()
        elif mode == "all":
            self.get_all_training_points(random_shift_voxels=18)
        else:
            raise Exception(f"mode {mode} not recognized")

    def write_out_annotations(self, output_directory, annotation_ids):
        annotation_type = "line"
        os.makedirs(f"{output_directory}/spatial0")
        if annotation_type == "line":
            coords_to_write = 6
        else:
            coords_to_write = 3

        annotations = (
            np.column_stack((self.annotation_starts, self.annotation_ends))
            * self.voxel_size[0]
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
            "dimensions": {"z": [1, "nm"], "y": [1, "nm"], "x": [1, "nm"]},
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
        print(f"annotations: {precomputed_path}")

    def visualize_removed_annotations(self, roi, radius):
        def add_segmentation_layer(state, data, name):
            dimensions = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"], units="nm", scales=3 * [self.voxel_size[0]]
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
                        scales=3 * [self.voxel_size[0]],
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
        for id, annotation_start, annotation_end in tqdm(
            zip(
                list(range(1, len(self.annotation_starts) + 1)),
                self.annotation_starts,
                self.annotation_ends,
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
                        voxels_in_cylinder[:, 0] - expanded_offset[0],
                        voxels_in_cylinder[:, 1] - expanded_offset[1],
                        voxels_in_cylinder[:, 2] - expanded_offset[2],
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
        annotation_datetime = (
            self.roi_calculator.annotation_csvs[0]
            .split("annotations_")[-1]
            .split(".csv")[0]
        )
        url = f"https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B0.0%2C0.0%2C0.0%5D%2C%22crossSectionScale%22:1%2C%22projectionScale%22:16384%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22n5://https://cellmap-vm1.int.janelia.org/nrs/data/{self.dataset}/{self.dataset}.n5/{self.raw_dataset_name}/%22%2C%22tab%22:%22source%22%2C%22name%22:%22fibsem-uint8%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22precomputed://https://cellmap-vm1.int.janelia.org/dm11/{self.username}/neuroglancer_annotations/{self.organelle}/splitting/{self.dataset}/bounding_boxes%22%2C%22tab%22:%22rendering%22%2C%22shader%22:%22%5Cnvoid%20main%28%29%20%7B%5Cn%20%20setColor%28prop_box_color%28%29%5Cn%20%20%20%20%20%20%20%20%20%20%29%3B%5Cn%7D%5Cn%22%2C%22name%22:%22bounding_boxes%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22n5://https://cellmap-vm1.int.janelia.org/nrs/{self.username}/cellmap/{self.organelle}/{self.organelle}.n5/{self.dataset}/%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%5D%2C%22name%22:%22{self.organelle}%22%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22precomputed://https://cellmap-vm1.int.janelia.org/dm11/{self.username}/neuroglancer_annotations/{annotation_datetime}%22%2C%22tab%22:%22source%22%2C%22name%22:%22{annotation_datetime}%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%2220230830_155757%22%7D%2C%22layout%22:%224panel%22%7D"
        print(url)

    def get_neuroglancer_url(self):
        state = parse_url(self.roi_calculator.neuroglancer_url)
        for name, zarr_path in zip(
            ["mask", "annotations_as_cylinders", "training_points"],
            [
                self.output_mask_zarr,
                self.output_gt_zarr,
                self.output_training_points_zarr,
            ],
        ):
            zarr_path = (
                zarr_path.replace("/nrs/cellmap", "nrs/")
                .replace("/dm11/cellmap", "dm11/")
                .replace("/prfs/cellmap", "prfs/")
            )
            zarr_path = "zarr://https://cellmap-vm1.int.janelia.org/" + zarr_path
            state.layers[name] = neuroglancer.SegmentationLayer(
                source=zarr_path + "/" + self.dataset
            )
        if len(self.removed_ids):
            print(f"{len(self.removed_ids)=}")
            state.layers["removed_annotations"] = neuroglancer.AnnotationLayer(
                source=f"{self.output_annotations_directory}/removed_annotations"
            )
            state.layers["kept_annotations"] = neuroglancer.AnnotationLayer(
                source=f"{self.output_annotations_directory}/kept_annotations"
            )
        print(f"Neuroglancer url:\n{neuroglancer.to_url(state)}")

    def standard_processing(self):
        print("Extract information from annotations:")
        self.extract_annotation_information()
        print("Write annotations as cylinders:")
        self.write_annotations_as_cylinders_and_get_intersections(radius=self.radius)
        print("Write intersections mask:")
        self.write_intersection_mask()
        print("Remove validation/test annotations from training:")
        self.remove_validation_or_test_annotations_from_training(
            self.training_point_selection_mode
        )
        print("Write training points:")
        self.write_training_points()

        if self.removed_ids:
            removed_annotations_dir = (
                f"{self.output_annotations_directory}/removed_annotations"
            )
            kept_annotations_dir = (
                f"{self.output_annotations_directory}/kept_annotations"
            )

            for annotations_dir in [removed_annotations_dir, kept_annotations_dir]:
                if os.path.isdir(annotations_dir):
                    shutil.rmtree(annotations_dir)

            self.write_out_annotations(
                output_directory=f"{self.output_annotations_directory}/removed_annotations",
                annotation_ids=self.removed_ids,
            )
            self.write_out_annotations(
                output_directory=f"{self.output_annotations_directory}/kept_annotations",
                annotation_ids=[
                    id
                    for id in range(1, len(self.annotation_starts) + 1)
                    if id not in self.removed_ids
                ],
            )
        self.get_neuroglancer_url()
        # self.get_neuroglancer_view()

    def save(self, output_path):
        # save to pkl file
        with open(output_path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def create_dacapo_run(
        self,
        lr=5e-5,
        batch_size=2,
        lsds_to_affs_weight_ratio=0.5,
        validation_interval=5000,
        snapshot_interval=10000,
        iterations=200000,
        repetitions=3,
        start_config=CosemStartConfig("setup04", "1820500"),
    ):
        DacapoRunBuilder(
            self.dataset,
            self.organelle,
            mask_zarr=self.output_mask_zarr,
            gt_zarr=self.output_gt_zarr,
            training_points=self.training_points,
            training_point_selection_mode=self.training_point_selection_mode,
            validation_rois_dict=self.roi_calculator.rois_dict["validation"],
            base_lr=lr,
            batch_size=batch_size,
            lsds_to_affs_weight_ratio=lsds_to_affs_weight_ratio,
            validation_interval=validation_interval,
            snapshot_interval=snapshot_interval,
            iterations=iterations,
            repetitions=repetitions,
            start_config=start_config,
        )
