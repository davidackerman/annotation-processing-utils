import daisy
from funlib.persistence import Array, open_ds
from funlib.geometry import Roi, Coordinate
import numpy as np
import tempfile
import pickle
import os
import scipy
import json
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# encoder for uint64 from https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class InstanceSegmentationOverlap:
    def __init__(
        self,
        gt_array: Array,
        test_array: Array,
        mask_array: Array = None,
        total_roi: Roi = None,
        log_dir: str = None,
        num_workers: int = 10,
    ):
        self.gt_array = gt_array
        self.test_array = test_array
        if total_roi:
            self.total_roi = total_roi
        else:
            self.total_roi = test_array.roi
        self.mask_array = mask_array
        self.read_write_roi: Roi = Roi(
            (0, 0, 0), self.gt_array.chunk_shape * self.gt_array.voxel_size
        )
        self.num_workers = num_workers
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            daisy.logging.set_log_basedir(log_dir)

    # function that returns a python dictionary
    def __get_overlap_dict_blockwise(
        self,
        block: daisy.Block,
        tmpdirname,
    ):
        _, block_id = block.block_id
        gt_block = self.gt_array.to_ndarray(block.read_roi, fill_value=0)
        test_block = self.test_array.to_ndarray(block.read_roi, fill_value=0)

        if self.mask_array:
            mask_roi = block.read_roi.snap_to_grid(self.mask_array.voxel_size)
            # assume isotropic
            mask_voxel_size = self.mask_array.voxel_size[0]
            gt_voxel_size = self.gt_array.voxel_size[0]
            scale_factor = int(mask_voxel_size / gt_voxel_size)
            mask_block = self.mask_array.to_ndarray(mask_roi)
            mask_block = (
                mask_block.repeat(scale_factor, axis=0)
                .repeat(scale_factor, axis=1)
                .repeat(scale_factor, axis=2)
            )
            mask_begin_voxels = (block.read_roi.begin - mask_roi.begin) / gt_voxel_size
            mask_end_voxels = mask_begin_voxels + Coordinate(gt_block.shape)
            mask_block = mask_block[
                mask_begin_voxels[0] : mask_end_voxels[0],
                mask_begin_voxels[1] : mask_end_voxels[1],
                mask_begin_voxels[2] : mask_end_voxels[2],
            ]

            test_block = np.multiply(test_block, mask_block)
        out_dict = {}
        # taken from funlib.evaluate detection
        # change logical_and to logical_or since we want total counts
        # prediction_mask = (
        #     prediction_mask.repeat(16, axis=0).repeat(16, axis=1).repeat(16, axis=2)
        # )

        # test_components = np.multiply(test_components, prediction_mask)

        # NOTE: had to chnage logical_and to logical_or since we are doing this in chunks
        # and want to make sure we have all our ids for downstream processing
        both_fg_mask = np.logical_or(gt_block > 0, test_block > 0)
        both_fg_true = gt_block[both_fg_mask].ravel()
        both_fg_test = test_block[both_fg_mask].ravel()
        if both_fg_true.size > 0:
            pairs, counts = np.unique(
                np.array([both_fg_true, both_fg_test]), axis=1, return_counts=True
            )

            for gt_id, test_id, count in zip(pairs[0], pairs[1], counts):
                out_dict[(gt_id, test_id)] = count

        with open(f"{tmpdirname}/block_{block_id}.pkl", "wb") as fp:
            pickle.dump(out_dict, fp)

    def __combine_block_dicts(self, tmpdirname):
        gt_test_counts = {}
        test_ids = {0}
        gt_ids = {0}
        for block_filename in os.listdir(tmpdirname):
            with open(f"{tmpdirname}/{block_filename}", "rb") as f:
                block_dict = pickle.load(f)
                if block_dict:
                    for (gt_id, test_id), v in block_dict.items():
                        gt_ids.add(gt_id)
                        test_ids.add(test_id)
                        gt_test_counts[(gt_id, test_id)] = (
                            gt_test_counts.get((gt_id, test_id), 0) + v
                        )

        # vs = np.array(list(test_id_counts.values()))
        test_id_renumbering = {test_id: i for i, test_id in enumerate(test_ids)}
        gt_id_renumbering = {gt_id: i for i, gt_id in enumerate(gt_ids)}
        # n_test = len(test_ids) - 1
        # n_gt = len(gt_ids) - 1
        gt_test_overlaps = np.zeros(
            (len(gt_id_renumbering), len(test_id_renumbering)), dtype=np.int64
        )
        for (gt_id, test_id), v in gt_test_counts.items():
            # TODO: Does it matter the order? In Funke lab stuff it is test, gt...
            gt_test_overlaps[gt_id_renumbering[gt_id], test_id_renumbering[test_id]] = v

        combined_dict = {
            "gt_test_counts": gt_test_counts,
            "gt_id_renumbering": gt_id_renumbering,
            "test_id_renumbering": test_id_renumbering,
            "gt_test_overlaps": gt_test_overlaps,
        }
        return combined_dict

    def get_overlap_dict(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            task = daisy.Task(
                total_roi=self.total_roi,
                read_roi=self.read_write_roi,
                write_roi=self.read_write_roi,
                process_function=lambda b: self.__get_overlap_dict_blockwise(
                    b, tmpdirname
                ),
                num_workers=self.num_workers,
                task_id="block_processing",
                fit="shrink",
            )
            # add export of scores
            daisy.run_blockwise([task])
            overlap_dict = self.__combine_block_dicts(tmpdirname)
        return overlap_dict


class InstanceSegmentationScorer:
    def __init__(self, overlap_dict):
        self.gt_test_overlaps = overlap_dict["gt_test_overlaps"]
        self.gt_test_counts = overlap_dict["gt_test_counts"]
        self.gt_id_renumbering = overlap_dict["gt_id_renumbering"]
        self.test_id_renumbering = overlap_dict["test_id_renumbering"]

    def __get_matches(self, array_to_optimize):
        gt_idxs, test_idxs = scipy.optimize.linear_sum_assignment(
            array_to_optimize, maximize=True
        )
        matches = [
            (gt_idx + 1, test_idx + 1)  # add one for background
            for gt_idx, test_idx in zip(gt_idxs, test_idxs)
            if array_to_optimize[gt_idx, test_idx]
            > 0  # f1 and iou require at least one voxel overlap
        ]
        return matches

    def __get_f1_score(self):
        # overlap f1 score
        gt_test_overlaps_without_background = self.gt_test_overlaps[1:, 1:]
        matches = self.__get_matches(gt_test_overlaps_without_background)
        n_gt, n_test = gt_test_overlaps_without_background.shape
        tp = len(matches)
        fp = n_test - tp
        fn = n_gt - tp
        f1_score = tp / (tp + 0.5 * (fp + fn))

        # iou score
        iou = self.__get_iou()
        average_iou = np.mean(
            [iou[gt_idx - 1, test_idx - 1] for (gt_idx, test_idx) in matches]
        )

        # relevant ids
        all_test_ids = list(self.test_id_renumbering.keys())
        tp_test_ids = set(all_test_ids[test_idx] for (_, test_idx) in matches)
        fp_test_ids = set(all_test_ids) - set([0])
        fp_test_ids -= tp_test_ids

        all_gt_ids = list(self.gt_id_renumbering.keys())
        tp_gt_ids = set(all_gt_ids[gt_idx] for (gt_idx, _) in matches)
        fn_gt_ids = set(all_gt_ids) - set([0])
        fn_gt_ids -= tp_gt_ids

        tp_gt_test_id_pairs = [
            [all_gt_ids[gt_idx], all_test_ids[test_idx]]
            for (gt_idx, test_idx) in matches
        ]

        output_dict = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1_score": f1_score,
            "iou": average_iou,
            "tp_gt_test_id_pairs": tp_gt_test_id_pairs,
            "fp_test_ids": list(fp_test_ids),
            "fn_gt_ids": list(fn_gt_ids),
        }

        return output_dict

    def __get_iou(self):
        # iou score
        gt_test_overlaps_without_background = self.gt_test_overlaps[1:, 1:]
        gt_volumes = np.sum(self.gt_test_overlaps, axis=1)[1:]  # ignore background
        test_volumes = np.sum(self.gt_test_overlaps, axis=0)[1:]  # ignore background
        # need to subtract overlap otherwise double count it
        union = (
            np.expand_dims(gt_volumes, 1) + np.expand_dims(test_volumes, 0)
        ) - gt_test_overlaps_without_background
        iou = gt_test_overlaps_without_background / union
        return iou

    def __get_iou_score(self):
        iou = self.__get_iou()
        matches = self.__get_matches(iou)
        average_iou = np.mean(
            [iou[gt_idx - 1, test_idx - 1] for (gt_idx, test_idx) in matches]
        )
        return {"iou_score": average_iou}

    def get_scores(self):
        f1_score_dict = self.__get_f1_score()
        iou_score_dict = self.__get_iou_score()
        return {"f1_score_info": f1_score_dict, "iou_score_info": iou_score_dict}


class InstanceSegmentationOverlapAndScorer:
    def __init__(
        self,
        gt_array: Array,
        test_array: Array,
        output_directory: str,
        mask_array: Array = None,
        log_dir: str = None,
        num_workers: int = 10,
    ):
        self.gt_array = gt_array
        self.test_array = test_array
        self.mask_array = mask_array
        self.output_directory = output_directory
        self.num_workers = num_workers
        self.log_dir = log_dir

    def process(self):
        iso = InstanceSegmentationOverlap(
            gt_array=self.gt_array,
            test_array=self.test_array,
            mask_array=self.mask_array,
            log_dir=self.log_dir,
            num_workers=self.num_workers,
        )
        overlap_dict = iso.get_overlap_dict()

        iss = InstanceSegmentationScorer(overlap_dict)
        scores_dict = iss.get_scores()

        os.makedirs(self.output_directory, exist_ok=True)
        with open(f"{self.output_directory}/scores.json", "w") as fp:
            json.dump(scores_dict, fp, cls=NpEncoder)
