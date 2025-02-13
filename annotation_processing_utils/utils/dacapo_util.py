from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    IntensityAugmentConfig,
)
from dacapo.experiments.tasks import AffinitiesTaskConfig
from funlib.geometry.coordinate import Coordinate
import math

from pathlib import Path
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    IntensitiesArrayConfig,
    CropArrayConfig,
)
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from funlib.geometry import Roi
from dacapo.experiments import RunConfig
from dacapo.experiments.starts import StartConfig
from dacapo.store.create_store import create_config_store
import getpass

from decimal import Decimal


def scientific_to_plain_string(number):
    # Convert to a Decimal and quantize it to the smallest necessary precision
    decimal_number = Decimal(number).normalize()
    # Convert to string, ensuring no trailing zeros
    plain_string = (
        str(decimal_number).rstrip("0").rstrip(".")
        if "." in str(decimal_number)
        else str(decimal_number)
    )
    return plain_string


class DacapoRunBuilder:
    def __init__(
        self,
        dataset=None,
        organelle=None,
        mask_zarr=None,
        gt_zarr=None,
        training_points=None,
        training_point_selection_mode=None,
        validation_rois_dict=None,
        datasplit_config=None,
        base_lr=2.5e-5,
        batch_size=2,
        lsds_to_affs_weight_ratio=0.5,
        validation_interval=5000,
        snapshot_interval=10000,
        iterations=200000,
        repetitions=3,
        start_config=None,
    ):
        lr = base_lr * batch_size
        config_store = create_config_store()
        self.create_task(lsds_to_affs_weight_ratio)

        if datasplit_config:
            self.datasplit_config = datasplit_config
        else:
            self.create_datasplit(
                dataset,
                organelle,
                gt_zarr,
                mask_zarr,
                training_points,
                training_point_selection_mode,
                validation_rois_dict,
                config_store,
            )

        self.architecture_config = self.create_architecture()
        self.create_trainer(lr, batch_size, snapshot_interval)
        self.create_run(
            self.task_config,
            self.datasplit_config,
            self.architecture_config,
            self.trainer_config,
            iterations,
            validation_interval,
            repetitions,
            start_config,
            config_store,
        )

    def create_trainer(self, lr, batch_size=2, snapshot_interval=5000):
        lr_str = scientific_to_plain_string(str(lr))
        self.trainer_config = GunpowderTrainerConfig(
            name=f"default_trainer_lr_{lr_str}_bs_{batch_size}",
            batch_size=batch_size,
            learning_rate=lr,
            augments=[
                ElasticAugmentConfig(
                    control_point_spacing=(100, 100, 100),
                    control_point_displacement_sigma=(10.0, 10.0, 10.0),
                    rotation_interval=(0, math.pi / 2.0),
                    subsample=8,
                    uniform_3d_rotation=True,
                ),
                IntensityAugmentConfig(
                    scale=(0.7, 1.3),
                    shift=(-0.2, 0.2),
                    clip=True,
                ),
            ],
            num_data_fetchers=20,
            snapshot_interval=snapshot_interval,
            min_masked=0.05,
        )

    def create_task(self, lsds_to_affs_weight_ratio):
        self.task_config = AffinitiesTaskConfig(
            name=f"3d_lsdaffs_weight_ratio_{lsds_to_affs_weight_ratio}",
            neighborhood=[
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (3, 0, 0),
                (0, 3, 0),
                (0, 0, 3),
                (9, 0, 0),
                (0, 9, 0),
                (0, 0, 9),
            ],
            lsds=True,
            lsds_to_affs_weight_ratio=lsds_to_affs_weight_ratio,
        )

    @staticmethod
    def create_architecture():
        return CNNectomeUNetConfig(
            name="unet",
            input_shape=Coordinate(216, 216, 216),
            eval_shape_increase=Coordinate(72, 72, 72),
            fmaps_in=1,
            num_fmaps=12,
            fmaps_out=72,
            fmap_inc_factor=6,
            downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
        )

    def create_datasplit(
        self,
        dataset,
        organelle,
        gt_zarr,
        mask_zarr,
        training_points,
        training_point_selection_mode,
        validation_rois_dict,
        config_store=None,
    ):
        # use pseudorandom centers
        raw_config = ZarrArrayConfig(
            name="raw",
            file_name=Path(f"/nrs/cellmap/data/{dataset}/{dataset}.zarr"),
            dataset="recon-1/em/fibsem-uint8/s0",
        )
        # We get an error without this, and will suggests having it as such https://cell-map.slack.com/archives/D02KBQ990ER/p1683762491204909
        raw_config = IntensitiesArrayConfig(
            name="raw", source_array_config=raw_config, min=0, max=255
        )

        gt_config = ZarrArrayConfig(
            name=organelle,
            file_name=Path(gt_zarr),
            dataset=dataset + "/s0",
        )

        # mask out regions of overlapping plasmodesmata
        mask_config = ZarrArrayConfig(
            name="mask",
            file_name=Path(mask_zarr),
            dataset=dataset + "/s0",
        )

        validation_data_config = []
        for roi_name, roi in validation_rois_dict.items():
            val_gt_config = CropArrayConfig(
                f"{roi_name}_gt", source_array_config=gt_config, roi=roi
            )
            validation_data_config.append(
                RawGTDatasetConfig(
                    f"val_{roi_name}",
                    raw_config=raw_config,
                    gt_config=val_gt_config,
                    mask_config=mask_config,
                )
            )

        training_data_config = RawGTDatasetConfig(
            f"train",
            raw_config=raw_config,
            gt_config=gt_config,
            sample_points=training_points,
            mask_config=mask_config,
        )
        self.datasplit_config = TrainValidateDataSplitConfig(
            name=f"{dataset}_{organelle}_{training_point_selection_mode}_training_points",
            train_configs=[training_data_config],
            validate_configs=validation_data_config,
        )

        # store it so that can combine later
        if config_store:
            config_store.store_datasplit_config(self.datasplit_config)

    def create_run(
        self,
        task_config,
        datasplit_config,
        architecture_config,
        trainer_config,
        iterations=500000,
        validation_interval=5000,
        repetitions=3,
        start_config=None,
        config_store=None,
    ):
        # make validation interval huge so don't have to deal with validation until after the fact
        validation_interval = validation_interval
        for i in range(repetitions):
            run_config = RunConfig(
                name=("_").join(
                    [
                        "scratch" if start_config is None else "finetuned",
                        task_config.name,
                        datasplit_config.name,
                        architecture_config.name,
                        trainer_config.name,
                    ]
                )
                + f"__{i}",
                task_config=task_config,
                datasplit_config=datasplit_config,
                architecture_config=architecture_config,
                trainer_config=trainer_config,
                num_iterations=iterations,
                validation_interval=validation_interval,
                repetition=i,
                start_config=start_config,
            )

            if config_store:
                config_store.store_run_config(run_config)
