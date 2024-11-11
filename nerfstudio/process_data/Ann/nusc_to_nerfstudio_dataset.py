# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processes an image sequence to a nerfstudio compatible dataset."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, List

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.process_data.Ann import hloc_utils

import numpy as np
import pyquaternion
import torch
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
import pickle
import json


@dataclass
class NuscToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process nuScenes into a nerfstudio dataset.

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    scene_names: Optional[List[str]] = field(default_factory=list)
    """Name of the scenes."""
    centroid_name: str = "0"
    """Name of centroid"""
    location: str = "singapore-onenorth"
    """Name of location"""
    cameras: Tuple[Literal["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], ...] = (
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    """Which cameras to use."""

    @property
    def absolute_colmap_model_path(self) -> Path:
        return self.absolute_colmap_path / "sparse/0"

    @property
    def absolute_colmap_path(self) -> Path:
        if len(self.cameras) == 1:
            return self.output_dir / "colmap" / f"{self.location}_c{self.centroid_name}" / f"{self.cameras[0]}"
        else:
            return self.output_dir / "colmap" / f"{self.location}_c{self.centroid_name}"

    def main(self) -> None:
        """Process nuScenes into a nerfstudio dataset."""
        
        transform1 = torch.tensor(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32
        )
        transform2 = torch.tensor(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32
        )

        summary_log = []

        sample_data_list = []

        with open(os.path.join("./nerfstudio/scripts/datasets/",
            f"{self.location}",
            f"{self.location}_centroids.json"), 'r') as f:
            scene_dict = json.load(f)
            # [self.scene_names.extend(v) for k, v in scene_dict.items()]
            self.scene_names.extend(scene_dict.get(self.centroid_name, []))
        
        assert len(self.scene_names) != 0, f"centroid {self.centroid_name} out of bound"
        
        for scene_name in self.scene_names:
            with open(os.path.join(str(self.data), "PreSight", f'{scene_name}.pkl'), 'rb') as f:
                sample_data_list.extend(pickle.load(f))

        # sort by timestamp
        sample_data_list.sort(key=lambda x: x["timestamp"])
        
        # get image filenames and camera data
        self.image_filenames = []
        # data_dir = str(self.config.data_dir.resolve())
        for i, sample_data in enumerate(sample_data_list):
            if sample_data['channel'] not in self.cameras:
                continue

            ego_pose = torch.tensor(sample_data['ego2global'], dtype=torch.float32)
            cam_pose = torch.tensor(sample_data['cam2ego'], dtype=torch.float32)
            pose = ego_pose @ cam_pose

            # rotate to opencv frame
            pose = transform1 @ pose

            # convert from opencv camera to nerfstudio camera
            pose[0:3, 1:3] *= -1
            pose = pose[np.array([1, 0, 2, 3]), :]
            pose[2, :] *= -1

            # rotate to z-up in viewer
            pose = transform2 @ pose
            img_fpath = Path(sample_data['filename'])
            self.image_filenames.append(img_fpath.relative_to(self.data))
        
        num_frames = len(self.image_filenames)
        summary_log.append(f"Starting with {num_frames} images")

        # Run COLMAP
        if not self.skip_colmap:
            require_cameras_exist = True
            self._run_colmap()
            # Colmap uses renamed images
            image_rename_map = None

        # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp

        if require_cameras_exist and not (self.absolute_colmap_model_path / "cameras.bin").exists():
            raise RuntimeError(f"Could not find existing COLMAP results ({self.absolute_colmap_model_path / 'cameras.bin'}).")

        summary_log += self._save_transforms(
            num_frames,
            image_id_to_depth_path,
            None,
            image_rename_map,
        )

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)

    def _run_colmap(self, mask_path: Optional[Path] = None):
        """
        Args:
            mask_path: Path to the camera mask. Defaults to None.
        """
        self.absolute_colmap_path.mkdir(parents=True, exist_ok=True)

        (
            sfm_tool,
            feature_type,
            matcher_type,
        ) = process_data_utils.find_tool_feature_matcher_combination(
            self.sfm_tool, self.feature_type, self.matcher_type
        )
        # check that sfm_tool is hloc if using refine_pixsfm
        if self.refine_pixsfm:
            assert sfm_tool == "hloc", "refine_pixsfm only works with sfm_tool hloc"

        # set the image_dir if didn't copy
        # if self.skip_image_processing:
        #     image_dir = self.data
        # else:
        #     image_dir = self.image_dir
        image_dir = self.data

        if sfm_tool == "colmap":
            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                camera_mask_path=mask_path,
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                refine_intrinsics=self.refine_intrinsics,
                colmap_cmd=self.colmap_cmd,
            )
        elif sfm_tool == "hloc":
            if mask_path is not None:
                raise RuntimeError("Cannot use a mask with hloc. Please remove the cropping options " "and try again.")

            assert feature_type is not None
            assert matcher_type is not None
            assert matcher_type != "NN"  # Only used for colmap.
            # TODO
            # maybe run hloc scene by scene, then transform extracted features by first frame's transform?
            hloc_utils.run_hloc(
                image_dir=image_dir,
                references=self.image_filenames,
                colmap_dir=self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                verbose=self.verbose,
                matching_method=self.matching_method,
                feature_type=feature_type,
                matcher_type=matcher_type,
                refine_pixsfm=self.refine_pixsfm,
            )
        else:
            raise RuntimeError("Invalid combination of sfm_tool, feature_type, and matcher_type, " "exiting")
