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
from typing import Literal, Optional, Tuple, List, Dict

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.process_data.Ann import colmap_utils, hloc_utils

import numpy as np
import pyquaternion
import torch
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
import pickle
import json

from pyquaternion import Quaternion


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
            return self.output_dir / self.sfm_tool / f"{self.location}_c{self.centroid_name}" / f"{self.cameras[0]}"
        else:
            return self.output_dir / self.sfm_tool / f"{self.location}_c{self.centroid_name}"

    def _save_transforms(
        self,
        num_frames: int,
        scene_name: Optional[str] = None,
        image_id_to_depth_path: Optional[Dict[int, Path]] = None,
        camera_mask_path: Optional[Path] = None,
        image_rename_map: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Save colmap transforms into the output folder

        Args:
            image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
            image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        """
        summary_log = []
        absolute_colmap_model_path = self.absolute_colmap_path / scene_name / "sparse/0" if scene_name is not None else self.absolute_colmap_path / "sparse/0"
        if (absolute_colmap_model_path / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    recon_dir=absolute_colmap_model_path,
                    image_dir=self.data,
                    output_dir=self.absolute_colmap_path / scene_name if scene_name is not None else self.absolute_colmap_path,
                    image_id_to_depth_path=image_id_to_depth_path,
                    camera_mask_path=camera_mask_path,
                    image_rename_map=image_rename_map,
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_frames, num_matched_frames))

        else:
            CONSOLE.log(
                "[bold yellow]Warning: Could not find existing COLMAP results. " "Not generating transforms.json"
            )
        return summary_log

    def main(self) -> None:
        """Process nuScenes into a nerfstudio dataset."""
        
        transform1 = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        transform2 = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        self.refine_intrinsics = False

        with open(os.path.join("./nerfstudio/scripts/datasets/",
            f"{self.location}",
            f"{self.location}_centroids.json"), 'r') as f:
            scene_dict = json.load(f)
            # [self.scene_names.extend(v) for k, v in scene_dict.items()]
            self.scene_names.extend(scene_dict.get(self.centroid_name, []))
        
        assert len(self.scene_names) != 0, f"centroid {self.centroid_name} out of bound"
        
        # ######## recon by each scene ########
        # for scene_name in self.scene_names:
        #     summary_log = []
        #     sample_data_list = []

        #     require_cameras_exist = False
        #     if self.colmap_model_path != ColmapConverterToNerfstudioDataset.default_colmap_path():
        #         if not self.skip_colmap:
        #             raise RuntimeError("The --colmap-model-path can only be used when --skip-colmap is not set.")
        #         if not (self.output_dir / self.colmap_model_path).exists():
        #             raise RuntimeError(f"The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist.")
        #         require_cameras_exist = True

        #     summary_log.append(f"Reconstructing {scene_name} in {self.location}")

        #     with open(os.path.join(str(self.data), "PreSight", f'{scene_name}.pkl'), 'rb') as f:
        #         sample_data_list.extend(pickle.load(f))

        #     # sort by timestamp
        #     sample_data_list.sort(key=lambda x: x["timestamp"])
            
        #     # get image filenames and camera data
        #     self.image_filenames = []
        #     # data_dir = str(self.config.data_dir.resolve())
        #     init_model_path = self.absolute_colmap_path / scene_name / "init"
        #     init_model_path.mkdir(parents=True, exist_ok=True)
        #     if (init_model_path / "points3D.txt").exists():
        #         (init_model_path / "points3D.txt").unlink()
        #     with open(init_model_path / "points3D.txt", 'w') as f:
        #         pass
        #     cam_ids = {cam: self.cameras.index(cam)+1 for cam in self.cameras}
        #     cam_intrins_added = [] # ID: [fx fy cx cy]
            
        #     if (init_model_path / "cameras.txt").exists():
        #         (init_model_path / "cameras.txt").unlink()
        #     fcam = open(init_model_path / "cameras.txt", 'w') # CAMERA_ID CAMERA_MODELS[self.camera_type] w h fx fy cx cy
            
        #     if (init_model_path / "images.txt").exists():
        #         (init_model_path / "images.txt").unlink()
        #     with open(init_model_path / "images.txt", 'w') as f: # ID qw qx qy qz tx ty tz CAMERA_ID filename
        #         for i, sample_data in enumerate(sample_data_list):
        #             # create initial model with sample_data['cam_intrinsic'] 3x3
        #             # sample_data['ego2global'] @ sample_data['cam2ego']
        #             if sample_data['channel'] not in self.cameras:
        #                 continue

        #             channel = sample_data['channel']
        #             if cam_ids[channel] not in cam_intrins_added:
        #                 cam_intrins_added.append(cam_ids[channel])
        #                 intrins = sample_data['cam_intrinsic']
        #                 tmp = " ".join(map(str, (cam_ids[channel], CAMERA_MODELS[self.camera_type].value, 1600, 900, intrins[0][0], intrins[1][1], intrins[0][2], intrins[1][2], 0, 0, 0, 0)))
        #                 fcam.writelines(tmp+"\n")

        #             ego_pose = sample_data['ego2global']
        #             cam_pose = sample_data['cam2ego']
        #             pose = ego_pose @ cam_pose # cam2world
        #             pose = np.linalg.inv(pose) # world2cam

        #             # # rotate to opencv frame
        #             # pose = transform1 @ pose

        #             # # convert from opencv camera to nerfstudio camera
        #             # pose[0:3, 1:3] *= -1
        #             # pose = pose[np.array([1, 0, 2, 3]), :]
        #             # pose[2, :] *= -1

        #             # # rotate to z-up in viewer
        #             # pose = transform2 @ pose
        #             img_fpath = Path(sample_data['filename'])
        #             self.image_filenames.append(img_fpath.relative_to(self.data))

        #             quat = Quaternion(matrix=pose)
        #             tmp = " ".join(map(str, (i+1, *quat, *pose[:3, 3], cam_ids[channel], img_fpath.relative_to(self.data))))
        #             f.writelines(tmp+"\n\n")
        #     fcam.close()
            
        #     num_frames = len(self.image_filenames)
        #     summary_log.append(f"Starting with {num_frames} images")

        #     image_rename_map: Optional[Dict[str, str]] = None

        #     # Run COLMAP
        #     if not self.skip_colmap:
        #         require_cameras_exist = True
        #         self._run_colmap(scene_name)
        #         # Colmap uses renamed images
        #         image_rename_map = None

        #     # Export depth maps
        #     image_id_to_depth_path, log_tmp = self._export_depth()
        #     summary_log += log_tmp

        #     if require_cameras_exist and not (self.absolute_colmap_path / scene_name / "sparse/0/cameras.bin").exists():
        #         raise RuntimeError(f"Could not find existing COLMAP results ({self.absolute_colmap_path / scene_name / 'sparse/0/cameras.bin'}).")

        #     summary_log += self._save_transforms(
        #         num_frames,
        #         scene_name,
        #         image_id_to_depth_path,
        #         None,
        #         image_rename_map,
        #     )

        #     CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        #     for summary in summary_log:
        #         CONSOLE.log(summary)
            
        ######## recon all scenes as a single model ######## 
        summary_log = []
        sample_data_list = [] 

        summary_log.append(f"Reconstructing {len(self.scene_names)} scenes in {self.location}")

        require_cameras_exist = False
        if self.colmap_model_path != ColmapConverterToNerfstudioDataset.default_colmap_path():
            if not self.skip_colmap:
                raise RuntimeError("The --colmap-model-path can only be used when --skip-colmap is not set.")
            if not (self.output_dir / self.colmap_model_path).exists():
                raise RuntimeError(f"The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist.")
            require_cameras_exist = True

        for scene_name in self.scene_names:

            with open(os.path.join(str(self.data), "PreSight", f'{scene_name}.pkl'), 'rb') as f:
                sample_data_list.extend(pickle.load(f))

            # sort by timestamp
            sample_data_list.sort(key=lambda x: x["timestamp"])   

        init_model_path = self.absolute_colmap_path / "init"
        init_model_path.mkdir(parents=True, exist_ok=True)
        if (init_model_path / "points3D.txt").exists():
            (init_model_path / "points3D.txt").unlink()
        with open(init_model_path / "points3D.txt", 'w') as f:
            pass
        cam_ids = {cam: self.cameras.index(cam)+1 for cam in self.cameras}
        cam_intrins_added = [] # ID: [fx fy cx cy]
        
        if (init_model_path / "cameras.txt").exists():
            (init_model_path / "cameras.txt").unlink()
        fcam = open(init_model_path / "cameras.txt", 'w') # CAMERA_ID CAMERA_MODELS[self.camera_type] w h fx fy cx cy
        
        if (init_model_path / "images.txt").exists():
            (init_model_path / "images.txt").unlink()
        with open(init_model_path / "images.txt", 'w') as fimg: # ID qw qx qy qz tx ty tz CAMERA_ID filename                
            # get image filenames and camera data          
            self.image_filenames = []
            for i, sample_data in enumerate(sample_data_list):
                # create initial model with sample_data['cam_intrinsic'] 3x3
                # sample_data['ego2global'] @ sample_data['cam2ego']
                if sample_data['channel'] not in self.cameras:
                    continue

                channel = sample_data['channel']
                if cam_ids[channel] not in cam_intrins_added:
                    cam_intrins_added.append(cam_ids[channel])
                    intrins = sample_data['cam_intrinsic']
                    tmp = " ".join(map(str, (cam_ids[channel], CAMERA_MODELS[self.camera_type].value, 1600, 900, intrins[0][0], intrins[1][1], intrins[0][2], intrins[1][2], 0, 0, 0, 0)))
                    fcam.writelines(tmp+"\n")

                ego_pose = sample_data['ego2global']
                cam_pose = sample_data['cam2ego']
                pose = ego_pose @ cam_pose # cam2world
                pose = np.linalg.inv(pose) # world2cam

                # # rotate to opencv frame
                # pose = transform1 @ pose

                # # convert from opencv camera to nerfstudio camera
                # pose[0:3, 1:3] *= -1
                # pose = pose[np.array([1, 0, 2, 3]), :]
                # pose[2, :] *= -1

                # # rotate to z-up in viewer
                # pose = transform2 @ pose
                img_fpath = Path(sample_data['filename'])
                self.image_filenames.append(img_fpath.relative_to(self.data))

                quat = Quaternion(matrix=pose)
                tmp = " ".join(map(str, (i+1, *quat, *pose[:3, 3], cam_ids[channel], img_fpath.relative_to(self.data))))
                fimg.writelines(tmp+"\n\n")
        fcam.close()
        
        num_frames = len(self.image_filenames)
        summary_log.append(f"Starting with {num_frames} images")

        image_rename_map: Optional[Dict[str, str]] = None

        # Run COLMAP
        if not self.skip_colmap:
            require_cameras_exist = True
            self._run_colmap()
            # Colmap uses renamed images
            image_rename_map = None

        # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp

        if require_cameras_exist and not (self.absolute_colmap_path / "sparse/0/cameras.bin").exists():
            raise RuntimeError(f"Could not find existing COLMAP results ({self.absolute_colmap_path / 'sparse/0/cameras.bin'}).")

        summary_log += self._save_transforms(
            num_frames,
            image_id_to_depth_path,
            None,
            image_rename_map,
        )

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)

    def _run_colmap(self, scene_name: Optional[str] = None, mask_path: Optional[Path] = None):
        """
        Args:
            mask_path: Path to the camera mask. Defaults to None.
        """
        self.absolute_colmap_path.mkdir(parents=True, exist_ok=True)
        if scene_name is not None:
            (self.absolute_colmap_path / scene_name).mkdir(parents=True, exist_ok=True)

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
                references=self.image_filenames,
                colmap_dir=self.absolute_colmap_path / scene_name if scene_name is not None else self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                reference_model_dir=self.absolute_colmap_path / scene_name / "init"
                    if scene_name is not None else self.absolute_colmap_path / "init",
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
                colmap_dir=self.absolute_colmap_path / scene_name if scene_name is not None else self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                reference_model_dir=self.absolute_colmap_path / scene_name / "init"
                    if scene_name is not None else self.absolute_colmap_path / "init",
                verbose=self.verbose,
                matching_method=self.matching_method,
                feature_type=feature_type,
                matcher_type=matcher_type,
                refine_pixsfm=self.refine_pixsfm,
            )
        else:
            raise RuntimeError("Invalid combination of sfm_tool, feature_type, and matcher_type, " "exiting")
