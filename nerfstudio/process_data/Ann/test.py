import os
import sys
from pathlib import Path
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py
from typing import List

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

    
from nuscenes.map_expansion.map_api import *
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.bitmap import BitMap

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize map')
    parser.add_argument('--dataroot', 
                        type=str, 
                        default='./data/nuscenes',
                        help='root of nuScenes')
    parser.add_argument('--version',
                        type=str,
                        default='v1.0-trainval',
                        help='version of nuScenes')
    parser.add_argument('--map-name', 
                        type=str, 
                        default='singapore-onenorth',
                        help='map name')
    parser.add_argument('--scene',
                        type=str,
                        nargs='*',
                        default=None,
                        help='scene name to render')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=True)
    nusc_map = NuScenesMap(dataroot=args.dataroot, map_name=args.map_name)
    bitmap = BitMap('/home/user/data/juyeb/dataset/nuscenes', args.map_name, 'basemap')

    camera_types = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
    ]
    
    myjet = np.array([[0.        , 0.        , 0.5       ],
                      [0.        , 0.        , 0.99910873],
                      [0.        , 0.37843137, 1.        ],
                      [0.        , 0.83333333, 1.        ],
                      [0.30044276, 1.        , 0.66729918],
                      [0.66729918, 1.        , 0.30044276],
                      [1.        , 0.90123457, 0.        ],
                      [1.        , 0.48002905, 0.        ],
                      [0.99910873, 0.07334786, 0.        ],
                      [0.5       , 0.        , 0.        ]])

    feature_path = 'data/nuscenes/colmap/singapore-onenorth_c0/CAM_FRONT/scene-0962/features.h5'

    args.out_dir = os.path.dirname(feature_path)
    os.makedirs(args.out_dir, exist_ok=True)
    map_path = os.path.join(args.out_dir, f'{args.map_name}_map.jpg')
    
    scenes_to_render = [scene for scene in nusc.scene if nusc.get('log', scene['log_token'])['location'] == args.map_name]
    print(f'rendering {len(scenes_to_render)} scenes...')
    with h5py.File(feature_path, "r") as fd:
        for i, scene in (enumerate(scenes_to_render)):
            scene_name = scene['name']
            if args.scene is not None and scene_name not in args.scene:
                continue
            scene_dir = os.path.join(args.out_dir, scene_name)
            # os.makedirs(scene_dir, exist_ok=True)
            sample_tokens = nusc.field2token('sample', 'scene_token', scene['token'])
            vout = cv2.VideoWriter(os.path.join(scene_dir, 'surround_img.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 2, (2700, 1200))
            for sample_token in sample_tokens:
                sample_record = nusc.get('sample', sample_token)
                timestamp = sample_record['timestamp']
                out_dir = os.path.join(scene_dir, str(timestamp))
                # os.makedirs(out_dir, exist_ok=True)
                surround_img_path = os.path.join(out_dir, 'surround_img.jpg')
                im_surround = None
                # read image if already exists, otherwise render and save
                if os.path.exists(out_dir) and os.path.split(surround_img_path)[-1] in os.listdir(out_dir):
                    im_surround = cv2.imread(surround_img_path)
                else:
                    im_front = []
                    im_back = []
                    for i, camera_type in enumerate(camera_types):
                        sample_record = nusc.get('sample', sample_token)
                        cam_token = sample_record['data'][camera_type]
                        cam_record = nusc.get('sample_data', cam_token)
                        cam_path = nusc.get_sample_data_path(cam_token)
                        cam_path_rel = Path(cam_path).relative_to(args.dataroot)
                        cam_path_rel = cam_path_rel.__str__()
                        if cam_path_rel not in fd:
                            continue
                        grp = fd[cam_path_rel]
                        keypoints = grp['keypoints'].__array__().astype(np.float32)
                        scores = grp['scores'].__array__().astype(np.float32)
                        
                        out = cv2.imread(cam_path)
                        for pt, score in zip(keypoints, scores):
                            clr = myjet[int(np.clip(np.floor(score*10), 0, 9)), :]*255
                            pt = (int(round(pt[0])), int(round(pt[1])))
                            cv2.circle(out, pt, 3, clr, -1, lineType=16)
                        out = cv2.resize(out, (900, 600))
                        cam_out_path = os.path.join(out_dir, f'{camera_type}.jpg')
                        os.makedirs(out_dir, exist_ok=True)
                        cv2.imwrite(cam_out_path, out)
                        if i < 3:
                            im_front.append(out)
                        else:
                            im_back.append(out)
                    if len(im_front) != 3 and len(im_back) != 3:
                        continue
                    im_front = np.hstack(im_front)
                    im_back = np.hstack(im_back)
                    im_surround = np.vstack([im_front, im_back])
                    cv2.imwrite(surround_img_path, im_surround)
                if vout.isOpened() and im_surround is not None:
                    vout.write(im_surround)
            if vout.isOpened():
                vout.release()
    return

if __name__ == "__main__":
    main()