import os
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class NuScenesDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        num_samples=0,
        vis_scene_index=-1,
        phase='train',
        return_keys=[
            'img',
            'projection_mat',
            'image_wh',
            'occ_label',
            'occ_xyz',
            'occ_cam_mask',
            'ori_img',
            'cam_positions',
            'focal_positions'
        ],
    ):
        self.data_path = data_root
        
        # Check if we're using mini dataset
        mini_dataset = 'mini' in data_root or 'v1.0-mini' in data_root
        print(f"Detected {'mini' if mini_dataset else 'full'} NuScenes dataset")
        
        # Try to load the specified pickle file
        try:
            if os.path.exists(imageset):
                print(f"Loading data from {imageset}")
                data = mmengine.load(imageset)
                self.scene_infos = data['infos']
                self.keyframes = data['metadata']
            else:
                print(f"Warning: {imageset} not found. Creating simple dataset structure for NuScenes dataset.")
                # Create a simple structure based on directory scanning
                data = self._create_simple_dataset_info(mini=mini_dataset, phase=phase)
                self.scene_infos = data['infos']
                self.keyframes = data['metadata']
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating simple dataset structure as fallback.")
            data = self._create_simple_dataset_info(mini=mini_dataset, phase=phase)
            self.scene_infos = data['infos']
            self.keyframes = data['metadata']
        
        # Debugging info
        print(f"Dataset phase: {phase}")
        print(f"Loaded {len(self.scene_infos)} scenes")
        print(f"Loaded {len(self.keyframes)} keyframes")
        if len(self.keyframes) > 0:
            first_keyframe = self.keyframes[0]
            print(f"First keyframe type: {type(first_keyframe)}")
            print(f"First keyframe value: {first_keyframe}")
            
            # Check if all keyframes have the same structure
            keyframe_types = set(type(kf) for kf in self.keyframes[:min(10, len(self.keyframes))])
            print(f"Keyframe types (first 10): {keyframe_types}")
        
        # Define a safer sorting key
        def safe_sort_key(x):
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                return str(x[0]) + "{:0>3}".format(str(x[1]))
            elif isinstance(x, (list, tuple)) and len(x) == 1:
                return str(x[0]) + "000"
            elif isinstance(x, str):
                return x + "000"
            else:
                return str(x) + "000"
        
        try:
            self.keyframes = sorted(self.keyframes, key=safe_sort_key)
        except Exception as e:
            print(f"Warning: Failed to sort keyframes: {e}")
            # Keep original order if sorting fails
        
        self.data_aug_conf = data_aug_conf
        self.test_mode = (phase != 'train')
        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.return_keys = return_keys
        
        if vis_scene_index >= 0:
            try:
                frame = self.keyframes[vis_scene_index]
                scene_token = frame[0] if isinstance(frame, (list, tuple)) and len(frame) > 0 else str(frame)
                num_frames = len(self.scene_infos.get(scene_token, []))
                self.keyframes = [(scene_token, i) for i in range(num_frames)]
                print(f'Scene length: {len(self.keyframes)}')
            except Exception as e:
                print(f"Error processing vis_scene_index: {e}")
                # Keep original keyframes if processing fails
        elif vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

    def _create_simple_dataset_info(self, mini=True, phase='val'):
        """
        Create a simple dataset structure when pkl files are not available.
        This is useful for testing with mini dataset.
        """
        import os
        from nuscenes.nuscenes import NuScenes
        
        print(f"Creating simple dataset info for {'mini' if mini else 'full'} NuScenes dataset")
        version = 'v1.0-mini' if mini else 'v1.0-trainval'
        
        try:
            # Initialize NuScenes
            nusc = NuScenes(version=version, dataroot=self.data_path, verbose=True)
            
            # Create basic structure
            data = {
                'infos': {},
                'metadata': []
            }
            
            # Get scenes to use based on phase
            if mini:
                # For mini dataset, use all scenes
                scenes = nusc.scene
            else:
                # For full dataset, split scenes by phase
                if phase == 'train':
                    scenes = [s for s in nusc.scene if s['name'].startswith('scene-')][:700]  # First 700 for training
                else:
                    scenes = [s for s in nusc.scene if s['name'].startswith('scene-')][700:]  # Rest for validation
            
            print(f"Processing {len(scenes)} scenes for {phase} set")
            
            # Populate with scene information
            for scene in scenes:
                scene_token = scene['token']
                data['infos'][scene_token] = []
                
                # Get first sample in scene
                sample_token = scene['first_sample_token']
                
                # Add samples to scene info
                frame_idx = 0
                while sample_token:
                    sample = nusc.get('sample', sample_token)
                    
                    # Create sample info dict
                    sample_info = {
                        'token': sample_token,
                        'timestamp': sample['timestamp'],
                        'data': {}
                    }
                    
                    # Add sensor data
                    for sensor in ['LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                                  'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                        if sensor in sample['data']:
                            sensor_token = sample['data'][sensor]
                            sample_data = nusc.get('sample_data', sensor_token)
                            
                            # Get pose and calibration
                            ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
                            calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                            
                            sample_info['data'][sensor] = {
                                'filename': sample_data['filename'],
                                'pose': {
                                    'translation': ego_pose['translation'],
                                    'rotation': ego_pose['rotation']
                                },
                                'calib': {
                                    'camera_intrinsic': calib.get('camera_intrinsic', None),
                                    'translation': calib['translation'],
                                    'rotation': calib['rotation']
                                }
                            }
                    
                    # Create fake occupancy path (you'll need to generate or get these)
                    sample_info['occ_path'] = f"samples/{sample_token}.pcd.bin.npy"
                    
                    # Add to scene infos
                    data['infos'][scene_token].append(sample_info)
                    
                    # Add to metadata (as a proper tuple)
                    data['metadata'].append((scene_token, frame_idx))
                    
                    # Move to next sample
                    sample_token = sample['next']
                    frame_idx += 1
                    
            print(f"Created dataset with {len(data['metadata'])} frames across {len(data['infos'])} scenes")
            return data
            
        except Exception as e:
            print(f"Error creating dataset info: {e}")
            # Return minimal empty structure as fallback
            return {'infos': {}, 'metadata': []}

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):
        scene_token, index = self.keyframes[index]
        info = deepcopy(self.scene_infos[scene_token][index])
        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)
        
        return_dict = {k: input_dict[k] for k in self.return_keys}
        return return_dict
    
    def get_data_info(self, info):
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        ego2image_rts = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['data']['LIDAR_TOP']['pose']['translation']).T

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))

            img2global = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global

            lidar2img_rts.append(lidar2img)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

            img2lidar = np.linalg.inv(lidar2global) @ img2global
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            cam_position = img2lidar @ viewpad @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = img2lidar @ viewpad @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
            
        input_dict =dict(
            # sample_idx=info["token"],
            sample_idx=info.get("token", ""),
            # occ_path=info["occ_path"],
            occ_path=info.get("occ_path", ""),
            timestamp=info["timestamp"] / 1e6,
            img_filename=image_paths,
            pts_filename=os.path.join(self.data_path, info['data']['LIDAR_TOP']['filename']),
            ego2lidar=ego2lidar,
            lidar2img=np.asarray(lidar2img_rts),
            ego2img=np.asarray(ego2image_rts),
            cam_positions=np.asarray(cam_positions),
            focal_positions=np.asarray(focal_positions))

        return input_dict

    def __len__(self):
        return len(self.keyframes)