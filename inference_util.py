import os
import time
import argparse
import numpy as np
import torch
import cv2
from pyquaternion import Quaternion
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
from PIL import Image
import mmcv
from copy import deepcopy


class EnvironmentManager:
    """Handles environment setup, configuration, distributed processing, and logging."""

    def __init__(self, local_rank, args):
        """
        Initialize environment settings and logging.

        Args:
            local_rank (int): Local rank for distributed processing
            args (argparse.Namespace): Command line arguments
        """
        self.local_rank = local_rank
        self.args = args
        self.distributed = False
        self.cfg = None
        self.logger = None

        # Setup environment in sequence
        self._setup_environment()
        self._load_config()
        self._setup_logging()

    def _setup_environment(self):
        """Set up environment variables and random seeds for reproducibility."""
        set_random_seed(self.args.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _load_config(self):
        """Load configuration from file."""
        self.cfg = Config.fromfile(self.args.py_config)
        self.cfg.work_dir = self.args.work_dir

        # Ensure work directory exists
        os.makedirs(self.args.work_dir, exist_ok=True)

    def _setup_logging(self):
        """Set up logging with timestamp."""
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(self.args.work_dir, f'{timestamp}.log')
        self.logger = MMLogger('selfocc', log_file=log_file)
        MMLogger._instance_dict['selfocc'] = self.logger

        self.logger.info(f"Configuration:\n{self.cfg.pretty_text}")

    def is_master(self):
        """Check if this process is the master process."""
        return self.local_rank == 0

    def get_logger(self):
        """Get the logger instance."""
        return self.logger

    def get_config(self):
        """Get the configuration instance."""
        return self.cfg


class ModelManager:
    """Handles model creation, initialization, and checkpoint loading."""

    def __init__(self, env_manager):
        """
        Initialize the model manager.

        Args:
            env_manager (EnvironmentManager): Environment manager instance
        """
        self.env = env_manager
        self.cfg = env_manager.get_config()
        self.logger = env_manager.get_logger()
        self.local_rank = env_manager.local_rank

        self.model = None
        self.raw_model = None

    def initialize(self):
        """Initialize the model and load weights if specified."""
        self._build_model()
        self._load_checkpoint()
        return self.model

    def _build_model(self):
        """Build and initialize the segmentation model."""
        # Import model definitions (ensure this is available in your project)
        try:
            import model
        except ImportError:
            self.logger.warning("Could not import model module. Assuming model definitions are registered.")

        # Build model from config
        self.model = build_segmentor(self.cfg.model)
        self.model.init_weights()

        # Log model size
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of trainable parameters: {n_parameters:,}')

        # Move model to GPU
        self.model = self.model.cuda()
        self.raw_model = self.model

        self.logger.info('Model initialization complete')

    def _load_checkpoint(self):
        """Resume from checkpoint or load pre-trained weights."""
        # Override with command line argument if provided
        if self.env.args.resume_from:
            self.cfg.resume_from = self.env.args.resume_from

        self.logger.info(f'Resume from: {self.cfg.resume_from}')
        self.logger.info(f'Work directory: {self.env.args.work_dir}')

        # Load checkpoint if available
        if self.cfg.resume_from and os.path.exists(self.cfg.resume_from):
            self._load_from_checkpoint(self.cfg.resume_from)
        # Otherwise load from pretrained weights if specified
        elif hasattr(self.cfg, 'load_from') and self.cfg.load_from:
            self._load_from_pretrained(self.cfg.load_from)

    def _load_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        try:
            map_location = 'cpu'
            ckpt = torch.load(checkpoint_path, map_location=map_location)
            self.raw_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
            self.logger.info(f'Successfully resumed from {checkpoint_path}')
        except Exception as e:
            self.logger.error(f'Failed to load checkpoint: {str(e)}')
            raise

    def _load_from_pretrained(self, pretrained_path):
        """Load pretrained weights."""
        try:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)

            try:
                load_info = self.raw_model.load_state_dict(state_dict, strict=False)
                self.logger.info(f'Loaded pretrained weights: {load_info}')
            except Exception:
                # Try with weight refinement if regular loading fails
                from misc.checkpoint_util import refine_load_from_sd
                refined_state_dict = refine_load_from_sd(state_dict)
                load_info = self.raw_model.load_state_dict(refined_state_dict, strict=False)
                self.logger.info(f'Loaded pretrained weights with refinement: {load_info}')
        except Exception as e:
            self.logger.error(f'Failed to load pretrained weights: {str(e)}')
            raise

    def get_model(self):
        """Get the initialized model."""
        return self.model


# Data Pipeline Components

class LoadMultiViewImageFromFiles:
    """Load multi-view images from files for inference."""
    
    def __init__(self, to_float32=True, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type
        
    def __call__(self, image_paths):
        """
        Load images from the provided paths.
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            dict: Dictionary containing loaded images and metadata
        """
        results = {'img_filename': image_paths}
        
        # Load all images
        img = np.stack([mmcv.imread(name, self.color_type) for name in image_paths], axis=-1)
        
        if self.to_float32:
            img = img.astype(np.float32)
            
        results['filename'] = image_paths
        # Unravel to list for processing
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['ori_img'] = deepcopy(img)
        results['img_shape'] = [x.shape[:2] for x in results['img']]
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        
        # Default normalization config (will be updated by NormalizeMultiviewImage)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False
        )
        
        return results


class NormalizeMultiviewImage:
    """Normalize images with mean, std values."""
    
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        
    def __call__(self, results):
        """
        Normalize each image in the results dict.
        
        Args:
            results (dict): Dictionary containing images to normalize
            
        Returns:
            dict: Dictionary with normalized images
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        
        return results


class DefaultFormatBundle:
    """Format images for model input."""
    
    def __call__(self, results):
        """
        Convert images to the format expected by the model.
        
        Args:
            results (dict): Dictionary containing images
            
        Returns:
            dict: Dictionary with formatted images
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # Process multiple images in a single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                
            results['img'] = torch.from_numpy(imgs)
            
        return results


class NuScenesAdaptor:
    """Adapt NuScenes data format for the model."""
    
    def __init__(self, use_ego=False, num_cams=6):
        self.projection_key = 'ego2img' if use_ego else 'lidar2img'
        self.num_cams = num_cams
        
    def __call__(self, results):
        """
        Adapt the NuScenes data format for the model.
        
        Args:
            results (dict): Dictionary containing data
            
        Returns:
            dict: Dictionary with adapted data
        """
        if self.projection_key in results:
            results["projection_mat"] = np.float32(
                np.stack(results[self.projection_key])
            )
            
        if "img_shape" in results:
            results["image_wh"] = np.ascontiguousarray(
                np.array(results["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
            )
            
        return results


# Get meshgrid for occupancy coordinates
def get_meshgrid(ranges, grid, reso):
    """
    Generate meshgrid for occupancy coordinates.
    
    Args:
        ranges (list): Range limits [x_min, y_min, z_min, x_max, y_max, z_max]
        grid (list): Grid dimensions [x_dim, y_dim, z_dim]
        reso (float): Resolution
        
    Returns:
        np.ndarray: Coordinates array
    """
    xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
    yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
    zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

    xxx = xxx[:, None, None].expand(*grid)
    yyy = yyy[None, :, None].expand(*grid)
    zzz = zzz[None, None, :].expand(*grid)

    xyz = torch.stack([xxx, yyy, zzz], dim=-1).numpy()
    return xyz  # x, y, z, 3


def run_inference(model, images, camera_params=None, device='cuda'):
    """
    Run inference on given images using the loaded model.
    
    Args:
        model (nn.Module): Loaded GaussianFormer model
        images (torch.Tensor or list): Input images with shape [N, C, H, W] or list of images
                                      where N is batch size, C is channels (3 for RGB)
        camera_params (dict, optional): Camera parameters including projection matrices, etc.
                                       If None, default parameters will be used
        device (str): Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        dict: Dictionary containing:
            - 'final_occ': Final occupancy predictions with shape [B, 200, 200, 16, C] 
                          where C is the number of classes (17 typically)
            - Other model outputs like Gaussian parameters if available
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Process input images
    if isinstance(images, list):
        # If input is a list of images, stack them
        if all(isinstance(img, np.ndarray) for img in images):
            # Convert numpy arrays to tensors if needed
            images = [torch.from_numpy(img.transpose(2, 0, 1)).float() for img in images]
        images = torch.stack(images)
        
        # Add batch dimension if not present
        if len(images.shape) == 4:  # [N, C, H, W] -> [1, N, C, H, W]
            images = images.unsqueeze(0)
    elif isinstance(images, torch.Tensor) and len(images.shape) == 4:
        # Single batch of N cameras: [N, C, H, W] -> [1, N, C, H, W]
        images = images.unsqueeze(0)
    
    # Move to device
    images = images.to(device)
    
    # Prepare metadata dict
    if camera_params is None:
        # Default camera parameters for 6-camera NuScenes setup
        # These are example values and should be replaced with actual camera parameters
        image_height, image_width = images.shape[-2], images.shape[-1]
        batch_size = images.shape[0]
        num_cams = images.shape[1]
        
        # Create dummy projection matrices (identity transforms)
        projection_matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, num_cams, 1, 1).float().to(device)
        image_wh = torch.tensor([[image_width, image_height]]).repeat(batch_size, num_cams, 1).float().to(device)
        
        # Create occupancy grid coordinates (match the model's expectations)
        grid_size = 0.5
        x = torch.arange(-50, 50, grid_size) + grid_size/2
        y = torch.arange(-50, 50, grid_size) + grid_size/2
        z = torch.arange(-5, 3, grid_size) + grid_size/2
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        occ_xyz = torch.stack([xx, yy, zz], dim=-1)
        
        metas = {
            'projection_mat': projection_matrices,
            'image_wh': image_wh,
            'occ_xyz': occ_xyz.reshape(1, 200, 200, 16, 3).repeat(batch_size, 1, 1, 1, 1).to(device),
            'cam_positions': torch.zeros(batch_size, num_cams, 3).to(device),
            'focal_positions': torch.zeros(batch_size, num_cams, 3).to(device)
        }
    else:
        # Use provided camera parameters
        metas = {}
        batch_size = images.shape[0]
        
        for key, value in camera_params.items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value).float()
                
                # Ensure batch dimension
                if key in ['projection_mat', 'image_wh', 'cam_positions', 'focal_positions']:
                    if len(tensor_value.shape) == 3 and key == 'projection_mat':  # [num_cams, 4, 4]
                        tensor_value = tensor_value.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                    elif len(tensor_value.shape) == 2 and key in ['image_wh', 'cam_positions', 'focal_positions']:
                        tensor_value = tensor_value.unsqueeze(0).repeat(batch_size, 1, 1)
                
                # Special handling for occ_xyz which has a specific expected shape
                if key == 'occ_xyz' and len(tensor_value.shape) == 5:  # [1, 200, 200, 16, 3]
                    if tensor_value.shape[0] != batch_size:
                        tensor_value = tensor_value.repeat(batch_size, 1, 1, 1, 1)
                        
                metas[key] = tensor_value.to(device)
            elif isinstance(value, torch.Tensor):
                # Same checks as above but for tensors
                tensor_value = value.float()
                
                # Ensure batch dimension
                if key in ['projection_mat', 'image_wh', 'cam_positions', 'focal_positions']:
                    if len(tensor_value.shape) == 3 and key == 'projection_mat':
                        tensor_value = tensor_value.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                    elif len(tensor_value.shape) == 2 and key in ['image_wh', 'cam_positions', 'focal_positions']:
                        tensor_value = tensor_value.unsqueeze(0).repeat(batch_size, 1, 1)
                
                if key == 'occ_xyz' and len(tensor_value.shape) == 5:
                    if tensor_value.shape[0] != batch_size:
                        tensor_value = tensor_value.repeat(batch_size, 1, 1, 1, 1)
                        
                metas[key] = tensor_value.to(device)
            else:
                metas[key] = value
    
    # Run inference
    with torch.no_grad():
        results = model(imgs=images, metas=metas)
    
    return results


def prepare_nuscenes_camera_params(nusc, sample, camera_types):
    """
    Helper function to prepare camera parameters from NuScenes data.
    
    Args:
        nusc (NuScenes): NuScenes instance
        sample (dict): NuScenes sample
        camera_types (list): List of camera names
        
    Returns:
        dict: Camera parameters required for inference
    """
    results = {}
    
    # Initialize lidar2img and ego2img lists
    results['lidar2img'] = []
    results['ego2img'] = []
    
    # Get lidar2cam matrices for each camera
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    # Get lidar to global transform
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    l2e_r = Quaternion(lidar_cs['rotation']).rotation_matrix
    l2e_t = np.array(lidar_cs['translation'])
    e2g_r = Quaternion(lidar_pose['rotation']).rotation_matrix
    e2g_t = np.array(lidar_pose['translation'])
    
    l2e = np.eye(4)
    l2e[:3, :3] = l2e_r
    l2e[:3, 3] = l2e_t
    
    e2g = np.eye(4)
    e2g[:3, :3] = e2g_r
    e2g[:3, 3] = e2g_t
    
    lidar2global = e2g @ l2e
    results['ego2lidar'] = l2e
    
    # Set image shape
    results['img_shape'] = []
    
    # Process each camera
    for cam in camera_types:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Calculate camera to ego transform
        c2e_r = Quaternion(cam_cs['rotation']).rotation_matrix
        c2e_t = np.array(cam_cs['translation'])
        c2e = np.eye(4)
        c2e[:3, :3] = c2e_r
        c2e[:3, 3] = c2e_t
        
        # Calculate ego to global transform
        e2g_r = Quaternion(cam_pose['rotation']).rotation_matrix
        e2g_t = np.array(cam_pose['translation'])
        e2g = np.eye(4)
        e2g[:3, :3] = e2g_r
        e2g[:3, 3] = e2g_t
        
        # Calculate camera to global transform
        cam2global = e2g @ c2e
        
        # Calculate camera intrinsics
        intrinsic = np.array(cam_cs['camera_intrinsic'])
        viewpad = np.eye(4)
        viewpad[:3, :3] = intrinsic
        
        # Calculate lidar to image projection
        lidar2cam = np.linalg.inv(cam2global) @ lidar2global
        lidar2img = viewpad @ lidar2cam
        results['lidar2img'].append(lidar2img)
        
        # Calculate ego to image projection
        ego2cam = np.linalg.inv(cam2global) @ e2g
        ego2img = viewpad @ ego2cam
        results['ego2img'].append(ego2img)
        
        # Store image dimensions
        width, height = cam_data['width'], cam_data['height']
        results['img_shape'].append((height, width))
    
    # Create occupancy grid coordinates
    grid_size = 0.5
    ranges = [-50, -50, -5.0, 50, 50, 3.0]
    grid = [200, 200, 16]
    
    occ_xyz = get_meshgrid(ranges, grid, grid_size)
    results['occ_xyz'] = occ_xyz.reshape(1, 200, 200, 16, 3)
    
    # Apply data pipeline transformations
    # This adapts the data to the format expected by the model
    adaptor = NuScenesAdaptor(use_ego=False, num_cams=6)
    results = adaptor(results)
    
    return results


def inference_example(sample_idx=0, save_output=True, output_dir=None):
    """
    Example function showing how to use run_inference with sample images from a NuScenes scene.
    
    Args:
        sample_idx (int): Index of the NuScenes sample to use
        save_output (bool): Whether to save the prediction visualization
        output_dir (str): Directory to save the output visualization (defaults to out/viz)
        
    Returns:
        dict: Model predictions
    """
    print("Starting inference example...")
    from nuscenes.nuscenes import NuScenes
    
    # Initialize environment and model
    args = argparse.Namespace(
        py_config='config/nuscenes_gs25600_solid.py',
        work_dir='out',
        resume_from='downloads/nonempty/nonempty.pth',
        seed=42,
        gpus=torch.cuda.device_count(),
        vis_occ=False
    )
    
    if output_dir is None:
        output_dir = os.path.join(args.work_dir, 'viz')
    os.makedirs(output_dir, exist_ok=True)
    
    env_manager = EnvironmentManager(0, args)
    model_manager = ModelManager(env_manager)
    model_manager.initialize()
    model = model_manager.get_model()
    
    # Load a sample from NuScenes
    print("Loading NuScenes data...")
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes/', verbose=False)
    
    # Get the requested sample
    if sample_idx >= len(nusc.sample):
        print(f"Warning: Sample index {sample_idx} out of range, using first sample")
        sample_idx = 0
        
    sample = nusc.sample[sample_idx]
    print(f"Running inference on sample {sample_idx}, token: {sample['token']}")
    
    # Camera types used in NuScenes
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                  'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Prepare image paths
    image_paths = []
    for cam in camera_types:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = os.path.join('data/nuscenes/', cam_data['filename'])
        image_paths.append(img_path)
        print(f"Image path: {img_path}")
    
    # Setup data pipeline
    print("Setting up data pipeline...")
    # 1. Load images using LoadMultiViewImageFromFiles
    image_loader = LoadMultiViewImageFromFiles(to_float32=True)
    results = image_loader(image_paths)
    
    # 2. Get camera parameters
    print("Preparing camera parameters...")
    camera_params = prepare_nuscenes_camera_params(nusc, sample, camera_types)
    
    # Add camera parameters to results
    for key, value in camera_params.items():
        results[key] = value
    
    # 3. Normalize images
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    )
    normalizer = NormalizeMultiviewImage(**img_norm_cfg)
    results = normalizer(results)
    
    # 4. Format data
    formatter = DefaultFormatBundle()
    results = formatter(results)
    
    # Run inference
    print("Running model inference...")
    if isinstance(results['img'], torch.Tensor):
        # Model expects [B, N, C, H, W]
        images = results['img']
        if len(images.shape) == 4:  # [N, C, H, W]
            images = images.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image format: {type(results['img'])}")
    
    # Prepare camera parameters for run_inference
    inference_camera_params = {
        'projection_mat': results['projection_mat'],
        'image_wh': results['image_wh'] if 'image_wh' in results else np.array([[1600, 900]] * 6),
        'occ_xyz': results['occ_xyz'] if 'occ_xyz' in results else None,
    }
    
    # Run inference
    inference_results = run_inference(model, images, inference_camera_params)
    
    # Process the results
    occupancy_pred = inference_results['final_occ'][0]  # First batch item
    
    # Convert to numpy if needed
    if isinstance(occupancy_pred, torch.Tensor):
        occupancy_pred = occupancy_pred.cpu().numpy()
    
    print(f"Occupancy prediction shape: {occupancy_pred.shape}")
    
    # Visualize results if requested
    if save_output:
        try:
            from vis import save_occ
            
            vis_dir = output_dir
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save prediction visualization
            save_occ(
                vis_dir,
                occupancy_pred.reshape(1, 200, 200, 16),
                f'sample_{sample_idx}_pred',
                True, 0
            )
            print(f"Visualization saved to {vis_dir}/sample_{sample_idx}_pred.png")
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
    
    return inference_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GaussianFormer inference')
    parser.add_argument('--py-config', type=str, default='config/nuscenes_gs25600_solid.py',
                      help='Path to config file')
    parser.add_argument('--work-dir', type=str, default='out',
                      help='Working directory')
    parser.add_argument('--resume-from', type=str, default='downloads/nonempty/nonempty.pth',
                      help='Path to checkpoint file')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed')
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(),
                      help='Number of GPUs to use')
    parser.add_argument('--sample-idx', type=int, default=0,
                      help='Sample index to use for inference example')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save visualization outputs')
    parser.add_argument('--save-output', action='store_true',
                      help='Save visualization output')
    
    args = parser.parse_args()
    
    # Run inference example
    print(f"Running inference on sample {args.sample_idx}...")
    results = inference_example(
        sample_idx=args.sample_idx,
        save_output=args.save_output,
        output_dir=args.output_dir
    )
    
    print("Inference completed successfully!") 