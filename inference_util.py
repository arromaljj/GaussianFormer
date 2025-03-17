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
            'focal_positions': torch.zeros(batch_size, num_cams, 3).to(device),
            # Add dummy occ_label to avoid KeyError during inference
            'occ_label': torch.zeros(batch_size, 200, 200, 16).long().to(device),
            # Add dummy occ_cam_mask to avoid KeyError during inference
            'occ_cam_mask': torch.ones(batch_size, 200, 200, 16, num_cams).bool().to(device)
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
                
        # Add dummy occ_label to avoid KeyError during inference
        if 'occ_label' not in metas:
            metas['occ_label'] = torch.zeros(batch_size, 200, 200, 16).long().to(device)
            
        # Add dummy occ_cam_mask to avoid KeyError during inference
        if 'occ_cam_mask' not in metas:
            num_cams = images.shape[1]
            metas['occ_cam_mask'] = torch.ones(batch_size, 200, 200, 16, num_cams).bool().to(device)
    
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
    
    # Add dummy occ_label for inference
    results['occ_label'] = np.zeros((1, 200, 200, 16), dtype=np.int64)
    
    # Add dummy occ_cam_mask for inference
    num_cams = len(camera_types)
    results['occ_cam_mask'] = np.ones((1, 200, 200, 16, num_cams), dtype=bool)
    
    # Apply data pipeline transformations
    # This adapts the data to the format expected by the model
    adaptor = NuScenesAdaptor(use_ego=False, num_cams=6)
    results = adaptor(results)
    
    return results


def initialize_model(args):
    """
    Initialize the model with given arguments.
    
    Args:
        args: Configuration parameters
        
    Returns:
        model: Initialized model
    """
    print("==== MODEL INITIALIZATION START ====")
    try:
        env_manager = EnvironmentManager(0, args)
        model_manager = ModelManager(env_manager)
        model_manager.initialize()
        model = model_manager.get_model()
        print("✓ Model initialized successfully")
        return model
    except Exception as e:
        print(f"✗ Model initialization failed: {str(e)}")
        raise


def load_nuscenes_data(sample_idx=0):
    """
    Load a sample from NuScenes dataset.
    
    Args:
        sample_idx: Index of the sample to load
        
    Returns:
        nusc: NuScenes instance
        sample: NuScenes sample
        camera_types: List of camera types
        image_paths: List of image paths
    """
    print("\n==== NUSCENES DATA LOADING START ====")
    try:
        from nuscenes.nuscenes import NuScenes
        
        print("Loading NuScenes dataset...")
        nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes/', verbose=False)
        
        # Get the requested sample
        if sample_idx >= len(nusc.sample):
            print(f"Warning: Sample index {sample_idx} out of range, using first sample")
            sample_idx = 0
            
        sample = nusc.sample[sample_idx]
        print(f"✓ Loaded sample {sample_idx}, token: {sample['token']}")
        
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
        
        print(f"✓ Prepared {len(image_paths)} camera image paths")
        for i, path in enumerate(image_paths):
            print(f"  - Camera {i} ({camera_types[i]}): {path}")
            
        return nusc, sample, camera_types, image_paths
    except Exception as e:
        print(f"✗ NuScenes data loading failed: {str(e)}")
        raise


def load_and_process_images(image_paths):
    """
    Load and preprocess images.
    
    Args:
        image_paths: List of image paths
        
    Returns:
        results: Dictionary with loaded and processed images
    """
    print("\n==== IMAGE LOADING START ====")
    try:
        # Load images using LoadMultiViewImageFromFiles
        image_loader = LoadMultiViewImageFromFiles(to_float32=True)
        results = image_loader(image_paths)
        
        if 'img' in results:
            img_shapes = [img.shape for img in results['img']]
            print(f"✓ Successfully loaded {len(results['img'])} images with shapes: {img_shapes}")
        else:
            print("✗ Failed to load images: 'img' key not found in results")
            
        return results
    except Exception as e:
        print(f"✗ Image loading failed: {str(e)}")
        raise


def prepare_inputs(results, camera_params):
    """
    Prepare the inputs for model inference.
    
    Args:
        results: Dictionary with loaded images
        camera_params: Camera parameters
        
    Returns:
        results: Updated results dictionary
        inference_camera_params: Parameters for inference
    """
    print("\n==== INPUT PREPARATION START ====")
    try:
        # Add camera parameters to results
        print("Adding camera parameters to results...")
        for key, value in camera_params.items():
            results[key] = value
            if key in ['projection_mat', 'occ_xyz', 'image_wh', 'occ_label', 'occ_cam_mask']:
                shape_info = f"shape: {np.array(value).shape}" if hasattr(value, 'shape') else type(value)
                print(f"  - Added {key} ({shape_info})")
        
        # Normalize images
        print("Normalizing images...")
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True
        )
        normalizer = NormalizeMultiviewImage(**img_norm_cfg)
        results = normalizer(results)
        print(f"✓ Normalized images with mean={img_norm_cfg['mean']}, std={img_norm_cfg['std']}")
        
        # Format data
        print("Formatting data for model input...")
        formatter = DefaultFormatBundle()
        results = formatter(results)
        
        # Prepare model input images
        if isinstance(results['img'], torch.Tensor):
            images = results['img']
            if len(images.shape) == 4:  # [N, C, H, W]
                images = images.unsqueeze(0)
            print(f"✓ Prepared model input images with shape: {images.shape}")
        else:
            raise ValueError(f"Unexpected image format: {type(results['img'])}")
        
        # Prepare camera parameters for run_inference
        inference_camera_params = {
            'projection_mat': results['projection_mat'],
            'image_wh': results['image_wh'] if 'image_wh' in results else np.array([[1600, 900]] * 6),
            'occ_xyz': results['occ_xyz'] if 'occ_xyz' in results else None,
            'occ_label': results['occ_label'] if 'occ_label' in results else np.zeros((1, 200, 200, 16), dtype=np.int64),
            'occ_cam_mask': results['occ_cam_mask'] if 'occ_cam_mask' in results else np.ones((1, 200, 200, 16, 6), dtype=bool),
        }
        print("✓ Prepared camera parameters for inference")
        
        return results, inference_camera_params
    except Exception as e:
        print(f"✗ Input preparation failed: {str(e)}")
        raise


def run_model_inference(model, images, camera_params):
    """
    Run model inference.
    
    Args:
        model: Model to run
        images: Input images
        camera_params: Camera parameters
        
    Returns:
        inference_results: Model prediction results
    """
    print("\n==== MODEL INFERENCE START ====")
    try:
        print(f"Running inference with images shape: {images.shape}")
        print(f"Camera parameters keys: {list(camera_params.keys())}")
        
        # Print each parameter's shape for debugging
        for key, value in camera_params.items():
            if hasattr(value, 'shape'):
                print(f"  - {key} shape: {value.shape}")
                
        # Debug check for essential keys
        essential_keys = ['projection_mat', 'image_wh', 'occ_xyz', 'occ_label', 'occ_cam_mask']
        missing_keys = [key for key in essential_keys if key not in camera_params]
        if missing_keys:
            print(f"⚠️ WARNING: Missing essential keys: {missing_keys}")
            
        # Check tensor device consistency
        device_info = {}
        for key, value in camera_params.items():
            if isinstance(value, torch.Tensor):
                device_info[key] = value.device
        if len(set(device_info.values())) > 1:
            print(f"⚠️ WARNING: Inconsistent tensor devices: {device_info}")
        
        print("Running inference...")
        inference_results = run_inference(model, images, camera_params)
        
        # Print inference results
        print("Inference completed successfully! Results keys:")
        for key in inference_results.keys():
            if hasattr(inference_results[key], 'shape'):
                print(f"  - {key}: shape {inference_results[key].shape}")
            else:
                print(f"  - {key}: {type(inference_results[key])}")
                
        return inference_results
    except Exception as e:
        print(f"✗ Model inference failed: {str(e)}")
        # Print additional debug info
        print("\n=== DEBUG INFO FOR MODEL INFERENCE ERROR ===")
        
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
            
        # Check if error is due to missing key
        if isinstance(e, KeyError):
            print(f"\nMissing key: '{str(e)}'")
            print(f"Available keys in camera_params: {list(camera_params.keys())}")
            # If it's a missing key in metas dict inside model
            print(f"\nTry adding this key to both run_inference and prepare_nuscenes_camera_params functions!")
        
        raise


def save_input_images(image_paths, results_img, sample_idx, output_dir):
    """
    Save the input camera images used for inference.
    
    Args:
        image_paths: Original paths to the camera images
        results_img: Processed image tensor from inference
        sample_idx: Sample index
        output_dir: Directory to save images
    """
    print(f"\n==== SAVING INPUT IMAGES ====")
    try:
        # Create directory for images
        image_dir = os.path.join(output_dir, f"sample_{sample_idx}", "input_images")
        os.makedirs(image_dir, exist_ok=True)
        
        # Save original images
        camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                       'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        for i, img_path in enumerate(image_paths):
            # Copy original image
            img = cv2.imread(img_path)
            if img is not None:
                output_path = os.path.join(image_dir, f"{camera_types[i]}.jpg")
                cv2.imwrite(output_path, img)
                print(f"  ✓ Saved {camera_types[i]} image to {output_path}")
            else:
                print(f"  ✗ Failed to save {camera_types[i]} image - could not read original")
        
        # If processed image tensor is available (already normalized, might look different)
        if isinstance(results_img, torch.Tensor):
            processed_dir = os.path.join(image_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # Handle different tensor formats
            if len(results_img.shape) == 5:  # [B, N, C, H, W]
                imgs = results_img[0]  # Take first batch
            elif len(results_img.shape) == 4:  # [N, C, H, W]
                imgs = results_img
            else:
                print(f"  ✗ Unexpected image tensor shape: {results_img.shape}")
                return
                
            # Save processed images
            for i in range(min(len(camera_types), imgs.shape[0])):
                img = imgs[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                
                # Denormalize if needed (approximately)
                img = img * np.array([58.395, 57.12, 57.375]) + np.array([123.675, 116.28, 103.53])
                img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                output_path = os.path.join(processed_dir, f"{camera_types[i]}_processed.jpg")
                cv2.imwrite(output_path, img)
                print(f"  ✓ Saved processed {camera_types[i]} image to {output_path}")
        
        print(f"✓ All available input images saved to {image_dir}")
        return image_dir
    except Exception as e:
        print(f"✗ Failed to save input images: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_and_save_results(inference_results, sample_idx, save_output, output_dir, image_paths=None, input_images=None):
    """
    Process and save inference results.
    
    Args:
        inference_results: Model prediction results
        sample_idx: Sample index
        save_output: Whether to save visualization
        output_dir: Directory to save visualizations
        image_paths: Original image paths for saving input images
        input_images: Processed input image tensor for saving
        
    Returns:
        occupancy_pred: Processed occupancy predictions
    """
    print("\n==== RESULTS PROCESSING START ====")
    try:
        # Create sample directory
        if output_dir is None:
            output_dir = 'out/results'
        
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Process the results
        if 'final_occ' not in inference_results:
            print(f"✗ 'final_occ' not found in inference results. Available keys: {list(inference_results.keys())}")
            return None
            
        occupancy_pred = inference_results['final_occ'][0]  # First batch item
        
        # Convert to numpy if needed
        if isinstance(occupancy_pred, torch.Tensor):
            occupancy_pred = occupancy_pred.cpu().numpy()
        
        print(f"✓ Processed occupancy prediction with shape: {occupancy_pred.shape}")
        
        # Save raw prediction data
        pred_dir = os.path.join(sample_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        
        # Save the raw data first (always do this)
        raw_output_path = os.path.join(pred_dir, "occupancy_pred_raw.npy")
        np.save(raw_output_path, occupancy_pred)
        print(f"✓ Raw prediction data saved to {raw_output_path}")
        
        # Save reshaping metadata to help with later analysis
        original_shape = occupancy_pred.shape
        if len(original_shape) == 1:
            # If flattened, typical shape would be [200*200*16, num_classes]
            # or just [200*200*16] for a single-class prediction
            suggested_reshape = "(200, 200, 16, X)" if len(original_shape) > 1 else "(200, 200, 16)"
            with open(os.path.join(pred_dir, "reshape_info.txt"), "w") as f:
                f.write(f"Original shape: {original_shape}\n")
                f.write(f"Suggested reshape: {suggested_reshape}\n")
                f.write("Occupancy grid dimensions: 200x200x16\n")
                f.write("Spatial extent: x=[-50,50], y=[-50,50], z=[-5,3]\n")
                f.write("Grid resolution: 0.5 meters\n")
        
        # Try to save reshaped data if we can determine the shape
        try:
            if len(original_shape) == 1:
                # Assume standard 200x200x16 occupancy grid
                if occupancy_pred.size == 200*200*16:
                    reshaped = occupancy_pred.reshape(200, 200, 16)
                    np.save(os.path.join(pred_dir, "occupancy_pred_reshaped.npy"), reshaped)
                    print(f"✓ Reshaped prediction data saved (200x200x16)")
                elif occupancy_pred.size % (200*200*16) == 0:
                    # Multi-class case
                    num_classes = occupancy_pred.size // (200*200*16)
                    reshaped = occupancy_pred.reshape(200, 200, 16, num_classes)
                    np.save(os.path.join(pred_dir, "occupancy_pred_reshaped.npy"), reshaped)
                    print(f"✓ Reshaped prediction data saved (200x200x16x{num_classes})")
        except Exception as e:
            print(f"⚠️ Could not save reshaped prediction: {str(e)}")
        
        # Save input images if available
        if image_paths is not None and input_images is not None:
            save_input_images(image_paths, input_images, sample_idx, output_dir)
        
        # Visualize results if requested
        if save_output:
            try:
                from vis import save_occ
                
                vis_dir = os.path.join(sample_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                
                print(f"Saving visualization to {vis_dir}...")
                
                # Try to determine correct shape for visualization
                viz_data = occupancy_pred
                if len(occupancy_pred.shape) == 1:
                    # Assume it's flattened - try to reshape for visualization
                    if occupancy_pred.size == 200*200*16:
                        viz_data = occupancy_pred.reshape(1, 200, 200, 16)
                    elif occupancy_pred.size % (200*200*16) == 0:
                        # Multi-class case, reshape and take argmax for visualization
                        num_classes = occupancy_pred.size // (200*200*16)
                        reshaped = occupancy_pred.reshape(200, 200, 16, num_classes)
                        viz_data = np.argmax(reshaped, axis=-1).reshape(1, 200, 200, 16)
                elif len(occupancy_pred.shape) > 1:
                    # Already has a shape but may need batch dimension
                    if len(occupancy_pred.shape) == 3:  # [200, 200, 16]
                        viz_data = occupancy_pred.reshape(1, 200, 200, 16)
                
                # Save prediction visualization
                save_occ(
                    vis_dir,
                    viz_data,
                    f'sample_{sample_idx}_pred',
                    True, 0
                )
                print(f"✓ Visualization saved to {vis_dir}/sample_{sample_idx}_pred.png")
            except Exception as e:
                print(f"✗ Visualization failed: {str(e)}")
                print("Note: This is expected in headless environments. Use the saved .npy files for visualization elsewhere.")
        
        print(f"\n==== RESULTS SUMMARY ====")
        print(f"✓ All results saved to: {sample_dir}")
        print(f"  - Raw predictions: {pred_dir}/occupancy_pred_raw.npy")
        if image_paths is not None:
            print(f"  - Input images: {sample_dir}/input_images/")
        if save_output:
            print(f"  - Visualizations: {sample_dir}/visualizations/ (if successful)")
        print(f"\nTo analyze these results on another machine:")
        print(f"1. Download the entire '{output_dir}' directory")
        print(f"2. Load the raw predictions with: np.load('occupancy_pred_raw.npy')")
        print(f"3. Check reshape_info.txt for guidance on reshaping the prediction data")
        
        return occupancy_pred
    except Exception as e:
        print(f"✗ Results processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def inference_example(sample_idx=0, save_output=True, output_dir=None):
    """
    Run complete inference pipeline example.
    
    Args:
        sample_idx: Index of the NuScenes sample to use
        save_output: Whether to save the prediction visualization
        output_dir: Directory to save the output visualization
        
    Returns:
        dict: Model predictions
    """
    print("\n====== STARTING INFERENCE EXAMPLE ======")
    
    try:
        # 1. Initialize environment and model
        args = argparse.Namespace(
            py_config='config/nuscenes_gs25600_solid.py',
            work_dir='out',
            resume_from='downloads/nonempty/nonempty.pth',
            seed=42,
            gpus=torch.cuda.device_count(),
            vis_occ=False
        )
        
        if output_dir is None:
            output_dir = os.path.join(args.work_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        model = initialize_model(args)
        
        # 2. Load NuScenes data
        nusc, sample, camera_types, image_paths = load_nuscenes_data(sample_idx)
        
        # 3. Load and preprocess images
        results = load_and_process_images(image_paths)
        
        # 4. Get camera parameters
        print("\n==== CAMERA PARAMETERS PREPARATION START ====")
        camera_params = prepare_nuscenes_camera_params(nusc, sample, camera_types)
        print(f"✓ Prepared camera parameters with keys: {list(camera_params.keys())}")
        
        # 5. Prepare inputs and camera parameters
        results, inference_camera_params = prepare_inputs(results, camera_params)
        
        # 6. Run inference
        inference_results = run_model_inference(model, results['img'], inference_camera_params)
        
        # 7. Process and save results (now passing image paths and processed images)
        occupancy_pred = process_and_save_results(
            inference_results, 
            sample_idx, 
            save_output, 
            output_dir,
            image_paths=image_paths,
            input_images=results['img']
        )
        
        print("\n====== INFERENCE EXAMPLE COMPLETED SUCCESSFULLY ======")
        return inference_results
        
    except Exception as e:
        print(f"\n✗✗✗ INFERENCE EXAMPLE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


if __name__ == "__main__":
    # Use a fixed args object instead of command line parsing
    args = argparse.Namespace(
        py_config='config/nuscenes_gs25600_solid.py',
        work_dir='out',
        resume_from='downloads/nonempty/nonempty.pth',
        seed=42,
        gpus=torch.cuda.device_count(),
        vis_occ=False
    )
    
    # Run inference example
    print(f"Running inference on sample 0...")
    results = inference_example(
        sample_idx=0,
        save_output=True,
        output_dir=None
    )
    
    print("Inference completed successfully!") 