#%%
import time
import argparse
import os
import os.path as osp
import torch
import numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
import warnings
warnings.filterwarnings("ignore")


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
        self._setup_distributed()
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

    def _setup_distributed(self):
        """Set up distributed data parallel processing if multiple GPUs are available."""
        if self.args.gpus > 1:
            self.distributed = True

            # Get environment variables for distributed setup
            ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
            port = os.environ.get("MASTER_PORT", "20507")
            hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
            rank = int(os.environ.get("RANK", 0))  # node id
            gpus = torch.cuda.device_count()  # gpus per node

            if self.local_rank == 0:
                print(f"Initializing DDP: tcp://{ip}:{port}")

            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{ip}:{port}",
                world_size=hosts * gpus,
                rank=rank * gpus + self.local_rank
            )

            world_size = dist.get_world_size()
            self.cfg.gpu_ids = range(world_size)
            torch.cuda.set_device(self.local_rank)

            # Suppress prints on non-master processes
            if self.local_rank != 0:
                import builtins
                builtins.print = self.pass_print
        else:
            self.distributed = False

    def _setup_logging(self):
        """Set up logging with timestamp."""
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(self.args.work_dir, f'{timestamp}.log')
        self.logger = MMLogger('selfocc', log_file=log_file)
        MMLogger._instance_dict['selfocc'] = self.logger

        if self.local_rank == 0:
            self.logger.info(f"Configuration:\n{self.cfg.pretty_text}")

    def pass_print(self, *args, **kwargs):
        """Empty print function for non-master processes."""
        pass

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
        self.distributed = env_manager.distributed
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

        # Setup distributed training if needed
        if self.distributed:
            self._setup_distributed_model()
        else:
            self.model = self.model.cuda()
            self.raw_model = self.model

        self.logger.info('Model initialization complete')

    def _setup_distributed_model(self):
        """Setup model for distributed training with SyncBN if needed."""
        # Convert to SyncBN if specified
        if self.cfg.get('syncBN', True):
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.logger.info('Converted to SyncBatchNorm')

        # Setup DDP
        find_unused_parameters = self.cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        self.model = ddp_model_module(
            self.model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )
        self.raw_model = self.model.module

    def _load_checkpoint(self):
        """Resume from checkpoint or load pre-trained weights."""
        # Check for latest checkpoint
        self.cfg.resume_from = ''
        if osp.exists(osp.join(self.env.args.work_dir, 'latest.pth')):
            self.cfg.resume_from = osp.join(self.env.args.work_dir, 'latest.pth')

        # Override with command line argument if provided
        if self.env.args.resume_from:
            self.cfg.resume_from = self.env.args.resume_from

        self.logger.info(f'Resume from: {self.cfg.resume_from}')
        self.logger.info(f'Work directory: {self.env.args.work_dir}')

        # Load checkpoint if available
        if self.cfg.resume_from and osp.exists(self.cfg.resume_from):
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


class DatasetManager:
    """Handles dataset loading and preparation."""

    def __init__(self, env_manager):
        """
        Initialize the dataset manager.

        Args:
            env_manager (EnvironmentManager): Environment manager instance
        """
        self.env = env_manager
        self.cfg = env_manager.get_config()
        self.logger = env_manager.get_logger()
        self.distributed = env_manager.distributed

        self.train_loader = None
        self.val_loader = None

    def load_datasets(self, val_only=True):
        """
        Load datasets and create data loaders.

        Args:
            val_only (bool): Whether to only load validation data

        Returns:
            tuple: (train_loader, val_loader) - DataLoader instances
        """
        try:
            from dataset import get_dataloader

            self.train_loader, self.val_loader = get_dataloader(
                self.cfg.train_dataset_config,
                self.cfg.val_dataset_config,
                self.cfg.train_loader,
                self.cfg.val_loader,
                dist=self.distributed,
                val_only=val_only
            )

            self.logger.info('Dataset loaded successfully')

            if val_only:
                self.logger.info(f'Validation dataset size: {len(self.val_loader)}')
            else:
                self.logger.info(f'Training dataset size: {len(self.train_loader)}')
                self.logger.info(f'Validation dataset size: {len(self.val_loader)}')

            return self.train_loader, self.val_loader
        except Exception as e:
            self.logger.error(f'Failed to load dataset: {str(e)}')
            raise

    def get_val_loader(self):
        """Get validation data loader."""
        return self.val_loader

#%%
class Evaluator:
    """Handles model evaluation and metrics computation."""

    def __init__(self, env_manager, model_manager, dataset_manager):
        """
        Initialize the evaluator.

        Args:
            env_manager (EnvironmentManager): Environment manager instance
            model_manager (ModelManager): Model manager instance
            dataset_manager (DatasetManager): Dataset manager instance
        """
        self.env = env_manager
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager

        self.cfg = env_manager.get_config()
        self.logger = env_manager.get_logger()
        self.local_rank = env_manager.local_rank
        self.args = env_manager.args

        self.model = model_manager.get_model()
        self.val_loader = dataset_manager.get_val_loader()

    def evaluate(self):
        """
        Run model evaluation.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print_freq = self.cfg.print_freq

        # Initialize metrics
        miou_metric = self._setup_metrics()

        # Set model to evaluation mode
        self.model.eval()
        os.environ['eval'] = 'true'

        self.logger.info('Starting evaluation...')
        from pprint import pprint
        pprint(enumerate(self.val_loader))
        with torch.no_grad():
            for i_iter_val, data in enumerate(self.val_loader):
                # Process batch
                result_dict = self._process_batch(data)

                # Handle occupancy predictions
                if 'final_occ' in result_dict:
                    self._process_occupancy(result_dict, miou_metric, i_iter_val)

                # Log progress
                if i_iter_val % print_freq == 0 and self.env.is_master():
                    self.logger.info(f'[EVAL] Iter {i_iter_val:5d}')

        # Compute and log final metrics
        miou, iou2 = miou_metric._after_epoch()
        self.logger.info(f'Evaluation results - mIoU: {miou:.4f}, iou2: {iou2:.4f}')
        miou_metric.reset()

        return {'miou': miou, 'iou2': iou2}

    def _setup_metrics(self):
        """Setup evaluation metrics."""
        from misc.metric_util import MeanIoU

        class_names = [
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'
        ]

        miou_metric = MeanIoU(
            list(range(1, 17)),  # Classes
            17,                  # Number of classes
            class_names,         # Class names for logging
            True,                # Use class weights
            17,                  # Ignore index
            filter_minmax=False  # Don't filter min/max values
        )
        miou_metric.reset()
        return miou_metric

    def _process_batch(self, data):
        """
        Process a single batch of data.

        Args:
            data (dict): Batch data dictionary

        Returns:
            dict: Result dictionary from model inference
        """
        # Move tensors to GPU
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()

        # Extract images and forward pass
        input_imgs = data.pop('img')
        return self.model(imgs=input_imgs, metas=data)

    def _process_occupancy(self, result_dict, miou_metric, iter_idx):
        """
        Process occupancy predictions and update metrics.

        Args:
            result_dict (dict): Model output dictionary
            miou_metric (MeanIoU): Metric instance
            iter_idx (int): Iteration index for visualization
        """
        for idx, pred in enumerate(result_dict['final_occ']):
            pred_occ = pred
            gt_occ = result_dict['sampled_label'][idx]
            occ_mask = result_dict['occ_mask'][idx].flatten()

            # Visualize occupancy if requested
            if self.args.vis_occ:
                self._visualize_occupancy(pred_occ, gt_occ, iter_idx)

            # Update metrics
            miou_metric._after_step(pred_occ, gt_occ, occ_mask)

    def _visualize_occupancy(self, pred_occ, gt_occ, iter_idx):
        """
        Visualize occupancy predictions.

        Args:
            pred_occ (torch.Tensor): Predicted occupancy
            gt_occ (torch.Tensor): Ground truth occupancy
            iter_idx (int): Iteration index for naming
        """
        try:
            from vis import save_occ

            vis_dir = os.path.join(self.args.work_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)

            # Save prediction visualization
            save_occ(
                vis_dir,
                pred_occ.reshape(1, 200, 200, 16),
                f'val_{iter_idx}_pred',
                True, 0
            )

            # Save ground truth visualization
            save_occ(
                vis_dir,
                gt_occ.reshape(1, 200, 200, 16),
                f'val_{iter_idx}_gt',
                True, 0
            )
        except Exception as e:
            self.logger.warning(f'Visualization failed: {str(e)}')

import argparse
import torch

# Create a dummy args object with the desired attributes.
args = argparse.Namespace(
    py_config='config/nuscenes_gs25600_solid.py',
    work_dir='out',
    resume_from='downloads/nonempty/nonempty.pth',
    seed=42,
    gpus=torch.cuda.device_count(),
    vis_occ=False
)




#%%

def main(local_rank, args):
    """
    Main entry point for model evaluation.

    Args:
        local_rank (int): Local rank for distributed processing
        args (argparse.Namespace): Command line arguments
    """
    # Initialize environment, model, and dataset managers
    env_manager = EnvironmentManager(local_rank, args)
    model_manager = ModelManager(env_manager)
    # dataset_manager = DatasetManager(env_manager)

    # Initialize components
    model_manager.initialize()
    # dataset_manager.load_datasets(val_only=True)

    # # Evaluate the model
    # evaluator = Evaluator(env_manager, model_manager, dataset_manager)
    # results = evaluator.evaluate()

    # # Log final results
    # env_manager.get_logger().info(f"Evaluation complete. Final results: {results}")


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
        
        # Create dummy projection matrices (identity transforms)
        projection_matrices = torch.eye(4).unsqueeze(0).repeat(6, 1, 1).float().to(device)
        image_wh = torch.tensor([[image_width, image_height]]).repeat(6, 1).float().to(device)
        
        metas = {
            'projection_mat': projection_matrices,
            'image_wh': image_wh,
            # Add other necessary metadata with default values
            'occ_xyz': torch.zeros(1, 200, 200, 16, 3).to(device),  # Occupancy grid coordinates
            'cam_positions': torch.zeros(1, 6, 3).to(device),  # Camera positions
            'focal_positions': torch.zeros(1, 6, 3).to(device)  # Focal positions
        }
    else:
        # Use provided camera parameters
        metas = {}
        for key, value in camera_params.items():
            if isinstance(value, np.ndarray):
                metas[key] = torch.from_numpy(value).to(device)
            elif isinstance(value, torch.Tensor):
                metas[key] = value.to(device)
            else:
                metas[key] = value
    
    # Run inference
    with torch.no_grad():
        results = model(imgs=images, metas=metas)
    
    return results


def inference_example():
    """
    Example function showing how to use run_inference with sample images from a NuScenes scene.
    
    Returns:
        dict: Model predictions
    """
    import cv2
    import os
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    
    # Initialize environment and model
    args = argparse.Namespace(
        py_config='config/nuscenes_gs25600_solid.py',
        work_dir='out',
        resume_from='downloads/nonempty/nonempty.pth',
        seed=42,
        gpus=torch.cuda.device_count(),
        vis_occ=False
    )
    
    env_manager = EnvironmentManager(0, args)
    model_manager = ModelManager(env_manager)
    model_manager.initialize()
    model = model_manager.get_model()
    
    # Load a sample from NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes/', verbose=False)
    
    # Get a sample
    sample = nusc.sample[0]  # Use the first sample
    
    # Load all 6 camera images
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                  'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    images = []
    for cam in camera_types:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = os.path.join('data/nuscenes/', cam_data['filename'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            # Create a blank image as fallback
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Resize image to expected dimensions
        img = cv2.resize(img, (1600, 900))
        
        # Convert to float32 (required by model)
        img = img.astype(np.float32)
        
        # Apply normalization using the same parameters as in training
        # From surroundocc.py: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = (img - mean) / std
        
        images.append(img)
    
    # Get camera calibration and other metadata
    camera_params = {}
    
    # Create projection matrices
    camera_params['projection_mat'] = np.zeros((6, 4, 4), dtype=np.float32)
    camera_params['image_wh'] = np.array([[1600, 900]] * 6, dtype=np.float32)
    
    # Get lidar2cam matrices for each camera
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    # Create occupancy grid coordinates
    grid_size = 0.5
    x = np.arange(-50, 50, grid_size) + grid_size/2
    y = np.arange(-50, 50, grid_size) + grid_size/2
    z = np.arange(-5, 3, grid_size) + grid_size/2
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    occ_xyz = np.stack([xx, yy, zz], axis=-1)
    camera_params['occ_xyz'] = occ_xyz.reshape(1, 200, 200, 16, 3)
    
    # Set camera and focal positions (these would normally come from calibration)
    camera_params['cam_positions'] = np.zeros((1, 6, 3), dtype=np.float32)
    camera_params['focal_positions'] = np.zeros((1, 6, 3), dtype=np.float32)
    
    # Calculate actual projection matrices for each camera
    for i, cam in enumerate(camera_types):
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Calculate view matrix (world to camera)
        # Convert quaternion to rotation matrix
        R_cam = Quaternion(cam_cs['rotation']).rotation_matrix  # Use pyquaternion to convert
        t_cam = np.array(cam_cs['translation'])
        
        view_mat = np.eye(4)
        view_mat[:3, :3] = R_cam
        view_mat[:3, 3] = t_cam
        
        # Calculate projection matrix (camera to image)
        intrinsic = np.array(cam_cs['camera_intrinsic'])
        proj_mat = np.eye(4)
        proj_mat[:3, :3] = intrinsic
        
        # Combined matrix
        camera_params['projection_mat'][i] = proj_mat @ np.linalg.inv(view_mat)
        
        # Store camera position
        camera_params['cam_positions'][0, i] = t_cam
        camera_params['focal_positions'][0, i] = t_cam + R_cam @ np.array([0, 0, 0.0055])
    
    # Run inference
    results = run_inference(model, images, camera_params)
    
    # Process the results
    occupancy_pred = results['final_occ'][0]  # First batch item
    
    # Convert to numpy if needed
    if isinstance(occupancy_pred, torch.Tensor):
        occupancy_pred = occupancy_pred.cpu().numpy()
    
    print(f"Occupancy prediction shape: {occupancy_pred.shape}")
    
    return results

# %%
env_manager = EnvironmentManager(0, args)
model_manager = ModelManager(env_manager)
# %%
dataset_manager = DatasetManager(env_manager)
dataset_manager.load_datasets(val_only=True)
# %%
evaluator = Evaluator(env_manager, model_manager, dataset_manager)
# %%
# results = evaluator.evaluate()

# %%

inference_example()


# %%


# %%
