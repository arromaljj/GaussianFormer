#!/usr/bin/env python

#%%

# coding: utf-8
"""
MMSegmentation Inference Tool (Version 3) - NuScenes GS6400 Config

This script provides a specific interface for performing inference with the NuScenes GS6400 configuration.
It uses the MMSegmentation lower-level APIs for inference with the custom config.

Usage:
    # Import and use in your code:
    from inference_v3 import NuScenesGS6400Inference
    
    # Initialize the inference model
    inference = NuScenesGS6400Inference(checkpoint="path/to/checkpoint.pth")
    
    # Run inference on an image
    result = inference.run_inference("path/to/image.jpg", show=True)
"""

import os
import numpy as np
import torch
from mmengine import Config
from mmseg.apis import init_model, inference_model, show_result_pyplot


class NuScenesGS6400Inference:
    """
    Class for inference with the NuScenes GS6400 configuration.
    """

    def __init__(self, checkpoint=None, device='cuda'):
        """
        Initialize the inference interface for NuScenes GS6400 config.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. If None, uses default from config.
            device (str, optional): Device to run inference on. Defaults to 'cuda'.
        """
        self.device = device
        self.checkpoint = checkpoint
        self.config_path = 'config/prob/nuscenes_gs6400.py'
        self.model = None
        
        # Initialize the model
        self.init_model()
            
    def init_model(self):
        """Initialize model with the NuScenes GS6400 config."""
        print(f"Initializing model from config: {self.config_path}")
        
        # Load the config file and initialize the model
        try:
            self.model = init_model(
                self.config_path,
                self.checkpoint,
                device=self.device
            )
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    
    def run_inference(self, img_path, show=False, output_dir=None):
        """
        Run inference using the NuScenes GS6400 model.
        
        Args:
            img_path (str): Path to image file.
            show (bool): Whether to show the result. Defaults to False.
            output_dir (str, optional): Directory to save visualization results.
            
        Returns:
            Result of the inference.
        """
        if self.model is None:
            self.init_model()
            
        # Run inference
        result = inference_model(self.model, img_path)
        
        # Visualize result if requested
        if show or output_dir:
            out_file = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_file = os.path.join(output_dir, os.path.basename(img_path))
                
            vis_result = show_result_pyplot(
                self.model, 
                img_path, 
                result, 
                show=show,
                out_file=out_file
            )
            
        return result

#%% 

# Example usage
if __name__ == '__main__':
    # Simple args object example - modify these values as needed
    class Args:
        def __init__(self):
            self.img_path = None  # Path to your image
            self.checkpoint = None  # Path to your checkpoint or None to use default
            self.show = True  # Whether to display results
            self.output_dir = 'results'  # Directory to save results
    
    # Create args object
    args = Args()
    
    # If you have an image path, you can run inference
    if args.img_path:
        # Initialize the inference model
        inference = NuScenesGS6400Inference(checkpoint=args.checkpoint)
        
        # Run inference
        result = inference.run_inference(
            args.img_path,
            show=args.show,
            output_dir=args.output_dir
        )
        print("Inference completed successfully")
    else:
        print("No image path provided. Please set Args.img_path to run inference.")
        
        # Just initialize the model without running inference
        inference = NuScenesGS6400Inference(checkpoint=args.checkpoint)
        print("Model initialized successfully, but no inference performed.") 
# %%
