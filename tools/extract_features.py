import os
import sys
import argparse
import torch
import h5py
import numpy as np
from tqdm import tqdm
from transformers import DINOv3ViTModel, AutoImageProcessor
import torchvision.transforms.functional as TF

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from semgaze.datasets.gazefollow import GazeFollowDataset
from PIL import Image

def extract_features(args):
    """
    Extracts features and head crops from the GazeFollow dataset and saves them to disk.
    """

    print(f"Starting feature extraction for split: {args.split}")

    # Initialize DINO model and processor
    print(f"Initializing DINOv3 model '{args.model_name}' and processor...")
    try:
        model = DINOv3ViTModel.from_pretrained(args.model_name)
        processor = AutoImageProcessor.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error initializing model from Hugging Face: {e}")
        return

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    # Initialize Dataset
    print("Initializing GazeFollow dataset...")
    try:
        dataset = GazeFollowDataset(
            root=args.root,
            root_project=args.root_project,
            root_heads=args.root_heads,
            split=args.split,
            transform=None,  # Important: We want the raw PIL images
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
        
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Features will be saved to: {output_dir}")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Extracting features for {args.split} split"):
            sample = dataset[i]
            
            image_path = sample["path"]
            
            # Define output path
            feature_file_path = os.path.join(output_dir, os.path.splitext(image_path)[0] + ".h5")

            # Skip if feature file already exists
            if os.path.exists(feature_file_path):
                continue
                
            image = sample["image"]

            # Process main image for DINOv3 features
            inputs = processor(images=image, return_tensors="pt", do_rescale=True, do_resize=True, do_normalize=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            image_features = outputs.last_hidden_state.squeeze(0).cpu().half().numpy()

            # Process head crops
            head_crops_pil = sample["heads"]  # This is a list of PIL images
            head_crops_np = []
            for head_pil in head_crops_pil:
                head_resized = TF.resize(head_pil, (224, 224), antialias=True)
                head_np = np.array(head_resized)
                head_crops_np.append(head_np)
            
            if head_crops_np:
                all_head_crops = np.stack(head_crops_np)
            else:
                all_head_crops = np.zeros((0, 224, 224, 3), dtype=np.uint8)

            # Ensure the directory for the feature file exists
            os.makedirs(os.path.dirname(feature_file_path), exist_ok=True)

            # Save features and head crops to H5 file
            with h5py.File(feature_file_path, 'w') as f:
                f.create_dataset('image_features', data=image_features)
                f.create_dataset('head_crops', data=all_head_crops)
                f.attrs['path'] = image_path

    print(f"Feature extraction for split '{args.split}' completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract DINOv3 features for the GazeFollow dataset.")
    parser.add_argument('--root', type=str, default='/home/elicer/semgaze/data/gazefollow_extended',
                        help='Root directory of the GazeFollow dataset.')
    parser.add_argument('--root_project', type=str, default='/home/elicer/semgaze',
                        help='Root directory of the project.')
    parser.add_argument('--root_heads', type=str, default='/home/elicer/semgaze/data/GazeFollow-head',
                        help='Root directory for head detections.')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split to process.')
    parser.add_argument('--model_name', type=str, default='facebook/dinov3-vitl16-pretrain-lvd1689m',
                        help='The DINOv3 model to use for feature extraction.')
    parser.add_argument('--output_dir', type=str, default='/home/elicer/semgaze/data/features',
                        help='Directory to save the extracted features.')

    args = parser.parse_args()
    extract_features(args)
