import argparse
import os
import subprocess
from PIL import Image

def is_valid_image(image_path):
    """Check if the image is valid."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is a valid image
        return True
    except (IOError, SyntaxError) as e:
        print(f"Skipping corrupted image {image_path}: {e}")
        return False

def run_inference(cfg, checkpoint, indir, outdir, save_img, pose_track):
    """Run inference on the images located in the 'indir'."""
    command = [
        "python", "scripts/demo_inference.py",
        "--cfg", cfg,
        "--checkpoint", checkpoint,
        "--indir", indir,
        "--outdir", outdir
    ]
    if pose_track:
        command.append("--pose_track")
    if save_img:
        command.append("--save_img")

    print(f"Running inference on: {indir}")
    subprocess.run(command)

def main():
    """Main function to parse arguments and run inference."""
    parser = argparse.ArgumentParser(description="Run AlphaPose inference on multiple 'front_RGB' directories")
    parser.add_argument('--cfg', type=str, required=True, help='experiment configure file name')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint file name')
    parser.add_argument('--base_dir', type=str, required=True, help='base directory containing subdirectories for inference')
    parser.add_argument('--save_img', default=False, action='store_true', help='save result as image')
    parser.add_argument('--pose_track', dest='pose_track', help='track humans in video with reid', action='store_true', default=False)

    args = parser.parse_args()

    # Iterate through directories to find all 'front_RGB' folders
    for root, dirs, files in os.walk(args.base_dir):
        for dir in dirs:
            if dir == 'front_RGB':
                indir = os.path.join(root, dir)
                outdir = root  # Parent directory of 'front_RGB'
                
                print(f"Processing directory: {indir}")  # Log the path of the directory being processed
                
                # Ensure we don't process images in 'trash' subdirectories
                valid_images = []
                for subdir, _, img_files in os.walk(indir):
                    if 'trash' not in os.path.relpath(subdir, indir):
                        for img_file in img_files:
                            img_path = os.path.join(subdir, img_file)
                            if is_valid_image(img_path):
                                valid_images.append(img_path)
                
                if valid_images:
                    # Create a temporary directory to store valid images
                    temp_dir = os.path.join(indir, 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    for img_path in valid_images:
                        temp_img_path = os.path.join(temp_dir, os.path.basename(img_path))
                        # Check if the file or symlink already exists
                        if not os.path.exists(temp_img_path):
                            os.symlink(img_path, temp_img_path)
                        else:
                            print(f"Skipping existing file/symlink: {temp_img_path}")
                    
                    run_inference(args.cfg, args.checkpoint, temp_dir, outdir, args.save_img, args.pose_track)
                    
                    # Clean up the temporary directory
                    for img_file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, img_file))
                    os.rmdir(temp_dir)
                else:
                    print(f"No valid images found in: {indir}")

if __name__ == "__main__":
    main()
