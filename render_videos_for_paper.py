import os
import sys
import argparse
import cv2
sys.path.append('ailia-models/util')
from point_e.util.point_cloud import PointCloud
from point_e.util.plotting import render_point_cloud_video

def frames_to_video(input_dir, output_file, fps=20):
    """
    Renders a video from frames stored in a directory.
    
    Parameters:
        input_dir (str): Path to the directory containing frames in PNG format.
        output_file (str): Path to the output video file.
        fps (int): Frames per second for the output video.
    
    Returns:
        None
    """
    # Get a sorted list of frame file paths
    frames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")])
    
    if not frames:
        raise ValueError("No PNG files found in the specified directory.")
    
    # Read the first frame to determine video properties
    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    size = (width, height)
    
    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 videos
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, size)
    
    # Write each frame to the video
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        video_writer.write(frame)
    
    # Release the VideoWriter
    video_writer.release()
    print(f"Video successfully saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Render videos for paper.')
    parser.add_argument('input_folder', type=str, help='Input folder containing .npz files')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames (default: 100)')
    args = parser.parse_args()

    input_folder = args.input_folder
    num_frames = args.num_frames

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npz'):
                print(f'Rendering video for - {os.path.join(root, file)}')
                npz_file = os.path.join(root, file)           
                output_folder = os.path.join(root, file[:-4])
                # check if video already exists
                if os.path.exists(output_folder): #and file[:-4] != "changeit":
                    if os.path.exists(os.path.join(output_folder, 'images', 'frame_00049.png')):
                        print(f'Frames exist, re-rendering vid for {npz_file}')
                        frames_to_video(os.path.join(output_folder, 'images'), os.path.join(output_folder, 'video.mp4'))
                        continue
                    if os.path.exists(os.path.join(output_folder, 'video.mp4')):
                        print(f'Video already exists for {npz_file}')
                        continue
                os.makedirs(output_folder, exist_ok=True)
                try:
                    render_point_cloud_video(npz_file, output_folder, num_frames)
                except Exception as e:
                    print(f'Error rendering: {e}')

if __name__ == '__main__':
    main()