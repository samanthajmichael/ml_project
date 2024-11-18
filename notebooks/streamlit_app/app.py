import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

def display_grid(frames, timestamps, cols=3):
    """Display frames in a grid using Streamlit columns"""
    # Limit to 12 frames to prevent performance issues
    frames = frames[:12]
    timestamps = timestamps[:12]
    
    # Create rows of columns
    for i in range(0, len(frames), cols):
        # Create columns for this row
        columns = st.columns(cols)
        
        # Fill each column with an image
        for col_idx in range(cols):
            frame_idx = i + col_idx
            if frame_idx < len(frames):
                with columns[col_idx]:
                    st.image(frames[frame_idx], 
                            caption=f"Scene at {timestamps[frame_idx]:.2f}s",
                            use_column_width=True)

def sample_frames(cap, total_frames, max_frames=1000):
    """Sample frames evenly throughout the video"""
    if total_frames <= max_frames:
        return range(total_frames)
    
    # Calculate step size to get max_frames samples
    step = total_frames // max_frames
    return range(0, total_frames, step)

def process_video_frame(frame):
    """Convert frame to grayscale and return it"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def main():
    st.title("Video Scene Detection")
    st.write("Upload a video to detect scene changes for match cutting")
    
    # Set up session state for storing results
    if 'scene_changes' not in st.session_state:
        st.session_state.scene_changes = []
    
    # File uploader
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        # Create two columns for controls and preview
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("Video Controls")
            threshold = st.slider("Scene Change Sensitivity", 
                                min_value=10, 
                                max_value=100, 
                                value=30)
            
            sample_rate = st.slider("Sampling Rate (%)", 
                                  min_value=1, 
                                  max_value=100, 
                                  value=10,
                                  help="Lower value = faster processing but might miss some scenes")
            
            process_button = st.button('Process Video')
        
        # Save uploaded video to temporary file
        if process_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_file.read())
                video_path = tfile.name
            
            try:
                # Open video file
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("Error: Couldn't open video file")
                else:
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    with col2:
                        st.write(f"FPS: {fps}")
                        st.write(f"Total Frames: {total_frames}")
                    
                    # Calculate frames to sample based on sample_rate
                    max_frames = int((total_frames * sample_rate) / 100)
                    frame_indices = sample_frames(cap, total_frames, max_frames)
                    
                    # Create progress bar
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    # Process frames
                    prev_frame = None
                    scene_changes = []
                    frames_processed = 0
                    
                    for frame_idx in frame_indices:
                        # Set frame position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if not ret:
                            break
                        
                        # Update progress
                        frames_processed += 1
                        progress = int((frames_processed / len(frame_indices)) * 100)
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing frame {frames_processed}/{len(frame_indices)}")
                        
                        if prev_frame is not None:
                            # Process frames
                            curr_gray = process_video_frame(frame)
                            prev_gray = process_video_frame(prev_frame)
                            
                            # Calculate difference
                            frame_diff = cv2.absdiff(curr_gray, prev_gray)
                            mean_diff = np.mean(frame_diff)
                            
                            # If difference is above threshold, we have a scene change
                            if mean_diff > threshold:
                                timestamp = frame_idx / fps
                                scene_changes.append((timestamp, frame))
                                
                                # Display frame where scene change was detected
                                with col2:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    st.image(frame_rgb, 
                                           caption=f"Scene change at {timestamp:.2f} seconds",
                                           use_container_width=True)
                        
                        prev_frame = frame
                    
                    # Store results in session state
                    st.session_state.scene_changes = scene_changes
                    
                    # Display results
                    if scene_changes:
                        st.success(f"Found {len(scene_changes)} scene changes!")
                        
                        # Create download button for timestamps
                        timestamp_text = "\n".join([f"Scene change at {t:.2f} seconds" for t, _ in scene_changes])
                        st.download_button(
                            "Download Timestamps",
                            timestamp_text,
                            file_name="scene_timestamps.txt",
                            mime="text/plain"
                        )
                        
                        # Option to view all scenes in a grid
                        if st.button("Show All Scene Changes in Grid"):
                            cols = st.columns(3)
                            for idx, (timestamp, frame) in enumerate(scene_changes):
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                cols[idx % 3].image(frame_rgb, 
                                                  caption=f"Scene at {timestamp:.2f}s",
                                                  use_container_width=True)
                    else:
                        st.warning("No scene changes detected. Try adjusting the sensitivity.")
                
                cap.release()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Cleanup
                try:
                    os.unlink(video_path)
                except:
                    pass

if __name__ == "__main__":
    main()
