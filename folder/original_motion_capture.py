# import cv2
# import mediapipe as mp

# # Initialize MediaPipe solutions.
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# # Open the video
# cap = cv2.VideoCapture('merged_video.mp4')

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', 'AVC1', etc.
# out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# # Check if video opened successfully
# if not cap.isOpened():
#     print("Error opening video stream or file")

# # Read until video is completed
# while cap.isOpened():
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret:
#         # Convert the BGR image to RGB.
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the image and find poses
#         results = pose.process(image)
        
#         # Draw the pose annotation on the image.
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
#         out.write(frame)
        
#     else:
#         break

# # When everything done, release the video capture and video write objects
# cap.release()
# out.release()
# # Closes all the frames
# cv2.destroyAllWindows()


import os
import json
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
import requests
from pydub import AudioSegment
import string
nltk.download('wordnet')
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip
file_path = './WLASL_v0.3.json'
import cv2
import mediapipe as mp
with open(file_path, 'r') as file:
    json_data = json.load(file)

videoADD = []


API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

def genlandmk():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

    # Open the video
    cap = cv2.VideoCapture('merged_video.mp4')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', 'AVC1', etc.
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and find poses
            results = pose.process(image)
            
            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            out.write(frame)
            
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    json_response = response.json()
    transcribed_text = json_response.get('text', 'Text not found')
    
    # Formatting the text
    formatted_text = transcribed_text.lower()  # Convert to lowercase
    formatted_text = formatted_text.replace('-', ' ')  # Replace hyphens with spaces
    
    # Remove punctuation (excluding hyphens since they are already replaced)
    translator = str.maketrans('', '', string.punctuation)
    formatted_text = formatted_text.translate(translator)

    return formatted_text


def stream_litSETUP():
    st.title('Video Display App')
    uploaded_file = st.file_uploader("Choose an audio file")
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        save_path = "saved_audio_file." + uploaded_file.name.split('.')[-1]
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        output = query(save_path)
        words = formatSignLang(output)
        for i in range(0,len(words)):
            exists(words[i])
        print(videoADD)
        create_VIDEO()
        genlandmk()
        video_file = 'output_video.mp4'
        video_file = open(video_file, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

def formatSignLang(sentence):
    be_verbs = {'am', 'is', 'are', 'was', 'were'}
    articles = {'a', 'an', 'the'}
    words = sentence.split()
    filtered_words = [word for word in words if word not in be_verbs and word not in articles]
    lemmatizer = WordNetLemmatizer()
    present_tense_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words] 
    
    return present_tense_words


def exists(value):
    contains_value = any(entry["gloss"] == value for entry in json_data)
    if contains_value == False:
        print(value)
        return 0

    video_directory = "./videos"
    found = False  # Flag to indicate if the video is found

    for entry in json_data:
        if entry["gloss"] == value:
            for instance in entry["instances"]:
                video_id = instance["video_id"]
                videoIDSTR = str(video_id)
                videoIDSTRMP4 = videoIDSTR + ".mp4"
                video_path = os.path.join(video_directory,videoIDSTRMP4)
                
                print(f"Checking {video_path}: ", os.path.exists(video_path))
                
                if os.path.exists(video_path):
                    print("Video found: " + videoIDSTRMP4)
                    found = True
                    break  
        if found:
            videoADD.append(video_path)
            break
    print(videoIDSTRMP4)
    return videoIDSTRMP4

def resize_clip(clip, width, height):
    return clip.resize((width, height))

def create_VIDEO():

    # Resize video clips to a common resolution
    width = 1280  # Common width
    height = 720  # Common height
    for i in range(0, len(videoADD)):
        videoADD[i] = resize_clip(VideoFileClip(videoADD[i]), width, height)
    

    # Concatenate video clips with the static image in between
    final_video = concatenate_videoclips(videoADD)

    # Export the final video
    final_video.write_videofile("merged_video.mp4", codec="libx264")

stream_litSETUP()