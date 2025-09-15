import os
import json
import moviepy as mp
import cv2
import numpy as np
import torch
import shutil
import tkinter as tk
from tkinter import SE, Tk, filedialog, messagebox, Button, Entry
import os
from datetime import datetime
from PIL import Image
from collections import Counter
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from huggingface_hub import login
from sklearn.cluster import KMeans
from pathlib import Path
from transformers import pipeline
from transformers import AutoTokenizer
from difflib import SequenceMatcher
import re
import logging
import chardet
import subprocess


# Setup environment and logging
os.environ["IMAGEMAGICK_BINARY"] = "auto"
logging.getLogger("moviepy").setLevel(logging.ERROR)

# Load models and processors
device = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP (image captioning) model for scene descriptions
login("...") #INSERT HUGGINGFACE TOKEN HERE
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# CLIP model for scene classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)

# Emotion classifier for scene analysis (Updated classifier)
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

# Scene labels for CLIP model
scene_labels = [
    "Office", "Apartment", "Kitchen", "Bathroom", "Street", "Forest", "Beach",
    "Shopping mall", "Train station", "Hospital", "Warehouse", "Ruins",
    "Library", "Space Station", "Airport Terminal", "Mountain Trail", 
    "Restaurant", "Concert Stage", "Parking Garage", "Underground Tunnel",
    "Desert", "Battlefield", "Pool", "Art Gallery", "Rubble", "Highway",
    "Bridge", "Gas Station", "Countryside", "Cityscape",
    "Night club", "Garden", "Rooftop Terrace", "Subway Station",
    "School", "Tower", "Outer Space", "Bedroom", "Living Room", "Arena", "Village",
    "Wedding", "Car", "Van", "Gas station", "Cave", "Hotel", "Cabin", "City",
    "Space Ship", "Bar", "Cafe", "New York", "Desert", "Temple", "Swamp", "City Ruins",
    "Classroom", "Library", "Roadside", "Laboratory", "Waterfall", "Cliffside", "River",
    "Field", "Jungle", "Lighthouse", "City Square", "Suburban Neighborhood", "Skyscrapper Rooftop",
    "House Rooftop", "Shopping Street", "Mall", "Parking lot", "Construction Site",
    "Dinning Room", "Museum", "Theater", "Cinema", "Chapel", "Church", "Park", "Zoo",
    "Alleyway", "Train Station", "Bus Stop", "Amusement Park", "Factory", "Prison", "Cemetery",
    "Military Base", "White House", "Oval Office", "Fire Station", "Police Station","Racetrack","Stadium",
    "College", "Study Room", "Farm", "Computer Lab", "Prom", "Airplane", "Moon", "Cafeteria","Savannah",
    "Underwater", "High Sea","Jail","Courtroom","School Bus","Subway","Los Angeles","Castle",
    "Sewage System", "Diner", "Basement", "Dungeon", "Newsroom", "Grocery Store", "News Stand","Disco", 
    "Beach Resort", "Ski Resort", "Amusement Arcade", "Bowling Alley", "Ice Cream Parlor", 
    "Pet Store", "Fireworks Stand", "Car Dealership", "Bookstore", "Music Store", "Sports Store", "Antique Shop",
    "Flower Shop", "Hardware Store", "Pharmacy", "Beauty Salon", "Barbershop", "Tattoo Parlor"
]


def convert_to_mp4(input_video_path, output_video_path):
    try:
        command = [
            'ffmpeg', '-y',
            '-i', input_video_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_video_path
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            yield f"FFmpeg: {line.strip()}"
        process.wait()

        if process.returncode != 0:
            yield "Error: FFmpeg conversion failed."
            raise RuntimeError("FFmpeg failed")

        # Delete original input video after successful conversion
        os.remove(input_video_path)
        yield f"Conversion complete. Original video deleted: {input_video_path}"

    except Exception as e:
        yield f"Error during conversion: {str(e)}"



def chunk_text(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks


def classify_emotions_in_chunks(text):
    chunks = chunk_text(text, emotion_tokenizer, max_length=512)

    emotion_scores = {}

    # Process all chunks as a batch
    results = emotion_classifier(chunks, top_k=5, truncation=True)

    for chunk_result in results:
        for result in chunk_result:
            label = result["label"]
            score = result["score"]
            emotion_scores[label] = emotion_scores.get(label, 0) + score

    # Sort scores from highest to lowest
    sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)

    # Remove 'neutral' if another strong emotion is present
    if "neutral" in emotion_scores and len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0.1:
        sorted_emotions = [(label, score) for label, score in sorted_emotions if label != "neutral"]

    return dict(sorted_emotions)


def timestamp_to_seconds(timestamp):
    t = datetime.strptime(timestamp, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000

def seconds_to_timestamp(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def convert_timestamp_to_seconds(timestamp):
    timestamp = timestamp.replace('.', ',')
    h, m, s = timestamp.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def extract_dominant_colors_with_percentage(frame, n_colors=5, saturation_threshold=0.2):
    frame_resized = cv2.resize(frame, (150, 150))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2HSV)
    hsv_pixels = frame_hsv.reshape((-1, 3))
    saturation = hsv_pixels[:, 1] / 255.0
    mask = saturation > saturation_threshold
    filtered_hsv_pixels = hsv_pixels[mask]
    if len(filtered_hsv_pixels) < n_colors:
        filtered_hsv_pixels = hsv_pixels

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(filtered_hsv_pixels)
    color_counts = Counter(labels)

    total_pixels = len(filtered_hsv_pixels)
    color_percentages = {i: (count / total_pixels) * 100 for i, count in color_counts.items()}

    ordered_hsv_colors = [kmeans.cluster_centers_[i] for i, _ in color_counts.most_common()]
    ordered_rgb_colors = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0] for color in ordered_hsv_colors]
    hex_colors = [f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in ordered_rgb_colors]

    color_with_percentage = list(zip(hex_colors, [color_percentages[i] for i, _ in color_counts.most_common()]))
    return color_with_percentage

def get_scene_probabilities(frame):
    try:
        pil_frame = Image.fromarray(frame)  # convert NumPy to PIL
        inputs = clip_processor(text=scene_labels, images=pil_frame, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
    except Exception as e:
        print("Error during CLIP inference:", e)
        raise
    return probs

def estimate_time_of_day(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[..., 2].mean()
    if brightness > 180:
        return "morning"
    elif brightness > 120:
        return "afternoon"
    elif brightness > 60:
        return "evening"
    else:
        return "night"

def parse_srt(srt_file):
    if not os.path.exists(srt_file):
        print(f"Error: Subtitle file '{srt_file}' not found.")

    # Try reading the file with multiple encodings
    encodings = ['utf-8', 'latin1', 'utf-16']
    srt_data = None
    for encoding in encodings:
        try:
            with open(srt_file, 'r', encoding=encoding) as file:
                srt_data = file.read()
                break
        except (UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Error reading file with {encoding} encoding. Trying next encoding...")
            continue

    if srt_data is None:
        print("Unable to read the subtitle file with supported encodings.")
        exit(1)

    srt_pattern = re.compile(r"(\d+)\n(\d{2}:\d{2}:\d{2}[.,]\d{3}) --> (\d{2}:\d{2}:\d{2}[.,]\d{3})\n(.+?)(?=\n{2}|\Z)", re.DOTALL)
    subtitle_matches = srt_pattern.findall(srt_data)

    subtitles = []
    for match in subtitle_matches:
        start_time = match[1].replace('.', ',')
        end_time = match[2].replace('.', ',')
        text = match[3].replace('\n', ' ').strip()

        start_seconds = convert_timestamp_to_seconds(start_time)
        end_seconds = convert_timestamp_to_seconds(end_time)

        subtitles.append((start_seconds, end_seconds, text))

    return subtitles


def extract_frames(video_path, frame_rate=1, start_time=None, end_time=None):
    video = cv2.VideoCapture(video_path)
    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames

    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    interval = max(1, int(fps / frame_rate))

    while frame_count < end_frame:
        ret, frame = video.read()
        if not ret:
            break
        if (frame_count - start_frame) % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1

    video.release()
    return frames

def get_dialogue_for_scene(subtitles, start_time, end_time):
    dialogue = []
    for subtitle in subtitles:
        subtitle_start = subtitle[0]  # start time is the first element in the tuple
        subtitle_end = subtitle[1]    # end time is the second element
        subtitle_text = subtitle[2]   # text is the third element
        
        if start_time <= subtitle_start < end_time:
            dialogue.append(subtitle_text)
    
    return " ".join(dialogue)

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() > threshold

def describe_frames_batch(frames):
    descriptions = []
    last_desc = ""
    
    for frame in frames:
        inputs = blip_processor(images=frame, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        description = blip_processor.decode(out[0], skip_special_tokens=True).strip()

        # Skip if too similar to the previous one
        if not is_similar(description, last_desc):
            descriptions.append(description)
            last_desc = description

    # Final deduplication and joining
    unique_descriptions = []
    for desc in descriptions:
        if all(not is_similar(desc, existing) for existing in unique_descriptions):
            unique_descriptions.append(desc)

    return ". ".join(unique_descriptions) + "." if unique_descriptions else ""

def save_scene_metadata_json(video_path, scene_list):
    metadata = {
        "video_path": video_path,
        "total_scenes": len(scene_list),
        "scenes": []
    }
    for scene in scene_list:
        scene_metadata = {
            "start_time": scene[0],
            "end_time": scene[1],
            "scene_description": "",
            "dominant_colors": [],
            "time_of_day": "",
            "location": "",
            "dialogue": "",
            "emotion_analysis": ""
        }
        metadata["scenes"].append(scene_metadata)
    json_path = os.path.splitext(video_path)[0] + "_metadata.json"
    with open(json_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

def process_video_with_segments(video_path, metadata, subtitles, cancel_event=None):
    total = len(metadata["movie"]["scenes"])
    for idx, scene in enumerate(metadata["movie"]["scenes"], 1):

        if cancel_event and cancel_event.is_set():
            yield "Canceled."
            return
        
        start_time = scene["start_time"]
        end_time = scene["end_time"]

        try:
            yield f"Scene {idx}/{total}: Extracting frames..."
            frames = extract_frames(video_path, start_time=start_time, end_time=end_time)
            if not frames:
                yield f"Scene {idx}/{total}: No frames extracted."
                continue

            yield f"Scene {idx}/{total}: Describing frames..."
            descriptions = describe_frames_batch(frames)
            scene["scene_description"] = descriptions

            yield f"Scene {idx}/{total}: Extracting colors..."
            scene["dominant_colors"] = extract_dominant_colors_with_percentage(frames[0])

            yield f"Scene {idx}/{total}: Estimating time of day..."
            scene["time_of_day"] = estimate_time_of_day(frames[0])

            yield f"Scene {idx}/{total}: Classifying location..."
            scene_probs = get_scene_probabilities(frames[0])
            top_indices = np.argsort(scene_probs)[::-1][:5]
            scene["location"] = [
                {"scene": scene_labels[i], "score": float(scene_probs[i])}
                for i in top_indices
            ]

            yield f"Scene {idx}/{total}: Extracting dialogue..."
            scene["dialogue"] = get_dialogue_for_scene(subtitles, start_time, end_time)

            combined_text = f"{scene['dialogue']} {scene['scene_description']}".strip()
            if combined_text:
                yield f"Scene {idx}/{total}: Classifying emotion..."
                scene["emotion"] = classify_emotions_in_chunks(combined_text)
            else:
                scene["emotion"] = {}

            yield f"Scene {idx}/{total}: Done."

        except Exception as e:
            print(f"Error processing scene {scene['segment']}: {e}")
            scene["error"] = str(e)
            yield f"Scene {idx}/{total}: Error: {str(e)}"

    yield "All scenes processed"
    return metadata


def find_movie_file(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.mp4', '.mkv', '.avi')):
            return os.path.join(directory, filename)
    return None

def srt_to_vtt(srt_path, vtt_path):
    encoding = detect_encoding(srt_path)
    with open(srt_path, 'r', encoding=encoding) as srt_file:
        srt_content = srt_file.read()

    # Filter out numbering lines (only lines with digits)
    lines = srt_content.splitlines()
    filtered_lines = [line for line in lines if not re.fullmatch(r'\d+', line.strip())]

    filtered_content = "\n".join(filtered_lines)

    # Convert timecode commas to dots
    vtt_content = re.sub(
        r"(\d{2}:\d{2}:\d{2}),(\d{3})",
        r"\1.\2",
        filtered_content
    )

    # Add the WEBVTT header
    vtt_content = "WEBVTT\n\n" + vtt_content

    with open(vtt_path, 'w', encoding='utf-8') as vtt_file:
        vtt_file.write(vtt_content)
        print(f"CONVERTERD TO VTT")

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # sample first 10k bytes for detection
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    if not encoding or encoding.lower() == 'ascii':
        # fallback to latin-1 to avoid decode errors for unknown or ascii-only files
        encoding = 'latin-1'
    return encoding

def segment_movie(video_path, BW=False):
    # no yield inside here
    threshold = 5.0 if BW else 25.0
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    serializable_scene_list = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        serializable_scene_list.append((start_time, end_time))
    return serializable_scene_list

def analyze_movie_and_update_json_streaming(video_path, subtitle_file, genre=None, year=None, media_type=None, series=None, season=None, episode=None, title=None,BW=False,cancel_event=None):

    movie_code = Path(video_path).parts[-2]

    if cancel_event and cancel_event.is_set():
        yield "Canceled."
        return
        
    yield "Segmenting movie..."
    scene_list = segment_movie(video_path, BW)
    yield f"Detected {len(scene_list)} scenes"
            
    if cancel_event and cancel_event.is_set():
        yield "Canceled."
        return
    
    yield f"Detected {len(scene_list)} scenes"

    metadata = {
        "movie": {
            "genre": genre or ["", "", ""],
            "year": year or "",
            "type": media_type or "",
            "series": series or "",
            "season": season or "",
            "episode": episode or "",
            "title": title or "",
            "total_scenes": len(scene_list),
            "scenes": []
        }
    }

    total_scenes = len(scene_list)
    for i, scene in enumerate(scene_list, 1):
        if cancel_event and cancel_event.is_set():
            yield "Canceled."
            return
        
        metadata["movie"]["scenes"].append({
            "id": i,
            "segment": [seconds_to_timestamp(scene[0]), seconds_to_timestamp(scene[1])],
            "start_time": scene[0],
            "end_time": scene[1],
            "scene_description": "",
            "dominant_colors": [],
            "time_of_day": "",
            "location": "",
            "dialogue": "",
            "emotion": {}
        })

        if cancel_event and cancel_event.is_set():
            yield "Canceled."
            return
        #yield f"Prepared scene {i}/{total_scenes}"

    if subtitle_file:
        yield "Parsing subtitles..."
        subtitles = parse_srt(subtitle_file)
        srt_to_vtt(subtitle_file, f"./movies/{movie_code}/subtitles.vtt")
        yield f"Parsed {len(subtitles)} subtitles"
    else:
        subtitles = []
        yield "No subtitles provided; skipping subtitle parsing."
    
    if cancel_event and cancel_event.is_set():
        yield "Canceled."
        return


    yield "Enriching metadata..."
    for message in process_video_with_segments(video_path, metadata, subtitles, cancel_event):
        if cancel_event and cancel_event.is_set():
            yield "Canceled."
            return
        yield message

    if cancel_event and cancel_event.is_set():
        yield "Canceled."
        return
    
    json_path = f"./data/{movie_code}.json"
    with open(json_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    yield "Metadata saved"
    yield "Analysis complete"

