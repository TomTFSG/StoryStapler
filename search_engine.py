import os
import pickle
import json
import hashlib
import time
import subprocess
import uuid
import numpy as np
import math
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy import VideoFileClip, concatenate_videoclips, TextClip, ColorClip, CompositeVideoClip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-mpnet-base-v2", device=device)
video_formats = ['movie.mp4']
MAX_QUEUE = 5

defined_scenes = {}
scene_embeddings = {}
dialogue_embeddings = {}
emotion_list = []
location_list = []
genre_list = []
year_list = []
scene_count = 0
selected_queue = []  # (video_path, start_sec, end_sec)

cache_file = "./cache/scene_embeds_cache.pkl"

def is_data_loaded():
    # Check if key data structures are populated
    return bool(defined_scenes) and bool(scene_embeddings) and bool(dialogue_embeddings)

def print_progress(prefix, i, total, every=100):
    if i % every == 0 or i == total - 1:
        print(f"{prefix} {i+1}/{total} ({(i+1)/total*100:.2f}%)")

def get_embedding(text):
    return model.encode(text, convert_to_numpy=True)

def parse_time_to_seconds(time_str):
    dt = datetime.strptime(time_str, "%H:%M:%S.%f")
    return dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6

def get_video_file_path(folder_name):
    for fmt in video_formats:
        path = f"./movies/{folder_name}/{fmt}"
        if os.path.exists(path):
            return path
    return None

def hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_scene_data(folder_path):
    global defined_scenes, scene_embeddings, emotion_list, location_list, genre_list, year_list, scene_count
    
    print(f"Starting to load scenes from: {folder_path}")

    scenes_local = {}
    emotions_found = set()
    locations_found = set()
    genres_found = set()
    years_found = set()

    scene_count = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'movie' not in data or 'scenes' not in data['movie']:
                    print(f"Malformed JSON structure in: {filename}")
                    continue

                movie = data['movie']
                title = movie.get('title', '')
                raw_genres = movie.get('genre', [])
                genres = raw_genres if isinstance(raw_genres, list) else [g.strip() for g in raw_genres.split(',')]
                genres_found.update(genres)

                year = movie.get('year', None)
                try:
                    year = int(year)
                    years_found.add(year)
                except (TypeError, ValueError):
                    pass

                type = movie.get('type', None)
                series = movie.get('series', None)
                season = movie.get('season', None)
                episode = movie.get('episode', None)
                moviecode = os.path.splitext(filename)[0]
                total_scenes = movie.get('total_scenes', None)

                for scene in movie['scenes']:
                    scene_count += 1
                    print(f"Loaded scene {scene_count}", end="\r", flush=True)

                    sid = f"{filename} - Scene {scene.get('id', 'unknown')}"
                    desc = scene.get('scene_description', '')
                    segment = scene.get('segment', ['00:00:00.000', '00:00:00.000'])
                    dlg = scene.get('dialogue', '')
                    if isinstance(dlg, list):
                        dlg = ' '.join(dlg)
                    emotions = scene.get('emotion', {})
                    emotion_labels = list(emotions.keys())
                    locations = [l.get('scene', '') for l in scene.get('location', [])]

                    emotions_found.update(emotions.keys())
                    locations_found.update(locations)

                    scenes_local[sid] = {
                        'total_scenes': total_scenes,
                        'moviecode': moviecode,
                        'description': desc,
                        'segment': segment,
                        'dialogue': dlg,
                        'title': title,
                        'year': year,
                        'type': type,
                        'series': series,
                        'season': season,
                        'episode': episode,
                        'emotion': emotion_labels,
                        'location': locations,
                        'genres': genres
                    }

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Load existing cache or create empty
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)  # cached: dict sid -> (hash, embedding)
    else:
        cached = {}

    # Identify scenes needing new embeddings
    new_ids = []
    new_texts = []

    for sid, scene in scenes_local.items():
        current_hash = hash_text(scene['description'])
        if sid not in cached or cached[sid][0] != current_hash:
            new_ids.append(sid)
            new_texts.append(scene['description'])

    # Encode new or updated scenes
    if new_texts:
        print(f"\nEncoding {len(new_texts)} new/updated scene descriptions...")
        new_embeds = model.encode(
            new_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=64,
            num_workers=16
        )
        for sid, emb, desc_text in zip(new_ids, new_embeds, new_texts):
            cached[sid] = (hash_text(desc_text), emb)

    # Save updated cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(cached, f)

    # Prepare embeddings dict for use
    scene_embeds = {sid: emb for sid, (h, emb) in cached.items()}

    # Update globals
    defined_scenes = scenes_local
    scene_embeddings = scene_embeds
    emotion_list = sorted(emotions_found)
    location_list = sorted(locations_found)
    genre_list = sorted(genres_found)
    year_list = sorted(set(int(y) for y in years_found if isinstance(y, int) or (isinstance(y, str) and y.isdigit())))

    print(f"Finished loading {len(defined_scenes)} scenes.")

def load_dialogue_embeddings():
    global dialogue_embeddings

    cache_file = "./cache/dialogue_embeds_cache.pkl"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    print("Loading dialogue embeddings...")

    # Load existing cache or empty
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)  # cached: dict sid -> (hash, embedding)
    else:
        cached = {}

    new_ids = []
    new_texts = []

    # Check which dialogues need new embeddings
    for sid, scene in defined_scenes.items():
        dlg = scene.get('dialogue', '')
        current_hash = hash_text(dlg)
        if sid not in cached or cached[sid][0] != current_hash:
            new_ids.append(sid)
            new_texts.append(dlg)

    # Encode new or updated dialogues
    if new_texts:
        print(f"Encoding {len(new_texts)} new/updated dialogues...")
        new_embeds = model.encode(
            new_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=64,
            num_workers=16
        )
        for sid, emb, dlg_text in zip(new_ids, new_embeds, new_texts):
            cached[sid] = (hash_text(dlg_text), emb)

        # Save updated cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(cached, f)

    # Prepare embeddings dict for use
    dialogue_embeddings = {sid: emb for sid, (h, emb) in cached.items()}

    print(f"Loaded {len(dialogue_embeddings)} dialogue embeddings.")

def find_best_matches(query, embeddings_dict, batch_size=5000):
    q = get_embedding(query)
    sids = list(embeddings_dict.keys())
    sims = []

    for i in range(0, len(sids), batch_size):
        batch_sids = sids[i:i+batch_size]
        batch_matrix = np.stack([embeddings_dict[sid] for sid in batch_sids])
        batch_sims = cosine_similarity([q], batch_matrix)[0]
        sims.extend(zip(batch_sids, batch_sims))

    return sorted(sims, key=lambda x: x[1], reverse=True)

def build_result_list(matches, data_dict, top_n=300):
    out = []
    for sid, score in matches[:top_n]:
        d = data_dict[sid]
        out.append({
            "total_scenes": d['total_scenes'],
            'scene_id': sid,
            'moviecode': d['moviecode'],
            'title': d['title'],
            'segment': d['segment'],
            'score': float(score),
            'year': d.get('year'),
            'type': d.get('type'),
            'series': d.get('series'),
            'season': d.get('season'),
            'episode': d.get('episode'),
            'desc': d['description'][:150],
            'dialogue': d['dialogue'][:150],
            'emotion': d['emotion'],
            'location': d['location'],
            'genres': d['genres']
        })
    return out

def search_scenes(query, weight=0.5, emotion_filter='All', location_filter='All',
                  genre_filter='All', year_start=0, year_end=9999):
    if not query:
        return []

    desc_scores = dict(find_best_matches(query, scene_embeddings))
    dlg_scores = dict(find_best_matches(query, dialogue_embeddings))

    combined = {}
    total = len(defined_scenes)
    for i, sid in enumerate(defined_scenes.keys()):
        desc_score = desc_scores.get(sid, 0.0)
        dlg_score = dlg_scores.get(sid, 0.0)
        final_score = (dlg_score * (1 - weight)) + (desc_score * weight)
        combined[sid] = final_score
        #print_progress("Scoring", i, total, every=500)

    matches = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    filtered = []
    for sid, sc in matches:
        data = defined_scenes[sid]
        emo_dict = data['emotion']
        loc_list = data['location']
        genres = data['genres']
        year = data.get('year')
        if year is None or not (year_start <= year <= year_end):
            continue
        if (
            (emotion_filter == 'All' or any(emotion_filter.lower() == e.lower() for e in emo_dict)) and 
            (location_filter == 'All' or location_filter.lower() in [loc.lower() for loc in loc_list]) and
            (genre_filter == 'All' or genre_filter.lower() in [g.lower() for g in genres])
        ):
            filtered.append((sid, sc))
    return build_result_list(filtered, defined_scenes)

def add_scene_to_queue(scene_id, custom_start=None, custom_end=None, queue=None):
    scene = defined_scenes.get(scene_id)
    if not scene:
        return False, "Scene not found"

    segment = scene.get('segment', ['00:00:00.000', '00:00:00.000'])
    start_default = parse_time_to_seconds(segment[0])
    end_default = parse_time_to_seconds(segment[1])

    start = float(custom_start) if custom_start is not None else start_default
    end = float(custom_end) if custom_end is not None else end_default

    movie_code = scene.get('moviecode', 'unknown')
    video_path = get_video_file_path(movie_code)

    if start >= end:
        return False, "Start time must be before end time"

    if video_path is None or not os.path.exists(video_path):
        return False, "Video file not found"

    if queue is not None:
        queue.append((video_path, start, end))
    else:
        return False, "Queue not provided"

    return True, f"Scene {scene_id} queued with time {start:.2f} - {end:.2f}"

def even_width_clip(clip, target_height=720):
    aspect_ratio = clip.w / clip.h
    new_width = int(round(target_height * aspect_ratio))
    if new_width % 2 != 0:
        new_width += 1
    return clip.resized(height=target_height, width=new_width)



def stitch_and_save(scene_queue, title_text="My Story"):
    if not scene_queue:
        return False, "No clips in queue"

    clips = []
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uid = str(uuid.uuid4())[:8]
        filename_base = f"{timestamp}_{uid}"
        os.makedirs('results', exist_ok=True)

        tmp_output = f'results/tmp_{filename_base}.mp4'
        final_output = f'results/stitched_{filename_base}.mp4'

        # --- Title Scene ---
        
        w, h = 1280, 720
        duration = 2  # seconds

        bg = ColorClip(size=(w, h), color=(0, 0, 0), duration=duration)
        bg = ColorClip(size=(w, h), color=(0, 0, 0), duration=duration)
        txt = TextClip(
            font="./static/css/fonts/IBM_Plex_Sans/static/IBMPlexSans_Condensed-Bold.ttf",
            text=title_text,
            font_size=70,
            color='white',
            size=(w, h),
            method="caption",
            duration=duration
        ).with_position("center")

        title_clip = CompositeVideoClip([bg, txt]).with_fps(24).with_audio(None)
        clips.append(title_clip)

        # --- User-selected scenes ---
        for path, st, ed in scene_queue:
            clip = VideoFileClip(path).subclipped(st, ed)
            clip = even_width_clip(clip)
            clips.append(clip)

        # Concatenate all
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            tmp_output,
            codec='libx264',
            audio_codec='aac',
            fps=24,
            threads=4,
            preset='medium'
        )
        final_clip.close()
        for c in clips:
            c.close()

        # FFmpeg re-encode
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', tmp_output,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            final_output
        ]
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            return False, f"FFmpeg encoding failed:\n{result.stdout}"

        os.remove(tmp_output)
        return True, final_output

    except Exception as e:
        return False, str(e)



# Load data on import
load_scene_data('data')
load_dialogue_embeddings()
