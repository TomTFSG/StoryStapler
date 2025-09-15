from flask import Flask, Response, stream_with_context, request, jsonify, render_template, send_file,send_from_directory
import os, shutil
from werkzeug.utils import secure_filename
from tempfile import NamedTemporaryFile
import search_engine
import os
import json
import subprocess
import threading
import uuid
from pathlib import Path
from collections import Counter
from search_engine import is_data_loaded, defined_scenes,scene_embeddings,dialogue_embeddings,load_dialogue_embeddings,load_scene_data
from video_processing import analyze_movie_and_update_json_streaming,convert_to_mp4
job_cancellations = {}
active_queues = {}
app = Flask(__name__)
MOVIES = './movies'
DATA = './data'
os.makedirs(MOVIES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        'data_loaded': is_data_loaded(),
        'scene_count': len(defined_scenes) if defined_scenes else 0
    })

@app.route('/analyze_select')
def analyze_select():
    return render_template('analyze.html')

@app.route('/search_select')
def search_select():
    return render_template('search.html')

@app.route('/data_select')
def data_select():
    return render_template('data.html')

@app.route('/analyze', methods=['POST'])
def upload_files():
    video = request.files.get('video')
    subtitle = request.files.get('subtitle')
    movie_code = request.form.get('movie_code')
    job_id = request.form.get('job_id') or str(uuid.uuid4())
    if not video or not movie_code:
        return jsonify({"error": "Video and movie code required"}), 400

    # Prepare movie paths
    movie_dir = Path(MOVIES) / movie_code
    movie_dir.mkdir(parents=True, exist_ok=True)

    original_ext = Path(video.filename).suffix.lower()
    temp_video_path = movie_dir / f'temp_input{original_ext}'
    movie_video = movie_dir / 'movie.mp4'
    movie_sub = movie_dir / 'subtitles.srt'

    # Save files to disk
    video.save(temp_video_path)
    if subtitle:
        subtitle.save(movie_sub)
    else:
        movie_sub = None

    # Wait loop to confirm the file is written before FFmpeg touches it
    for _ in range(5):
        if temp_video_path.exists() and temp_video_path.stat().st_size > 0:
            break
        time.sleep(0.5)
    else:
        return jsonify({"error": "Failed to save video file properly."}), 500

    # Capture metadata from form
    meta_args = {
        'genre': ','.join(filter(None, [
            request.form.get('genre1'),
            request.form.get('genre2'),
            request.form.get('genre3')
        ])),
        'year': request.form.get('year'),
        'media_type': request.form.get('type'),
        'series': request.form.get('series'),
        'season': request.form.get('season'),
        'episode': request.form.get('episode'),
        'title': request.form.get('title'),
        'BW': request.form.get('BW', 'false').lower() == 'true',
    }
    cancel_event = threading.Event()
    job_cancellations[job_id] = cancel_event
    def generate():
        yield "event: status\ndata: Verifying video file...\n\n"

        if not is_valid_mp4(temp_video_path):
            yield f"event: error\ndata: Uploaded video file is not a valid MP4 or is corrupt: {temp_video_path}\n\n"
            return

        yield "event: status\ndata: Converting video to MP4 format...\n\n"

        
        for message in convert_to_mp4(str(temp_video_path), str(movie_video)):
            print (f"event: status\ndata: {message}\n\n")

        yield "event: status\ndata: Starting analysis...\n\n"
        print(f"Analyzing movie: {movie_video} with subtitles: {movie_sub} and metadata: {meta_args}")

        for message in analyze_movie_and_update_json_streaming(
            str(movie_video),
            str(movie_sub) if movie_sub else None,
            cancel_event=cancel_event,
            **meta_args):
            if job_cancellations[job_id].is_set():
                yield "event: status\ndata: Canceled by user.\n\n"
                return
            yield f"event: status\ndata: {message}\n\n"

        yield "event: done\ndata: FINISHED\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream', headers={"X-Job-ID": job_id})

def is_valid_mp4(filepath):
    """Check if the given video file is a valid MP4."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return result.returncode == 0
    except Exception as e:
        print("ffprobe check failed:", e)
        return False

@app.route('/cancel', methods=['POST'])
def cancel_job():
    job_id = request.form.get('job_id')
    if job_id in job_cancellations:
        job_cancellations[job_id].set()
        return jsonify({'success': True, 'message': 'Job canceled'}), 200
    else:
        return jsonify({'success': False, 'message': 'Invalid job ID'}), 400
    
@app.route('/movies/<moviecode>/<filename>')
def serve_movie(moviecode, filename):
    # Sanitize inputs to prevent path traversal
    moviecode = os.path.basename(moviecode)
    filename = os.path.basename(filename)

    file_path = os.path.join('movies', moviecode, filename)

    if not os.path.isfile(file_path):
        abort(404)

    return send_file(file_path, mimetype='video/mp4', conditional=True)

@app.route('/results/<path:filename>')
def download_file(filename):
    return send_from_directory('results', filename)

@app.route('/search', methods=['GET'])
def search():
    print("Received search request with args:", request.args)
    query = request.args.get('query', '')
    weight = float(request.args.get('weight', 0.5))
    emotion = request.args.get('emotion', 'All')
    location = request.args.get('location', 'All')
    genre = request.args.get('genre', 'All')
    year_start = int(request.args.get('year_start', '0'))
    year_end = int(request.args.get('year_end', '9999'))
    results = search_engine.search_scenes(query, weight, emotion, location, genre, year_start, year_end)
    return jsonify(results)

@app.route('/queue', methods=['POST'])
def queue_scene():
    data = request.get_json(force=True)
    scene_id = data.get('scene_id')
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    queue_id = data.get('queue_id') or str(uuid.uuid4())

    # Ensure valid times
    try:
        custom_start = float(start_time) if start_time is not None else None
        custom_end = float(end_time) if end_time is not None else None
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'Invalid start or end time format'}), 400

    # Get or initialize the queue
    queue = active_queues.setdefault(queue_id, [])

    # Add scene to queue
    success, message = search_engine.add_scene_to_queue(
        scene_id,
        custom_start=custom_start,
        custom_end=custom_end,
        queue=queue
    )

    status = 200 if success else 400
    return jsonify({'success': success, 'message': message, 'queue_id': queue_id}), status


@app.route('/queue/status', methods=['GET'])
def queue_status():
    queue_data = []
    for idx, (video_path, start, end) in enumerate(selected_queue):
        queue_data.append({
            'index': idx,
            'video_path': video_path,
            'start': start,
            'end': end,
            'duration': round(end - start, 2)
        })

    return jsonify({
        'success': True,
        'queued_scenes': queue_data,
        'count': len(queue_data)
    })

@app.route('/unqueue', methods=['POST'])
def unqueue_scene():
    data = request.json
    scene_id = data.get('scene_id')
    # Remove from queue storage here
    return jsonify({'success': True})


@app.route("/genre-distribution")
def genre_distribution():
    genre_counter = Counter()
    data_folder = DATA
    broken_files = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(data_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    #print(f"🔍 Attempting to load: {filename}")
                    movie_data = json.load(f)
                    raw_genres = movie_data.get("movie", {}).get("genre", [])

                    genres = []
                    if isinstance(raw_genres, str):
                        genres = [g.strip().title() for g in raw_genres.split(',')]
                    elif isinstance(raw_genres, list):
                        for g in raw_genres:
                            if isinstance(g, str):
                                genres.extend([x.strip().title() for x in g.split(',')])

                    genre_counter.update(genres)

            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
                broken_files.append((filename, str(e)))

    if broken_files:
        return jsonify({
            "error": "Some files failed to load.",
            "broken_files": broken_files
        }), 500

    genre_data = [{"genre": genre, "count": count} for genre, count in genre_counter.items()]
    genre_data.sort(key=lambda x: x["count"], reverse=True)
    return jsonify(genre_data)

@app.route("/maingenre-distribution")
def main_genre_distribution():
    genre_counter = Counter()
    data_folder = DATA
    broken_files = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(data_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    #print(f"🔍 Attempting to load: {filename}")
                    movie_data = json.load(f)
                    raw_genres = movie_data.get("movie", {}).get("genre", [])

                    genres = []
                    if isinstance(raw_genres, str):
                        genres = [g.strip().title() for g in raw_genres.split(',') if g.strip()]
                    elif isinstance(raw_genres, list):
                        for g in raw_genres:
                            if isinstance(g, str):
                                genres.extend([x.strip().title() for x in g.split(',') if x.strip()])

                    if genres:
                        main_genre = genres[0]
                        genre_counter[main_genre] += 1

            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
                broken_files.append((filename, str(e)))

    if broken_files:
        return jsonify({
            "error": "Some files failed to load.",
            "broken_files": broken_files
        }), 500

    genre_data = [{"genre": genre, "count": count} for genre, count in genre_counter.items()]
    genre_data.sort(key=lambda x: x["count"], reverse=True)
    return jsonify(genre_data)


@app.route("/emotion-distribution")
def emotion_distribution():
    emotion_data = []
    for filename in os.listdir(DATA):
        if filename.endswith(".json"):
            with open(os.path.join(DATA, filename), 'r') as f:
                movie_data = json.load(f)
                title = movie_data.get("movie", {}).get("title", "Unknown Title")
                scenes = movie_data.get("movie", {}).get("scenes", [])
                counter = Counter()
                for scene in scenes:
                    emotions = scene.get("emotion", {})
                    if emotions:
                        dominant = max(emotions.items(), key=lambda x: x[1])[0]
                        counter[dominant.title()] += 1
                for emotion, count in counter.items():
                    emotion_data.append({"title": title, "emotion": emotion, "count": count})
    return jsonify(emotion_data)

@app.route("/color-distribution")
def color_distribution():
    color_data = []
    for filename in os.listdir(DATA):
        if filename.endswith(".json"):
            with open(os.path.join(DATA, filename), 'r') as f:
                movie_data = json.load(f)
                title = movie_data.get("movie", {}).get("title", "Unknown Title")
                scenes = movie_data.get("movie", {}).get("scenes", [])
                for scene in scenes:
                    start_time = scene.get("start_time", 0)
                    end_time = scene.get("end_time", 0)
                    dominant_colors = scene.get("dominant_colors", [])
                    if dominant_colors:
                        color_data.append({
                            "title": title,
                            "start_time": start_time,
                            "end_time": end_time,
                            "dominant_colors": dominant_colors
                        })
    return jsonify(color_data)

@app.route('/videos/<path:filename>')
def download_video(filename):
    return send_file(f"videos/{filename}", as_attachment=True)


@app.route("/location-distribution")
def location_distribution():
    location_data = []
    for filename in os.listdir(DATA):
        if filename.endswith(".json"):
            with open(os.path.join(DATA, filename), "r") as f:
                movie_data = json.load(f)
                title = movie_data.get("movie", {}).get("title", "Unknown Title")
                counter = Counter()
                for scene in movie_data.get("movie", {}).get("scenes", []):
                    top_location = scene.get("location", [])
                    if isinstance(top_location, list) and top_location:
                        loc = top_location[0].get("scene", "").lower()
                        if loc:
                            counter[loc] += 1
                for loc, count in counter.items():
                    location_data.append({"title": title, "location": loc, "count": count})
    return jsonify(location_data)

@app.route('/stitch', methods=['POST'])
def stitch():
    data = request.get_json(force=True)
    queue_id = data.get('queue_id')
    title_text = data.get('title', "My Story")

    if not queue_id or queue_id not in active_queues:
        return jsonify({'success': False, 'message': 'Invalid or missing queue ID'}), 400

    queue = active_queues.pop(queue_id)

    success, result_path_or_error = search_engine.stitch_and_save(queue, title_text)
    
    if success:
        return jsonify({'success': True, 'video_path': result_path_or_error})
    else:
        return jsonify({'success': False, 'message': result_path_or_error}), 500
    
@app.route('/video', methods=['GET'])
def serve_video():
    video_path = 'results/stitched.mp4'
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'message': 'No stitched video found'}), 404
    return send_file(video_path, mimetype='video/mp4')

@app.route('/filters', methods=['GET'])
def get_filters():
    return jsonify({
        'emotions': ['All'] + search_engine.emotion_list,
        'locations': ['All'] + sorted(set(loc.title() for loc in search_engine.location_list)),
        'genres': ['All'] + search_engine.genre_list,
        'years': search_engine.year_list
    })

@app.route('/reload_scenes', methods=['POST'])
def reload_scenes():
    try:
        search_engine.load_scene_data('data')
        search_engine.load_dialogue_embeddings()
        return jsonify({'success': True, 'message': 'Scenes and dialogue embeddings reloaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/test_search')
def test_search():
    results = search_engine.search_scenes("hello", 0.5, "All", "All", "All", 1900, 2100)
    return jsonify(results)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)