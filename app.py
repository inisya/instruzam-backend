from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Instruments et leurs caracteristiques spectrales
INSTRUMENTS = {
    'Piano':       {'low': 0.15, 'mid': 0.40, 'high': 0.30, 'vhigh': 0.15},
    'Guitare':     {'low': 0.10, 'mid': 0.35, 'high': 0.40, 'vhigh': 0.15},
    'Violon':      {'low': 0.05, 'mid': 0.25, 'high': 0.45, 'vhigh': 0.25},
    'Basse':       {'low': 0.55, 'mid': 0.30, 'high': 0.10, 'vhigh': 0.05},
    'Batterie':    {'low': 0.40, 'mid': 0.30, 'high': 0.20, 'vhigh': 0.10},
    'Saxophone':   {'low': 0.10, 'mid': 0.50, 'high': 0.30, 'vhigh': 0.10},
    'Trompette':   {'low': 0.08, 'mid': 0.35, 'high': 0.40, 'vhigh': 0.17},
    'Flute':       {'low': 0.02, 'mid': 0.15, 'high': 0.45, 'vhigh': 0.38},
    'Violoncelle': {'low': 0.25, 'mid': 0.40, 'high': 0.25, 'vhigh': 0.10},
    'Synthetiseur':{'low': 0.20, 'mid': 0.30, 'high': 0.30, 'vhigh': 0.20},
    'Harmonica':   {'low': 0.05, 'mid': 0.45, 'high': 0.35, 'vhigh': 0.15},
    'Harpe':       {'low': 0.10, 'mid': 0.30, 'high': 0.38, 'vhigh': 0.22},
    'Accordeon':   {'low': 0.12, 'mid': 0.48, 'high': 0.28, 'vhigh': 0.12},
    'Clarinette':  {'low': 0.08, 'mid': 0.42, 'high': 0.35, 'vhigh': 0.15},
    'Trombone':    {'low': 0.20, 'mid': 0.45, 'high': 0.25, 'vhigh': 0.10},
}

EMOJIS = {
    'Piano': '🎹', 'Guitare': '🎸', 'Violon': '🎻', 'Basse': '🎸',
    'Batterie': '🥁', 'Saxophone': '🎷', 'Trompette': '🎺', 'Flute': '🎶',
    'Violoncelle': '🎻', 'Synthetiseur': '🎹', 'Harmonica': '🎵',
    'Harpe': '🎼', 'Accordeon': '🪗', 'Clarinette': '🎵', 'Trombone': '🎺',
}

FAMILLES = {
    'Piano': 'Cordes frappees', 'Guitare': 'Cordes', 'Violon': 'Cordes',
    'Basse': 'Cordes', 'Batterie': 'Percussions', 'Saxophone': 'Vents',
    'Trompette': 'Cuivres', 'Flute': 'Vents', 'Violoncelle': 'Cordes',
    'Synthetiseur': 'Electronique', 'Harmonica': 'Vents', 'Harpe': 'Cordes',
    'Accordeon': 'Vents', 'Clarinette': 'Vents', 'Trombone': 'Cuivres',
}

def analyze_spectrum(audio_bytes):
    # Convertir bytes PCM16 en tableau numpy
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    if len(samples) < 1024:
        return None
    
    # FFT
    fft_size = 4096
    chunk = samples[:fft_size] if len(samples) >= fft_size else np.pad(samples, (0, fft_size - len(samples)))
    
    # Fenetre de Hann pour reduire les fuites spectrales
    window = np.hanning(fft_size)
    spectrum = np.abs(np.fft.rfft(chunk * window))
    spectrum = spectrum / (np.max(spectrum) + 1e-10)
    
    # Energie par bande de frequence
    n = len(spectrum)
    low   = np.mean(spectrum[:int(n*0.08)])   # 0-400Hz
    mid   = np.mean(spectrum[int(n*0.08):int(n*0.30)])  # 400Hz-2kHz
    high  = np.mean(spectrum[int(n*0.30):int(n*0.65)])  # 2kHz-5kHz
    vhigh = np.mean(spectrum[int(n*0.65):])   # 5kHz+
    
    total = low + mid + high + vhigh + 1e-10
    return {
        'low':   float(low / total),
        'mid':   float(mid / total),
        'high':  float(high / total),
        'vhigh': float(vhigh / total),
    }

def match_instruments(bands):
    if bands is None:
        return []
    
    scores = {}
    for name, profile in INSTRUMENTS.items():
        # Distance euclidienne entre profil et mesure
        dist = sum((bands[k] - profile[k])**2 for k in bands)
        similarity = max(0, 1 - dist * 4)
        scores[name] = similarity
    
    # Garder uniquement les instruments avec un score significatif
    threshold = max(scores.values()) * 0.4
    results = []
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        if score >= threshold and len(results) < 4:
            pct = min(99, max(20, int(score * 100)))
            results.append({
                'name': name,
                'emoji': EMOJIS.get(name, '🎵'),
                'pct': pct,
                'famille': FAMILLES.get(name, 'Inconnu'),
                'type': 'Acoustique',
            })
    
    return results

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    # Verifier niveau sonore minimum
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    
    if rms < 500:
        return jsonify({'error': 'silence', 'message': 'Aucun son detecte'}), 200
    
    bands = analyze_spectrum(audio_bytes)
    instruments = match_instruments(bands)
    
    return jsonify({
        'status': 'success',
        'instruments': instruments,
        'bands': bands,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '1.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
