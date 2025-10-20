import argparse
import math
import os
import random
import struct
import sys
import time
import wave
from dataclasses import dataclass
from typing import List, Tuple


# --------------------------
# Basic audio/music utilities
# --------------------------

SAMPLE_RATE = 44100
MASTER_VOL = 0.9  # Master volume (before normalization)



def midi_to_freq(midi: int) -> float:
	"""Convert MIDI note number to frequency (A4=440Hz)."""
	return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def clamp(v: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, v))



def adsr_envelope(n_samples: int, sr: int, attack: float, decay: float, sustain_level: float, release: float) -> List[float]:
	"""Generate an ADSR envelope in seconds for given sample count and sample rate."""
	attack_samples = int(sr * max(0.0, attack))
	decay_samples = int(sr * max(0.0, decay))
	release_samples = int(sr * max(0.0, release))

	sustain_samples = max(0, n_samples - (attack_samples + decay_samples + release_samples))
	env: List[float] = []

	# Attack: 0 -> 1
	for i in range(attack_samples):
		env.append(i / max(1, attack_samples))

	# Decay: 1 -> sustain_level
	for i in range(decay_samples):
		if decay_samples == 0:
			env.append(sustain_level)
		else:
			env.append(1.0 + (sustain_level - 1.0) * (i / decay_samples))

	# Sustain: sustain_level
	for _ in range(sustain_samples):
		env.append(sustain_level)

	# Release: sustain_level -> 0
	for i in range(release_samples):
		if release_samples == 0:
			env.append(0.0)
		else:
			env.append(sustain_level * (1.0 - i / release_samples))

	# Align to n_samples length
	if len(env) < n_samples:
		env += [0.0] * (n_samples - len(env))
	elif len(env) > n_samples:
		env = env[:n_samples]
	return env



def osc_sample(phase: float, wave_type: str) -> float:
	"""Generate one sample for given waveform. phase in [0, 1)."""
	wave_type = wave_type.lower()
	x = phase - math.floor(phase)
	if wave_type in ("sine", "sin"):
		return math.sin(2 * math.pi * x)
	if wave_type in ("square", "sqr"):
		return 1.0 if x < 0.5 else -1.0
	if wave_type in ("saw", "sawtooth"):
		return 2.0 * (x - 0.5)
	if wave_type in ("triangle", "tri"):
		return 4.0 * abs(x - 0.5) - 1.0
	# Default to sine
	return math.sin(2 * math.pi * x)



def synth_note(freq: float, duration_sec: float, wave_type: str, 
			   attack: float, decay: float, sustain: float, release: float, 
			   volume: float = 1.0, vibrato_hz: float = 0.0, vibrato_depth: float = 0.0) -> List[float]:
	"""Synthesize a single note and return float samples in [-1, 1]."""
	n = max(1, int(SAMPLE_RATE * duration_sec))
	env = adsr_envelope(n, SAMPLE_RATE, attack, decay, sustain, release)
	out: List[float] = [0.0] * n

	phase = 0.0
	phase_inc = freq / SAMPLE_RATE

	for i in range(n):
		# Simple vibrato
		f = freq
		if vibrato_hz > 0.0 and vibrato_depth > 0.0:
			f *= 1.0 + vibrato_depth * math.sin(2 * math.pi * vibrato_hz * (i / SAMPLE_RATE))
			phase_inc = f / SAMPLE_RATE

		s = osc_sample(phase, wave_type) * env[i] * volume
		out[i] = s
		phase += phase_inc
	return out



def mix_inplace(target: List[float], source: List[float], start: int, gain: float = 1.0) -> None:
	"""Mix source into target starting at sample index 'start'. Extends target when needed."""
	if start < 0:
		# Clip negative start
		source = source[-start:]
		start = 0
	end = start + len(source)
	if end > len(target):
		target.extend([0.0] * (end - len(target)))
	for i, s in enumerate(source):
		target[start + i] += s * gain


def normalize_buffer(buf: List[float], peak: float = 0.95) -> List[float]:
	if not buf:
		return buf
	m = max(abs(x) for x in buf)
	if m == 0:
		return buf
	scale = peak / m
	return [x * scale for x in buf]


def write_wav_mono(path: str, samples: List[float], sample_rate: int = SAMPLE_RATE) -> None:
	"""Write 16-bit PCM mono WAV file."""
	samples_i16 = [int(clamp(x, -1.0, 1.0) * 32767) for x in samples]
	with wave.open(path, 'wb') as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)  # 16-bit
		wf.setframerate(sample_rate)
		wf.writeframes(b''.join(struct.pack('<h', s) for s in samples_i16))


# --------------------------
# Scale and chord utilities
# --------------------------

SCALE_INTERVALS = {
	"major": [0, 2, 4, 5, 7, 9, 11],            # Major scale
	"minor": [0, 2, 3, 5, 7, 8, 10],            # Natural minor
	"harmonic_minor": [0, 2, 3, 5, 7, 8, 11],   # Harmonic minor
	"phrygian": [0, 1, 3, 5, 7, 8, 10],         # Phrygian mode
	"dorian": [0, 2, 3, 5, 7, 9, 10],           # Dorian mode
	"pentatonic_major": [0, 2, 4, 7, 9],
	"pentatonic_minor": [0, 3, 5, 7, 10],
}


def build_scale(root_midi: int, scale_name: str, octaves: int = 2) -> List[int]:
	"""Build multiple-octave scale notes for melody generation."""
	intervals = SCALE_INTERVALS[scale_name]
	out: List[int] = []
	for octv in range(octaves):
		base = root_midi + octv * 12
		out.extend(base + i for i in intervals)
	return out


def chord_from_degree(root_midi: int, scale_name: str, degree: int) -> Tuple[int, int, int]:
	"""Build a triad (root, third, fifth) in MIDI by scale degree (0-based)."""
	intervals = SCALE_INTERVALS[scale_name]
	if len(intervals) < 7:
		# For pentatonics, use simplified power chord (root + perfect fifth + octave)
		root = root_midi + intervals[degree % len(intervals)]
		fifth = root + 7
		octave = root + 12
		return (root, fifth, octave)
	deg = degree % 7
	# Take 1, 3, 5 degrees (wrapping)
	degs = [deg, (deg + 2) % 7, (deg + 4) % 7]
	notes = [root_midi + intervals[d] for d in degs]
	return (notes[0], notes[1], notes[2])


# --------------------------
# Mood configuration
# --------------------------

@dataclass
class MoodConfig:
	name: str
	scale: str
	bpm: int
	root_midi: int
	waveform: str
	melody_density: float  # 0.0~1.0，越大旋律越密
	chord_progression: List[int]  # 0基度数序列
	attack: float
	decay: float
	sustain: float
	release: float
	vibrato_hz: float = 0.0
	vibrato_depth: float = 0.0


def get_mood_config(mood: str) -> MoodConfig:
	m = (mood or "").strip().lower()
	# Synonym mapping (English only)
	aliases = {
		"happy": ["happy", "joy", "cheerful", "glad", "delighted"],
		"sad": ["sad", "down", "blue", "unhappy", "melancholy"],
		"calm": ["calm", "relax", "peace", "peaceful", "serene"],
		"energetic": ["energetic", "energy", "excited", "lively", "upbeat"],
		"angry": ["angry", "mad", "rage", "furious", "irate"],
		"romantic": ["romantic", "love", "tender", "affectionate", "lovely"],
		"mysterious": ["mysterious", "mystery", "dark", "enigmatic", "eerie"],
		# Newly added negative moods
		"anxious": ["anxious", "anxiety", "nervous", "restless", "uneasy"],
		"depressed": ["depressed", "depression", "despair", "hopeless", "downcast"],
		"lonely": ["lonely", "lonesome", "isolated", "solitary", "alone"],
		"fearful": ["fearful", "afraid", "scared", "terrified", "frightened"],
		"tense": ["tense", "stress", "stressed", "tight", "strained"],
		"gloomy": ["gloomy", "bleak", "somber", "dismal", "murky"],
		"bored": ["bored", "boring", "apathetic", "indifferent", "listless"],
		"frustrated": ["frustrated", "irritated", "annoyed", "upset", "resentful"],
	}
	key = None
	for k, words in aliases.items():
		if m in words:
			key = k
			break
	if key is None:
		# Fallback: substring contains
		for k, words in aliases.items():
			if any(w in m for w in words):
				key = k
				break
	if key is None:
		key = "happy"

	if key == "happy":
		return MoodConfig(
			name="happy", scale="major", bpm=random.randint(120, 138), root_midi=60, waveform="triangle",
			melody_density=0.6, chord_progression=[0, 4, 5, 3],  # I V vi IV (0-based: I=0, V=4, vi=5, IV=3)
			attack=0.01, decay=0.08, sustain=0.8, release=0.12, vibrato_hz=5.0, vibrato_depth=0.003
		)
	if key == "sad":
		return MoodConfig(
			name="sad", scale="minor", bpm=random.randint(72, 88), root_midi=57, waveform="sine",
			melody_density=0.35, chord_progression=[0, 5, 2, 6],  # i VI III VII (relative minor)
			attack=0.02, decay=0.15, sustain=0.7, release=0.4, vibrato_hz=5.5, vibrato_depth=0.004
		)
	if key == "calm":
		return MoodConfig(
			name="calm", scale="pentatonic_major", bpm=random.randint(60, 74), root_midi=60, waveform="sine",
			melody_density=0.25, chord_progression=[0, 3, 1, 4],
			attack=0.03, decay=0.2, sustain=0.85, release=0.6, vibrato_hz=5.0, vibrato_depth=0.002
		)
	if key == "energetic":
		return MoodConfig(
			name="energetic", scale="minor", bpm=random.randint(144, 160), root_midi=57, waveform="saw",
			melody_density=0.8, chord_progression=[0, 3, 4, 3],
			attack=0.005, decay=0.05, sustain=0.7, release=0.08, vibrato_hz=6.0, vibrato_depth=0.002
		)
	if key == "angry":
		return MoodConfig(
			name="angry", scale="phrygian", bpm=random.randint(140, 158), root_midi=52, waveform="square",
			melody_density=0.7, chord_progression=[0, 1, 0, 4],
			attack=0.003, decay=0.06, sustain=0.6, release=0.06
		)
	if key == "romantic":
		return MoodConfig(
			name="romantic", scale="major", bpm=random.randint(84, 98), root_midi=60, waveform="saw",
			melody_density=0.45, chord_progression=[0, 3, 4, 0],
			attack=0.015, decay=0.12, sustain=0.8, release=0.35, vibrato_hz=5.2, vibrato_depth=0.004
		)
	if key == "mysterious":
		return MoodConfig(
			name="mysterious", scale="harmonic_minor", bpm=random.randint(76, 90), root_midi=57, waveform="triangle",
			melody_density=0.4, chord_progression=[0, 6, 4, 5],
			attack=0.015, decay=0.1, sustain=0.75, release=0.25
		)

	# Newly added negative moods
	if key == "anxious":
		return MoodConfig(
			name="anxious", scale="phrygian", bpm=random.randint(148, 168), root_midi=55, waveform="saw",
			melody_density=0.75, chord_progression=[0, 1, 2, 1],
			attack=0.004, decay=0.06, sustain=0.65, release=0.07, vibrato_hz=6.5, vibrato_depth=0.003
		)
	if key == "depressed":
		return MoodConfig(
			name="depressed", scale="minor", bpm=random.randint(60, 72), root_midi=57, waveform="sine",
			melody_density=0.25, chord_progression=[0, 5, 2, 6],
			attack=0.02, decay=0.18, sustain=0.7, release=0.5, vibrato_hz=5.2, vibrato_depth=0.004
		)
	if key == "lonely":
		return MoodConfig(
			name="lonely", scale="dorian", bpm=random.randint(68, 80), root_midi=57, waveform="triangle",
			melody_density=0.3, chord_progression=[0, 3, 4, 3],
			attack=0.018, decay=0.14, sustain=0.78, release=0.4, vibrato_hz=5.0, vibrato_depth=0.003
		)
	if key == "fearful":
		return MoodConfig(
			name="fearful", scale="phrygian", bpm=random.randint(76, 92), root_midi=52, waveform="square",
			melody_density=0.4, chord_progression=[0, 1, 6, 1],
			attack=0.008, decay=0.12, sustain=0.7, release=0.25, vibrato_hz=6.8, vibrato_depth=0.005
		)
	if key == "tense":
		return MoodConfig(
			name="tense", scale="harmonic_minor", bpm=random.randint(128, 144), root_midi=55, waveform="saw",
			melody_density=0.7, chord_progression=[0, 6, 5, 1],
			attack=0.006, decay=0.08, sustain=0.65, release=0.1, vibrato_hz=6.0, vibrato_depth=0.003
		)
	if key == "gloomy":
		return MoodConfig(
			name="gloomy", scale="minor", bpm=random.randint(72, 84), root_midi=55, waveform="triangle",
			melody_density=0.35, chord_progression=[0, 6, 4, 5],
			attack=0.015, decay=0.12, sustain=0.75, release=0.35
		)
	if key == "bored":
		return MoodConfig(
			name="bored", scale="pentatonic_major", bpm=random.randint(60, 70), root_midi=60, waveform="sine",
			melody_density=0.15, chord_progression=[0, 0, 3, 0],
			attack=0.02, decay=0.1, sustain=0.8, release=0.45
		)
	if key == "frustrated":
		return MoodConfig(
			name="frustrated", scale="minor", bpm=random.randint(120, 140), root_midi=55, waveform="square",
			melody_density=0.65, chord_progression=[0, 4, 0, 3],
			attack=0.004, decay=0.07, sustain=0.65, release=0.08
		)

	# 兜底
	return MoodConfig(
		name="happy", scale="major", bpm=128, root_midi=60, waveform="triangle",
		melody_density=0.6, chord_progression=[0, 4, 5, 3],
		attack=0.01, decay=0.08, sustain=0.8, release=0.12
	)


# --------------------------
# Composition and rendering
# --------------------------

def compose_and_render(cfg: MoodConfig, duration_sec: float = 30.0) -> List[float]:
	"""Render mono audio samples for the given mood config and duration."""
	spb = 60.0 / cfg.bpm  # seconds per beat
	beats_per_bar = 4
	bar_sec = spb * beats_per_bar
	total_bars = max(1, int(math.ceil(duration_sec / bar_sec)))

	# Melody scale (two octaves)
	scale_notes = build_scale(cfg.root_midi + 12, cfg.scale, octaves=2)

	# Output buffer
	out: List[float] = []

	# Simple delay (echo) parameters to add space
	delay_sec = 60.0 / cfg.bpm  # one beat delay
	delay_samples = int(SAMPLE_RATE * delay_sec)
	delay_gain = 0.22

	# Melody timing: 2 steps per beat (eighth notes)
	steps_per_beat = 2
	step_sec = spb / steps_per_beat

	# Bass root an octave below
	bass_root = cfg.root_midi - 12

	# Main loop: per-bar generation
	for bar_idx in range(total_bars):
		degree = cfg.chord_progression[bar_idx % len(cfg.chord_progression)]
		chord_root, chord_third, chord_fifth = chord_from_degree(cfg.root_midi, cfg.scale, degree)

		# Pad chords: one chord per bar, sustained
		chord_freqs = [midi_to_freq(chord_root), midi_to_freq(chord_third), midi_to_freq(chord_fifth)]
		chord_note = synth_note(
			freq=chord_freqs[0], duration_sec=bar_sec, wave_type=cfg.waveform,
			attack=cfg.attack * 1.5, decay=cfg.decay * 1.2, sustain=cfg.sustain, release=cfg.release * 1.5,
			volume=0.25, vibrato_hz=cfg.vibrato_hz * 0.6, vibrato_depth=cfg.vibrato_depth * 0.5,
		)
		# Layer 3rd and 5th with lower volume
		chord_third_note = synth_note(chord_freqs[1], bar_sec, cfg.waveform, cfg.attack, cfg.decay, cfg.sustain, cfg.release, volume=0.17)
		chord_fifth_note = synth_note(chord_freqs[2], bar_sec, cfg.waveform, cfg.attack, cfg.decay, cfg.sustain, cfg.release, volume=0.17)

		bar_start = int(len(out))
		mix_inplace(out, chord_note, bar_start)
		mix_inplace(out, chord_third_note, bar_start)
		mix_inplace(out, chord_fifth_note, bar_start)

		# Bass: root at bar start, one note every half bar
		bass_freq = midi_to_freq(bass_root + (chord_root - cfg.root_midi))
		bass_note_len = bar_sec / 2
		for k in range(2):
			bass = synth_note(
				freq=bass_freq, duration_sec=bass_note_len, wave_type="triangle",
				attack=0.005, decay=0.06, sustain=0.7, release=0.08, volume=0.22
			)
			mix_inplace(out, bass, bar_start + int(k * bass_note_len * SAMPLE_RATE))

		# Melody: random walk around chord tones
		steps_per_bar = beats_per_bar * steps_per_beat
		for step in range(steps_per_bar):
			t0 = bar_start + int(step * step_sec * SAMPLE_RATE)
			place_prob = cfg.melody_density
			if random.random() < place_prob:
				# Prefer chord tones (root/third/fifth) near current register
				chord_tones = [chord_root + 12, chord_third + 12, chord_fifth + 12]
				prefer = random.choice(chord_tones)
				# Choose the closest scale note to 'prefer'
				candidate = min(scale_notes, key=lambda n: abs(n - prefer) + random.random() * 0.5)
				freq = midi_to_freq(candidate)
				# Duration: mostly 1/8, occasionally tie to 1/4
				dur = step_sec if random.random() < 0.75 else step_sec * 2
				dur = min(dur, bar_sec - step * step_sec)  # do not exceed bar
				mel = synth_note(
					freq=freq, duration_sec=dur, wave_type=cfg.waveform,
					attack=cfg.attack, decay=cfg.decay, sustain=cfg.sustain, release=cfg.release,
					volume=0.28, vibrato_hz=cfg.vibrato_hz, vibrato_depth=cfg.vibrato_depth
				)
				mix_inplace(out, mel, t0)

		# Delay: copy current bar to delay position with reduced gain
		if delay_samples > 0:
			bar_end = bar_start + int(bar_sec * SAMPLE_RATE)
			echo_segment = out[bar_start:bar_end]
			mix_inplace(out, echo_segment, bar_start + delay_samples, delay_gain)

	# Trim/pad to target duration
	target_len = int(duration_sec * SAMPLE_RATE)
	if len(out) < target_len:
		out.extend([0.0] * (target_len - len(out)))
	elif len(out) > target_len:
		out = out[:target_len]

	# Normalize and apply master volume
	out = normalize_buffer(out, peak=MASTER_VOL)
	return out


def save_and_maybe_play(samples: List[float], outfile: str, play: bool = True) -> str:
	write_wav_mono(outfile, samples, SAMPLE_RATE)
	print(f"Saved: {outfile}")
	if play:
		try:
			if sys.platform.startswith("win"):
				import winsound
				winsound.PlaySound(outfile, winsound.SND_FILENAME)
			else:
				print("Non-Windows platform detected, skipping playback.")
		except Exception as e:
			print(f"Playback failed: {e}")
	return outfile


def parse_args(argv: List[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Generate 30s mood-based music (WAV).")
	p.add_argument("--mood", type=str, default=None, help="Mood: happy/sad/calm/energetic/angry/romantic/mysterious")
	p.add_argument("--duration", type=float, default=30.0, help="Duration in seconds (default 30)")
	p.add_argument("--outfile", type=str, default=None, help="Output file path (.wav)")
	p.add_argument("--no-play", action="store_true", help="Save only, do not play")
	return p.parse_args(argv)


def main(argv: List[str]):
	args = parse_args(argv)

	if args.mood is None:
		# Interactive input
		try:
			user_mood = input(
				"Please enter your current mood (happy/sad/calm/energetic/angry/romantic/mysterious/anxious/depressed/lonely/fearful/tense/gloomy/bored/frustrated): "
			).strip()
		except EOFError:
			user_mood = "happy"
	else:
		user_mood = args.mood

	cfg = get_mood_config(user_mood)
	duration = clamp(args.duration if args.duration else 30.0, 5.0, 180.0)

	print(f"Mood: {cfg.name} | Scale: {cfg.scale} | BPM: {cfg.bpm} | Duration: {duration}s")

	samples = compose_and_render(cfg, duration)

	# Output path
	out_dir = os.getcwd()
	if args.outfile:
		outfile = args.outfile
		if not outfile.lower().endswith(".wav"):
			outfile += ".wav"
	else:
		ts = time.strftime("%Y%m%d_%H%M%S")
		outfile = os.path.join(out_dir, f"output_{cfg.name}_{ts}.wav")

	save_and_maybe_play(samples, outfile, play=not args.no_play)


if __name__ == "__main__":
	main(sys.argv[1:])

