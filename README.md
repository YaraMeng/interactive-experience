# 30-second Mood-based Music Generator

This repo contains two scripts that synthesize 30-second mono WAVs matching a given mood using only Python standard libraries (no external deps):

- `test1.py` (Chinese prompts)
- `Emotional musicalization.py` (English prompts and comments)

## Usage

Run interactively (asks for mood):

```powershell
python .\Emotional musicalization.py
```

Specify mood and use default 30s:

```powershell
python .\Emotional musicalization.py --mood happy
```

Specify mood and custom duration, save only:

```powershell
python .\Emotional musicalization.py --mood sad --duration 30 --no-play
```

Specify output filename:

```powershell
python .\Emotional musicalization.py --mood calm --outfile my_calm_music.wav
```

Tip: On Windows the script tries to play the WAV via `winsound`. Use `--no-play` to disable playback.

## Supported moods (high-level feel)

- Positive/neutral:
	- happy: major, faster BPM, triangle; I–V–vi–IV; lively
	- calm: pentatonic major, slow, sine; sparse melody
	- energetic: minor, very fast, saw; dense, punchy
	- romantic: major, mid-slow, saw with vibrato; tender
	- mysterious: harmonic minor, mid-slow, triangular; suspense

- Negative (new):
	- sad: natural minor, slow, sine; soft envelope
	- anxious: phrygian, fast, saw; tense, restless
	- depressed: minor, very slow, sine; heavy tail
	- lonely: dorian, slow-mid, triangle; wistful
	- fearful: phrygian, slow-mid, square; uneasy with vibrato
	- tense: harmonic minor, fast, saw; pressure
	- gloomy: minor, slow-mid, triangle; somber
	- bored: pentatonic major, slow, sine; low density and repetition
	- frustrated: minor, fast, square; edgy

## Implementation notes

- 44.1 kHz, 16-bit PCM, mono.
- Oscillators: sine/square/saw/triangle with ADSR envelope and simple delay.
- Composition: 4 beats per bar, chord pad + probabilistic scale-based melody + simple bass.

Ideas to extend:
- Custom chord progressions and rhythms per mood.
- Add percussive noise layers.
- Stereo output and panning.
