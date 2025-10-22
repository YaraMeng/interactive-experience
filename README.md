# 30-second Mood-based Music Generator


This repo contains a scripts that synthesize 30-second mono WAVs matching a given mood using only Python standard libraries (no external deps):

- `Emotional musicalization.py` (main script, English prompts and comments, supports batch/all moods)

## Usage


## Usage

**Run interactively (asks for mood):**

```powershell
python .\Emotional musicalization.py
```

**Specify mood and use default 30s:**

```powershell
python .\Emotional musicalization.py --mood happy
```

**Specify mood and custom duration, save only:**

```powershell
python .\Emotional musicalization.py --mood sad --duration 30 --no-play
```

**Specify output filename:**

```powershell
python .\Emotional musicalization.py --mood calm --outfile my_calm_music.wav
```

**Batch generate all moods (one-click, 30s each, save to `music folder`):**

```powershell
python .\Emotional musicalization.py --all-moods --duration 30 --no-play
```

**Mario style generator (asks for mood, Mario game style):**

```powershell
python .\mario_musicalization.py
```

> Tip: On Windows the script tries to play the WAV via `winsound`. Use `--no-play` to disable playback.


## Supported moods (all available moods)

| Mood (English) | 中文 | Scale/Feel | Description |
|---|---|---|---|
| happy | 快乐 | major | lively, bright, I–V–vi–IV |
| calm | 平静 | pentatonic major | slow, sparse, gentle |
| energetic | 充满活力 | minor | very fast, punchy, dense |
| romantic | 浪漫 | major | mid-slow, vibrato, tender |
| mysterious | 神秘 | harmonic minor | suspense, triangular |
| sad | 悲伤 | natural minor | slow, soft, gentle tail |
| anxious | 焦虑 | phrygian | fast, tense, restless |
| depressed | 沮丧 | minor | very slow, heavy, deep tail |
| lonely | 孤独 | dorian | slow-mid, wistful, triangle |
| fearful | 恐惧 | phrygian | slow-mid, uneasy, vibrato |
| tense | 紧张 | harmonic minor | fast, pressure, saw |
| gloomy | 忧郁 | minor | slow-mid, somber, triangle |
| bored | 无聊 | pentatonic major | slow, repetitive, low density |
| frustrated | 挫败 | minor | fast, edgy, square |
| angry | 愤怒 | phrygian | fast, aggressive, square |

> All moods above are supported in both `Emotional musicalization.py` and `mario_musicalization.py` (Mario style has its own mapping).


## Features & Implementation notes

- 44.1 kHz, 16-bit PCM, mono output.
- All standard library, no external dependencies.
- Oscillators: sine, square, saw, triangle with ADSR envelope and simple delay.
- Chord pad, probabilistic scale-based melody, simple bass, and dynamic drum patterns.
- Output always saved to `music folder` (auto-created).
- Batch mode: one-click generate all moods with `--all-moods`.
- Mario style generator: Mario game-inspired scales, chords, rhythm, and drums.

### Ideas to extend
- Custom chord progressions and rhythms per mood.
- Add percussive noise layers.
- Stereo output and panning.
