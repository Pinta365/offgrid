# Audio test assets

Speech and tone inputs for audio/ASR model tests.

| File | Format | Rate | Channels | Duration | Size (bytes) |
|---|---|---|---|---|---|
| `jfk.wav` | WAV / PCM s16le | 16000 Hz | 1 (mono) | 11.000 s | 352,078 |
| `jfk.mp3` | MP3 | 16000 Hz | 1 (mono) | 11.000 s | 64,016 |
| `jfk.flac` | FLAC | 16000 Hz | 1 (mono) | 11.000 s | 203,353 |
| `tone.wav` | WAV / PCM s16le | 16000 Hz | 1 (mono) | 2.000 s | 64,044 |

## jfk.wav

Public-domain sample from whisper.cpp (`samples/jfk.wav`), JFK's 1961 inaugural
address. Transcript:

> And so my fellow Americans, ask not what your country can do for you, ask what
> you can do for your country.

Source: https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav

`jfk.mp3` and `jfk.flac` are transcodes of `jfk.wav` made with ffmpeg.

## tone.wav

Generated with Python's standard library (`wave` + `struct` + `math`): a 440 Hz
sine wave, 2 seconds, 16 kHz mono, 16-bit PCM, amplitude 0.3.
