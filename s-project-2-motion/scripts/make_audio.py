"""Original quiet five-second logo sound design; no third-party samples."""
from pathlib import Path
import wave
import numpy as np

RATE = 48000
DURATION = 5
ROOT = Path(__file__).resolve().parent.parent
EVENTS = [0.56 + i * 0.106 for i in range(10)]


def make_audio(destination):
    audio = np.zeros((RATE * DURATION, 2), dtype=np.float64)
    rng = np.random.default_rng(20261023)

    def add(signal, onset, pan=0):
        start = round(onset * RATE)
        size = min(len(signal), len(audio) - start)
        # Equal-power, restrained stereo spread.
        angle = (pan + 1) * np.pi / 4
        audio[start:start + size, 0] += signal[:size] * np.cos(angle)
        audio[start:start + size, 1] += signal[:size] * np.sin(angle)

    # Soft tactile pulse; no sharp attack and no sustained bass.
    t = np.arange(round(.66 * RATE)) / RATE
    envelope = (1 - np.exp(-t / .014)) * np.exp(-t / .13)
    pulse = .085 * envelope * (np.sin(2 * np.pi * 58 * t) + .18 * np.sin(2 * np.pi * 116 * t))
    pulse *= np.minimum(1, (.66 - t) / .025)
    add(pulse, .09)

    # Ten gentle, low-level connection ticks synced to the staggered reveal.
    for i, onset in enumerate(EVENTS):
        t = np.arange(round(.052 * RATE)) / RATE
        noise = rng.normal(0, 1, len(t))
        smooth = np.convolve(noise, np.ones(7) / 7, mode='same')
        envelope = (1 - np.exp(-t / .0018)) * np.exp(-t / .009)
        signal = .018 * envelope * (smooth + .3 * np.sin(2 * np.pi * (680 + i * 23) * t))
        add(signal, onset, pan=(i % 3 - 1) * .16)

    # Barely audible resolving tone, fading out before the final static hold ends.
    t = np.arange(round(.94 * RATE)) / RATE
    envelope = (1 - np.exp(-t / .055)) * np.exp(-t / .21)
    envelope *= np.minimum(1, (.94 - t) / .09)
    tone = .012 * envelope * (np.sin(2 * np.pi * 440 * t) + .24 * np.sin(2 * np.pi * 660 * t))
    add(tone, 3.55)

    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) < 1, 'Clipping'
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(audio * 32767).astype('<i2')
    with wave.open(str(destination), 'wb') as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(pcm.tobytes())
    print(f'{destination}: {DURATION}s, {RATE} Hz stereo, peak {20*np.log10(np.max(np.abs(audio))):.1f} dBFS')


if __name__ == '__main__':
    make_audio(ROOT / 'assets' / 'sound-design.wav')
