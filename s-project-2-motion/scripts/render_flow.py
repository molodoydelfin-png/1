"""Long, fluid symbol turn with exact ray proportions and the returning reveal cursor."""
import argparse
import json
import subprocess

import numpy as np

import render_clean as clean
from render_clean import ROOT, ramp
from render_sweep import JOIN, encode_preview_4k
from render_turn import TurnAnimation


PREFIX = 's-project-2-flow'
TURN_START = .18
TURN_END = 3.30
START_ANGLE = np.deg2rad(-250.)


class FlowAnimation(TurnAnimation):
    """A rigid constellation turn with the reference video's soft, long settle."""

    @staticmethod
    def angle(t):
        # A partial turn makes the changing orientation readable.  Both ends have
        # zero velocity, while the long interval keeps the middle speed restrained.
        return START_ANGLE * (1 - ramp(t, TURN_START, TURN_END - TURN_START))

    def base_growth(self, t):
        # The reference expands quickly at first and then covers the last distance
        # progressively more slowly.  A short gate removes the exponential's hard
        # initial velocity; the final blend reaches the approved geometry exactly.
        if t <= TURN_START:
            return 0.
        if t >= TURN_END:
            return 1.
        x = np.clip((t - TURN_START) / (TURN_END - TURN_START), 0, 1)
        exponential = (1 - np.exp(-2.4 * x)) / (1 - np.exp(-2.4))
        gate = ramp(t, TURN_START, .32)
        settle = ramp(t, 2.75, TURN_END - 2.75)
        return float(exponential * gate * (1 - settle) + settle)

    def scales(self, t):
        # Every spoke uses the same multiplier, so the original asymmetry and all
        # relative lengths stay intact instead of producing independent wobbles.
        return np.full(10, self.base_growth(t), dtype=np.float64)

    def node_growth(self, t, growth):
        # Heads reach their source size together and never inflate during the turn.
        head = ramp(t, TURN_START, .90)
        return np.full(10, head, dtype=np.float64)

    def node_positions(self, t, growth):
        vectors = (self.endpoints - self.center) * growth[:, None]
        if t >= TURN_END:
            return self.center + vectors
        theta = self.angle(t)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        return self.center + vectors @ rotation.T


def render(four_k=False):
    prefix = PREFIX
    if four_k:
        clean.K = 3840 / 1448
        clean.OUTPUT_SIZE = (3840, 2160)
        clean.OUTPUT_SCALE = 2
        clean.STAR_SSAA = 4
        prefix += '-4k'

    animation = FlowAnimation(ROOT / 'assets/source.png')
    frames = ROOT / f'.cache/{prefix}-frames'
    frames.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'exports'
    for n in range(150):
        animation.frame(n / 30).save(frames / f'{n:04d}.png', compress_level=2)
        if n % 30 == 0:
            print(f'Fluid reference-paced turn {n // 30}/5 seconds', flush=True)
    animation.final.save(out / f'{prefix}-final.png')

    audio = ROOT / 'assets/sound-design.wav'
    if four_k:
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '30',
            '-i', str(frames / '%04d.png'), '-i', str(audio), '-map', '0:v', '-map', '1:a',
            '-frames:v', '150', '-t', '5', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
            '-auto-alt-ref', '0', '-b:v', '0', '-crf', '0', '-lossless', '1', '-row-mt', '1',
            '-cpu-used', '4', '-threads', '2', '-c:a', 'libopus', '-b:a', '128k',
            str(out / f'{prefix}-alpha.webm')
        ], check=True)
        encode_preview_4k(frames, out / f'{prefix}-preview.mp4', audio)
    else:
        clean.encode(frames, out, audio, prefix=prefix, preview_crf=8,
                     alpha_crf=0, prores_q=6, alpha_lossless=True)

    info = {
        'duration': 5, 'fps': 30, 'frames': 150, 'size': list(clean.OUTPUT_SIZE),
        'motion_reference': 'long low-jerk radial expansion and ease-out timing only',
        'symbol_turn_degrees': 250, 'turn_start': TURN_START, 'turn_end': TURN_END,
        'relative_ray_lengths_preserved_during_motion': True,
        'independent_ray_oscillation': False, 'node_size_pulsation': False,
        'rotates_text': False, 'cursor_rejoined_at': JOIN, 'final_hold_start': 3.7,
        'matches_approved_final': True, 'extra_fragments': 0,
        'quality': 'analytic geometry, area antialiasing, lossless transparent master'
    }
    (out / f'{prefix}-render-info.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--4k', action='store_true', dest='four_k')
    render(parser.parse_args().four_k)
