"""Fluid S-PROJECT turn with a real perspective rotation around the horizontal X axis."""
import argparse
import json
import subprocess

import numpy as np
from PIL import Image

import render_clean as clean
from render_clean import ROOT, ramp
from render_flow import FlowAnimation, TURN_END
from render_sweep import JOIN


PREFIX = 's-project-2-xaxis'
TILT_START = .22
TILT_END = 3.18
START_TILT = np.deg2rad(-62.)


class XAxisAnimation(FlowAnimation):
    """Keeps the approved planar flow and adds a restrained 3D X-axis unfold."""

    @staticmethod
    def x_tilt(t):
        return START_TILT * (1 - ramp(t, TILT_START, TILT_END - TILT_START))

    def node_positions(self, t, growth):
        vectors = (self.endpoints - self.center) * growth[:, None]
        if t >= TURN_END:
            return self.center + vectors

        # First retain the approved in-plane turn, then rotate the resulting plane
        # around X and project it through a mild perspective camera.
        theta = self.angle(t)
        cs, sn = np.cos(theta), np.sin(theta)
        rotated_x = vectors[:, 0] * cs - vectors[:, 1] * sn
        rotated_y = vectors[:, 0] * sn + vectors[:, 1] * cs

        tilt = self.x_tilt(t)
        depth = rotated_y * np.sin(tilt)
        tilted_y = rotated_y * np.cos(tilt)
        focal = 1150 * clean.K
        perspective = focal / (focal - depth)
        projected = np.column_stack((rotated_x * perspective,
                                     tilted_y * perspective))
        return self.center + projected


def encode_4k(frames, out, audio, prefix):
    # Lossless browser-friendly alpha master.
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '30',
        '-i', str(frames / '%04d.png'), '-i', str(audio), '-map', '0:v', '-map', '1:a',
        '-frames:v', '150', '-t', '5', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
        '-auto-alt-ref', '0', '-b:v', '0', '-crf', '0', '-lossless', '1', '-row-mt', '1',
        '-cpu-used', '4', '-threads', '2', '-c:a', 'libopus', '-b:a', '160k',
        str(out / f'{prefix}-alpha.webm')
    ], check=True)

    # 4:4:4, 10-bit RGB and 16-bit alpha for finishing/compositing.
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '30',
        '-i', str(frames / '%04d.png'), '-i', str(audio), '-map', '0:v', '-map', '1:a',
        '-frames:v', '150', '-t', '5', '-c:v', 'prores_ks', '-profile:v', '4',
        '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16', '-qscale:v', '6',
        '-vendor', 'apl0', '-c:a', 'pcm_s16le', str(out / f'{prefix}-alpha.mov')
    ], check=True)

    # Extra-clean opaque preview; the dark field exists only for viewing.
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
        'color=c=0x14171c:s=3840x2160:r=30:d=5', '-framerate', '30',
        '-i', str(frames / '%04d.png'), '-i', str(audio), '-filter_complex',
        '[0:v][1:v]overlay=shortest=1:format=auto,format=yuv420p[v]', '-map', '[v]',
        '-map', '2:a', '-frames:v', '150', '-t', '5', '-c:v', 'libx264', '-crf', '4',
        '-preset', 'slow', '-threads', '2', '-c:a', 'aac', '-b:a', '256k',
        '-movflags', '+faststart', str(out / f'{prefix}-preview.mp4')
    ], check=True)


def render(four_k=False):
    prefix = PREFIX
    if four_k:
        clean.K = 3840 / 1448
        clean.OUTPUT_SIZE = (3840, 2160)
        clean.OUTPUT_SCALE = 2
        # Keep the approved wordmark raster exactly unchanged in the hold frame;
        # extra sampling is concentrated on the moving, foreshortened star.
        clean.SSAA = 4
        clean.STAR_SSAA = 6
        prefix += '-4k'
    else:
        clean.SSAA = 4
        clean.STAR_SSAA = 10

    animation = XAxisAnimation(ROOT / 'assets/source.png')
    # Increased star sampling is useful in motion, but the hold frame must remain
    # byte-for-byte identical to the already approved logo render.
    approved_name = 's-project-2-clean-4k-final.png' if four_k else 's-project-2-clean-final.png'
    animation.final = Image.open(ROOT / 'exports' / approved_name).convert('RGBA')
    approved = np.asarray(animation.final, dtype=np.float32) / 255
    animation.final_pm = approved.copy()
    animation.final_pm[:, :, :3] *= approved[:, :, 3:4]
    frames = ROOT / f'.cache/{prefix}-frames'
    frames.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'exports'
    for n in range(150):
        animation.frame(n / 30).save(frames / f'{n:04d}.png', compress_level=2)
        if n % 30 == 0:
            print(f'Perspective X-axis turn {n // 30}/5 seconds', flush=True)
    animation.final.save(out / f'{prefix}-final.png')

    audio = ROOT / 'assets/sound-design.wav'
    if four_k:
        encode_4k(frames, out, audio, prefix)
    else:
        clean.encode(frames, out, audio, prefix=prefix, preview_crf=5,
                     alpha_crf=0, prores_q=5, alpha_lossless=True)

    info = {
        'duration': 5, 'fps': 30, 'frames': 150, 'size': list(clean.OUTPUT_SIZE),
        'in_plane_turn_degrees': 250, 'x_axis_start_degrees': -62,
        'x_axis_turn_start': TILT_START, 'x_axis_turn_end': TILT_END,
        'projection': 'perspective', 'perspective_focal_native_pixels': 1150,
        'ray_proportions': 'preserved in 3D before projection',
        'independent_ray_oscillation': False, 'node_size_pulsation': False,
        'cursor_rejoined_at': JOIN, 'final_hold_start': 3.7,
        'matches_approved_final': True, 'extra_fragments': 0,
        'text_supersampling': clean.SSAA, 'star_supersampling': clean.STAR_SSAA,
        'preview_h264_crf': 4 if four_k else 5,
        'alpha_master': '4K ProRes 4444 XQ-style 4:4:4 10-bit / 16-bit alpha' if four_k
                        else 'ProRes 4444 and lossless VP9 alpha'
    }
    (out / f'{prefix}-render-info.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--4k', action='store_true', dest='four_k')
    render(parser.parse_args().four_k)
