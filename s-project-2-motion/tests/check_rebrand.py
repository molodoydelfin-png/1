"""Validate the silent old-to-new presentation render."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from render_rebrand import (RebrandAnimation, PREFIX, FPS, FRAMES, SIZE, DURATION,
                            DETACH, SCAN_START, SCAN_END, JOIN, BG_START, BG_END)


count = 0


def check(name, condition):
    global count
    assert condition, name
    count += 1
    print('PASS', name, flush=True)


a = RebrandAnimation()
old = np.asarray(a.old_full.convert('RGB'))
check('opening is the exact reconstructed old identity on its light field',
      np.array_equal(np.asarray(a.frame(0)), old) and np.array_equal(np.asarray(a.frame(1.0)), old))

for t in np.linspace(1.2, DETACH, 100):
    _, points, _, radii, _ = a.star_state(t)
    assert np.isfinite(points).all() and np.isfinite(radii).all() and radii.min() > 0
check('ten-ray symbol rebuild remains finite and coherent through perspective motion', True)

start_center, start_points, _, _, _ = a.star_state(1.25)
quarter_center, quarter_points, _, _, _ = a.star_state(1.25+2.90*.25)
end_center, end_points, _, _, _ = a.star_state(DETACH)
start_vectors = start_points-start_center
quarter_vectors = quarter_points-quarter_center
end_vectors = end_points-end_center
check('symbol performs one coherent horizontal turn and returns to its exact orientation',
      np.sign(start_vectors[0, 0]) == np.sign(quarter_vectors[0, 0]) and
      np.max(np.abs(end_points-a.new_endpoints)) < 1e-6)
check('perspective squeeze stays restrained instead of collapsing the star',
      .60*np.ptp(start_points[:, 0]) < np.ptp(quarter_points[:, 0]) <
      .90*np.ptp(start_points[:, 0]) and
      np.ptp(quarter_points[:, 1]) > .85*np.ptp(start_points[:, 1]))

check('cursor launches from a real symbol node',
      np.array_equal(a.cursor_position(DETACH), a.star_state(DETACH)[1][9]))
positions = np.array([a.cursor_position(t) for t in np.linspace(SCAN_START, SCAN_END, 100)])
check('conversion cursor scans monotonically from right to left', np.diff(positions[:, 0]).max() <= 1e-8)

old_alpha = a.text_alpha(SCAN_START-.1)
new_alpha = a.text_alpha(SCAN_END+.1)
check('text conversion starts from old contours and ends on the new custom wordmark',
      np.array_equal(old_alpha, a.old_word_alpha) and np.array_equal(new_alpha, a.new_word_alpha))

new_points = a.new_endpoints
moving = a.star_state((SCAN_START+SCAN_END)/2)[1]
settled = a.star_state(JOIN)[1]
check('star has restrained micro-motion during text conversion and settles exactly',
      np.max(np.linalg.norm(moving-new_points, axis=1)) > 1 and
      np.max(np.abs(settled-new_points)) < 1e-6)

samples = np.array([a.colors(t)[0] for t in np.linspace(BG_START, BG_END, 140)])
check('background transition is continuous and spread across more than three seconds',
      np.max(np.abs(np.diff(samples.astype(int), axis=0))) <= 5 and
      np.mean(samples[0]) > 240 and np.mean(samples[-1]) < 30)

exact = np.asarray(a.exact_final.convert('RGB'))
for t in [8.05, 8.3, 8.6, (FRAMES-1)/FPS]:
    assert np.array_equal(np.asarray(a.frame(t)), exact)
check('final hold is the exact approved new identity on graphite', True)

def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration',
        '-of', 'json', str(path)
    ]))


for name, size in [(f'{PREFIX}-4k.mp4', (3840, 2160)),
                   (f'{PREFIX}-fullhd.mp4', (1920, 1080))]:
    info = probe(ROOT / 'exports' / name)
    videos = [stream for stream in info['streams'] if stream['codec_type'] == 'video']
    check(f'{name}: H.264, native dimensions, 30 fps and {FRAMES} frames',
          len(videos) == 1 and videos[0]['codec_name'] == 'h264' and
          videos[0]['pix_fmt'] == 'yuv420p' and
          (videos[0]['width'], videos[0]['height']) == size and
          videos[0]['r_frame_rate'] == '30/1' and videos[0]['nb_read_frames'] == str(FRAMES))
    check(f'{name}: deliberately contains no audio stream',
          not any(stream['codec_type'] == 'audio' for stream in info['streams']))
    check(f'{name}: exact {DURATION}-second container duration',
          abs(float(info['format']['duration'])-DURATION) < .01)

print(f'\n{count} rebrand presentation checks passed.')
