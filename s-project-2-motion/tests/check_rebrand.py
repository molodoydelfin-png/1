"""Validate the silent old-to-new presentation render."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from render_rebrand import RebrandAnimation, PREFIX, FPS, FRAMES, SIZE


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

for t in np.linspace(1.2, 3.6, 100):
    _, points, _, radii, _ = a.star_state(t)
    assert np.isfinite(points).all() and np.isfinite(radii).all() and radii.min() > 0
check('ten-ray symbol rebuild remains finite and coherent through perspective motion', True)

check('cursor launches from a real symbol node',
      np.array_equal(a.cursor_position(3.0), a.star_state(3.0)[1][9]))
positions = np.array([a.cursor_position(t) for t in np.linspace(4.0, 5.35, 100)])
check('conversion cursor scans monotonically from right to left', np.diff(positions[:, 0]).max() <= 1e-8)

old_alpha = a.text_alpha(3.9)
new_alpha = a.text_alpha(5.4)
check('text conversion starts from old contours and ends on the new custom wordmark',
      np.array_equal(old_alpha, a.old_word_alpha) and np.array_equal(new_alpha, a.new_word_alpha))

exact = np.asarray(a.exact_final.convert('RGB'))
for t in [6.0, 6.5, 7.0, 239/30]:
    assert np.array_equal(np.asarray(a.frame(t)), exact)
check('last two seconds are the exact approved new identity on graphite', True)

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
    check(f'{name}: H.264, native dimensions, 30 fps and 240 frames',
          len(videos) == 1 and videos[0]['codec_name'] == 'h264' and
          videos[0]['pix_fmt'] == 'yuv420p' and
          (videos[0]['width'], videos[0]['height']) == size and
          videos[0]['r_frame_rate'] == '30/1' and videos[0]['nb_read_frames'] == str(FRAMES))
    check(f'{name}: deliberately contains no audio stream',
          not any(stream['codec_type'] == 'audio' for stream in info['streams']))
    check(f'{name}: exact eight-second container duration', abs(float(info['format']['duration'])-8) < .01)

print(f'\n{count} rebrand presentation checks passed.')
