"""Checks for the reference-paced turn and its delivered media."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as nd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from render_clean import CleanAnimation
from render_flow import FlowAnimation, PREFIX, TURN_START, TURN_END, START_ANGLE
from render_sweep import DETACH, JOIN, CURSOR_NODE


count = 0


def check(name, value):
    global count
    assert value, name
    count += 1
    print('PASS', name, flush=True)


a = FlowAnimation(ROOT / 'assets/source.png')
approved = CleanAnimation(ROOT / 'assets/source.png')
reference = np.asarray(approved.final)

times = np.linspace(0, TURN_END, 1501)
angles = np.array([a.angle(t) for t in times])
speeds = np.diff(angles) / np.diff(times)
check('symbol makes one continuous readable 250-degree turn',
      np.isclose(angles[0], START_ANGLE) and np.isclose(angles[-1], 0) and
      speeds.min() >= -1e-8 and np.rad2deg(speeds.max()) < 151)
check('turn eases to a true stop instead of bouncing',
      np.rad2deg(speeds[-1]) < .01 and np.all(np.diff(angles) >= -1e-10))

growth = np.array([a.base_growth(t) for t in times])
check('radial emergence is monotonic with a long soft tail',
      growth[0] == 0 and growth[-1] == 1 and np.diff(growth).min() >= -1e-10 and
      .45 < a.base_growth(1.) < .58 and .89 < a.base_growth(2.5) < .96)

source_lengths = np.linalg.norm(a.endpoints - a.center, axis=1)
for t in [.55, 1., 1.45, 2., 2.8]:
    current = np.linalg.norm(a.node_positions(t, a.scales(t)) - a.center, axis=1)
    ratio = current / source_lengths
    assert np.ptp(ratio) < 1e-6
check('all ten rays preserve their exact relative lengths throughout the turn', True)
check('heads become stable without size pulsation',
      np.array_equal(a.node_growth(1.2, a.scales(1.2)), np.ones(10)) and
      np.array_equal(a.node_growth(2.5, a.scales(2.5)), np.ones(10)))

epsilon = 1e-4
check('cursor detaches from the moving sphere without a position jump',
      np.array_equal(a.cursor_position(DETACH), a.endpoint(CURSOR_NODE, DETACH)))
incoming = (a.endpoint(CURSOR_NODE, DETACH) - a.endpoint(CURSOR_NODE, DETACH - epsilon)) / epsilon
outgoing = (a.cursor_position(DETACH + epsilon) - a.cursor_position(DETACH)) / epsilon
check('cursor inherits the fluid turn velocity at launch', np.linalg.norm(incoming - outgoing) < .5)
for boundary in [DETACH, JOIN]:
    sprite, x, y = a.cursor_layer(boundary)
    target, xx, yy = a.node_sprite(a.cursor_position(boundary), a.radii[CURSOR_NODE], 1.)
    assert (x, y) == (xx, yy) and np.array_equal(sprite, target)
check('node material stays identical at both cursor handoffs', True)

check('approved final composition remains pixel-identical',
      np.array_equal(np.asarray(a.final), reference))
for t in [3.7, 4., 4.5, 149 / 30]:
    assert np.array_equal(np.asarray(a.frame(t)), reference)
check('final hold is exact, sharp and static', True)

for frame in range(1, 66):
    rgba = np.asarray(Image.open(ROOT / f'.cache/{PREFIX}-frames/{frame:04d}.png'))
    assert nd.label(rgba[:, :, 3] > 3, structure=np.ones((3, 3)))[1] <= 2
check('rendered early frames contain no stray pieces', True)


def probe(file):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration',
        '-of', 'json', str(file)
    ]))


for name, size in [(f'{PREFIX}-alpha.webm', (1920, 1080)),
                   (f'{PREFIX}-alpha.mov', (1920, 1080)),
                   (f'{PREFIX}-4k-alpha.webm', (3840, 2160))]:
    path = ROOT / 'exports' / name
    info = probe(path)
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    check(f'{name}: native size, 30 fps and 150 frames',
          (video['width'], video['height']) == size and video['r_frame_rate'] == '30/1' and
          video['nb_read_frames'] == '150')
    decoder = ['-c:v', 'libvpx-vp9'] if path.suffix == '.webm' else []
    raw = subprocess.check_output([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', *decoder, '-i', str(path),
        '-ss', '4.5', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'
    ])
    decoded = np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 4)
    target = reference if size == (1920, 1080) else np.asarray(
        Image.open(ROOT / 'exports/s-project-2-clean-4k-final.png'))
    difference = np.abs(decoded[:, :, 3].astype(np.int16) - target[:, :, 3].astype(np.int16))
    check(f'{name}: real transparency and intact final silhouette',
          decoded[0, 0, 3] == 0 and decoded[:, :, 3].max() == 255 and
          (difference.max() <= 1 if path.suffix == '.webm' else difference.mean() < .1))
    check(f'{name}: soundtrack retained', any(s['codec_type'] == 'audio' for s in info['streams']))

for prefix, size in [(PREFIX, (1920, 1080)), (PREFIX + '-4k', (3840, 2160))]:
    info = probe(ROOT / f'exports/{prefix}-preview.mp4')
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    check(f'{prefix}: preview is sharp native H.264',
          video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p' and
          (video['width'], video['height']) == size)

print(f'\n{count} fluid-turn checks passed.')
