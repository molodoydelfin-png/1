"""Geometry, quality and delivery checks for the perspective X-axis version."""
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
from render_xaxis import XAxisAnimation, PREFIX, TILT_START, TILT_END, START_TILT
from render_flow import TURN_END
from render_sweep import DETACH, JOIN, CURSOR_NODE


count = 0


def check(name, value):
    global count
    assert value, name
    count += 1
    print('PASS', name, flush=True)


a = XAxisAnimation(ROOT / 'assets/source.png')
approved = CleanAnimation(ROOT / 'assets/source.png')
reference = np.asarray(approved.final)

times = np.linspace(0, TILT_END, 1501)
tilts = np.array([a.x_tilt(t) for t in times])
check('X-axis tilt moves monotonically from -62 degrees to face-on',
      np.isclose(tilts[0], START_TILT) and np.isclose(tilts[-1], 0) and
      np.diff(tilts).min() >= -1e-10)
check('X-axis turn has zero-speed easing at both ends',
      np.rad2deg(np.diff(tilts[:2])[0] / np.diff(times[:2])[0]) < .01 and
      np.rad2deg(np.diff(tilts[-2:])[0] / np.diff(times[-2:])[0]) < .01)

for t in [.55, 1., 1.45, 2., 2.8]:
    positions = a.node_positions(t, a.scales(t))
    assert np.isfinite(positions).all()
    assert not np.allclose(positions, a.center + (a.endpoints - a.center) * a.scales(t)[:, None])
check('rendered endpoints receive genuine perspective X-axis projection', True)

for t in np.linspace(.3, 3.45, 120):
    positions = a.node_positions(t, a.scales(t))
    left, top, right, bottom = a.star_box
    assert np.min(positions[:, 0] - a.radii) > left + 2
    assert np.max(positions[:, 0] + a.radii) < right - 2
    assert np.min(positions[:, 1] - a.radii) > top + 2
    assert np.max(positions[:, 1] + a.radii) < bottom - 2
check('perspective silhouette stays inside high-resolution antialiasing margins', True)

check('cursor detaches from its projected sphere without a jump',
      np.array_equal(a.cursor_position(DETACH), a.endpoint(CURSOR_NODE, DETACH)))
epsilon = 1e-4
incoming = (a.endpoint(CURSOR_NODE, DETACH) - a.endpoint(CURSOR_NODE, DETACH - epsilon)) / epsilon
outgoing = (a.cursor_position(DETACH + epsilon) - a.cursor_position(DETACH)) / epsilon
check('cursor inherits the X-axis motion at launch', np.linalg.norm(incoming - outgoing) < .5)
for boundary in [DETACH, JOIN]:
    sprite, x, y = a.cursor_layer(boundary)
    target, xx, yy = a.node_sprite(a.cursor_position(boundary), a.radii[CURSOR_NODE], 1.)
    assert (x, y) == (xx, yy) and np.array_equal(sprite, target)
check('sphere material stays identical at cursor handoffs', True)

check('all rotation resolves to the approved front view',
      a.x_tilt(TURN_END) == 0 and
      np.array_equal(a.node_positions(TURN_END, a.scales(TURN_END)), a.endpoints))
check('approved final frame remains pixel-identical', np.array_equal(np.asarray(a.final), reference))
for t in [3.7, 4., 4.5, 149 / 30]:
    assert np.array_equal(np.asarray(a.frame(t)), reference)
check('final hold is sharp and static', True)

for frame in range(1, 66):
    rgba = np.asarray(Image.open(ROOT / f'.cache/{PREFIX}-frames/{frame:04d}.png'))
    assert nd.label(rgba[:, :, 3] > 3, structure=np.ones((3, 3)))[1] <= 2
check('actual rendered frames contain no stray pieces', True)


def probe(file):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration',
        '-of', 'json', str(file)
    ]))


deliveries = [
    (f'{PREFIX}-alpha.webm', (1920, 1080), 'vp9'),
    (f'{PREFIX}-alpha.mov', (1920, 1080), 'prores'),
    (f'{PREFIX}-4k-alpha.webm', (3840, 2160), 'vp9'),
    (f'{PREFIX}-4k-alpha.mov', (3840, 2160), 'prores')
]
for name, size, codec in deliveries:
    path = ROOT / 'exports' / name
    info = probe(path)
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    check(f'{name}: native size, 30 fps, 150 frames and expected codec',
          (video['width'], video['height']) == size and video['r_frame_rate'] == '30/1' and
          video['nb_read_frames'] == '150' and video['codec_name'] == codec)
    if size == (3840, 2160) and codec == 'prores':
        check('4K finishing master is 4:4:4 10-bit ProRes 4444',
              video['pix_fmt'] == 'yuva444p12le' and video['profile'] == '4444')
    decoder = ['-c:v', 'libvpx-vp9'] if path.suffix == '.webm' else []
    raw = subprocess.check_output([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', *decoder, '-i', str(path),
        '-ss', '4.5', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'
    ])
    decoded = np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 4)
    target = reference if size == (1920, 1080) else np.asarray(
        Image.open(ROOT / 'exports/s-project-2-clean-4k-final.png'))
    difference = np.abs(decoded[:, :, 3].astype(np.int16) - target[:, :, 3].astype(np.int16))
    check(f'{name}: transparency and final silhouette survive decoding',
          decoded[0, 0, 3] == 0 and decoded[:, :, 3].max() == 255 and
          (difference.max() <= 1 if path.suffix == '.webm' else difference.mean() < .1))

for prefix, size in [(PREFIX, (1920, 1080)), (PREFIX + '-4k', (3840, 2160))]:
    info = probe(ROOT / f'exports/{prefix}-preview.mp4')
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    check(f'{prefix}: preview is native H.264',
          video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p' and
          (video['width'], video['height']) == size)

print(f'\n{count} X-axis quality checks passed.')
