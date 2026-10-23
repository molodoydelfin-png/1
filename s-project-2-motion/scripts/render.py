"""Animate original source pixels. The logo is never recreated or typeset.

Masks only control which source pixels are visible. The final 16:9 frame is
the original PNG under a single uniform scale and a crop of empty margins.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage as nd
from make_audio import make_audio

ROOT = Path(__file__).resolve().parent.parent
SIZE = (1920, 1080)
FPS = 30
DURATION = 5.0
CENTER = np.array([389.0, 526.0], dtype=np.float32)
# Measured on assets/source.png. The source's ten directions are retained.
ENDPOINTS = np.array([
    [501.0, 364.5], [395.0, 385.0], [281.0, 438.0], [213.0, 524.0],
    [329.0, 551.0], [311.0, 618.0], [391.0, 659.0], [478.0, 673.5],
    [466.0, 558.0], [469.0, 487.0],
], dtype=np.float32)
ROI = (145, 300, 1340, 735)
CROP_TOP = 112.25


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease(t, start, duration):
    u = np.clip((t - start) / duration, 0, 1)
    if u <= 0:
        return 0.0
    if u >= 1 - 1e-10:
        return 1.0
    return float(u * u * u * (u * (6 * u - 15) + 10))


def ray_progress(t, index):
    u = np.clip((t - (.5 + index * .055)) / .805, 0, 1)
    if u <= 0:
        return 0.0
    if u >= 1 - 1e-10:
        return 1.0
    # A minute variation of velocity; no geometric overshoot or visible bounce.
    base = u * u * u * (u * (6 * u - 15) + 10)
    return float(np.clip(base + .008 * np.sin(4*np.pi*u) * u*u*(1-u)*(1-u), 0, 1))


class OriginalPixelAnimation:
    def __init__(self, source):
        self.source_path = Path(source)
        self.source = Image.open(source).convert('RGB')
        if self.source.size != (1448, 1086):
            raise ValueError('This measured animation requires the supplied 1448×1086 source.png; do not silently resize a different logo.')
        self.original = np.asarray(self.source)
        x0, y0, x1, y1 = ROI
        self.crop = self.original[y0:y1, x0:x1].copy()
        foreground = self.crop.min(axis=2) < 224
        # Fill only the background hidden behind the existing logo.
        # The actual graphic is kept in self.crop, untouched.
        self.active = nd.binary_dilation(foreground, iterations=12)
        self.base = self._clean_background(foreground)
        self.background_delta = self.crop.astype(np.float32) - self.base
        self.background_delta[self.active] = 0
        self.ys, self.xs = np.where(self.active)
        self.x = self.xs.astype(np.float32) + x0
        self.y = self.ys.astype(np.float32) + y0
        self.points = np.column_stack((self.x, self.y))
        self.delta = self.crop[self.ys, self.xs].astype(np.float32) - self.base[self.ys, self.xs]
        self.symbol = self.x < 520
        self.s_letter = (self.x >= 520) & (self.x < 691)
        self.hyphen = (self.x >= 691) & (self.x < 735)
        self.project = self.x >= 735
        offsets = self.points - CENTER
        self.radial = np.linalg.norm(offsets, axis=1)
        vectors = ENDPOINTS - CENTER
        self.lengths = np.linalg.norm(vectors, axis=1)
        units = vectors / self.lengths[:, None]
        distances = []
        for unit, length in zip(units, self.lengths):
            projection = np.clip(offsets @ unit, 0, length)
            distances.append(np.linalg.norm(offsets - projection[:, None] * unit, axis=1))
        self.ray_owner = np.argmin(np.stack(distances), axis=0)
        node_distances = np.linalg.norm(self.points[:, None, :] - ENDPOINTS[None, :, :], axis=2)
        self.node_owner = np.argmin(node_distances, axis=1)
        self.on_node = (node_distances.min(axis=1) <= 30) & self.symbol
        self.on_ray = self.symbol & ~self.on_node
        self.s_clock = self._s_clock()
        self.project_clock = np.clip((self.x - 742) / (1292 - 742), 0, 1)
        self.symbol_local = np.zeros(self.crop.shape[:2], dtype=bool)
        self.symbol_local[self.ys[self.symbol], self.xs[self.symbol]] = True
        self.reference = self._to_frame(self.source)

    def _clean_background(self, foreground):
        # Bright bevel pixels must disappear along with the dark artwork.
        # Fit a smooth background from unoccupied pixels; sampling the edge of
        # a hidden letter would accidentally leave a bright letter-shaped halo.
        height, width = foreground.shape
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        x = (xx-width/2)/width
        y = (yy-height/2)/height
        basis = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1)
        excluded = nd.binary_dilation(foreground, iterations=32)
        sample = np.zeros_like(foreground)
        sample[::7, ::7] = True
        sample &= ~excluded
        design = basis[sample].astype(np.float64)
        colors = self.crop[sample].astype(np.float64)
        weights = np.ones(len(colors))
        for _ in range(4):
            weighted = np.sqrt(weights)[:, None]
            coefficients = np.linalg.lstsq(design*weighted, colors*weighted, rcond=None)[0]
            residual = np.linalg.norm(colors-design@coefficients, axis=1)
            weights = np.minimum(1, 3/np.maximum(residual, .001))
        repaired = (basis@coefficients).astype(np.float32)
        edge = np.minimum.reduce([xx, yy, width-1-xx, height-1-yy])
        feather = smoothstep(edge/24)[:, :, None]
        return np.clip(repaired*feather+self.crop*(1-feather),0,255)

    def _s_clock(self):
        # A reveal guide over the ORIGINAL custom S, not a replacement outline.
        # The three original cut strokes remain untouched in the source image.
        guide = np.array([
            [688, 482], [660, 484], [635, 489], [607, 499], [587, 509], [580, 514],
            [584, 518], [581, 522], [600, 528], [623, 535], [647, 542], [664, 547],
            [654, 554], [659, 560], [649, 569], [630, 578], [602, 587], [564, 595], [523, 598],
        ], dtype=np.float32)
        dense = []
        for a, b in zip(guide[:-1], guide[1:]):
            steps = max(2, int(np.linalg.norm(b-a)*2))
            dense.extend(np.linspace(a, b, steps, endpoint=False))
        dense.append(guide[-1])
        dense = np.asarray(dense)
        # KD tree limits memory and deterministically maps pixels to stroke time.
        from scipy.spatial import cKDTree
        tree = cKDTree(dense)
        _, index = tree.query(self.points)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        return (arc[index] / arc[-1]).astype(np.float32)

    def _to_frame(self, native):
        # Uniform 1920/1448 scale; crop is entirely outside the actual logo.
        # The source's presentation label at y≈940 is outside this 16:9 crop.
        inverse_scale = self.source.width / SIZE[0]
        return native.transform(SIZE, Image.Transform.AFFINE,
                                (inverse_scale, 0, 0, 0, inverse_scale, CROP_TOP),
                                resample=Image.Resampling.BICUBIC)

    def state(self, t):
        return {
            'time': float(t),
            'rays': [ray_progress(t, i) for i in range(10)],
            'spheres': [ease(t, 1.2+i*.075, .325) for i in range(10)],
            's': ease(t, 2.5, .68),
            'hyphen': ease(t, 3.05, .25),
            'project': ease(t, 3.12, .58),
            'settle': self.settle(t),
        }

    @staticmethod
    def settle(t):
        if t <= 2 or t >= 3:
            return (0., 0.)
        u = t - 2
        window = np.sin(np.pi*u)**2
        # Values in OUTPUT pixels: a single coherent, tiny translation.
        return (float(.55*np.sin(2*np.pi*u)*window), float(-1.4*window))

    def alpha(self, t):
        state = self.state(t)
        alpha = np.zeros(len(self.x), dtype=np.float32)
        for i, progress in enumerate(state['rays']):
            select = self.on_ray & (self.ray_owner == i)
            if progress > 0:
                alpha[select] = smoothstep((self.lengths[i]*progress - self.radial[select] + 1.1) / 2.2)
        for i, progress in enumerate(state['spheres']):
            alpha[self.on_node & (self.node_owner == i)] = progress
        center = ease(t, 0, .36)
        pulse = 1 + .035*np.sin(np.pi*np.clip(t/.5,0,1))**2
        center_mask = self.symbol & (self.radial <= 3.1*pulse)
        alpha[center_mask] = np.maximum(alpha[center_mask], center)
        alpha[self.s_letter] = smoothstep((state['s']*1.11 - self.s_clock[self.s_letter])/.11)
        alpha[self.hyphen] = state['hyphen']
        alpha[self.project] = smoothstep((state['project']*1.15 - self.project_clock[self.project])/.15)
        # Restore the few pale edge/background pixels that have no ray centerline
        # gradually, not with a last-frame pop.
        finish = ease(t, 3.5, .2)
        alpha += (1-alpha)*finish
        if t >= 3.7 - 1e-10:
            # All layers fully restored. This also restores any background-edge
            # pixels repaired during the extraction of the raster artwork.
            alpha[:] = 1
        return alpha

    def frame(self, t):
        alpha = self.alpha(t)
        layer = self.background_delta * ease(t, 0, 3.7)
        layer[self.ys, self.xs] = self.delta * alpha[:, None]
        dx, dy = self.settle(t)
        if dx or dy:
            symbol_layer = np.where(self.symbol_local[:, :, None], layer, 0)
            layer -= symbol_layer
            native_scale = self.source.width / SIZE[0]
            layer += nd.shift(symbol_layer, (dy*native_scale, dx*native_scale, 0),
                              order=1, mode='constant', cval=0, prefilter=False)
        pixels = np.rint(np.clip(self.base+layer, 0, 255)).astype(np.uint8)
        native = self.source.copy()
        native.paste(Image.fromarray(pixels), ROI[:2])
        return self._to_frame(native)


def render(source, destination):
    animation = OriginalPixelAnimation(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio_path = ROOT / 'assets/sound-design.wav'
    make_audio(audio_path)
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', '1920x1080', '-r', str(FPS), '-i', 'pipe:0',
        '-i', str(audio_path), '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-threads', '2', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', str(DURATION),
        '-movflags', '+faststart', str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for n in range(round(DURATION*FPS)):
            process.stdin.write(animation.frame(n/FPS).tobytes())
            if n % FPS == 0:
                print(f'Render {n//FPS}/{int(DURATION)} seconds', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('FFmpeg video encoding failed')
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    final = animation.frame(5)
    if not np.array_equal(np.asarray(final), np.asarray(animation.reference)):
        raise AssertionError('Final frame does not match the uniformly reframed source')
    final.save(destination.parent/'s-project-2-final.png')
    metadata = {
        'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'source_size': list(animation.source.size), 'size': list(SIZE), 'fps': FPS,
        'duration': DURATION, 'frames': round(DURATION*FPS),
        'center': CENTER.tolist(), 'endpoints': ENDPOINTS.tolist(),
        'source_crop': [0, CROP_TOP, 1448, CROP_TOP+1080*1448/1920],
        'uniform_scale': 1920/1448,
        'final_png_matches_reframed_source_exactly': True,
        'note': 'Original raster pixels; no replacement font, no regenerated logo. MP4 uses ordinary lossy H.264 compression.',
    }
    (destination.parent/'render-info.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    silent = destination.with_name(destination.stem+'-silent.mp4')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(destination),
                    '-map','0:v:0','-c:v','copy','-an','-movflags','+faststart',str(silent)],check=True)
    print(f'Complete: {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'assets/source.png')
    parser.add_argument('--output', type=Path, default=ROOT/'exports/s-project-2.mp4')
    args = parser.parse_args()
    render(args.source, args.output)
