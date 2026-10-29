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


def ray_progress(t, index=0):
    """One shared radial spring, anchored at the original center; no angle order."""
    u = np.clip((t - .35) / 1.9, 0, 1)
    if u <= 0:
        return .025
    if u >= 1 - 1e-10:
        return 1.0
    spring = 1 - np.exp(-8*u)*(np.cos(8*u)+np.sin(8*u))
    # Gently eliminate the remaining oscillation and its velocity at the end.
    settle = ease(u, .78, .22)
    spring = spring*(1-settle)+settle
    return float(.025+.975*spring)


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
        self.base = self._clean_background(foreground)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        self.symbol = xx < 520
        self.s_letter = (xx >= 520) & (xx < 691)
        self.project = xx >= 691  # Hyphen and PROJECT are one intact fading layer.
        # Isolate actual dark artwork, not rectangular light background patches.
        # Signed color differences preserve the original spheres and graphite.
        contrast = (self.base-self.crop.astype(np.float32)).mean(axis=2)
        matte = smoothstep((contrast-3)/12)
        self.ink = np.minimum(self.crop.astype(np.float32)-self.base,0)*matte[:,:,None]
        # A compact moving tile with margins for the ~4.2% spring overshoot.
        self.star_box = (30,30,383,412)
        sx0,sy0,sx1,sy1 = self.star_box
        self.star = self.ink[sy0:sy1,sx0:sx1].copy()
        self.star *= self.symbol[sy0:sy1,sx0:sx1,None]
        self.star_center = np.array([CENTER[1]-y0-sy0,CENTER[0]-x0-sx0])
        central = ((xx-CENTER[0])**2+(yy-CENTER[1])**2)<=3.1**2
        self.central = self.ink*central[:,:,None]
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
            'spheres': [ease(t, .35, .55)]*10,
            's': ease(t, 2.15, 1.15),
            'hyphen': ease(t, 2.3, 1.25),
            'project': ease(t, 2.3, 1.25),
            'settle': (0.,0.),
        }

    @staticmethod
    def settle(t):
        return (0., 0.)

    def frame(self, t):
        if t >= 3.7-1e-10:
            return self.reference.copy()
        state = self.state(t)
        scale = state['rays'][0]
        opacity = state['spheres'][0]
        layer = np.zeros_like(self.base)
        if opacity:
            inverse = 1/scale
            matrix = np.diag([inverse,inverse,1.])
            offset = [*(self.star_center*(1-inverse)),0.]
            moving = nd.affine_transform(self.star,matrix,offset=offset,order=1,
                                        mode='constant',cval=0,prefilter=False)
            sx0,sy0,sx1,sy1 = self.star_box
            layer[sy0:sy1,sx0:sx1] = np.minimum(moving,0)*opacity
        # Keep the hub anchored; rays and spheres are the same moving layer.
        layer = np.minimum(layer,self.central*ease(t,0,.35))
        layer += self.ink*self.s_letter[:,:,None]*state['s']
        layer += self.ink*self.project[:,:,None]*state['project']
        composed = np.clip(self.base+layer,0,255)
        # Once the artwork is fully in place, restore the source's very pale
        # background texture uniformly. No ray sectors or moving white patches.
        finish = ease(t,3.2,.5)
        composed = composed*(1-finish)+self.crop*finish
        pixels = np.rint(np.clip(composed, 0, 255)).astype(np.uint8)
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
        'motion': 'shared radial spring; attached spheres; whole-letter and whole-word opacity, no stroke drawing or wiping',
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
    parser.add_argument('--output', type=Path, default=ROOT/'exports/s-project-2-radial.mp4')
    args = parser.parse_args()
    render(args.source, args.output)
