"""Silent 8-second old-to-new S-PROJECT identity transition."""
from copy import deepcopy
from pathlib import Path
import argparse
import io
import json
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

import render_clean as clean
from render_clean import CleanAnimation, ROOT, ramp


PREFIX = 's-project-rebrand-horizontal'
SIZE = (3840, 2160)
FPS = 30
DURATION = 9
FRAMES = FPS * DURATION
SCALE = 2.0  # motion geometry is authored in the 1920×1080 coordinate system
OLD_BG = np.array([247, 247, 247], dtype=np.float32)
NEW_BG = np.array([20, 23, 28], dtype=np.float32)
OLD_FG = np.array([9, 9, 9], dtype=np.float32)
NEW_FG = np.array([245, 245, 245], dtype=np.float32)

DETACH = 4.15
OUT_END = 4.95
SCAN_START = 5.15
SCAN_END = 7.00
JOIN = 7.65
BG_START = 3.80
BG_END = 7.55

OLD_CENTER = np.array([103.2 + 203 * 2.4, 61.2 + 205 * 2.4])
OLD_ENDPOINTS = np.array([
    [278, 85], [210, 102], [126, 147], [71, 204], [165, 220],
    [152, 267], [204, 303], [252, 314], [246, 225], [250, 183]
], dtype=np.float64) * 2.4 + np.array([103.2, 61.2])


def smooth01(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def bezier(p0, p1, p2, p3, u):
    return ((1-u)**3*p0 + 3*(1-u)**2*u*p1 +
            3*(1-u)*u*u*p2 + u**3*p3)


def render_old_svg(part=None):
    source = ET.parse(ROOT.parent / 's-project-motion/assets/logo.svg').getroot()
    root = deepcopy(source)
    for element in list(root):
        if element.get('data-background-rect') is not None and part is not None:
            root.remove(element)
    composition = next(element for element in root.iter()
                       if element.get('data-composition') is not None)
    for child in list(composition):
        if part == 'wordmark' and child.get('data-symbol') is not None:
            composition.remove(child)
        elif part == 'symbol' and child.get('data-wordmark') is not None:
            composition.remove(child)
    root.set('color', '#090909')
    pixels = clean.cairosvg.svg2png(bytestring=ET.tostring(root),
                                    output_width=SIZE[0], output_height=SIZE[1])
    return Image.open(io.BytesIO(pixels)).convert('RGBA')


class RebrandAnimation:
    def __init__(self):
        model = CleanAnimation(ROOT / 'assets/source.png')
        self.new_center = model.center.astype(np.float64)
        self.new_endpoints = model.endpoints.astype(np.float64)
        self.new_radii = model.radii.astype(np.float64)
        self.new_widths = model.widths.astype(np.float64)

        self.old_full = render_old_svg()
        self.old_word_alpha = np.asarray(render_old_svg('wordmark'))[:, :, 3]
        self.new_full = Image.open(ROOT / 'exports/s-project-2-clean-4k-final.png').convert('RGBA')
        new_pixels = np.asarray(self.new_full)
        self.new_word_alpha = new_pixels[:, :, 3].copy()
        self.new_word_alpha[:, :1374] = 0
        self.exact_final = Image.new('RGBA', SIZE, tuple(NEW_BG.astype(np.uint8)) + (255,))
        self.exact_final.alpha_composite(self.new_full)
        occupied = np.where(self.new_word_alpha > 8)
        self.text_bounds = (occupied[1].min(), occupied[0].min(),
                            occupied[1].max() + 1, occupied[0].max() + 1)

    def background_progress(self, t):
        return ramp(t, BG_START, BG_END-BG_START)

    def colors(self, t):
        background_progress = self.background_progress(t)
        foreground_progress = ramp(t, 5.85, 1.35)
        bg = np.rint(OLD_BG * (1-background_progress) + NEW_BG * background_progress).astype(np.uint8)
        fg = np.rint(OLD_FG * (1-foreground_progress) + NEW_FG * foreground_progress).astype(np.uint8)
        return tuple(bg), tuple(fg), background_progress

    def star_state(self, t):
        phase = ramp(t, 1.25, 2.90)
        morph = ramp(t, 1.50, 2.35)
        center = OLD_CENTER * (1-morph) + self.new_center * morph
        old_vectors = OLD_ENDPOINTS - OLD_CENTER
        new_vectors = self.new_endpoints - self.new_center
        vectors = old_vectors * (1-morph) + new_vectors * morph
        # Horizontal 3D turn around the vertical axis. Ray directions stay as one
        # coherent sign; only horizontal foreshortening, depth and light change.
        yaw = 2*np.pi*phase
        vectors *= 1 - .045*np.sin(np.pi*phase)**2
        source_x = vectors[:, 0].copy()
        x = source_x * (1 - .22*np.sin(yaw)**2)
        y = vectors[:, 1]
        depth = source_x * .28*np.sin(yaw)
        perspective = (1150 * clean.K) / (1150 * clean.K - depth)
        projected = np.column_stack((x * perspective, y * perspective))

        # During the rewrite the assembled system breathes as one rigid object;
        # no ray receives an independent wobble or changes its direction.
        micro = ramp(t, 4.30, .55) * (1-ramp(t, 7.00, .55))
        if micro > 0:
            breath = np.sin(2*np.pi*(t-4.30)/3.8)
            projected[:, 0] *= 1 - .018*micro*(.5+.5*breath)
            projected *= 1 + .005*micro*breath
        points = center + projected
        widths = np.full(10, 3.6 * 2.4) * (1-morph) + self.new_widths * morph
        radii = np.full(10, 7.1 * 2.4) * (1-morph) + self.new_radii * morph
        depth_scale = np.clip(1 + depth/(1150*clean.K)*.42, .92, 1.08)
        radii *= depth_scale
        widths *= np.clip(1 + depth/(1150*clean.K)*.16, .96, 1.04)
        return center, points, widths, radii, morph

    def cursor_position(self, t):
        start = self.star_state(DETACH)[1][9]
        target = self.new_endpoints[9]
        left, top, right, bottom = self.text_bounds
        middle = (top + bottom) / (2 * SCALE)
        far = np.array([right / SCALE + 52, middle])
        near = np.array([left / SCALE - 34, middle])
        if t <= DETACH:
            return start
        if t < OUT_END:
            u = ramp(t, DETACH, OUT_END - DETACH)
            return bezier(start, start + np.array([130, -30]),
                          far - np.array([190, 25]), far, u)
        if t < SCAN_START:
            return far
        if t < SCAN_END:
            return far + (near - far) * ramp(t, SCAN_START, SCAN_END - SCAN_START)
        if t < JOIN:
            u = ramp(t, SCAN_END, JOIN - SCAN_END)
            return bezier(near, near + np.array([0, -35]),
                          target + np.array([24, 0]), target, u)
        return target

    def cursor_shape(self, t):
        return ramp(t, OUT_END - .22, .25) * (1 - ramp(t, SCAN_END, .22))

    @staticmethod
    def node_sprite(position, radius, foreground, material):
        position = position * SCALE
        radius *= SCALE
        quality = 4
        padding = 4
        x0 = int(np.floor(position[0] - radius - padding))
        y0 = int(np.floor(position[1] - radius - padding))
        x1 = int(np.ceil(position[0] + radius + padding))
        y1 = int(np.ceil(position[1] + radius + padding))
        mask = Image.new('L', ((x1-x0)*quality, (y1-y0)*quality))
        draw = ImageDraw.Draw(mask)
        cx = (position[0]-x0)*quality
        cy = (position[1]-y0)*quality
        rr = radius*quality
        draw.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), fill=255)
        alpha = np.asarray(mask.resize((x1-x0, y1-y0), Image.Resampling.BOX),
                           dtype=np.float32) / 255
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        nx = (xx + .5 - position[0]) / radius
        ny = (yy + .5 - position[1]) / radius
        radial = np.clip(np.hypot(nx + .32, ny + .42) / 1.7, 0, 1)
        fg = np.array(foreground, dtype=np.float32)
        if fg.mean() < 128:
            rgb = fg[None, None, :] + material * (1-radial[:, :, None]) * 34
        else:
            rgb = fg[None, None, :] * (1 - material * .24 * radial[:, :, None])
        rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
        rgba[:, :, 3] = np.rint(alpha * 255).astype(np.uint8)
        return Image.fromarray(rgba, 'RGBA'), (x0, y0)

    def star_layer(self, t, foreground):
        center, points, widths, radii, material = self.star_state(t)
        left, top, right, bottom = [value * SCALE for value in (160, 170, 875, 895)]
        left, top, right, bottom = map(int, (left, top, right, bottom))
        quality = 4
        mask = Image.new('L', ((right-left)*quality, (bottom-top)*quality))
        draw = ImageDraw.Draw(mask)
        c = center * SCALE - np.array([left, top])
        for point, width in zip(points, widths):
            p = point * SCALE - np.array([left, top])
            draw.line([tuple(c*quality), tuple(p*quality)], fill=255,
                      width=max(1, round(width*SCALE*quality)))
        hub = ((4.3*(1-material) + 3.0*clean.K*material) * SCALE * quality)
        draw.ellipse((c[0]*quality-hub, c[1]*quality-hub,
                      c[0]*quality+hub, c[1]*quality+hub), fill=255)
        alpha = mask.resize((right-left, bottom-top), Image.Resampling.BOX)
        layer = Image.new('RGBA', (right-left, bottom-top), tuple(foreground) + (0,))
        layer.putalpha(alpha)
        for index, (point, radius) in enumerate(zip(points, radii)):
            if index == 9 and DETACH <= t < JOIN:
                continue
            turn_phase = ramp(t, 1.25, 2.90)
            yaw = 2*np.pi*turn_phase
            local_x = (self.new_endpoints-self.new_center)[index, 0]
            shimmer = .5 + .5*np.tanh(local_x/80*np.sin(yaw)*2)
            turn_active = np.sin(np.pi*turn_phase)**2
            node_material = np.clip(material*(.78+.34*shimmer*turn_active), 0, 1)
            sprite, (x, y) = self.node_sprite(point, radius, foreground, node_material)
            layer.alpha_composite(sprite, (x-left, y-top))
        return layer, (left, top)

    def cursor_layer(self, t, foreground):
        position = self.cursor_position(t) * SCALE
        radius = self.new_radii[9] * SCALE
        shape = self.cursor_shape(t)
        rx = radius * (1 - .86*shape)
        text_height = self.text_bounds[3] - self.text_bounds[1]
        ry = radius + (text_height*.54 - radius)*shape
        flight = DETACH < t < OUT_END
        returning = SCAN_END < t < JOIN
        trail_strength = 0.
        previous = position.copy()
        if flight:
            u = np.clip((t-DETACH)/(OUT_END-DETACH), 0, 1)
            trail_strength = np.sin(np.pi*u)
            previous = self.cursor_position(max(DETACH, t-.085)) * SCALE
        elif returning:
            u = np.clip((t-SCAN_END)/(JOIN-SCAN_END), 0, 1)
            trail_strength = np.sin(np.pi*u)
            previous = self.cursor_position(max(SCAN_END, t-.075)) * SCALE
        padding = 28
        x0 = int(np.floor(min(position[0]-rx, previous[0]-radius)-padding))
        y0 = int(np.floor(min(position[1]-ry, previous[1]-radius)-padding))
        x1 = int(np.ceil(max(position[0]+rx, previous[0]+radius)+padding))
        y1 = int(np.ceil(max(position[1]+ry, previous[1]+radius)+padding))
        quality = 4
        mask = Image.new('L', ((x1-x0)*quality, (y1-y0)*quality))
        draw = ImageDraw.Draw(mask)
        cx = (position[0]-x0)*quality; cy = (position[1]-y0)*quality
        if trail_strength > 1e-5:
            # A tapered sequence of round dabs reads as a detached ray of light,
            # and collapses back to the exact sphere at both handoffs.
            start = ((previous[0]-x0)*quality, (previous[1]-y0)*quality)
            end = ((position[0]-x0)*quality, (position[1]-y0)*quality)
            draw.line([start, end], fill=round(75*trail_strength),
                      width=max(1, round(radius*quality*.62)))
            draw.line([start, end], fill=round(185*trail_strength),
                      width=max(1, round(radius*quality*.14)))
        draw.ellipse((cx-rx*quality, cy-ry*quality,
                      cx+rx*quality, cy+ry*quality), fill=255)
        alpha = np.asarray(mask.resize((x1-x0, y1-y0), Image.Resampling.BOX),
                           dtype=np.float32) / 255
        halo = nd.gaussian_filter(alpha, 5.0) * (.18*shape + .24*trail_strength)
        combined = np.clip(alpha + halo, 0, 1)
        fg = np.array(foreground, dtype=np.float32)
        rgb = np.clip(alpha[:, :, None]*fg + halo[:, :, None]*255, 0, 255)
        rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.rint(rgb).astype(np.uint8)
        rgba[:, :, 3] = np.rint(combined*255).astype(np.uint8)
        return Image.fromarray(rgba, 'RGBA'), (x0, y0)

    def text_alpha(self, t):
        if t < SCAN_START:
            return self.old_word_alpha
        if t >= SCAN_END:
            return self.new_word_alpha
        head = self.cursor_position(t)[0] * SCALE
        xx = np.arange(SIZE[0], dtype=np.float32)[None, :]
        reveal = smooth01((xx - head + 34) / 68)
        return np.rint(self.old_word_alpha*(1-reveal) + self.new_word_alpha*reveal).astype(np.uint8)

    def pulse_layer(self, t, foreground):
        active = ramp(t, 1.18, .15) * (1-ramp(t, 1.30, .55))
        if active <= 0:
            return None, None
        center = OLD_CENTER * SCALE
        radius = (7 + 44*ramp(t, 1.18, .67))*SCALE
        padding = 8
        size = int(np.ceil((radius+padding)*2))
        quality = 4
        mask = Image.new('L', (size*quality, size*quality))
        draw = ImageDraw.Draw(mask)
        c = size*quality/2; r = radius*quality
        draw.ellipse((c-r, c-r, c+r, c+r), outline=round(70*active),
                     width=max(1, round(1.3*SCALE*quality)))
        alpha = mask.resize((size, size), Image.Resampling.BOX)
        layer = Image.new('RGBA', (size, size), tuple(foreground) + (0,))
        layer.putalpha(alpha)
        return layer, (round(center[0]-size/2), round(center[1]-size/2))

    def frame(self, t):
        if t < 1.18:
            return self.old_full.convert('RGB')
        background, foreground, _ = self.colors(t)
        frame = Image.new('RGBA', SIZE, background + (255,))

        text = Image.new('RGBA', SIZE, foreground + (0,))
        text.putalpha(Image.fromarray(self.text_alpha(t), 'L'))
        frame.alpha_composite(text)

        star, position = self.star_layer(t, foreground)
        frame.alpha_composite(star, position)

        pulse, pulse_position = self.pulse_layer(t, foreground)
        if pulse is not None:
            frame.alpha_composite(pulse, pulse_position)

        if DETACH <= t < JOIN:
            cursor, cursor_position = self.cursor_layer(t, foreground)
            frame.alpha_composite(cursor, cursor_position)

        finish = ramp(t, 7.65, .40)
        if finish > 0:
            frame = Image.blend(frame, self.exact_final, finish)
        return frame.convert('RGB')


def encode(frames, out, prefix):
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(FPS),
        '-i', str(frames / '%04d.png'), '-frames:v', str(FRAMES), '-t', str(DURATION),
        '-c:v', 'libx264', '-crf', '5', '-preset', 'slow', '-threads', '2',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an',
        str(out / f'{prefix}-4k.mp4')
    ], check=True)
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(FPS),
        '-i', str(frames / '%04d.png'), '-vf', 'scale=1920:1080:flags=lanczos',
        '-frames:v', str(FRAMES), '-t', str(DURATION), '-c:v', 'libx264', '-crf', '6',
        '-preset', 'slow', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-an', str(out / f'{prefix}-fullhd.mp4')
    ], check=True)


def render():
    animation = RebrandAnimation()
    frames = ROOT / f'.cache/{PREFIX}-4k-frames'
    frames.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'exports'
    for number in range(FRAMES):
        animation.frame(number / FPS).save(frames / f'{number:04d}.png', compress_level=2)
        if number % 30 == 0:
            print(f'Rebrand transition {number // 30}/{DURATION} seconds', flush=True)
    encode(frames, out, PREFIX)
    animation.frame((FRAMES-1)/FPS).save(out / f'{PREFIX}-final.png')
    info = {
        'duration': DURATION, 'fps': FPS, 'frames': FRAMES, 'size': list(SIZE),
        'audio': False, 'old_hold_end': 1.18, 'symbol_rebuild_end': DETACH,
        'cursor_detach': DETACH, 'text_conversion_start': SCAN_START,
        'text_conversion_end': SCAN_END, 'cursor_join': JOIN,
        'background_transition': [BG_START, BG_END],
        'symbol_turn': 'one smooth horizontal 360-degree Y-axis turn with 22% maximum foreshortening',
        'light_particle': 'one endpoint becomes a tapered light ray, rewrites the wordmark, and rejoins',
        'micro_motion': '1.8% rigid horizontal breathing during wordmark conversion',
        'exact_new_logo_hold_start': 8.05,
        'story': 'old identity contracts, unfolds in perspective, and is rewritten by one returning node'
    }
    (out / f'{PREFIX}-render-info.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    render()
