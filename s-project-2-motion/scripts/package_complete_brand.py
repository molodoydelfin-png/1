"""Build the complete static and motion delivery kit for the approved S-PROJECT logo."""
from copy import deepcopy
from pathlib import Path
import hashlib
import io
import shutil
import sys
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / '.cache/vector-libs'))
import cairosvg
DEST = ROOT / 'brand-assets' / 'S-PROJECT-COMPLETE'
ARCHIVE = ROOT.parent / 'S-PROJECT-brand-kit-8K-smooth.zip'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
CANVAS = (7680, 4320)
SCALE = CANVAS[0] / 1448
CROP_TOP = 112.25
PADDING = 96
DARK = (20, 23, 28, 255)


def select_part(source, part):
    root = deepcopy(source)
    for child in list(root):
        tag = child.tag.rsplit('}', 1)[-1]
        keep = ((tag == 'defs' and part != 'wordmark') or
                (tag == 'g' and part != 'symbol') or
                (tag in ('line', 'circle') and part != 'wordmark'))
        if not keep:
            root.remove(child)
    return root


def recolor_svg(root, color):
    if color == 'white':
        return
    for element in root.iter():
        tag = element.tag.rsplit('}', 1)[-1]
        if tag == 'line':
            element.set('stroke', '#000000')
        elif tag in ('circle', 'path'):
            element.set('fill', '#000000')
    for defs in root.findall(f'{{{NS}}}defs'):
        root.remove(defs)


def save_svg(source, box, destination, part, color):
    root = select_part(source, part)
    recolor_svg(root, color)
    x0, y0, x1, y1 = box
    root.set('width', str(x1 - x0))
    root.set('height', str(y1 - y0))
    root.set('viewBox',
             f'{x0 / SCALE:.9f} {y0 / SCALE + CROP_TOP:.9f} '
             f'{(x1 - x0) / SCALE:.9f} {(y1 - y0) / SCALE:.9f}')
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)

    parsed = ET.parse(destination).getroot()
    if part == 'symbol':
        assert len(parsed.findall(f'{{{NS}}}line')) == 10
        assert len(parsed.findall(f'{{{NS}}}circle')) == 10
        assert not parsed.findall(f'{{{NS}}}g')
    elif part == 'wordmark':
        assert not parsed.findall(f'{{{NS}}}line')
        assert not parsed.findall(f'{{{NS}}}circle')
        assert len(list(parsed.iter(f'{{{NS}}}path'))) == 11


def black_version(array):
    result = np.zeros_like(array)
    result[:, :, 3] = array[:, :, 3]
    return result


def on_background(image, color):
    background = Image.new('RGBA', image.size, color)
    return Image.alpha_composite(background, image).convert('RGB')


def padded_box(image):
    bbox = image.getchannel('A').getbbox()
    assert bbox is not None
    return (max(0, bbox[0] - PADDING), max(0, bbox[1] - PADDING),
            min(CANVAS[0], bbox[2] + PADDING), min(CANVAS[1], bbox[3] + PADDING))


def save_element(folder, name, part, white_array, vector):
    out = DEST / folder
    out.mkdir(parents=True, exist_ok=True)
    white_canvas = Image.fromarray(white_array, 'RGBA')
    black_canvas = Image.fromarray(black_version(white_array), 'RGBA')
    box = padded_box(white_canvas)
    white = white_canvas.crop(box)
    black = black_canvas.crop(box)

    white.save(out / f'{name}-white-transparent.png', dpi=(300, 300))
    black.save(out / f'{name}-black-transparent.png', dpi=(300, 300))
    white_canvas.save(out / f'{name}-white-8k-transparent.png', dpi=(300, 300))
    black_canvas.save(out / f'{name}-black-8k-transparent.png', dpi=(300, 300))
    on_background(white, DARK).save(out / f'{name}-white-on-dark.png', dpi=(300, 300))
    on_background(black, (255, 255, 255, 255)).save(
        out / f'{name}-black-on-white.png', dpi=(300, 300))
    save_svg(vector, box, out / f'{name}-white-transparent.svg', part, 'white')
    save_svg(vector, box, out / f'{name}-black-transparent.svg', part, 'black')

    for path in [out / f'{name}-white-transparent.png',
                 out / f'{name}-black-transparent.png',
                 out / f'{name}-white-8k-transparent.png',
                 out / f'{name}-black-8k-transparent.png']:
        saved = Image.open(path).convert('RGBA')
        assert saved.getchannel('A').getbbox() is not None
        assert saved.getpixel((0, 0))[3] == 0
    assert np.array_equal(np.asarray(white)[:, :, 3], np.asarray(black)[:, :, 3])
    print(f'{folder}: white + black, transparent PNG/SVG, 8K canvas and backgrounds')
    return white, black


def make_preview(elements):
    preview = Image.new('RGB', (1800, 1320), '#d9dadd')
    draw = ImageDraw.Draw(preview)
    draw.rectangle((40, 40, 1760, 610), fill='#14171c')
    draw.rectangle((40, 670, 1760, 1280), fill='white')
    draw.text((75, 65), 'S-PROJECT / WHITE', fill='#a9afb6')
    draw.text((75, 695), 'S-PROJECT / BLACK', fill='#62666b')
    for index, (_, white, black) in enumerate(elements):
        maximum = (1450, 235) if index == 0 else (700, 210)
        column = 0 if index != 2 else 1
        white_thumb = white.copy()
        black_thumb = black.copy()
        white_thumb.thumbnail(maximum, Image.Resampling.LANCZOS)
        black_thumb.thumbnail(maximum, Image.Resampling.LANCZOS)
        if index == 0:
            xw = (1800 - white_thumb.width) // 2
            xb = (1800 - black_thumb.width) // 2
            yw, yb = 150, 780
        else:
            center = 480 if column == 0 else 1320
            xw = center - white_thumb.width // 2
            xb = center - black_thumb.width // 2
            yw, yb = 390, 1040
        preview.paste(white_thumb, (xw, yw), white_thumb.getchannel('A'))
        preview.paste(black_thumb, (xb, yb), black_thumb.getchannel('A'))
    preview.save(DEST / 'PREVIEW.png')


def write_readme():
    (DEST / 'README-RU.txt').write_text('''S-PROJECT — полный комплект логотипа и анимации

01_full-logo — полный логотип.
02_symbol — графический знак: десять лучей и сфер.
03_wordmark — оригинальная надпись S-PROJECT отдельно.
04_motion — финальная анимация с вращением по X-оси.
05_source — исходное изображение, на основе которого собран комплект.

В папках 01–03 находятся одинаковые наборы:
  *-white-transparent.png — белая версия без подложки;
  *-black-transparent.png — чёрная версия без подложки;
  *-white-8k-transparent.png и *-black-8k-transparent.png — те же элементы
    на прозрачном холсте 7680×4320 в точном положении финальной композиции;
  *-white-transparent.svg и *-black-transparent.svg — векторные файлы без
    фона, которые можно увеличивать без потери качества;
  *-white-on-dark.png и *-black-on-white.png — готовые версии с подложкой.

PNG и SVG с пометкой transparent действительно не имеют фона. Белое лого
может казаться пустым в просмотрщике с белым фоном — откройте PREVIEW.png
или поместите файл поверх тёмного изображения.

04_motion:
  logo-4k-alpha.mov — основной мастер ProRes 4444, прозрачный фон;
  logo-4k-alpha.webm — облегчённый 4K-мастер с прозрачностью;
  preview-4k.mp4 — версия на тёмном фоне только для просмотра;
  sound-design.wav — звук отдельно.

Финальный кадр анимации совпадает с утверждённым логотипом. Для монтажа
используйте MOV; для сайта и быстрых пересылок удобнее WebM. MP4 содержит
тёмную подложку и предназначен только для предпросмотра.
''', encoding='utf-8')


def write_manifest():
    lines = ['S-PROJECT complete brand kit', 'SHA-256:']
    for file in sorted(DEST.rglob('*')):
        if file.is_file() and file.name != 'MANIFEST-SHA256.txt':
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            lines.append(f'{digest}  {file.relative_to(DEST)}')
    (DEST / 'MANIFEST-SHA256.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    vector_path = ROOT / 'exports/s-project-2-clean-4k-final.svg'
    rendered = cairosvg.svg2png(url=str(vector_path), output_width=CANVAS[0],
                               output_height=CANVAS[1])
    source = Image.open(io.BytesIO(rendered)).convert('RGBA')
    pixels = np.asarray(source)
    vector = ET.parse(vector_path).getroot()

    # Native artwork has an empty gap around x=518 between symbol and wordmark.
    split = round(518.35 * SCALE)
    symbol = pixels.copy()
    symbol[:, split:] = 0
    wordmark = pixels.copy()
    wordmark[:, :split] = 0
    assert np.array_equal(np.maximum(symbol[:, :, 3], wordmark[:, :, 3]), pixels[:, :, 3])
    assert not np.logical_and(symbol[:, :, 3] > 0, wordmark[:, :, 3] > 0).any()

    configured = [
        ('01_full-logo', 'logo', 'full', pixels),
        ('02_symbol', 'symbol', 'symbol', symbol),
        ('03_wordmark', 'wordmark', 'wordmark', wordmark),
    ]
    previews = []
    for folder, name, part, array in configured:
        white, black = save_element(folder, name, part, array, vector)
        previews.append((name, white, black))

    motion = DEST / '04_motion'
    motion.mkdir()
    motion_files = {
        'logo-4k-alpha.mov': 'exports/s-project-2-xaxis-4k-alpha.mov',
        'logo-4k-alpha.webm': 'exports/s-project-2-xaxis-4k-alpha.webm',
        'preview-4k.mp4': 'exports/s-project-2-xaxis-4k-preview.mp4',
        'sound-design.wav': 'assets/sound-design.wav',
    }
    for name, origin in motion_files.items():
        shutil.copy2(ROOT / origin, motion / name)
        assert (motion / name).stat().st_size == (ROOT / origin).stat().st_size

    source_folder = DEST / '05_source'
    source_folder.mkdir()
    shutil.copy2(ROOT / 'assets/source.png', source_folder / 'source-reference.png')

    make_preview(previews)
    write_readme()
    write_manifest()

    with ZipFile(ARCHIVE, 'w', ZIP_DEFLATED, compresslevel=9) as output:
        for file in sorted(DEST.rglob('*')):
            if file.is_file():
                output.write(file, arcname=str(file.relative_to(DEST.parent)))
    with ZipFile(ARCHIVE) as output:
        assert output.testzip() is None
        assert len(output.namelist()) == 32
        assert all(not name.startswith('/') and '..' not in Path(name).parts
                   for name in output.namelist())
    print(f'Validated complete kit: {len(list(DEST.rglob("*")))} entries')
    print(f'ZIP: {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
