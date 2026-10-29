"""Package approved brand elements as separate transparent PNG/SVG files."""
from pathlib import Path
from copy import deepcopy
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image,ImageDraw,ImageFont
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'brand-assets'/'S-PROJECT'
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
SCALE=3840/1448
CROP_TOP=112.25
PADDING=24


def cropped_svg(source,box,destination,part):
    root=deepcopy(source)
    for child in list(root):
        tag=child.tag.rsplit('}',1)[-1]
        keep=(tag=='defs' and part!='wordmark') or (tag=='g' and part!='symbol') or (tag in ['line','circle'] and part!='wordmark')
        if not keep:root.remove(child)
    x0,y0,x1,y1=box
    root.set('width',str(x1-x0));root.set('height',str(y1-y0))
    root.set('viewBox',f'{x0/SCALE:.9f} {y0/SCALE+CROP_TOP:.9f} {(x1-x0)/SCALE:.9f} {(y1-y0)/SCALE:.9f}')
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)
    if part=='symbol':
        assert len(root.findall(f'{{{NS}}}line'))==10
        assert len(root.findall(f'{{{NS}}}circle'))==10
        assert not root.findall(f'{{{NS}}}g')
    if part=='wordmark':
        assert not root.findall(f'{{{NS}}}line')
        assert not root.findall(f'{{{NS}}}circle')
        assert len(list(root.iter(f'{{{NS}}}path')))>0


def main():
    source=Image.open(ROOT/'exports/s-project-2-clean-4k-final.png').convert('RGBA')
    pixels=np.array(source)
    vector=ET.parse(ROOT/'exports/s-project-2-clean-4k-final.svg').getroot()
    split=round(520*SCALE)
    symbol=pixels.copy();symbol[:,split:]=0
    wordmark=pixels.copy();wordmark[:,:split]=0
    assert np.array_equal(np.maximum(symbol[:,:,3],wordmark[:,:,3]),pixels[:,:,3])
    merged=Image.alpha_composite(Image.fromarray(symbol),Image.fromarray(wordmark))
    visible=pixels[:,:,3]>0
    assert np.array_equal(np.asarray(merged)[visible],pixels[visible])

    variants=[('01_wordmark','wordmark',wordmark),('02_symbol','symbol',symbol),('03_full-logo','logo',pixels)]
    thumbs=[]
    for folder,name,array in variants:
        out=DEST/folder;out.mkdir(parents=True,exist_ok=True)
        image=Image.fromarray(array)
        bbox=image.getchannel('A').getbbox()
        assert bbox is not None
        box=(max(0,bbox[0]-PADDING),max(0,bbox[1]-PADDING),min(3840,bbox[2]+PADDING),min(2160,bbox[3]+PADDING))
        cropped=image.crop(box)
        cropped.save(out/f'{name}.png',dpi=(300,300))
        image.save(out/f'{name}-4k-canvas.png')
        cropped_svg(vector,box,out/f'{name}.svg','full' if name=='logo' else name)
        # Preserve the exact approved pixels. Cropping never resizes or redraws.
        assert np.array_equal(np.asarray(cropped),array[box[1]:box[3],box[0]:box[2]])
        assert cropped.getpixel((0,0))[3]==0
        assert Image.open(out/f'{name}.png').mode=='RGBA'
        thumbs.append((folder,cropped))
        print(f'{folder}: {cropped.width}×{cropped.height} PNG + SVG + aligned 4K canvas')

    video=DEST/'04_animation';video.mkdir(exist_ok=True)
    files={
        'logo-fullhd-alpha.mov':'exports/s-project-2-clean-alpha.mov',
        'logo-4k-alpha.webm':'exports/s-project-2-clean-4k-alpha.webm',
        'preview-4k.mp4':'exports/s-project-2-clean-4k-preview.mp4',
        'preview-fullhd.mp4':'exports/s-project-2-clean-preview.mp4',
        'sound-design.wav':'assets/sound-design.wav',
    }
    for name,origin in files.items():
        shutil.copy2(ROOT/origin,video/name)
        assert (video/name).stat().st_size==(ROOT/origin).stat().st_size

    preview=Image.new('RGB',(1600,1080),'#14171c')
    draw=ImageDraw.Draw(preview)
    for i,(name,image) in enumerate(thumbs):
        thumb=image.copy();thumb.thumbnail((1280,270),Image.Resampling.LANCZOS)
        x=(1600-thumb.width)//2;y=i*350+55+(270-thumb.height)//2
        preview.paste(thumb,(x,y),thumb.getchannel('A'))
        draw.text((45,i*350+20),name,fill='#a9afb6')
    preview.save(DEST/'PREVIEW-dark.png')
    (DEST/'README-RU.txt').write_text('''S-PROJECT — комплект утверждённого логотипа

01_wordmark — надпись S-PROJECT отдельно, с оригинальной буквой S и надрезами.
02_symbol — звёздочка отдельно: десять лучей и сфер.
03_full-logo — полный логотип.
04_animation — готовая общая заставка: прозрачный MOV Full HD, прозрачный
WebM 4K, предпросмотры MP4 в 4K/Full HD и отдельный звук WAV.

В каждой из первых трёх папок:
  *.png — прозрачная картинка, аккуратно обрезанная с небольшим полем;
  *.svg — самостоятельный векторный файл, масштабируется без пикселизации;
  *-4k-canvas.png — прозрачный холст 3840×2160 с исходным расположением.

Надпись и знак на файлах *-4k-canvas.png можно положить друг на друга
без перемещения: они точно соберутся в утверждённую композицию.
Обрезанные PNG не растягивались: сохранены пиксели последнего 4K-рендера.

Элементы белые/светлые. На белом фоне они могут быть не видны.
Для удобного просмотра открыт PREVIEW-dark.png. Тёмный фон есть только
на этом обзорном листе и в видео preview-*.mp4.

Для наложения анимации на своё видео используйте MOV или WebM,
а не MP4-предпросмотр. Все статичные PNG/SVG имеют прозрачный фон.
Белые буквы — готовые контуры; устанавливать шрифт не требуется.
''',encoding='utf-8')
    archive=ROOT.parent/'S-PROJECT-brand-kit.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=9) as output:
        for file in sorted(DEST.rglob('*')):
            if file.is_file():output.write(file,arcname=str(file.relative_to(DEST.parent)))
    with ZipFile(archive) as output:
        assert output.testzip() is None
        assert len(output.namelist())==16
    print(f'Validated ZIP: {archive}, {archive.stat().st_size} bytes')


if __name__=='__main__':main()
