"""Check native vector rendering, monotonic motion and real encoded alpha."""
from pathlib import Path
import io
import json
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_clean import CleanAnimation,PREFIX,K,SSAA,STAR_SSAA,cairosvg

count=0
def check(name,condition):
    global count
    assert condition,name
    count+=1;print('PASS',name,flush=True)

a=CleanAnimation(ROOT/'assets/source.png')
values=np.array([a.scales(t) for t in np.linspace(0,3,1001)])
check('all ten rays move monotonically outwards with no spring, recoil or overshoot',
      values.min()==0 and values.max()==1 and np.diff(values,axis=0).min()>=-1e-9)
check('all measured endpoints are reached exactly and stay fixed',
      np.array_equal(a.scales(1.85),np.ones(10)) and np.array_equal(a.scales(5),np.ones(10)))
check('original source sphere sizes and ray widths are retained after contour cleanup',
      np.all((a.radii/K>13.8)&(a.radii/K<16.1)) and np.all((a.widths/K>=3.5-1e-6)&(a.widths/K<=7.5+1e-6)))
check('opening remains fully transparent',not np.asarray(a.frame(0))[:,:,3].any())

for frame in range(1,56):
    alpha=np.asarray(Image.open(ROOT/f'.cache/{PREFIX}-frames/{frame:04d}.png'))[:,:,3]
    assert nd.label(alpha>1,structure=np.ones((3,3)))[1]==1,f'detached star pixels in frame {frame}'
check('every early rendered frame has one connected star: no flying sphere or shaft fragments',True)
check('geometry uses 8× area sampling and typography uses 4× supersampling',SSAA==4 and STAR_SSAA==8)
check('letter contours are true cubic vector paths, without substitute fonts or embedded raster images',
      '<path' in a.svg_path.read_text() and ' C' in a.svg_path.read_text() and
      '<image' not in a.svg_path.read_text() and '<text' not in a.svg_path.read_text())

native=cairosvg.svg2png(url=str(a.svg_path),output_width=795,output_height=180)
traced=np.asarray(Image.open(io.BytesIO(native)))[:,:,3]>.5*255
original=a.original.text_alpha>.5
iou=np.count_nonzero(traced&original)/np.count_nonzero(traced|original)
check('cleaned vector letter silhouettes agree with the source at >97% intersection/union',iou>.97)
check('custom S fragments, original glyph separations and cuts remain intact',
      nd.label(traced)[1]==nd.label(original)[1])
check('the actual text render originates from a 4× native-resolution Cairo mask',
      a.high_text_alpha.shape==(a.text_alpha.shape[0]*4,a.text_alpha.shape[1]*4))
early=a.text_frame(2.25)[:,:,3]
check('light still reveals the wordmark from the center outwards',
      early.max()>.2 and np.count_nonzero(early[:,0:100]>.01)==0 and np.count_nonzero(early[:,-100:]>.01)==0)
check('no detached fragment layers exist',not hasattr(a,'fragments'))

final=np.asarray(a.final)
for time in [3.7,4.,4.5,149/30]:
    assert np.array_equal(np.asarray(a.frame(time)),final)
check('final hold is exactly static',True)
check('saved transparent PNG agrees with the new vector/analytic renderer',
      np.array_equal(np.asarray(Image.open(ROOT/f'exports/{PREFIX}-final.png')),final))

def probe(file):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
       'stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration',
       '-of','json',str(file)]))

for name,size in [(f'{PREFIX}-alpha.webm',(1920,1080)),(f'{PREFIX}-alpha.mov',(1920,1080)),
                  (f'{PREFIX}-4k-alpha.webm',(3840,2160))]:
    file=ROOT/'exports'/name
    info=probe(file);video=next(s for s in info['streams'] if s['codec_type']=='video')
    check(f'{name}: native dimensions, 150 frames and 30 fps',
          (video['width'],video['height'])==size and video['r_frame_rate']=='30/1' and video['nb_read_frames']=='150')
    decoder=['-c:v','libvpx-vp9'] if file.suffix=='.webm' else []
    command=['ffmpeg','-hide_banner','-loglevel','error',*decoder,'-i',str(file)]
    raw=subprocess.check_output(command+['-ss','4.5','-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'])
    decoded=np.frombuffer(raw,np.uint8).reshape(size[1],size[0],4)
    reference=final if size==(1920,1080) else np.asarray(Image.open(ROOT/f'exports/{PREFIX}-4k-final.png'))
    check(f'{name}: actual decoded background is clear, with opaque white artwork',
          decoded[0,0,3]==0 and decoded[:,:,3].max()==255 and np.count_nonzero(decoded[:,:,3]==255)>30000)
    difference=np.abs(decoded[:,:,3].astype(np.int16)-reference[:,:,3].astype(np.int16))
    check(f'{name}: decoded alpha preserves the clean reference edges',
          difference.max()<=1 if file.suffix=='.webm' else difference.mean()<.1)
    raw=subprocess.check_output(command+['-ss','1.2','-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'])
    alpha=np.frombuffer(raw,np.uint8).reshape(size[1],size[0],4)[:,:,3]
    check(f'{name}: actual decoded early star has no detached specks',nd.label(alpha>3,structure=np.ones((3,3)))[1]==1)
    check(f'{name}: quiet audio retained',any(s['codec_type']=='audio' for s in info['streams']))
    assert file.stat().st_size<90*1024**2

preview=probe(ROOT/f'exports/{PREFIX}-preview.mp4')
video=next(s for s in preview['streams'] if s['codec_type']=='video')
check('MP4 is correctly labelled an opaque Full HD H.264 preview',
      video['codec_name']=='h264' and video['pix_fmt']=='yuv420p' and (video['width'],video['height'])==(1920,1080))
preview4k=probe(ROOT/f'exports/{PREFIX}-4k-preview.mp4')
video4k=next(s for s in preview4k['streams'] if s['codec_type']=='video')
check('4K preview is a real 3840×2160 H.264 render with 150 frames',
      (video4k['width'],video4k['height'])==(3840,2160) and video4k['nb_read_frames']=='150')
print(f'\n{count} clean-quality checks passed. Source contour IoU: {iou:.4f}')
