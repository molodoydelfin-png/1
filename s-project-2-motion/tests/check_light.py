"""Validate actual alpha after decoding both transparent video masters."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_light import LightAnimation, PREFIX
from render import ENDPOINTS, CENTER, CROP_TOP

count=0
def check(name,condition):
    global count
    assert condition,name
    count+=1;print('PASS',name,flush=True)

a=LightAnimation(ROOT/'assets/source.png')
check('transparent opening has no painted background or unrequested objects',not np.asarray(a.frame(0))[:,:,3].any())
source_dark=np.asarray(a.source).mean(2)<210
allowed=nd.binary_dilation(source_dark,iterations=3)
check('final matte excludes all raster background, including pale noisy islands',
      not ((a.alpha>.5/255)&~allowed).any())

waves=np.array([a.scales(t) for t in np.linspace(.75,2.1,80)])
check('length maximum travels through all ten directions, rather than uniform zooming',
      len(set(np.argmax(waves,axis=1)))==10 and np.max(np.ptp(waves,axis=1))>.6)
check('all original lengths settle exactly by 2.5 s',np.array_equal(a.scales(2.5),np.ones(10)))
for time in [.95,1.35,1.75,2.15]:
    pixels=np.asarray(a.frame(time))
    for endpoint,scale in zip(ENDPOINTS,a.scales(time)):
        point=CENTER+(endpoint-CENTER)*scale
        x=round(float(point[0])*1920/1448)
        y=round((float(point[1])-CROP_TOP)*1920/1448)
        assert pixels[y-2:y+3,x-2:x+3,3].max()>220
check('actual rendered spheres remain on their original angles while their lengths vary',True)

final=np.asarray(a.final)
for time in [3.7,4.,4.5,149/30,5.]:
    assert np.array_equal(np.asarray(a.frame(time)),final)
check('all light and fragment effects are gone during a perfectly static final hold',True)
check('eight original-glyph fragments detach and all disappear before the final hold',
      len(a.fragments)==8 and max(f['birth']+f['life'] for f in a.fragments)<3.7)
tx,ty,_,_=__import__('render_light').TEXT_BOX
height,width=a.text_alpha.shape
check('custom S and PROJECT silhouettes, hyphen and cuts are preserved in the final alpha',
      np.array_equal(a.final_native[ty:ty+height,tx:tx+width,3],a.text_alpha))
glyph=a.final_native[ty:ty+height,tx:tx+width]
solid=glyph[:,:,3]>.99
check('settled typography is pure white with no baked-in black matte',
      np.all(glyph[:,:,:3][solid]==glyph[:,:,3][solid,None]))
check('saved final PNG is RGBA and matches the renderer',
      Image.open(ROOT/f'exports/{PREFIX}-final.png').mode=='RGBA' and
      np.array_equal(np.asarray(Image.open(ROOT/f'exports/{PREFIX}-final.png')),final))

def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
      'stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration',
      '-of','json',str(path)]))

for extension in ['webm','mov']:
    path=ROOT/f'exports/{PREFIX}-alpha.{extension}'
    info=probe(path)
    video=next(s for s in info['streams'] if s['codec_type']=='video')
    check(f'{extension}: 1920×1080, 30 fps, 150 frames, five seconds of video',
          (video['width'],video['height'])==(1920,1080) and video['r_frame_rate']=='30/1' and
          video['nb_read_frames']=='150' and abs(float(info['format']['duration'])-5)<.02)
    decoder=['-c:v','libvpx-vp9'] if extension=='webm' else []
    raw=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error',*decoder,'-i',str(path),
                                 '-ss','4.5','-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'])
    decoded=np.frombuffer(raw,dtype=np.uint8).reshape(1080,1920,4)
    alpha=decoded[:,:,3]
    check(f'{extension}: decoded alpha is truly transparent outside the logo and opaque inside',
          alpha.min()==0 and alpha.max()==255 and alpha[0,0]==0 and np.count_nonzero(alpha==255)>30000)
    region=(alpha>0)|(final[:,:,3]>0)
    difference=np.abs(alpha.astype(np.int16)-final[:,:,3].astype(np.int16))
    check(f'{extension}: final decoded silhouette agrees with the transparent PNG',float(difference[region].mean())<3)
    check(f'{extension}: includes the original quiet soundtrack',any(s['codec_type']=='audio' for s in info['streams']))
    if extension=='mov':
        check('editing master is ProRes 4444 with alpha and below GitHub file-size limit',
              video['codec_name']=='prores' and video['profile']=='4444' and 'a' in video['pix_fmt'] and path.stat().st_size<90*1024**2)

preview=probe(ROOT/f'exports/{PREFIX}-preview.mp4')
video=next(s for s in preview['streams'] if s['codec_type']=='video')
check('MP4 is explicitly an opaque H.264 preview, not a transparency claim',
      video['codec_name']=='h264' and video['pix_fmt']=='yuv420p' and video['nb_read_frames']=='150')
print(f'\n{count} white/transparent edition checks passed.')
