"""Checks actual pixel output, measured geometry, timeline, audio and media."""
from pathlib import Path
import json
import subprocess
import sys
import wave
import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from render import OriginalPixelAnimation, ray_progress, ENDPOINTS, CENTER, ROI, CROP_TOP

checks = 0


def check(name, condition):
    global checks
    assert condition, name
    checks += 1
    print('PASS',name,flush=True)


a = OriginalPixelAnimation(ROOT/'assets/source.png')
native = np.asarray(a.source)
# Measure the actual spherical interiors; ignore the central ten-way crossing.
mask = native[330:720,175:520].mean(axis=2)<180
distance = nd.distance_transform_edt(mask)
peaks = (distance==nd.maximum_filter(distance,size=19)) & (distance>9)
labels, count = nd.label(peaks)
measured = []
for label in range(1,count+1):
    yy,xx = np.where(labels==label)
    point=np.array([xx.mean()+175,yy.mean()+330])
    if np.linalg.norm(point-CENTER)>25:
        measured.append(point)
check('10 original spheres are measured and all endpoint centers match within 1 source pixel',
      len(measured)==10 and max(min(np.linalg.norm(p-q) for q in ENDPOINTS) for p in measured)<1)

for t in np.linspace(0,3,301):
    state=a.state(t)
    assert np.ptp(state['rays'])==0 and np.ptp(state['spheres'])==0
check('all ten directions use the same spring and visibility; no circular sequence',True)
scales=np.array([ray_progress(t) for t in np.linspace(.35,2.25,1001)])
check('restrained outward-and-back motion: 3–5% overshoot, slight recoil, exact final size',
      1.03<scales.max()<1.05 and scales[scales.argmax()+1:].min()<1 and scales[-1]==1)
# Check the actual rendered positions, not just timeline metadata: every sphere
# and its attached ray must already exist together at a common partial radius.
t=.65
frame=np.asarray(a.frame(t))
growth=ray_progress(t)
for endpoint in ENDPOINTS:
    for fraction in [1.,.5]:
        point=CENTER+(endpoint-CENTER)*growth*fraction
        x=round(float(point[0])*1920/1448)
        y=round((float(point[1])-CROP_TOP)*1920/1448)
        assert frame[y-2:y+3,x-2:x+3].mean(axis=2).min()<225
check('rendered early frame has all 10 spheres and attached rays expanding together',True)
check('whole S and whole PROJECT fade smoothly; no writing, per-letter wipe or text translation',
      not hasattr(a,'s_clock') and not hasattr(a,'project_clock') and
      0<a.state(2.8)['s']<1 and 0<a.state(2.8)['project']<1 and
      a.state(2.8)['hyphen']==a.state(2.8)['project'] and a.state(3.55)['project']==1)
check('moving artwork cannot paint white patches: isolated signed pixel layers only darken the clean background',
      a.ink.max()<=0 and a.star.max()<=0 and np.isfinite(a.ink).all())
check('opening is a clean pale field with no hidden-letter silhouettes',np.asarray(a.frame(0)).min()>230)
blank=np.asarray(a.frame(0)).astype(np.int16)
for t in [.55,.65,.9,1.15,1.8,2.8,3.1]:
    pixels=np.asarray(a.frame(t)).astype(np.int16)
    dark=(blank-pixels).mean(axis=2)>10
    bright=(pixels-blank).max(axis=2)>3
    assert not (bright & ~nd.binary_dilation(dark,iterations=4)).any()
check('actual intermediate frames contain no detached white patches around the moving sign or fading text',True)

reference=np.asarray(a.reference)
for t in [3.7,4.,4.5,149/30,5.]:
    assert np.array_equal(np.asarray(a.frame(t)),reference)
check('entire final hold matches the uniformly reframed original PNG exactly',True)
near_final=np.asarray(a.frame(3.699)).astype(np.int16)
check('no final-frame pop: 3.699 s differs from the reference by at most 1 RGB level',
      np.abs(near_final-reference.astype(np.int16)).max()<=1)
check('final exported PNG equals the reframed original, including all original type cuts',
      np.array_equal(np.asarray(Image.open(ROOT/'exports/s-project-2-final.png')),reference))

check('center remains anchored: no rotation or drift; exact settled size from 2.25 s onward',
      ray_progress(2.25)==ray_progress(3)==1 and a.settle(1)==a.settle(3)==(0.,0.))
check('deterministic pixel render on repeated seeks',
      a.frame(1.57).tobytes()==a.frame(1.57).tobytes())

with wave.open(str(ROOT/'assets/sound-design.wav'),'rb') as wave_file:
    audio=np.frombuffer(wave_file.readframes(wave_file.getnframes()),dtype='<i2')
    check('original stereo sound: 48 kHz, exactly 5 s, no clipping, gentle ending',
          wave_file.getframerate()==48000 and wave_file.getnchannels()==2 and wave_file.getnframes()==240000 and
          np.abs(audio.astype(np.int32)).max()<32767 and np.count_nonzero(audio[-24000:])==0)

media=ROOT/'exports/s-project-2-radial.mp4'
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
   'stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels:format=duration',
   '-of','json',str(media)]))
video=next(s for s in info['streams'] if s['codec_type']=='video')
sound=next(s for s in info['streams'] if s['codec_type']=='audio')
check('encoded MP4: H.264, 1920×1080, 30 fps, 150 frames, exactly 5 s',
      video['codec_name']=='h264' and (video['width'],video['height'])==(1920,1080) and
      video['r_frame_rate']=='30/1' and video['nb_read_frames']=='150' and float(info['format']['duration'])==5.)
check('encoded MP4 has stereo AAC audio at 48 kHz',
      sound['codec_name']=='aac' and sound['channels']==2 and sound['sample_rate']=='48000')
silent=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type',
    '-of','json',str(ROOT/'exports/s-project-2-radial-silent.mp4')]))
check('silent alternative has only its video stream',len(silent['streams'])==1 and silent['streams'][0]['codec_type']=='video')
print(f'\n{checks} checks passed.')
