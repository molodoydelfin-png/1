"""Motion/return-cursor QA, original final image and decoded transparency."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_sweep import SweepAnimation,DETACH,OUT_END,RETURN,SCAN_END,JOIN,CURSOR_NODE,PREFIX
from render_clean import CleanAnimation

count=0
def check(name,condition):
    global count
    assert condition,name
    count+=1;print('PASS',name,flush=True)

a=SweepAnimation(ROOT/'assets/source.png')
approved=CleanAnimation(ROOT/'assets/source.png')
reference=np.asarray(approved.final)
check('final is pixel-identical to the approved clean logo, including white typography and cuts',
      np.array_equal(np.asarray(a.final),reference))
for time in [3.7,4.,4.5,149/30,5.]:assert np.array_equal(np.asarray(a.frame(time)),reference)
check('the entire final hold is fixed at the approved original pose',True)

times=np.linspace(.95,2.3,300)
deviations=np.array([a.scales(t)/a.base_growth(t)-1 for t in times])
check('near-opposite upper/lower rays grow and shrink in opposition',
      np.corrcoef(deviations[:,1],deviations[:,6])[0,1]<-.98)
values=np.array([a.scales(t) for t in np.linspace(0,4,801)])
check('wave is bounded, finite and smoothly settles into every original length',
      np.isfinite(values).all() and values.min()>=0 and values.max()<=1.161 and
      np.array_equal(a.scales(3.3),np.ones(10)))
check('sphere size follows shared deployment, not the ray-length oscillations',
      np.ptp(a.node_growth(1.8,a.scales(1.8)))==0 and a.node_growth(1.8,a.scales(1.8))[0]==1)
for time in np.linspace(.3,3.3,120):
    positions=a.center+(a.endpoints-a.center)*a.scales(time)[:,None]
    left,top,right,bottom=a.star_box
    assert np.min(positions[:,0]-a.radii)>left+2
    assert np.max(positions[:,0]+a.radii)<right-2
    assert np.min(positions[:,1]-a.radii)>top+2
    assert np.max(positions[:,1]+a.radii)<bottom-2
check('varying arms and spheres remain inside the anti-aliased star tile',True)

check('cursor launches from its own live node and returns to exactly its original endpoint',
      np.array_equal(a.cursor_position(DETACH),a.endpoint(CURSOR_NODE,DETACH)) and
      np.array_equal(a.cursor_position(JOIN),a.endpoints[CURSOR_NODE]))
epsilon=1e-4
incoming=(a.endpoint(CURSOR_NODE,DETACH)-a.endpoint(CURSOR_NODE,DETACH-epsilon))/epsilon
outgoing=(a.cursor_position(DETACH+epsilon)-a.cursor_position(DETACH))/epsilon
check('cursor inherits live node velocity without a launch hesitation',np.linalg.norm(incoming-outgoing)<.5)
for boundary in [OUT_END,RETURN,SCAN_END,JOIN]:
    v1=(a.cursor_position(boundary)-a.cursor_position(boundary-epsilon))/epsilon
    v2=(a.cursor_position(boundary+epsilon)-a.cursor_position(boundary))/epsilon
    assert np.linalg.norm(v1-v2)<1
check('turn, scan ending and reattachment are position/velocity continuous',True)
for boundary in [DETACH,JOIN]:
    sprite,x,y=a.cursor_sprite(boundary)
    expected,xx,yy=a.node_sprite(a.cursor_position(boundary),a.radii[CURSOR_NODE],1.)
    assert (x,y)==(xx,yy) and np.array_equal(sprite,expected)
check('cursor starts and ends with exactly the same sphere rasterizer and material',True)
check('the original node is absent only while its cursor is away',
      not a.skip_node(CURSOR_NODE,DETACH-.01) and a.skip_node(CURSOR_NODE,2.) and
      not a.skip_node(CURSOR_NODE,JOIN) and not any(a.skip_node(i,2.) for i in range(9)))

for time in [0.,1.8,2.05,RETURN]:assert not a.text_frame(time).any()
check('wordmark stays invisible until the cursor starts its return',True)
partial=a.text_frame(2.7)[:,:,3]
check('returning cursor reveals text from the right while the left stays unrevealed',
      partial[:,:partial.shape[1]//3].max()<.001 and partial[:,partial.shape[1]*2//3:].max()>.99)
check('text is fully restored before the cursor rejoins the sign',
      np.allclose(a.text_frame(JOIN)[:,:,3],a.text_alpha,atol=1e-6))

for frame in range(1,66):
    alpha=np.asarray(Image.open(ROOT/f'.cache/{PREFIX}-frames/{frame:04d}.png'))[:,:,3]
    assert nd.label(alpha>2,structure=np.ones((3,3)))[1]<=2
check('actual pre-text frames contain only the connected star and the one intended cursor',True)
for time in [1.7,2.1,2.7,3.4]:
    frame=np.asarray(Image.open(ROOT/f'.cache/{PREFIX}-frames/{round(time*30):04d}.png'))
    px,py=np.rint(a.cursor_position(time)).astype(int)
    assert frame[py-2:py+3,px-2:px+3,3].max()>230
check('actual rendered cursor follows its intended path at launch, turn and return',True)
check('saved final Full HD PNG remains identical to approved assets',
      np.array_equal(np.asarray(Image.open(ROOT/f'exports/{PREFIX}-final.png')),reference))

def probe(file):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
        'stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration',
        '-of','json',str(file)]))

for name,size in [(f'{PREFIX}-alpha.webm',(1920,1080)),(f'{PREFIX}-alpha.mov',(1920,1080)),
                  (f'{PREFIX}-4k-alpha.webm',(3840,2160))]:
    path=ROOT/'exports'/name
    info=probe(path);video=next(s for s in info['streams'] if s['codec_type']=='video')
    check(f'{name}: native size, 30 fps, 150 frames, five seconds',
          (video['width'],video['height'])==size and video['r_frame_rate']=='30/1' and
          video['nb_read_frames']=='150' and abs(float(info['format']['duration'])-5)<.02)
    decoder=['-c:v','libvpx-vp9'] if path.suffix=='.webm' else []
    base=['ffmpeg','-hide_banner','-loglevel','error',*decoder,'-i',str(path)]
    for moment in [2.4,4.5]:
        raw=subprocess.check_output(base+['-ss',str(moment),'-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'])
        decoded=np.frombuffer(raw,np.uint8).reshape(size[1],size[0],4)
        assert decoded[0,0,3]==0 and decoded[:,:,3].max()==255
        if moment==2.4:
            point=np.rint(a.cursor_position(moment)*(size[0]/1920)).astype(int)
            assert decoded[point[1]-2:point[1]+3,point[0]-2:point[0]+3,3].max()>220
        else:
            approved_ref=reference if size==(1920,1080) else np.asarray(Image.open(ROOT/'exports/s-project-2-clean-4k-final.png'))
            difference=np.abs(decoded[:,:,3].astype(np.int16)-approved_ref[:,:,3].astype(np.int16))
            assert difference.max()<=1 if path.suffix=='.webm' else difference.mean()<.1
    check(f'{name}: decoded active cursor and final pose have real, clean transparency',True)
    check(f'{name}: quiet audio is included',any(s['codec_type']=='audio' for s in info['streams']))
    assert path.stat().st_size<90*1024**2

for prefix,size in [(PREFIX,(1920,1080)),(PREFIX+'-4k',(3840,2160))]:
    info=probe(ROOT/f'exports/{prefix}-preview.mp4')
    video=next(s for s in info['streams'] if s['codec_type']=='video')
    check(f'{prefix}: opaque H.264 preview is the stated native size',
          video['codec_name']=='h264' and video['pix_fmt']=='yuv420p' and (video['width'],video['height'])==size)
print(f'\n{count} sweep/cursor checks passed.')
