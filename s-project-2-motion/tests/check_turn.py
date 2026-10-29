"""Actual angular motion, smooth handoffs and decoded transparent masters."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from render_turn import TurnAnimation,PREFIX
from render_clean import CleanAnimation
from render_sweep import DETACH,RETURN,JOIN,CURSOR_NODE

count=0
def check(name,value):
    global count
    assert value,name
    count+=1;print('PASS',name,flush=True)

a=TurnAnimation(ROOT/'assets/source.png')
approved=CleanAnimation(ROOT/'assets/source.png')
reference=np.asarray(approved.final)
angles=np.array([a.angle(t) for t in np.linspace(0,4,1001)])
check('symbol makes one genuine eased 360-degree turn',
      angles[0]==0 and np.isclose(angles[-1],2*np.pi) and np.diff(angles).min()>-1e-9 and
      np.pi/2<a.angle(1.8)<3*np.pi/2)
for t in [1.,1.7,2.2,2.7]:
    g=a.scales(t);positions=a.node_positions(t,g)
    original=(a.endpoints-a.center)*g[:,None]
    rotated=positions-a.center
    signed=np.arctan2(original[:,0]*rotated[:,1]-original[:,1]*rotated[:,0],np.sum(original*rotated,axis=1))
    expected=np.angle(np.exp(1j*a.angle(t)))
    assert np.max(np.abs(signed-expected))<1e-6
check('actual endpoint geometry rotates around the fixed hub; it is not only length pulsation',True)
for t in np.linspace(.3,3.5,140):
    positions=a.node_positions(t,a.scales(t));left,top,right,bottom=a.star_box
    assert np.min(positions[:,0]-a.radii)>left+2 and np.max(positions[:,0]+a.radii)<right-2
    assert np.min(positions[:,1]-a.radii)>top+2 and np.max(positions[:,1]+a.radii)<bottom-2
check('full rotating silhouette and antialiasing margins fit the enlarged tile',True)
check('gentle ray-length wave resolves into exact original lengths and orientation before cursor join',
      np.array_equal(a.scales(3.3),np.ones(10)) and
      np.array_equal(a.node_positions(3.3,a.scales(3.3)),a.endpoints))

for t in [2.2,2.4,2.6,2.8,3.,3.2]:
    star=a.star_frame(t)[:,:,3]
    text=a.text_frame(t)[:,:,3]
    frame=np.zeros((1080,1920),np.float32)
    x,y,_,_=a.star_box;frame[y:y+star.shape[0],x:x+star.shape[1]]=star
    overlap=frame[a.text_top:a.text_top+text.shape[0],a.text_left:a.text_left+text.shape[1]]
    assert not ((overlap>.03)&(text>.03)).any()
check('rotating sign does not collide with the visible wordmark',True)

epsilon=1e-4
check('cursor detaches at its rotated world position',
      np.array_equal(a.cursor_position(DETACH),a.endpoint(CURSOR_NODE,DETACH)))
incoming=(a.endpoint(CURSOR_NODE,DETACH)-a.endpoint(CURSOR_NODE,DETACH-epsilon))/epsilon
outgoing=(a.cursor_position(DETACH+epsilon)-a.cursor_position(DETACH))/epsilon
check('rotating endpoint velocity is inherited at cursor launch',np.linalg.norm(incoming-outgoing)<.5)
for boundary in [DETACH,JOIN]:
    sprite,x,y=a.cursor_layer(boundary)
    target,xx,yy=a.node_sprite(a.cursor_position(boundary),a.radii[CURSOR_NODE],1.)
    assert (x,y)==(xx,yy) and np.array_equal(sprite,target)
check('exposure disappears at circle handoffs, preventing a duplicate or changed sphere',True)
for t in [1.8,2.,2.4,2.8,3.1]:
    sprite,_,_=a.cursor_layer(t)
    assert sprite[:,:,3].min()>=0 and sprite[:,:,3].max()<=1
    assert np.max(sprite[:,:,:3]-sprite[:,:,3:4])<1e-6
check('cursor temporal smoothing uses valid premultiplied alpha',True)
check('text still waits for the cursor return and uses the unchanged original contours',
      not a.text_frame(RETURN).any() and np.array_equal(a.text_alpha,approved.text_alpha))
check('approved final composition remains pixel-identical',np.array_equal(np.asarray(a.final),reference))
for t in [3.7,4.,4.5,149/30]:assert np.array_equal(np.asarray(a.frame(t)),reference)
check('final hold is sharp, completely static and in the original orientation',True)

for frame in range(1,66):
    rgba=np.asarray(Image.open(ROOT/f'.cache/{PREFIX}-frames/{frame:04d}.png'))
    assert nd.label(rgba[:,:,3]>3,structure=np.ones((3,3)))[1]<=2
check('actual early frames contain no stray pieces beyond the sign and intended cursor',True)
for t in [1.,1.2,1.7,2.3,2.7]:
    frame=np.asarray(Image.open(ROOT/f'.cache/{PREFIX}-frames/{round(t*30):04d}.png'))
    positions=a.node_positions(t,a.scales(t))
    for i,p in enumerate(positions):
        if a.skip_node(i,t):continue
        px,py=np.rint(p).astype(int)
        assert frame[py-2:py+3,px-2:px+3,3].max()>220
check('actual rendered spheres follow the rotated paths rather than the old fixed angles',True)

def probe(file):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
       'stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration',
       '-of','json',str(file)]))

for name,size in [(f'{PREFIX}-alpha.webm',(1920,1080)),(f'{PREFIX}-alpha.mov',(1920,1080)),
                  (f'{PREFIX}-4k-alpha.webm',(3840,2160))]:
    path=ROOT/'exports'/name;info=probe(path)
    video=next(s for s in info['streams'] if s['codec_type']=='video')
    check(f'{name}: native resolution, 30 fps, 150 frames',
          (video['width'],video['height'])==size and video['r_frame_rate']=='30/1' and video['nb_read_frames']=='150')
    decoder=['-c:v','libvpx-vp9'] if path.suffix=='.webm' else []
    raw=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error',*decoder,'-i',str(path),
       '-ss','4.5','-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'])
    decoded=np.frombuffer(raw,np.uint8).reshape(size[1],size[0],4)
    target=reference if size==(1920,1080) else np.asarray(Image.open(ROOT/'exports/s-project-2-clean-4k-final.png'))
    difference=np.abs(decoded[:,:,3].astype(np.int16)-target[:,:,3].astype(np.int16))
    check(f'{name}: actual alpha is transparent and the final approved silhouette is intact',
          decoded[0,0,3]==0 and decoded[:,:,3].max()==255 and
          (difference.max()<=1 if path.suffix=='.webm' else difference.mean()<.1))
    check(f'{name}: quiet soundtrack retained',any(s['codec_type']=='audio' for s in info['streams']))
    assert path.stat().st_size<90*1024**2

for prefix,size in [(PREFIX,(1920,1080)),(PREFIX+'-4k',(3840,2160))]:
    info=probe(ROOT/f'exports/{prefix}-preview.mp4');video=next(s for s in info['streams'] if s['codec_type']=='video')
    check(f'{prefix}: H.264 preview is correctly native-sized and opaque',
          video['codec_name']=='h264' and video['pix_fmt']=='yuv420p' and (video['width'],video['height'])==size)
print(f'\n{count} actual-turn checks passed.')
