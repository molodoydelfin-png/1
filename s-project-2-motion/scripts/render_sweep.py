"""A coherent length wave and one logo node that becomes the reveal cursor."""
import argparse
import json
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as nd
import render_clean as clean
from render_clean import CleanAnimation,ROOT,ramp,over

PREFIX='s-project-2-sweep'
DETACH=1.45
OUT_END=2.1
RETURN=2.18
SCAN_END=3.3
JOIN=3.48
CURSOR_NODE=9


def bezier(p0,p1,p2,p3,u):
    return (1-u)**3*p0+3*(1-u)**2*u*p1+3*(1-u)*u*u*p2+u**3*p3


class SweepAnimation(CleanAnimation):
    def base_growth(self,t):
        return ramp(t,.22,.92)

    def scales(self,t):
        angles=np.arctan2(self.units[:,1],self.units[:,0])
        growth=self.base_growth(t)
        amplitude=.16*ramp(t,.35,.55)*(1-ramp(t,2.35,.95))
        phase=-2.1+2*np.pi*.42*(t-.35)
        return growth*(1+amplitude*np.cos(angles-phase))

    def node_growth(self,t,growth):
        # Only ray length oscillates; spheres do not inflate/deflate with the wave.
        return np.full(10,self.base_growth(t))

    def skip_node(self,index,t):
        return index==CURSOR_NODE and DETACH<=t<JOIN

    def endpoint(self,index,t):
        return self.center+(self.endpoints[index]-self.center)*self.scales(t)[index]

    def cursor_position(self,t):
        start=self.endpoint(CURSOR_NODE,DETACH)
        target=self.endpoints[CURSOR_NODE]
        occupied_y=np.where(self.text_alpha>.5)[0]
        middle=self.text_top+(occupied_y.min()+occupied_y.max())/2
        far=np.array([self.text_left+self.text_alpha.shape[1]+22*clean.K,middle])
        near=np.array([self.text_left-30*clean.K,middle])
        if t<=DETACH:return start
        if t<OUT_END:
            duration=OUT_END-DETACH
            u=ramp(t,DETACH,duration)
            # Inherit the live endpoint velocity; the launch has no tiny pause
            # when the node changes from the radial wave to the flight path.
            eps=1e-4
            velocity=(self.endpoint(CURSOR_NODE,DETACH+eps)-self.endpoint(CURSOR_NODE,DETACH-eps))/(2*eps)
            linear=np.clip((t-DETACH)/duration,0,1)
            correction=velocity*duration*linear*(1-linear)**3
            return bezier(start,start+np.array([140,-25])*clean.K,far-np.array([180,30])*clean.K,far,u)+correction
        if t<RETURN:return far
        if t<SCAN_END:
            return far+(near-far)*ramp(t,RETURN,SCAN_END-RETURN)
        if t<JOIN:
            u=ramp(t,SCAN_END,JOIN-SCAN_END)
            return bezier(near,near+np.array([0,-25])*clean.K,target+np.array([20,0])*clean.K,target,u)
        return target

    def cursor_shape(self,t):
        return ramp(t,DETACH+.1,.25)*(1-ramp(t,JOIN-.25,.25))

    def cursor_sprite(self,t):
        position=self.cursor_position(t)
        radius=self.radii[CURSOR_NODE]
        shape=self.cursor_shape(t)
        if shape<1e-8:
            return self.node_sprite(position,radius,1.)
        rx=radius*(1-.86*shape)
        occupied_y=np.where(self.text_alpha>.5)[0]
        height=occupied_y.max()-occupied_y.min()+1
        ry=radius+(height*.56-radius)*shape
        pad=8*clean.K
        x0=int(np.floor(position[0]-rx-pad));y0=int(np.floor(position[1]-ry-pad))
        x1=int(np.ceil(position[0]+rx+pad));y1=int(np.ceil(position[1]+ry+pad))
        q=clean.STAR_SSAA
        mask=Image.new('L',((x1-x0)*q,(y1-y0)*q))
        draw=ImageDraw.Draw(mask)
        cx=(position[0]-x0)*q;cy=(position[1]-y0)*q
        draw.ellipse((cx-rx*q,cy-ry*q,cx+rx*q,cy+ry*q),fill=255)
        alpha=np.asarray(mask.resize((x1-x0,y1-y0),Image.Resampling.BOX),dtype=np.float32)/255
        halo=nd.gaussian_filter(alpha,2.0*clean.K)*(.14*shape)
        combined=np.clip(alpha+halo,0,1)
        yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        nx=(xx+.5-position[0])/rx;ny=(yy+.5-position[1])/ry
        shade=(1-.24*np.clip(np.hypot(nx+.3,ny+.4)/1.7,0,1))*(1-shape)+shape
        tile=np.empty((*alpha.shape,4),np.float32)
        tile[:,:,:3]=np.clip(alpha[:,:,None]*shade[:,:,None]+halo[:,:,None],0,1)
        tile[:,:,3]=combined
        return tile,x0,y0

    def text_frame(self,t):
        if t>=3.7:return np.dstack([self.text_alpha]*4)
        if t<=RETURN:return np.zeros((*self.text_alpha.shape,4),np.float32)
        head=self.cursor_position(t)[0]-self.text_left
        xx=np.arange(self.text_alpha.shape[1],dtype=np.float32)[None,:]
        # The cursor returns right -> left; white type remains behind its edge.
        reveal=np.clip((xx-head+20*clean.K)/(40*clean.K),0,1)
        reveal=reveal*reveal*(3-2*reveal)
        active=1-ramp(t,SCAN_END,.18)
        pulse=np.exp(-.5*((xx-head)/(5*clean.K))**2)*active
        alpha=np.clip(self.text_alpha*(reveal+.35*pulse)+self.near*pulse*.8,0,1)
        return np.dstack([alpha]*4)

    def composite(self,t):
        canvas=np.zeros((clean.OUTPUT_SIZE[1],clean.OUTPUT_SIZE[0],4),np.float32)
        over(canvas,self.star_frame(t),self.star_box[0],self.star_box[1])
        over(canvas,self.text_frame(t),self.text_left,self.text_top)
        if DETACH<=t<JOIN:
            sprite,x,y=self.cursor_sprite(t)
            over(canvas,sprite,x,y)
        return canvas


def render(four_k=False):
    prefix=PREFIX
    if four_k:
        clean.K=3840/1448;clean.OUTPUT_SIZE=(3840,2160);clean.OUTPUT_SCALE=2
        clean.STAR_SSAA=4;prefix+='-4k'
    animation=SweepAnimation(ROOT/'assets/source.png')
    frames=ROOT/f'.cache/{prefix}-frames';frames.mkdir(parents=True,exist_ok=True)
    out=ROOT/'exports'
    for n in range(150):
        animation.frame(n/30).save(frames/f'{n:04d}.png',compress_level=2)
        if n%30==0:print(f'Sweep/cursor render {n//30}/5 seconds',flush=True)
    animation.final.save(out/f'{prefix}-final.png')
    audio=ROOT/'assets/sound-design.wav'
    if four_k:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','30','-i',str(frames/'%04d.png'),
            '-i',str(audio),'-map','0:v','-map','1:a','-frames:v','150','-t','5','-c:v','libvpx-vp9',
            '-pix_fmt','yuva420p','-auto-alt-ref','0','-b:v','0','-crf','0','-lossless','1','-row-mt','1',
            '-cpu-used','4','-threads','2','-c:a','libopus','-b:a','128k',str(out/f'{prefix}-alpha.webm')],check=True)
        encode_preview_4k(frames,out/f'{prefix}-preview.mp4',audio)
    else:
        clean.encode(frames,out,audio,prefix=prefix,preview_crf=8,alpha_crf=0,prores_q=6,alpha_lossless=True)
    info={'size':list(clean.OUTPUT_SIZE),'fps':30,'frames':150,'duration':5,
          'motion':'smooth opposing radial length wave, fixed angles, no whole-object rotation',
          'wave_amplitude':.16,'cursor_node':CURSOR_NODE,'detach':DETACH,'turn':OUT_END,
          'return_start':RETURN,'text_complete':SCAN_END,'cursor_rejoined':JOIN,'final_hold':3.7,
          'text':'returning cursor reveals original white wordmark right-to-left',
          'extra_fragments':0,'quality':'same vector contours and area-antialiasing as approved clean version'}
    (out/f'{prefix}-render-info.json').write_text(json.dumps(info,indent=2)+'\n')


def encode_preview_4k(frames,destination,audio):
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
        'color=c=0x14171c:s=3840x2160:r=30:d=5','-framerate','30','-i',str(frames/'%04d.png'),'-i',str(audio),
        '-filter_complex','[0:v][1:v]overlay=shortest=1:format=auto,format=yuv420p[v]',
        '-map','[v]','-map','2:a','-frames:v','150','-t','5','-c:v','libx264','-crf','8','-preset','medium',
        '-threads','2','-c:a','aac','-b:a','192k','-movflags','+faststart',str(destination)],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--4k',action='store_true',dest='four_k')
    render(parser.parse_args().four_k)
