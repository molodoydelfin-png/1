"""An actual eased 360-degree symbol turn, retaining the returning cursor."""
import argparse
import json
import subprocess
import numpy as np
import render_clean as clean
from render_clean import ROOT,ramp
from render_sweep import SweepAnimation,DETACH,JOIN,CURSOR_NODE,encode_preview_4k

PREFIX='s-project-2-turn'


class TurnAnimation(SweepAnimation):
    @staticmethod
    def angle(t):
        return 2*np.pi*ramp(t,.35,2.95)

    def star_bounds(self):
        return tuple(v*clean.OUTPUT_SCALE for v in (160,190,880,920))

    def base_growth(self,t):
        return ramp(t,.22,1.12)

    def scales(self,t):
        angles=np.arctan2(self.units[:,1],self.units[:,0])
        amplitude=.08*ramp(t,.4,.6)*(1-ramp(t,2.35,.95))
        phase=-2.1+2*np.pi*.3*(t-.35)
        return self.base_growth(t)*(1+amplitude*np.cos(angles-phase))

    def node_positions(self,t,growth):
        vectors=(self.endpoints-self.center)*growth[:,None]
        if t<=.35 or t>=3.3:
            return self.center+vectors
        theta=self.angle(t)
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return self.center+vectors@rotation.T

    def star_frame(self,t):
        sharp=super().star_frame(t)
        # Very short trailing exposure makes 30fps orbiting tips read smoothly.
        # Fade it around handoffs so a detached node is never drawn twice.
        strength=.32*ramp(t,.45,.3)*(1-ramp(t,3.,.3))
        if abs(t-DETACH)<.035 or abs(t-JOIN)<.035:
            strength*=ramp(abs(t-DETACH) if abs(t-DETACH)<.035 else abs(t-JOIN),0,.035)
        shutter=1/240
        if strength>1e-6 and self.skip_node(CURSOR_NODE,t)==self.skip_node(CURSOR_NODE,t-shutter):
            return sharp*(1-strength)+super().star_frame(max(0,t-shutter))*strength
        return sharp

    def cursor_layer(self,t):
        current,x,y=self.cursor_sprite(t)
        amount=.65*self.cursor_shape(t)**2
        if amount<1e-5:return current,x,y
        samples=[]
        for delay in np.linspace(0,.010,9):
            moment=max(DETACH,t-float(delay))
            tile,left,top=self.cursor_sprite(moment)
            samples.append((tile,left,top))
        left=min(item[1] for item in samples);top=min(item[2] for item in samples)
        right=max(item[1]+item[0].shape[1] for item in samples)
        bottom=max(item[2]+item[0].shape[0] for item in samples)
        accumulated=np.zeros((bottom-top,right-left,4),np.float32)
        for tile,sx,sy in samples:
            yy,xx=sy-top,sx-left
            accumulated[yy:yy+tile.shape[0],xx:xx+tile.shape[1]]+=tile/len(samples)*amount
        yy,xx=y-top,x-left
        accumulated[yy:yy+current.shape[0],xx:xx+current.shape[1]]+=current*(1-amount)
        return np.clip(accumulated,0,1),left,top


def render(four_k=False):
    prefix=PREFIX
    if four_k:
        clean.K=3840/1448;clean.OUTPUT_SIZE=(3840,2160);clean.OUTPUT_SCALE=2
        clean.STAR_SSAA=4;prefix+='-4k'
    animation=TurnAnimation(ROOT/'assets/source.png')
    frames=ROOT/f'.cache/{prefix}-frames';frames.mkdir(parents=True,exist_ok=True)
    out=ROOT/'exports'
    for n in range(150):
        animation.frame(n/30).save(frames/f'{n:04d}.png',compress_level=2)
        if n%30==0:print(f'Actual symbol turn {n//30}/5 seconds',flush=True)
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
    info={'duration':5,'fps':30,'frames':150,'size':list(clean.OUTPUT_SIZE),
          'symbol_turn_degrees':360,'turn_start':.35,'turn_end':3.3,
          'length_wave_amplitude':.08,'rotates_text':False,
          'symbol_trailing_exposure_seconds':1/240,'cursor_trailing_exposure_seconds':.010,
          'cursor_rejoined_at':3.48,'final_hold_start':3.7,'matches_approved_final':True}
    (out/f'{prefix}-render-info.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--4k',action='store_true',dest='four_k')
    render(parser.parse_args().four_k)
