"""Transparent white edition: travelling radial length wave and light tracing.

The custom S, PROJECT and their cuts are masks from the supplied PNG, never
replacement typography. Only the ten measured ray lengths change; no whole
symbol rotation is performed. Intermediate images use premultiplied alpha.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage as nd
from render import OriginalPixelAnimation, ROOT, CENTER, ENDPOINTS, ROI, CROP_TOP, SIZE, FPS

PREFIX = 's-project-2-light'
STAR_BOX = (100, 250, 585, 785)
TEXT_BOX = (515, 450, 1310, 630)


def ramp(t, start, duration):
    u = np.clip((t-start)/duration,0,1)
    return float(u*u*u*(u*(6*u-15)+10))


def clamp01(x):
    return np.clip(x,0,1)


def over(dst, src, x=0, y=0):
    """Alpha-over for premultiplied RGBA, clipping a small moving tile."""
    height,width=src.shape[:2]
    x0,y0=max(0,x),max(0,y)
    x1,y1=min(dst.shape[1],x+width),min(dst.shape[0],y+height)
    if x1<=x0 or y1<=y0:
        return
    tile=src[y0-y:y1-y,x0-x:x1-x]
    target=dst[y0:y1,x0:x1]
    target[:]=tile+target*(1-tile[:,:,3:4])


class LightAnimation:
    def __init__(self, source):
        original=OriginalPixelAnimation(source)
        self.source=original.source
        rgb=np.asarray(self.source).astype(np.float32)/255
        matte=np.zeros(rgb.shape[:2],np.float32)
        contrast=(original.base-original.crop.astype(np.float32)).mean(2)
        coverage=clamp01((contrast-7)/30)
        # A sharp material silhouette with antialiased boundaries; background
        # variations never become opaque islands on the transparent canvas.
        coverage=coverage*coverage*(3-2*coverage)
        x0,y0,x1,y1=ROI
        matte[y0:y1,x0:x1]=coverage
        self.alpha=matte
        self.original_rgb=rgb
        self.vectors=ENDPOINTS-CENTER
        self.lengths=np.linalg.norm(self.vectors,axis=1)
        self.units=self.vectors/self.lengths[:,None]
        self.angles=np.arctan2(self.units[:,1],self.units[:,0])
        self._build_star()
        self._build_text()
        self.final_native=self._final_native()
        self.final=self.to_output(self.final_native)

    @staticmethod
    def scales(t):
        growth=.035+.965*ramp(t,.25,.95)
        amplitude=.34*ramp(t,.35,.4)*(1-ramp(t,1.65,.85))
        phase=-2.4+2*np.pi*1.05*ramp(t,.3,2.15)
        angles=np.arctan2((ENDPOINTS-CENTER)[:,1],(ENDPOINTS-CENTER)[:,0])
        return growth*(1+amplitude*np.cos(angles-phase))

    @staticmethod
    def beam_position(t):
        return -70+935*ramp(t,1.85,1.65)

    def _build_star(self):
        x0,y0,x1,y1=STAR_BOX
        self.star_alpha=self.alpha[y0:y1,x0:x1].copy()
        yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        self.star_alpha[xx>=520]=0
        points=np.stack((xx-CENTER[0],yy-CENTER[1]),axis=-1)
        distances=[]
        normals=[]
        for unit,length in zip(self.units,self.lengths):
            projection=clamp01((points@unit)/length)*length
            distances.append(np.linalg.norm(points-projection[:,:,None]*unit,axis=2))
            normals.append(np.abs(points[:,:,0]*unit[1]-points[:,:,1]*unit[0]))
        owner=np.argmin(np.stack(distances),axis=0)
        node_distance=np.min(np.linalg.norm(
            np.stack((xx,yy),axis=-1)[:,:,None,:]-ENDPOINTS[None,None,:,:],axis=3),axis=2)
        self.shafts=[]
        for i in range(10):
            corridor=clamp01((4.7-normals[i])/1.1)
            self.shafts.append(self.star_alpha*(owner==i)*corridor*(node_distance>=14))
        self.local_center=np.array([CENTER[1]-y0,CENTER[0]-x0])
        self.node_tiles=[]
        for px,py in ENDPOINTS:
            nx,ny=int(np.floor(px))-18,int(np.floor(py))-18
            ay,ax=np.mgrid[ny:ny+37,nx:nx+37].astype(np.float32)
            circular=clamp01((16.5-np.hypot(ax-px,ay-py))/.8)
            alpha=self.alpha[ny:ny+37,nx:nx+37]*circular
            source=self.original_rgb[ny:ny+37,nx:nx+37].mean(2)
            # Original top-left lighting becomes a restrained pearl-white sphere.
            shade=.72+.28*clamp01((source-.035)/.43)
            tile=np.empty((37,37,4),np.float32)
            tile[:,:,:3]=alpha[:,:,None]*shade[:,:,None]
            tile[:,:,3]=alpha
            self.node_tiles.append((tile,np.array([px-nx,py-ny])))
        self.hub_radius=np.hypot(xx-CENTER[0],yy-CENTER[1])

    def _build_text(self):
        x0,y0,x1,y1=TEXT_BOX
        self.text_alpha=self.alpha[y0:y1,x0:x1].copy()
        self.text_alpha[:,:5]=0
        yy,xx=np.mgrid[:y1-y0,:x1-x0].astype(np.float32)
        self.text_x=xx;self.text_y=yy
        self.field=xx+.12*(yy-85)
        binary=self.text_alpha>.35
        self.edge=(binary & ~nd.binary_erosion(binary,iterations=1)).astype(np.float32)
        self.near=nd.gaussian_filter(self.edge,1.8)
        self.fragments=[]
        for i,desired in enumerate([76,160,251,337,425,520,615,735]):
            column=min(desired,self.text_alpha.shape[1]-5)
            candidates=np.argwhere(binary[:,column-3:column+4])
            if not len(candidates):
                continue
            ys=candidates[:,0]
            anchor_y=int(ys.min()+2 if i%2==0 else ys.max()-2)
            width=7+i%4
            height=3 if i%3 else 4
            left=column-width//2
            top=anchor_y-height//2
            fragment=self.text_alpha[top:top+height,left:left+width].copy()
            if not fragment.any():
                continue
            field=column+.12*(anchor_y-85)
            lo,hi=1.85,3.5
            for _ in range(24):
                middle=(lo+hi)/2
                if self.beam_position(middle)<field:lo=middle
                else:hi=middle
            self.fragments.append(dict(alpha=fragment,x=left,y=top,birth=(lo+hi)/2,
                                       life=.26+.025*(i%3),dy=(-1 if i%2==0 else 1)*(11+2*(i%4)),
                                       dx=-8-2*(i%3)))

    def _final_native(self):
        canvas=np.zeros((*self.alpha.shape,4),np.float32)
        x0,y0,x1,y1=STAR_BOX
        star=np.zeros((*self.star_alpha.shape,4),np.float32)
        star[:,:,:3]=self.star_alpha[:,:,None]*.94
        star[:,:,3]=self.star_alpha
        for (px,py),(tile,center) in zip(ENDPOINTS,self.node_tiles):
            nx=int(np.floor(px))-18-x0;ny=int(np.floor(py))-18-y0
            region=star[ny:ny+37,nx:nx+37]
            selected=tile[:,:,3]>1e-6
            color=np.divide(tile[:,:,:3],tile[:,:,3:4],out=np.zeros_like(tile[:,:,:3]),where=tile[:,:,3:4]>1e-6)
            region[:,:,:3][selected]=color[selected]*region[:,:,3:4][selected]
        over(canvas,star,x0,y0)
        tx,ty,_,_=TEXT_BOX
        text=np.dstack([self.text_alpha]*4)
        over(canvas,text,tx,ty)
        return canvas

    def star_frame(self,t):
        growth=.035+.965*ramp(t,.25,.95)
        node_scale=growth**.7
        widths=.35+.65*np.sqrt(growth)
        scales=self.scales(t)
        opacity=ramp(t,.2,.45)
        combined=np.zeros_like(self.star_alpha)
        for i,(unit,length,scale,shaft) in enumerate(zip(self.units,self.lengths,scales,self.shafts)):
            parallel=max(.025,(length*scale-14*node_scale)/(length-14))
            uyx=unit[::-1]
            vyx=np.array([-uyx[1],uyx[0]])
            inverse=np.outer(uyx,uyx)/parallel+np.outer(vyx,vyx)/widths
            offset=self.local_center-inverse@self.local_center
            moved=nd.affine_transform(shaft,inverse,offset=offset,order=1,mode='constant',cval=0,prefilter=False)
            np.maximum(combined,moved,out=combined)
        hub=self.star_alpha*clamp01((3+8*ramp(t,.15,.55)-self.hub_radius)/1.0)*ramp(t,0,.3)
        np.maximum(combined*opacity,hub,out=combined)
        canvas=np.empty((*combined.shape,4),np.float32)
        canvas[:,:,:3]=combined[:,:,None]*.94
        canvas[:,:,3]=combined
        x0,y0,_,_=STAR_BOX
        for endpoint,scale,(tile,center) in zip(ENDPOINTS,scales,self.node_tiles):
            position=CENTER+(endpoint-CENTER)*scale
            radius=20*node_scale+2
            left=int(np.floor(position[0]-radius));top=int(np.floor(position[1]-radius))
            right=int(np.ceil(position[0]+radius));bottom=int(np.ceil(position[1]+radius))
            yy,xx=np.mgrid[top:bottom,left:right].astype(np.float32)
            sy=(yy-position[1])/node_scale+center[1]
            sx=(xx-position[0])/node_scale+center[0]
            moved=np.stack([nd.map_coordinates(tile[:,:,c],[sy,sx],order=1,mode='constant',cval=0,prefilter=False)
                            for c in range(4)],axis=-1)*opacity
            over(canvas,moved,left-x0,top-y0)
        # Resolve to the precise source silhouette only once lengths have settled.
        resolve=ramp(t,2.5,.25)
        if resolve:
            target=self.final_native[y0:y0+canvas.shape[0],x0:x0+canvas.shape[1]]
            canvas=canvas*(1-resolve)+target*resolve
        return canvas

    def text_frame(self,t):
        width=self.text_alpha.shape[1];height=self.text_alpha.shape[0]
        canvas=np.zeros((height,width,4),np.float32)
        front=self.beam_position(t)
        active=ramp(t,1.8,.25)*(1-ramp(t,3.32,.36))
        reveal=clamp01((front-self.field+22)/44)
        reveal=reveal*reveal*(3-2*reveal)
        if t<1.85:reveal[:]=0
        pulse=np.exp(-.5*((self.field-front)/5.0)**2)*active
        fill=clamp01(self.text_alpha*(reveal+.52*pulse))
        outline=clamp01(self.edge*.17*active)
        glow=clamp01(self.near*(1.5*pulse+.12*active))
        alpha=clamp01(fill+outline+glow)
        canvas[:,:,:3]=alpha[:,:,None]
        canvas[:,:,3]=alpha
        return canvas

    def native_frame(self,t):
        if t>=3.7-1e-10:
            return self.final_native.copy()
        canvas=np.zeros_like(self.final_native)
        over(canvas,self.star_frame(t),STAR_BOX[0],STAR_BOX[1])
        tx,ty,_,_=TEXT_BOX
        over(canvas,self.text_frame(t),tx,ty)
        for fragment in self.fragments:
            u=(t-fragment['birth'])/fragment['life']
            if not 0<u<1:continue
            opacity=.65*np.sin(np.pi*u)*(1-u)
            alpha=fragment['alpha']*opacity
            tile=np.dstack([alpha]*4)
            movement=1-(1-u)**2
            x=round(tx+fragment['x']+fragment['dx']*movement)
            y=round(ty+fragment['y']+fragment['dy']*movement+4*u*u)
            over(canvas,tile,x,y)
        finish=ramp(t,3.48,.22)
        if finish:
            canvas=canvas*(1-finish)+self.final_native*finish
        return np.clip(canvas,0,1)

    @staticmethod
    def to_output(native):
        premult=np.rint(np.clip(native,0,1)*255).astype(np.uint8)
        image=Image.frombytes('RGBa',(premult.shape[1],premult.shape[0]),premult.tobytes())
        inv=1448/1920
        return image.transform(SIZE,Image.Transform.AFFINE,(inv,0,0,0,inv,CROP_TOP),
                               resample=Image.Resampling.BICUBIC).convert('RGBA')

    def frame(self,t):
        if t>=3.7-1e-10:return self.final.copy()
        return self.to_output(self.native_frame(t))


def encode(frames,out,audio):
    common=['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(FPS),'-i',str(frames/'%04d.png'),'-i',str(audio)]
    subprocess.run(common+['-map','0:v','-map','1:a','-frames:v','150','-t','5','-c:v','libvpx-vp9','-pix_fmt','yuva420p',
       '-auto-alt-ref','0','-b:v','0','-crf','16','-row-mt','1','-cpu-used','4','-threads','2',
       '-c:a','libopus','-b:a','128k',str(out/f'{PREFIX}-alpha.webm')],check=True)
    print('WebM alpha ready',flush=True)
    subprocess.run(common+['-map','0:v','-map','1:a','-frames:v','150','-t','5','-c:v','prores_ks','-profile:v','4',
       '-pix_fmt','yuva444p10le','-alpha_bits','16','-qscale:v','9','-threads','2','-c:a','pcm_s16le',
       str(out/f'{PREFIX}-alpha.mov')],check=True)
    print('ProRes 4444 alpha ready',flush=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
       'color=c=0x14171c:s=1920x1080:r=30:d=5','-framerate','30','-i',str(frames/'%04d.png'),'-i',str(audio),
       '-filter_complex','[0:v][1:v]overlay=shortest=1:format=auto,format=yuv420p[v]',
       '-map','[v]','-map','2:a','-frames:v','150','-t','5','-c:v','libx264','-crf','17','-preset','medium',
       '-threads','2','-c:a','aac','-b:a','192k','-movflags','+faststart',str(out/f'{PREFIX}-preview.mp4')],check=True)
    print('Dark MP4 preview ready (opaque preview only)',flush=True)


def render():
    animation=LightAnimation(ROOT/'assets/source.png')
    frames=ROOT/'.cache/light-frames';frames.mkdir(parents=True,exist_ok=True)
    out=ROOT/'exports';out.mkdir(exist_ok=True)
    for n in range(150):
        animation.frame(n/FPS).save(frames/f'{n:04d}.png',compress_level=2)
        if n%30==0:print(f'Render white/alpha: {n//30}/5 seconds',flush=True)
    animation.final.save(out/f'{PREFIX}-final.png')
    encode(frames,out,ROOT/'assets/sound-design.wav')
    info={'duration':5,'fps':30,'frames':150,'size':list(SIZE),
          'source':'assets/source.png','ray_count':10,'rotates_whole_symbol':False,
          'radial_wave_max_amplitude':.34,'lengths_settled_at':2.5,
          'text':'original custom letter silhouettes recolored white; light scan and temporary edge fragments',
          'temporary_fragments':len(animation.fragments),'all_effects_gone_at':3.7,
          'transparent':['WebM VP9 alpha','MOV ProRes 4444 alpha','final PNG'],
          'opaque_preview':'MP4 H.264 on #14171c; not a transparent master'}
    (out/'light-render-info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print('White transparent edition complete',flush=True)


if __name__=='__main__':render()
