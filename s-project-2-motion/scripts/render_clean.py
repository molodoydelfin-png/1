"""Clean deployment: vector-derived typography, analytic round nodes, no spring.

No moving source-image crops, angle waves, detached fragments or oscillation.
Glyph curves are traced from the supplied artwork, not a replacement typeface.
"""
from pathlib import Path
import io
import json
import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'.cache/vector-libs'))
import cairosvg
from svgpathtools import parse_path
from render import CENTER,ENDPOINTS,CROP_TOP,SIZE,FPS
from render_light import LightAnimation,over,ramp,encode,TEXT_BOX

PREFIX='s-project-2-clean'
K=1920/1448
SSAA=4
STAR_SSAA=8
OUTPUT_SIZE=SIZE
OUTPUT_SCALE=1
SVG_NS='http://www.w3.org/2000/svg'
ET.register_namespace('',SVG_NS)


def source_point(point):
    return np.array([point[0]*K,(point[1]-CROP_TOP)*K],dtype=np.float32)


def build_wordmark(original,force=False):
    destination=ROOT/'assets/wordmark-clean.svg'
    if destination.exists() and not force:return destination
    import vtracer
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    # A light subpixel blur removes the source raster staircase while retaining
    # all eleven intentional glyph pieces and cuts. Trace at 8×, then replace
    # the hundreds of microscopic trace corners with continuous cubic curves.
    smooth=nd.gaussian_filter(original.text_alpha.astype(np.float32),sigma=.55)
    high=Image.fromarray(np.rint(np.clip(smooth,0,1)*255).astype(np.uint8)).resize((6360,1440),Image.Resampling.BICUBIC)
    binary=np.asarray(high)>127
    input_path=cache/'wordmark-vector-input.png'
    traced=cache/'wordmark-vector-traced.svg'
    Image.fromarray(255-binary.astype(np.uint8)*255).save(input_path)
    vtracer.convert_image_to_svg_py(str(input_path),str(traced),colormode='binary',mode='spline',
        filter_speckle=2,corner_threshold=45,length_threshold=10.0,max_iterations=20,
        splice_threshold=60,path_precision=5)
    raw=ET.parse(traced).getroot()
    root=ET.Element(f'{{{SVG_NS}}}svg',{'viewBox':'0 0 795 180','width':'795','height':'180'})
    for element in raw.iter(f'{{{SVG_NS}}}path'):
        parsed=parse_path(element.get('d'))
        match=re.search(r'translate\(([-+0-9.eE]+)[, ]+([-+0-9.eE]+)\)',element.get('transform',''))
        offset=complex(float(match.group(1)),float(match.group(2))) if match else 0j
        commands=[]
        for subpath in parsed.continuous_subpaths():
            length=subpath.length()/8
            count=max(12,int(np.ceil(length/2.4)))
            points=np.array([(subpath.point(i/count)+offset)/8 for i in range(count)],dtype=np.complex128)
            points=nd.gaussian_filter1d(points.real,1.25,mode='wrap')+1j*nd.gaussian_filter1d(points.imag,1.25,mode='wrap')
            commands.append(f'M {points[0].real:.5f} {points[0].imag:.5f}')
            for i in range(count):
                previous=points[(i-1)%count];start=points[i]
                end=points[(i+1)%count];following=points[(i+2)%count]
                control1=start+(end-previous)/6
                control2=end-(following-start)/6
                commands.append(f'C {control1.real:.5f} {control1.imag:.5f} '
                                f'{control2.real:.5f} {control2.imag:.5f} '
                                f'{end.real:.5f} {end.imag:.5f}')
            commands.append('Z')
        ET.SubElement(root,f'{{{SVG_NS}}}path',{
            'd':' '.join(commands),'fill':'#ffffff','fill-rule':'evenodd'})
    assert len(root.findall(f'{{{SVG_NS}}}path'))==11
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)
    return destination


class CleanAnimation:
    def __init__(self,source):
        self.original=LightAnimation(source)
        self.svg_path=build_wordmark(self.original)
        self.glyphs=ET.parse(self.svg_path).getroot()
        self.center=source_point(CENTER)
        self.endpoints=np.array([source_point(p) for p in ENDPOINTS])
        self.lengths=np.linalg.norm(ENDPOINTS-CENTER,axis=1)
        self.units=(ENDPOINTS-CENTER)/self.lengths[:,None]
        self.radii,self.widths=self._measure_geometry()
        self._build_text()
        self.star_box=self.star_bounds()
        self.final_pm=self.composite(5)
        self.final=self.straight(self.final_pm)

    def _measure_geometry(self):
        radii=[];widths=[]
        alpha=self.original.alpha
        rr=np.linspace(10,18,161)
        angles=np.linspace(0,2*np.pi,100,endpoint=False)
        offsets=np.linspace(-7,7,113)
        for p,u in zip(ENDPOINTS,self.units):
            sampled=[]
            for angle in angles:
                direction=np.array([np.cos(angle),np.sin(angle)])
                if np.dot(direction,-u)>.8:continue
                xs=p[0]+direction[0]*rr;ys=p[1]+direction[1]*rr
                values=nd.map_coordinates(alpha,[ys,xs],order=1,mode='constant')
                hits=np.flatnonzero(values>=.5)
                if len(hits):sampled.append(rr[hits[-1]])
            radii.append(np.median(sampled))
            normal=np.array([-u[1],u[0]])
            measured=[]
            for fraction in [.35,.5,.65]:
                c=CENTER+(p-CENTER)*fraction
                xs=c[0]+normal[0]*offsets;ys=c[1]+normal[1]*offsets
                values=nd.map_coordinates(alpha,[ys,xs],order=1,mode='constant')
                measured.append(np.trapezoid(values,offsets))
            widths.append(np.clip(np.median(measured),3.5,7.5))
        return np.array(radii)*K,np.array(widths)*K

    def _build_text(self):
        tx,ty,rx,by=TEXT_BOX
        self.text_left=int(np.floor(tx*K));self.text_top=int(np.floor((ty-CROP_TOP)*K))
        width=int(np.ceil(rx*K))-self.text_left
        height=int(np.ceil((by-CROP_TOP)*K))-self.text_top
        paths=''.join(ET.tostring(child,encoding='unicode') for child in self.glyphs)
        source_x=self.text_left/K;source_y=self.text_top/K+CROP_TOP
        wrapper=(f'<svg xmlns="{SVG_NS}" viewBox="{source_x} {source_y} {width/K} {height/K}">'
                 f'<g transform="translate({tx} {ty})">{paths}</g></svg>')
        pixels=cairosvg.svg2png(bytestring=wrapper.encode(),output_width=width*SSAA,output_height=height*SSAA)
        high=Image.open(io.BytesIO(pixels)).getchannel('A')
        self.high_text_alpha=np.asarray(high)
        self.text_alpha=np.asarray(high.resize((width,height),Image.Resampling.BOX),dtype=np.float32)/255
        occupied=np.where(self.text_alpha>.5)[1]
        self.beam_center=float((occupied.min()+occupied.max())/2)
        self.beam_radius=float((occupied.max()-occupied.min())/2)
        yy,xx=np.mgrid[:height,:width].astype(np.float32)
        self.field=np.abs(xx-self.beam_center)
        self.edge=np.maximum(0,self.text_alpha-nd.grey_erosion(self.text_alpha,size=(3,3)))
        self.near=nd.gaussian_filter(self.edge,1.8*K)

    def scales(self,t):
        # Small length-based timing offsets; no angular sequence or reverse travel.
        rank=(self.lengths-self.lengths.min())/np.ptp(self.lengths)
        return np.array([ramp(t,.22+.05*r,1.48+.10*r) for r in rank])

    def node_growth(self,t,growth):
        return growth

    def star_bounds(self):
        return tuple(v*OUTPUT_SCALE for v in (210,265,730,800))

    def node_positions(self,t,growth):
        return self.center+(self.endpoints-self.center)*growth[:,None]

    def skip_node(self,index,t):
        return False

    def node_sprite(self,position,radius,opacity):
        """Shared subpixel sphere rasterizer, including detached-cursor joins."""
        quality=STAR_SSAA
        x0=int(np.floor(position[0]-radius-2));y0=int(np.floor(position[1]-radius-2))
        x1=int(np.ceil(position[0]+radius+2));y1=int(np.ceil(position[1]+radius+2))
        disk=Image.new('L',((x1-x0)*quality,(y1-y0)*quality))
        d=ImageDraw.Draw(disk)
        cx=(position[0]-x0)*quality;cy=(position[1]-y0)*quality;hr=radius*quality
        d.ellipse((cx-hr,cy-hr,cx+hr,cy+hr),fill=255)
        a=np.asarray(disk.resize((x1-x0,y1-y0),Image.Resampling.BOX),dtype=np.float32)/255*opacity
        yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        nx=(xx+.5-position[0])/radius;ny=(yy+.5-position[1])/radius
        shade=1-.24*np.clip(np.hypot(nx+.3,ny+.4)/1.7,0,1)
        tile=np.empty((*a.shape,4),np.float32)
        tile[:,:,:3]=a[:,:,None]*shade[:,:,None];tile[:,:,3]=a
        return tile,x0,y0

    def star_frame(self,t):
        left,top,right,bottom=self.star_box
        width=right-left;height=bottom-top
        quality=STAR_SSAA
        mask=Image.new('L',(width*quality,height*quality))
        draw=ImageDraw.Draw(mask)
        c=(self.center-np.array([left,top]))*quality
        growth=self.scales(t)
        positions=self.node_positions(t,growth)
        ray_opacity=ramp(t,.05,.4)
        for p,w in zip(positions,self.widths):
            endpoint=(p-np.array([left,top]))*quality
            draw.line([tuple(c),tuple(endpoint)],fill=round(255*ray_opacity),width=max(1,round(w*quality)))
        radius=3*K*quality
        draw.ellipse((c[0]-radius,c[1]-radius,c[0]+radius,c[1]+radius),fill=round(255*ramp(t,0,.3)))
        alpha=np.asarray(mask.resize((width,height),Image.Resampling.BOX),dtype=np.float32)/255
        canvas=np.empty((height,width,4),np.float32)
        canvas[:,:,:3]=alpha[:,:,None]*.94;canvas[:,:,3]=alpha
        heads=self.node_growth(t,growth)
        for index,(position,radius,g) in enumerate(zip(positions,self.radii,heads)):
            if self.skip_node(index,t):continue
            opacity=ramp(g,.2,.35)
            if opacity<=0:continue
            # Round heads are generated independently: no source-crop tails,
            # cropped ray tips or stretched sphere pixels can break off.
            r=radius*(.9+.1*g)
            tile,x0,y0=self.node_sprite(position,r,ramp(g,.2,.35))
            over(canvas,tile,x0-left,y0-top)
        return canvas

    def text_frame(self,t):
        front=-26*K+(self.beam_radius+52*K)*ramp(t,1.85,1.65)
        active=ramp(t,1.8,.25)*(1-ramp(t,3.32,.36))
        reveal=np.clip((front-self.field+22*K)/(44*K),0,1)
        reveal=reveal*reveal*(3-2*reveal)
        if t<1.85:reveal[:]=0
        pulse=np.exp(-.5*((self.field-front)/(5*K))**2)*active
        fill=np.clip(self.text_alpha*(reveal+.52*pulse),0,1)
        reached=np.clip((front+28*K-self.field)/(28*K),0,1)
        outline=self.edge*.17*active*reached
        glow=self.near*(1.5*pulse+.05*active*reveal)
        alpha=np.clip(fill+outline+glow,0,1)
        if t>=3.7:alpha=self.text_alpha
        return np.dstack([alpha]*4)

    def composite(self,t):
        canvas=np.zeros((OUTPUT_SIZE[1],OUTPUT_SIZE[0],4),np.float32)
        over(canvas,self.star_frame(t),self.star_box[0],self.star_box[1])
        over(canvas,self.text_frame(t),self.text_left,self.text_top)
        return canvas

    @staticmethod
    def straight(pm):
        alpha=np.clip(pm[:,:,3],0,1)
        colors=np.divide(pm[:,:,:3],alpha[:,:,None],out=np.zeros_like(pm[:,:,:3]),where=alpha[:,:,None]>1e-7)
        return Image.fromarray(np.rint(np.clip(np.dstack([colors,alpha]),0,1)*255).astype(np.uint8))

    def frame(self,t):
        if t>=3.7-1e-10:return self.final.copy()
        pm=self.composite(t)
        finish=ramp(t,3.48,.22)
        if finish:pm=pm*(1-finish)+self.final_pm*finish
        return self.straight(pm)

    def save_vector(self,path):
        # Same measured geometry and traced glyph contours; unlimited resolution.
        paths=''.join(ET.tostring(child,encoding='unicode') for child in self.glyphs)
        lines=''.join(f'<line x1="{CENTER[0]}" y1="{CENTER[1]}" x2="{p[0]}" y2="{p[1]}" stroke="#f0f0f0" stroke-width="{w/K}"/>'
                      for p,w in zip(ENDPOINTS,self.widths))
        nodes=''.join(f'<circle cx="{p[0]}" cy="{p[1]}" r="{r/K}" fill="url(#pearl)"/>' for p,r in zip(ENDPOINTS,self.radii))
        svg=(f'<svg xmlns="{SVG_NS}" width="{OUTPUT_SIZE[0]}" height="{OUTPUT_SIZE[1]}" viewBox="0 {CROP_TOP} 1448 814.5">'
             '<defs><radialGradient id="pearl" cx="35%" cy="30%" r="85%"><stop offset="0" stop-color="white"/>'
             '<stop offset="1" stop-color="#c2c2c2"/></radialGradient></defs>'
             f'{lines}{nodes}<g transform="translate(515 450)">{paths}</g></svg>')
        path.write_text(svg)


def render(four_k=False):
    global K,OUTPUT_SIZE,OUTPUT_SCALE,PREFIX,STAR_SSAA
    if four_k:
        K=3840/1448;OUTPUT_SIZE=(3840,2160);OUTPUT_SCALE=2
        PREFIX='s-project-2-clean-4k';STAR_SSAA=4
    animation=CleanAnimation(ROOT/'assets/source.png')
    frames=ROOT/f'.cache/{PREFIX}-frames';frames.mkdir(parents=True,exist_ok=True)
    out=ROOT/'exports'
    for n in range(150):
        animation.frame(n/FPS).save(frames/f'{n:04d}.png',compress_level=2)
        if n%30==0:print(f'Clean 4x-AA render: {n//30}/5 seconds',flush=True)
    animation.final.save(out/f'{PREFIX}-final.png')
    animation.save_vector(out/f'{PREFIX}-final.svg')
    if four_k:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','30','-i',str(frames/'%04d.png'),
            '-i',str(ROOT/'assets/sound-design.wav'),'-map','0:v','-map','1:a','-frames:v','150','-t','5',
            '-c:v','libvpx-vp9','-pix_fmt','yuva420p','-auto-alt-ref','0','-b:v','0','-crf','0','-lossless','1',
            '-row-mt','1','-cpu-used','4','-threads','2','-c:a','libopus','-b:a','128k',str(out/f'{PREFIX}-alpha.webm')],check=True)
        print('4K lossless VP9 alpha ready',flush=True)
        encode_four_k_preview(frames,out,ROOT/'assets/sound-design.wav')
    else:
        encode(frames,out,ROOT/'assets/sound-design.wav',prefix=PREFIX,preview_crf=8,alpha_crf=0,prores_q=6,alpha_lossless=True)
    info={'size':list(OUTPUT_SIZE),'fps':30,'frames':150,'duration':5,'text_supersampling':SSAA,
          'star_supersampling':STAR_SSAA,'downsampling':'area averaging, no ringing filter',
          'typography':'original contours traced to Bezier SVG; no font substitution',
          'motion':'monotonic radial deployment; no spring, reverse movement, oscillation or overshoot',
          'nodes':'independent analytic circles; no moving source-image fragments',
          'source_node_radii':(animation.radii/K).tolist(),'source_ray_widths':(animation.widths/K).tolist(),
          'preview_h264_crf':8,'alpha_vp9_lossless':True,'temporary_fragments':0,'final_hold_start':3.7}
    (out/f'{PREFIX}-render-info.json').write_text(json.dumps(info,indent=2)+'\n')


def encode_four_k_preview(frames,out,audio):
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
        'color=c=0x14171c:s=3840x2160:r=30:d=5','-framerate','30','-i',str(frames/'%04d.png'),'-i',str(audio),
        '-filter_complex','[0:v][1:v]overlay=shortest=1:format=auto,format=yuv420p[v]',
        '-map','[v]','-map','2:a','-frames:v','150','-t','5','-c:v','libx264','-crf','8','-preset','medium',
        '-threads','2','-c:a','aac','-b:a','192k','-movflags','+faststart',str(out/'s-project-2-clean-4k-preview.mp4')],check=True)
    print('Native 4K preview ready',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--4k',action='store_true',dest='four_k')
    args=parser.parse_args()
    render(args.four_k)
