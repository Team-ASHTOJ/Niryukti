#!/usr/bin/env python3
"""Deterministic Niryukti motion graphics. No downloaded art or solver dependency."""
from pathlib import Path
import argparse, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
GOLD=(207,172,116); IVORY=(236,233,227)
def ease(v):
    v=max(0,min(1,v)); return v*v*(3-2*v)
def window(t,a,b,c,d): return ease((t-a)/(b-a))*(1-ease((t-c)/(d-c)))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--width',type=int,default=1920); p.add_argument('--fps',type=int,default=30); p.add_argument('--preview',action='store_true'); args=p.parse_args()
    W=args.width; H=round(W*9/16); S=W/1920
    fonts=ROOT/'dashboard/static/fonts'
    def font(n,size): return ImageFont.truetype(str(fonts/n),max(1,round(size*S)))
    display=font('Sora-Variable.ttf',110); team=font('Sora-Variable.ttf',66); mono=font('font-0.ttf',19); small=font('font-2.ttf',28)
    yy,xx=np.mgrid[0:H,0:W]; rad=np.sqrt(((xx-W*.5)/W)**2+((yy-H*.43)/H)**2)
    light=np.maximum(0,1-rad*1.4)
    bg=np.stack([10+light*8,11+light*8,13+light*8],axis=2).astype('uint8')
    base=Image.fromarray(bg,'RGB').convert('RGBA')
    rng=np.random.default_rng(26119); points=rng.random((85,3))
    def text(layer,txt,y,f,opacity=1,color=IVORY,tracking=0):
        if opacity<=0: return
        d=ImageDraw.Draw(layer)
        if not tracking:
            box=d.textbbox((0,0),txt,font=f); width=box[2]-box[0]
            d.text(((W-width)/2,y*S),txt,font=f,fill=(*color,round(255*opacity)))
        else:
            widths=[d.textlength(c,font=f) for c in txt]; total=sum(widths)+(len(txt)-1)*tracking*S; x=(W-total)/2
            for c,cw in zip(txt,widths): d.text((x,y*S),c,font=f,fill=(*color,round(255*opacity))); x+=cw+tracking*S
    def mark(layer,cx,cy,size,opacity,progress):
        d=ImageDraw.Draw(layer); k=size/110*S; x=cx*S-size*S/2; y=cy*S-size*S/2
        def pts(v): return [(x+a*k,y+b*k) for a,b in v]
        ink=(*IVORY,round(255*opacity)); gold=(*GOLD,round(255*opacity))
        for a,b in [((7,0),(7,102)),((0,95),(92,95)),((98,6),(98,76))]:d.line(pts([a,b]),fill=ink,width=max(1,round(k)))
        for poly in [[(15,20),(39,44),(39,95),(15,71)],[(23,8),(38,8),(99,70),(99,88)]]:d.polygon(pts(poly),fill=ink)
        a=(15,9); b=(102,96); end=(a[0]+(b[0]-a[0])*progress,a[1]+(b[1]-a[1])*progress)
        d.line(pts([a,end]),fill=(13,14,16,round(255*opacity)),width=max(1,round(6*k)))
        d.line(pts([a,end]),fill=gold,width=max(1,round(1.5*k)))
        for u,v,f in [(15,9,0),(52,46,.42),(102,96,1)]:
            if progress>=f:
                px,py=pts([(u,v)])[0]; r=5*k; d.ellipse((px-r,py-r,px+r,py+r),fill=gold)
    def frame(t,kind):
        image=base.copy(); layer=Image.new('RGBA',(W,H)); d=ImageDraw.Draw(layer)
        # A slowly moving sparse lattice and convergence contours, with a quiet central safe area.
        for idx,(u,v,z) in enumerate(points):
            x=(u*1920+math.sin(t*.16+idx)*18)*S; y=(v*1080+math.cos(t*.13+idx)*12)*S
            alpha=round((16+z*23)*(1 if abs(u-.5)>.27 else .3)); r=(1+z)*S
            d.ellipse((x-r,y-r,x+r,y+r),fill=(*GOLD,alpha))
            if idx%4==0 and idx+1<len(points):
                u2,v2,_=points[idx+1]; d.line((x,y,u2*W,v2*H),fill=(*GOLD,10),width=1)
        for j in range(6):
            r=(310+j*65+t*3)*S; cx=W*.5; cy=H*.46
            d.ellipse((cx-r,cy-r*.38,cx+r,cy+r*.38),outline=(*GOLD,12),width=max(1,round(S)))
        if kind=='intro':
            o=window(t,.5,1.3,3.4,4.15)
            text(layer,'TEAM ASHTOJ',424+(1-o)*18,team,o,tracking=5)
            text(layer,'P R E S E N T S',534,mono,o*.8,GOLD)
            o=window(t,4,5.05,10.65,11.8)
            mark(layer,960,345,142,o,ease((t-4.5)/1.8))
            text(layer,'NIRYUKTI',486+(1-o)*24,display,o,tracking=12)
            text(layer,'INDEPENDENT SPARSE OPTIMIZATION ENGINE',646,mono,o*.85,GOLD,tracking=2)
            text(layer,'Mathematics in motion.',709,small,window(t,6.1,7,10.3,11.2)*.7)
            # Gold aperture opens into the screen recording; trim or cross dissolve at this point.
            q=window(t,10.5,11.1,11.4,12)
            d.line((W*.5-W*.34*q,H*.84,W*.5+W*.34*q,H*.84),fill=(*GOLD,round(150*q)),width=max(1,round(2*S)))
            fade=ease(t/.5)*(1-ease((t-11.4)/.6))
        else:
            o=window(t,.35,1.15,6.8,8)
            mark(layer,960,283,108,o,1)
            text(layer,'NIRYUKTI',420,display,o,tracking=12)
            text(layer,'Built to optimize. Designed to be inspected.',577,small,o*.9)
            d.line((W*.38,H*.62,W*.62,H*.62),fill=(*GOLD,round(120*o)),width=max(1,round(S)))
            text(layer,'TEAM ASHTOJ',708,mono,o,GOLD,tracking=3)
            text(layer,'pip install niryukti   /   npm install niryukti',771,mono,o*.65)
            fade=ease(t/.4)*(1-ease((t-7.2)/.8))
        image=Image.alpha_composite(image,layer).convert('RGB')
        if fade<1:image=Image.blend(Image.new('RGB',(W,H)),image,fade)
        return image
    if args.preview:
        for kind,t in [('intro',2),('intro',7.5),('outro',3)]:frame(t,kind).save(OUT/f'{kind}-{t:g}-preview.png')
        return
    for kind,duration in [('intro',12),('outro',8)]:
        target=OUT/f'niryukti-{kind}-1080p.mp4'
        cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for n in range(duration*args.fps):
                proc.stdin.write(frame(n/args.fps,kind).tobytes())
                if n%args.fps==0:print(f'{kind}: {n//args.fps}/{duration}s',flush=True)
        finally:proc.stdin.close()
        if proc.wait():raise RuntimeError('FFmpeg render failed')
        print(target,flush=True)
if __name__=='__main__':main()
