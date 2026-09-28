#!/usr/bin/env python3
"""Niryukti cinematic motion system: procedural 3D, particles and kinetic type.

Offline, deterministic artwork. The geometry is editorial, not solver telemetry.
Python + NumPy + Pillow + OpenCV + FFmpeg; project fonts only.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import math
import subprocess
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
GOLD = np.array([207., 172., 116.])
BRIGHT = np.array([249., 224., 180.])
IVORY = (236, 233, 227)
FPS = 60
DURATIONS = {'intro': 24, 'outro': 12}
cv2.setNumThreads(2)


def clamp(x):
    return max(0., min(1., x))


def smooth(x):
    x = clamp(x)
    return x*x*(3-2*x)


def out(x):
    return 1-(1-clamp(x))**3


def gate(t, start, attack, release, end):
    return smooth((t-start)/(attack-start))*(1-smooth((t-release)/(end-release)))


def rotate(points, yaw, pitch=0, roll=0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return np.asarray(points) @ (rz @ rx @ ry).T


class Film:
    def __init__(self, width):
        self.w = width
        self.h = int(round(width*9/16/2))*2
        self.s = width/1920
        y, x = np.mgrid[0:self.h, 0:self.w]
        r = ((x/self.w-.5)/.85)**2 + ((y/self.h-.45)/.8)**2
        light = np.clip(1-r, 0, 1)
        rng = np.random.default_rng(26119)
        noise = rng.uniform(-.6, .6, (self.h, self.w))
        self.base = np.stack([7+light*7+noise, 9+light*7+noise, 12+light*7+noise], 2).astype(np.uint8)
        self.dust = rng.uniform(0, 1, (210, 5))
        self.starts = rng.uniform([-1300, -850], [1300, 850], (220, 2))
        self.curves = rng.uniform(-1, 1, (220, 2))
        self.mark_segments = [
            [(7, 0), (7, 102)], [(0, 95), (92, 95)], [(98, 6), (98, 76)],
            [(15, 20), (39, 44), (39, 95), (15, 71), (15, 20)],
            [(23, 8), (38, 8), (99, 70), (99, 88), (23, 8)],
            [(15, 9), (102, 96)],
        ]
        contour = []
        for seg in self.mark_segments:
            for a, b in zip(seg, seg[1:]):
                for f in np.linspace(0, 1, 18):
                    contour.append(np.array(a)*(1-f)+np.array(b)*f)
        contour = np.asarray(contour)
        self.targets = contour[np.linspace(0, len(contour)-1, 220).astype(int)]
        self.targets = (self.targets-55)*2.45 + np.array([960, 332])
        self.preview_times = {'intro': [1.7, 3.6, 7.2, 11.8, 14.5, 18.6, 22.4], 'outro': [2.5, 5.5, 9]}

    def pt(self, v):
        return tuple(np.rint(np.asarray(v)*self.s).astype(int))

    def line(self, pts, intensity=.3, width=1, glow=False, color=GOLD, close=False):
        pts = np.asarray(pts)
        if len(pts)<2 or intensity <= .001:
            return
        tone = np.asarray(color)
        floor = np.array([14.,16.,19.]) if np.sum(tone)>100 else np.zeros(3)
        c = tuple(int(v) for v in np.clip(floor+(tone-floor)*intensity, 0, 255))
        cv2.polylines(self.canvas, [np.rint(pts*self.s).astype(np.int32)], close, c, max(1, round(width*self.s)), cv2.LINE_AA)
        if glow:
            cv2.polylines(self.bloom, [np.rint(pts*self.s/4).astype(np.int32)], close, c, max(1, round(width*self.s/4)), cv2.LINE_AA)

    def dot(self, p, radius=2, intensity=1, glow=False, color=GOLD):
        if intensity <= .003: return
        tone = np.asarray(color)
        floor = np.array([14.,16.,19.]) if np.sum(tone)>100 else np.zeros(3)
        c = tuple(int(v) for v in np.clip(floor+(tone-floor)*intensity, 0, 255))
        cv2.circle(self.canvas, self.pt(p), max(1, round(radius*self.s)), c, -1, cv2.LINE_AA)
        if glow:
            cv2.circle(self.bloom, tuple(np.rint(np.asarray(p)*self.s/4).astype(int)), max(1, round(radius*self.s/3)), c, -1, cv2.LINE_AA)

    def projection(self, points, center=(960, 480), zoom=1080, distance=5.2):
        p = np.asarray(points)
        depth = np.maximum(.8, distance-p[:, 2])
        return np.stack([center[0]+p[:, 0]*zoom/depth, center[1]-p[:, 1]*zoom/depth], 1)

    @lru_cache(maxsize=40)
    def font(self, size, kind='display', weight=400):
        name = {'display':'Sora-Variable.ttf', 'mono':'font-0.ttf', 'body':'font-2.ttf'}[kind]
        f = ImageFont.truetype(str(ROOT/'dashboard/static/fonts'/name), max(1, round(size*self.s)))
        if kind == 'display':
            f.set_variation_by_axes([weight])
        return f

    @lru_cache(maxsize=512)
    def glyph(self, char, size, kind, weight):
        f = self.font(size, kind, weight)
        advance = f.getlength(char)
        mask = Image.new('L', (max(1, math.ceil(advance)+8), round(size*1.7*self.s)))
        ImageDraw.Draw(mask).text((0,0), char, font=f, fill=255)
        return mask, advance

    def title(self, image, content, y, size, t, start, stop, spacing=4, color=IVORY, kind='display', weight=400, stagger=.045, attack=1.1):
        total_duration = attack+len(content)*stagger
        opacity = 1-smooth((t-stop)/.8)
        if t<start or opacity<=0:
            return
        glyphs = [self.glyph(c,size,kind,weight) for c in content]
        settle = out((t-start)/total_duration)
        tracking = (spacing+(1-settle)*9)*self.s
        total = sum(a for _, a in glyphs)+tracking*(len(glyphs)-1)
        x = (self.w-total)/2
        for i,(mask,advance) in enumerate(glyphs):
            q = out((t-start-i*stagger)/attack)
            alpha = q*opacity
            if alpha>0:
                yy = y*self.s + (1-q)*38*self.s - (1-opacity)*10*self.s
                scaled_mask = mask.point(lambda v: int(v*alpha))
                image.paste(tuple(color), (int(x),int(yy)), scaled_mask)
            x += advance+tracking

    def label(self, image, content, x, y, opacity, size=15, color=GOLD, align='left'):
        if opacity<=0:
            return
        f = self.font(size, 'mono')
        mask = Image.new('L', (int(f.getlength(content))+4, round(size*1.7*self.s)))
        ImageDraw.Draw(mask).text((0,0), content,font=f,fill=round(255*clamp(opacity)))
        if align=='right': x-=mask.width/self.s
        image.paste(tuple(int(v) for v in color), self.pt((x,y)), mask)

    def field(self, t, amount):
        # Depth-separated dust, fixed-size pixel marks, passing light packets.
        for i,(u,v,z,speed,phase) in enumerate(self.dust):
            x = (u*2100-90 + math.sin(t*.17+phase*6)*(12+z*32))
            y = ((v*1200-t*(4+speed*17))%1200)-60
            a = amount*(.17+z*.4)
            self.dot((x,y), .7+z*1.5, a)
            if i%13==0:
                self.line([(x,y),(x+12+z*20,y-3)],a*.55)
        # Architectural perspective floor with an animated scanning wave.
        horizon = 697
        for i in range(-15,16):
            self.line([(960+i*31,horizon),(960+i*210,1180)],amount*.10,color=np.array([108,111,119]))
        for k in range(12):
            z = ((k/12+t*.016)%1)**2
            y = horizon+z*500
            self.line([(0,y),(1920,y)],amount*(.055+.10*z),color=np.array([120,119,113]))
        for side in [0,1]:
            x0=92 if side==0 else 1640
            for row in range(8):
                for col in range(8):
                    active = (row*7+col*3)%11<3
                    scan = math.exp(-((row-(t*2.1)%11)/1.25)**2)
                    a=amount*(.11+.48*scan if active else .045)
                    x=x0+col*23; y=405+row*23
                    self.line([(x,y),(x+5,y)],a,2)
            self.line([(x0-12,397),(x0-18,397),(x0-18,583),(x0-12,583)],amount*.3)
            self.line([(x0+172,397),(x0+178,397),(x0+178,583),(x0+172,583)],amount*.3)

    def orbit(self, t, amount, radius=2.45, center=(960, 478), zoom=1120, build=1):
        angles = np.linspace(0, math.tau*build, 150)
        for j in range(5):
            points = np.stack([radius*np.cos(angles), radius*np.sin(angles), np.zeros_like(angles)],1)
            points = rotate(points,t*.10+j*.64,.45+j*.31,.15*math.sin(t*.2))
            projected = self.projection(points,center,zoom)
            self.line(projected,amount*(.15+.025*j),1)
            if build>.9:
                theta=(t*(.40+j*.055)+j*1.4)%math.tau
                seg=np.linspace(theta-.35,theta,18)
                points=rotate(np.stack([radius*np.cos(seg),radius*np.sin(seg),np.zeros_like(seg)],1),t*.10+j*.64,.45+j*.31,.15*math.sin(t*.2))
                projected=self.projection(points,center,zoom)
                self.line(projected,amount*.85,1.6,True)
                self.dot(projected[-1],2.8,amount,True,BRIGHT)

    def polytope(self,t,amount,zoom):
        if amount <= .003: return
        verts=[]
        for j in range(10):
            a=j*math.tau/10
            verts.append([1.5*math.cos(a),.78,1.5*math.sin(a)])
            verts.append([1.5*math.cos(a+.2),-.78,1.5*math.sin(a+.2)])
        verts.extend([[0,1.65,0],[0,-1.65,0]])
        world=rotate(verts,t*.32,.25+math.sin(t*.3)*.18,.15)
        p=self.projection(world,zoom=zoom)
        faces=[]
        for j in range(10):
            a=j*2; b=((j+1)%10)*2
            faces.extend([(20,a,b),(21,a+1,b+1),(a,b,b+1),(a,a+1,b+1)])
        # Depth-sorted translucent bronze facets give the opening object volume.
        facets=self.canvas.copy()
        for tri in sorted(faces,key=lambda f:float(np.mean(world[list(f),2]))):
            normal=np.cross(world[tri[1]]-world[tri[0]],world[tri[2]]-world[tri[0]])
            normal/=max(1e-8,np.linalg.norm(normal))
            illumination=.5+.5*abs(float(np.dot(normal,[.25,.65,.7])))
            colour=tuple(int(c*illumination) for c in [68,58,44])
            cv2.fillPoly(facets,[np.rint(p[list(tri)]*self.s).astype(np.int32)],colour,cv2.LINE_AA)
        self.canvas=cv2.addWeighted(self.canvas,1-.43*amount,facets,.43*amount,0)
        edges=[]
        for j in range(10):
            a=j*2; b=((j+1)%10)*2
            edges.extend([(a,b),(a+1,b+1),(a,a+1),(a,b+1),(20,a),(21,a+1)])
        for a,b in edges:
            depth=(world[a,2]+world[b,2])/4
            self.line([p[a],p[b]],amount*(.25+.18*(depth+1)),1.1)
        for i,pt in enumerate(p):self.dot(pt,2.4,amount*.7,True)
        # A bright path advances along actual polytope edges.
        path=[20,0,2,4,5,7,9,21]
        prog=(t*.65)% (len(path)-1)
        idx=int(prog); f=prog-idx
        if idx<len(path)-1:
            pt=p[path[idx]]*(1-f)+p[path[idx+1]]*f
            self.line([p[path[idx]],pt],amount,2.4,True,BRIGHT)
            self.dot(pt,4,amount,True,BRIGHT)

    def surface(self,t,amount):
        # A moving camera over a convex bowl, with a converging golden trajectory.
        def world(x,z):
            y=.14*(x*x+z*z)-1.0
            return np.stack([x,y,z],axis=-1)
        def project(p):
            return self.projection(rotate(p,.22+math.sin(t*.10)*.10,.45),center=(960,845),zoom=850,distance=7.8)
        xs=np.linspace(-4.8,4.8,60)
        for k,z in enumerate(np.linspace(-3.5,3.5,20)):
            p=project(world(xs,np.full_like(xs,z)))
            self.line(p,amount*(.12+.12*k/20),1)
        zs=np.linspace(-3.5,3.5,45)
        for x in np.linspace(-4.8,4.8,27):
            self.line(project(world(np.full_like(zs,x),zs)),amount*.18,1)
        f=np.linspace(0,1,100)
        radius=3.5*(1-f)
        theta=f*math.tau*1.25+t*.11
        p=project(world(radius*np.cos(theta),radius*np.sin(theta)))
        self.line(p,amount*.65,1.8,True)
        head=(t*.19)%1
        part=p[max(0,int(head*100)-10):max(2,int(head*100))]
        self.line(part,amount,2.5,True,BRIGHT)
        if len(part):self.dot(part[-1],3.5,amount,True,BRIGHT)
        self.dot(project(world(np.array([0.]),np.array([0.])))[0],4,amount*.9,True,BRIGHT)

    def assembled_mark(self,t,opacity,center=(960,332),size=224,progress=1):
        if opacity <= .003: return
        k=size/110
        def transform(seg):return (np.asarray(seg)-55)*k+np.asarray(center)
        # Gold construction strokes resolve into the solid coordinate-frame mark.
        for seg in self.mark_segments:
            pts=transform(seg)
            n=len(pts)-1
            q=progress*n
            idx=min(n-1,int(q))
            if progress>0:
                shown=list(pts[:idx+1])+[pts[idx]+(pts[idx+1]-pts[idx])*min(1,q-idx)]
                self.line(shown,opacity*.85,1.4,True)
        solid=smooth((progress-.60)/.40)*opacity
        for poly in self.mark_segments[3:5]:
            pts=np.rint(transform(poly)*self.s).astype(np.int32)
            cv2.fillPoly(self.canvas,[pts],tuple(int(c*solid) for c in IVORY),cv2.LINE_AA)
        for seg in self.mark_segments[:3]:self.line(transform(seg),opacity*.7,1,color=np.asarray(IVORY))
        pts=transform([(15,9),(102,96)])
        if solid>.1:
            self.line(pts,1,7,color=np.array([13,15,18]))
            self.line(pts,opacity,1.7,True)
        for point in [(15,9),(52,46),(102,96)]:self.dot(transform([point])[0],4.5*k,opacity,True)

    def assembly(self,t,amount):
        if amount <= .003: return
        q=out((t-13.05)/2.35)
        for i in range(len(self.starts)):
            lag=(i%17)*.014
            p=out((t-13.05-lag)/2.1)
            start=self.starts[i]+[960,450]
            target=self.targets[i]
            bend=self.curves[i]*220*math.sin(math.pi*p)
            position=start*(1-p)+target*p+bend
            oldp=max(0,p-.025*(1-p))
            previous=start*(1-oldp)+target*oldp+self.curves[i]*220*math.sin(math.pi*oldp)
            a=amount*(.40+.60*p)*(1-smooth((t-15.0)/.7))
            self.line([previous,position],a,1.2,True)
            self.dot(position,1.5,a,True)
        self.assembled_mark(t,amount,progress=q)

    def frame(self,t,kind):
        self.canvas=self.base.copy()
        self.bloom=np.zeros((max(1,self.h//4),max(1,self.w//4),3),np.uint8)
        text=[]
        if kind=='intro':
            alive=smooth((t-.5)/1.5)
            self.field(t,alive*(.6+.4*smooth((t-3)/2)))
            hook=gate(t,.2,1.5,4.2,5.25)
            # A single pulse grows into a volumetric mathematical structure.
            pulse=smooth((t-.15)/1.7)
            self.orbit(t,hook,zoom=500+650*out(t/3)+950*smooth((t-4.2)/1.1),build=pulse)
            self.polytope(t,hook,580+620*out(t/3)+1000*smooth((t-4.2)/1.1))
            for i in range(32):
                a=i*math.tau/32+t*.08
                r=30+out((t-.4-i*.012)/2.1)*580
                f=gate(t,.45+i*.015,1.2+i*.015,2.7,3.7)
                self.line([(960+math.cos(a)*r*.95,478+math.sin(a)*r*.7),(960+math.cos(a)*(r+45),478+math.sin(a)*(r+45)*.7)],f*.6,1.5,True)
            team=gate(t,4.5,5.4,9.15,10.1)
            self.orbit(t,team*.68,radius=3.6,zoom=1020,build=1)
            self.surface(t,team*.55)
            # Fine gold rules open out around the team name.
            spread=out((t-5.0)/1.2)
            for y in [386,674]:self.line([(960-390*spread,y),(960+390*spread,y)],team*.4,1)
            presents=gate(t,10.0,10.8,12.35,13.1)
            self.orbit(t,presents*.8,radius=1.0+smooth((t-10)/2)*.35,zoom=1080)
            self.dot((960,394),3,presents,True)
            self.line([(960,410),(960,458)],presents*.6,1,True)
            hero=gate(t,13.0,14.0,22.4,23.6)
            self.orbit(t,hero*.55,radius=3.4,center=(960,422),zoom=1180)
            self.surface(t,hero*.95)
            self.assembly(t,hero)
            # Two tracing rails flank the title without crossing its reading area.
            for side in [-1,1]:
                q=out((t-16)/1.5)
                pts=[(960+side*630,630),(960+side*745,630),(960+side*790,585),(960+side*855,585)]
                self.line(pts,hero*q*.55,1.4,True)
                self.dot(pts[-1],2,hero*q*.8,True)
            fade=1-smooth((t-23.5)/.5)
        else:
            alive=smooth(t/.7)
            self.field(t+24,alive)
            self.orbit(t+24,gate(t,0,.8,10.3,11.8)*.65,radius=3.4,center=(960,460),zoom=1150)
            self.surface(t+24,gate(t,.2,1.2,10,11.7)*.8)
            self.assembled_mark(t,gate(t,.15,1,10.4,11.6),center=(960,298),size=164,progress=out(t/1.6))
            spread=out((t-4)/1)
            self.line([(960-spread*400,658),(960+spread*400,658)],gate(t,4,4.8,10.4,11.6)*.5,1,True)
            fade=smooth(t/.3)*(1-smooth((t-11.3)/.7))
        # Optical bloom is restricted to bright trajectories and nodes.
        bloom=cv2.GaussianBlur(self.bloom,(0,0),3.1)
        bloom=cv2.resize(bloom,(self.w,self.h),interpolation=cv2.INTER_LINEAR)
        self.canvas=cv2.addWeighted(self.canvas,1,bloom,.65,0)
        image=Image.fromarray(self.canvas)
        if kind=='intro':
            self.title(image,'A WORLD OF POSSIBILITIES',774,19,t,1.5,3.65,3,GOLD.astype(int).tolist(),'mono',stagger=.018,attack=.65)
            self.title(image,'TEAM ASHTOJ',450,100,t,4.85,9.15,7,weight=500,attack=1.0)
            self.title(image,'IDEAS  /  MATHEMATICS  /  ENGINEERING',614,17,t,5.7,9.10,2,GOLD.astype(int).tolist(),'mono',stagger=.015,attack=.7)
            self.title(image,'PRESENTS',504,35,t,10.1,12.35,13,weight=300,stagger=.055,attack=.7)
            self.title(image,'NIRYUKTI',468,139,t,15.0,22.6,12,weight=500,stagger=.075,attack=1.2)
            self.title(image,'INDEPENDENT SPARSE OPTIMIZATION ENGINE',663,19,t,16.2,22.3,2,GOLD.astype(int).tolist(),'mono',stagger=.012,attack=.8)
            self.title(image,'MODEL    /    SOLVE    /    VERIFY',730,17,t,17.5,22.1,2,IVORY,'mono',stagger=.015,attack=.8)
            labels=gate(t,16,17,21.7,22.4)
            self.label(image,'SPARSE STRUCTURES',85,358,labels*.6,13)
            self.label(image,'VERIFIABLE SOLUTIONS',1835,358,labels*.6,13,align='right')
            self.label(image,'N / 01',85,943,gate(t,1,2,22,23)*.55,14)
            self.label(image,'MATHEMATICS IN MOTION',1835,943,gate(t,16,17,22,23)*.6,14,align='right')
            # A fast gold ribbon provides a visible editorial transition at the tail.
            wipe=smooth((t-22.5)/1.0)
            if 0<wipe<1:
                overlay=Image.new('RGBA',image.size)
                d=ImageDraw.Draw(overlay)
                x=-450+wipe*2900
                d.polygon([self.pt((x-90,-100)),self.pt((x+50,-100)),self.pt((x-440,1180)),self.pt((x-580,1180))],fill=(207,172,116,round(80*math.sin(math.pi*wipe))))
                image=Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')
        else:
            self.title(image,'NIRYUKTI',412,119,t,.5,10.2,11,weight=500,stagger=.07)
            self.title(image,'Built to optimize. Designed to be inspected.',568,29,t,2,10,0,IVORY,'body',stagger=.014,attack=.8)
            self.title(image,'TEAM ASHTOJ',701,23,t,4.1,10.2,5,GOLD.astype(int).tolist(),'mono',stagger=.05)
            self.title(image,'pip install niryukti    /    npm install niryukti',771,20,t,5.4,10.4,0,IVORY,'mono',stagger=.012,attack=.75)
            self.label(image,'INDEPENDENT OPTIMIZATION',85,945,gate(t,3,4,10,11)*.65,13)
            self.label(image,'THANK YOU FOR WATCHING',1835,945,gate(t,6,7,10,11)*.65,13,align='right')
        if fade<1:
            image=Image.blend(Image.new('RGB',image.size),image,clamp(fade))
        return image


def render(film,kind,fps,encoder):
    target=OUT/f'niryukti-{kind}-{film.h}p.mp4'
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{film.w}x{film.h}','-r',str(fps),'-i','-','-an']
    if encoder=='nvenc':cmd+=['-c:v','h264_nvenc','-preset','p6','-tune','hq','-rc','vbr','-cq','17','-b:v','0']
    else:cmd+=['-c:v','libx264','-preset','fast','-crf','17']
    cmd+=['-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    start=time.monotonic()
    try:
        for n in range(round(DURATIONS[kind]*fps)):
            proc.stdin.write(film.frame(n/fps,kind).tobytes())
            if n%fps==0:print(f'{kind} {n//fps}/{DURATIONS[kind]} s | render elapsed {time.monotonic()-start:.1f}s',flush=True)
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('FFmpeg failed')
    print(f'Finished: {target}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--fps',type=int,default=60)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--kind',choices=['intro','outro','both'],default='both')
    parser.add_argument('--encoder',choices=['cpu','nvenc'],default='cpu')
    args=parser.parse_args()
    if args.width<320 or args.width%2 or args.fps<=0:parser.error('Use an even width >= 320 and a positive FPS.')
    film=Film(args.width)
    kinds=['intro','outro'] if args.kind=='both' else [args.kind]
    if args.preview:
        thumbs=[]
        for kind in kinds:
            for t in film.preview_times[kind]:
                im=film.frame(t,kind)
                im.save(OUT/f'{kind}-{t:g}-preview.png')
                thumb=im.resize((640,360),Image.Resampling.LANCZOS)
                ImageDraw.Draw(thumb).text((14,14),f'{kind.upper()} / {t:04.1f}s',fill=IVORY)
                thumbs.append(thumb)
        sheet=Image.new('RGB',(640*2,360*math.ceil(len(thumbs)/2)))
        for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%2)*640,(i//2)*360))
        sheet.save(OUT/'storyboard.jpg',quality=94)
    else:
        for kind in kinds:render(film,kind,args.fps,args.encoder)

if __name__=='__main__':main()
