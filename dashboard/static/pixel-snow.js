'use strict';
// PixelSnow shader supplied from React Bits (https://reactbits.dev).
// Adapted to native WebGL2 for the existing dependency-free dashboard.
(()=>{
const fragmentSource=`#version 300 es

precision highp float;
precision highp int;
out vec4 snowColor;

uniform float uTime;
uniform vec2 uResolution;
uniform float uFlakeSize;
uniform float uMinFlakeSize;
uniform float uPixelResolution;
uniform float uSpeed;
uniform float uDepthFade;
uniform float uFarPlane;
uniform vec3 uColor;
uniform float uBrightness;
uniform float uGamma;
uniform float uDensity;
uniform float uVariant;
uniform float uDirection;

// Precomputed constants
#define PI 3.14159265
#define PI_OVER_6 0.5235988
#define PI_OVER_3 1.0471976
#define INV_SQRT3 0.57735027
#define M1 1597334677U
#define M2 3812015801U
#define M3 3299493293U
#define F0 2.3283064e-10

// Optimized hash - inline multiplication
#define hash(n) (n * (n ^ (n >> 15)))
#define coord3(p) (uvec3(p).x * M1 ^ uvec3(p).y * M2 ^ uvec3(p).z * M3)

// Precomputed camera basis vectors (normalized vec3(1,1,1), vec3(1,0,-1))
const vec3 camK = vec3(0.57735027, 0.57735027, 0.57735027);
const vec3 camI = vec3(0.70710678, 0.0, -0.70710678);
const vec3 camJ = vec3(-0.40824829, 0.81649658, -0.40824829);

// Precomputed branch direction
const vec2 b1d = vec2(0.574, 0.819);

vec3 hash3(uint n) {
  uvec3 hashed = hash(n) * uvec3(1U, 511U, 262143U);
  return vec3(hashed) * F0;
}

float snowflakeDist(vec2 p) {
  float r = length(p);
  float a = atan(p.y, p.x);
  a = abs(mod(a + PI_OVER_6, PI_OVER_3) - PI_OVER_6);
  vec2 q = r * vec2(cos(a), sin(a));
  float dMain = max(abs(q.y), max(-q.x, q.x - 1.0));
  float b1t = clamp(dot(q - vec2(0.4, 0.0), b1d), 0.0, 0.4);
  float dB1 = length(q - vec2(0.4, 0.0) - b1t * b1d);
  float b2t = clamp(dot(q - vec2(0.7, 0.0), b1d), 0.0, 0.25);
  float dB2 = length(q - vec2(0.7, 0.0) - b2t * b1d);
  return min(dMain, min(dB1, dB2)) * 10.0;
}

void main() {
  // Precompute reciprocals to avoid division
  float invPixelRes = 1.0 / uPixelResolution;
  float pixelSize = max(1.0, floor(0.5 + uResolution.x * invPixelRes));
  float invPixelSize = 1.0 / pixelSize;
  
  vec2 fragCoord = floor(gl_FragCoord.xy * invPixelSize);
  vec2 res = uResolution * invPixelSize;
  float invResX = 1.0 / res.x;

  vec3 ray = normalize(vec3((fragCoord - res * 0.5) * invResX, 1.0));
  ray = ray.x * camI + ray.y * camJ + ray.z * camK;

  // Precompute time-based values
  float timeSpeed = uTime * uSpeed;
  float windX = cos(uDirection) * 0.4;
  float windY = sin(uDirection) * 0.4;
  vec3 camPos = (windX * camI + windY * camJ + 0.1 * camK) * timeSpeed;
  vec3 pos = camPos;

  // Precompute ray reciprocal for strides
  vec3 absRay = max(abs(ray), vec3(0.001));
  vec3 strides = 1.0 / absRay;
  vec3 raySign = step(ray, vec3(0.0));
  vec3 phase = fract(pos) * strides;
  phase = mix(strides - phase, phase, raySign);

  // Precompute for intersection test
  float rayDotCamK = dot(ray, camK);
  float invRayDotCamK = 1.0 / rayDotCamK;
  float invDepthFade = 1.0 / uDepthFade;
  float halfInvResX = 0.5 * invResX;
  vec3 timeAnim = timeSpeed * 0.1 * vec3(7.0, 8.0, 5.0);

  float t = 0.0;
  for (int i = 0; i < 128; i++) {
    if (t >= uFarPlane) break;
    
    vec3 fpos = floor(pos);
    uint cellCoord = coord3(fpos);
    float cellHash = hash3(cellCoord).x;

    if (cellHash < uDensity) {
      vec3 h = hash3(cellCoord);
      
      // Optimized flake position calculation
      vec3 sinArg1 = fpos.yzx * 0.073;
      vec3 sinArg2 = fpos.zxy * 0.27;
      vec3 flakePos = 0.5 - 0.5 * cos(4.0 * sin(sinArg1) + 4.0 * sin(sinArg2) + 2.0 * h + timeAnim);
      flakePos = flakePos * 0.8 + 0.1 + fpos;

      float toIntersection = dot(flakePos - pos, camK) * invRayDotCamK;
      
      if (toIntersection > 0.0) {
        vec3 testPos = pos + ray * toIntersection - flakePos;
        float testX = dot(testPos, camI);
        float testY = dot(testPos, camJ);
        vec2 testUV = abs(vec2(testX, testY));
        
        float depth = dot(flakePos - camPos, camK);
        float flakeSize = max(uFlakeSize, uMinFlakeSize * depth * halfInvResX);
        
        // Avoid branching with step functions where possible
        float dist;
        if (uVariant < 0.5) {
          dist = max(testUV.x, testUV.y);
        } else if (uVariant < 1.5) {
          dist = length(testUV);
        } else {
          float invFlakeSize = 1.0 / flakeSize;
          dist = snowflakeDist(vec2(testX, testY) * invFlakeSize) * flakeSize;
        }

        if (dist < flakeSize) {
          float flakeSizeRatio = uFlakeSize / flakeSize;
          float intensity = exp2(-(t + toIntersection) * invDepthFade) *
                           min(1.0, flakeSizeRatio * flakeSizeRatio) * uBrightness;
          snowColor = vec4(uColor, pow(intensity, uGamma));
          return;
        }
      }
    }

    float nextStep = min(min(phase.x, phase.y), phase.z);
    vec3 sel = step(phase, vec3(nextStep));
    phase = phase - nextStep + strides * sel;
    t += nextStep;
    pos = mix(pos + ray * nextStep, floor(pos + ray * nextStep + 0.5), sel);
  }

  snowColor = vec4(0.0);
}
`;
const canvas=document.querySelector('#pixel-snow');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const gl=canvas.getContext('webgl2',{alpha:true,antialias:false,depth:false,stencil:false,powerPreference:'low-power',premultipliedAlpha:false});
if(!gl){canvas.hidden=true;return;}
function compile(type,source){
  const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
  if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){gl.deleteShader(shader);return null;}
  return shader;
}
const vertex=compile(gl.VERTEX_SHADER,`#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0.0,1.0);}`);
const fragment=compile(gl.FRAGMENT_SHADER,fragmentSource);
if(!vertex||!fragment){if(vertex)gl.deleteShader(vertex);if(fragment)gl.deleteShader(fragment);canvas.hidden=true;return;}
const program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);
gl.deleteShader(vertex);gl.deleteShader(fragment);
if(!gl.getProgramParameter(program,gl.LINK_STATUS)){gl.deleteProgram(program);canvas.hidden=true;return;}
gl.useProgram(program);
const uniform=name=>gl.getUniformLocation(program,name);
const settings={uFlakeSize:.01,uMinFlakeSize:1.25,uPixelResolution:300,uSpeed:1.1,uDepthFade:5,uFarPlane:10,uBrightness:.3,uGamma:.4545,uDensity:.25,uVariant:0,uDirection:80*Math.PI/180};
Object.entries(settings).forEach(([key,value])=>gl.uniform1f(uniform(key),value));
// Champagne flakes on graphite; deeper bronze on paper so they stay visible.
function tint(){const dark=document.documentElement.dataset.theme==='dark';gl.uniform3f(uniform('uColor'),...(dark?[.86,.74,.54]:[.6,.46,.26]));}
tint();
const timeUniform=uniform('uTime'),resolution=uniform('uResolution');
let timer=0,elapsed=0,lost=false,resizeTimer=0;
function allowed(){return effectsEnabled&&!reduced.matches&&!document.hidden&&!lost&&(!state.job||state.job.state==='finished');}
function draw(){if(lost)return;gl.uniform1f(timeUniform,elapsed);gl.drawArrays(gl.TRIANGLES,0,3);}
function frame(){timer=0;if(!allowed())return;elapsed+=1/24;draw();timer=setTimeout(frame,1000/24);}
function sync(){clearTimeout(timer);timer=0;if(allowed())frame();else if(!document.hidden)draw();}
function resize(){
  if(lost)return;
  // Fixed budget, independent of device pixel ratio and large desktop resolutions.
  const scale=Math.min(1,480/innerWidth,600/innerHeight);
  canvas.width=Math.max(1,Math.round(innerWidth*scale));canvas.height=Math.max(1,Math.round(innerHeight*scale));
  gl.viewport(0,0,canvas.width,canvas.height);gl.uniform2f(resolution,canvas.width,canvas.height);draw();
}
function queueResize(){clearTimeout(resizeTimer);resizeTimer=setTimeout(resize,100);}
canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();lost=true;clearTimeout(timer);canvas.hidden=true;});
// Context loss leaves the ordinary gradient/grid background available.
document.addEventListener('visibilitychange',sync);
document.addEventListener('solver-state',sync);
document.addEventListener('theme-change',()=>{if(!lost){tint();draw();}});
document.querySelector('#effects-toggle').addEventListener('click',sync);
reduced.addEventListener('change',sync);
window.addEventListener('resize',queueResize);
window.addEventListener('pagehide',()=>{clearTimeout(timer);clearTimeout(resizeTimer);});
window.addEventListener('pageshow',sync);
resize();sync();
})();
