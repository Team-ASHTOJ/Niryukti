/* Visual inspiration: Zepa UI wave-hero, https://zepa.design/components/wave-hero.
   Original reference credited to franky-adl (2026), MIT, fluid-cursor-demo.
   This dependency-free SVG implementation uses finite, event-triggered ripples. */
'use strict';
window.VantageWave=(()=>{
  let cleanup=()=>{};
  function mount(host,allowed){
    cleanup();
    if(!host)return;
    const ns='http://www.w3.org/2000/svg';
    const svg=document.createElementNS(ns,'svg');
    svg.setAttribute('viewBox','0 0 500 340');svg.setAttribute('aria-hidden','true');
    svg.classList.add('wave-field');
    const tiles=[],animations=new Set();
    for(let row=0;row<12;row++)for(let column=0;column<12;column++){
      const x=250+(column-row)*20,y=45+(column+row)*10;
      const group=document.createElementNS(ns,'g');
      const depth=13;
      const faces=[
        [`M${x-18} ${y}l18 9v${depth}l-18 -9Z`,'wave-left'],
        [`M${x+18} ${y}l-18 9v${depth}l18 -9Z`,'wave-right'],
        [`M${x} ${y-9}l18 9-18 9-18 -9Z`,'wave-top']
      ];
      for(const [d,className] of faces){const path=document.createElementNS(ns,'path');path.setAttribute('d',d);path.setAttribute('class',className);group.append(path);}
      svg.append(group);tiles.push({group,x,y});
    }
    host.prepend(svg);
    let visible=false,last=-Infinity;
    const stop=()=>{for(const animation of animations)animation.cancel();animations.clear();};
    const ripple=(x,y)=>{
      if(!visible||!allowed()||document.hidden||performance.now()-last<1100)return;
      last=performance.now();
      for(const tile of tiles){
        const distance=Math.hypot(tile.x-x,(tile.y-y)*1.5);
        const lift=12*Math.exp(-distance/350);
        const animation=tile.group.animate([
          {transform:'translateY(0)',opacity:1},
          {transform:`translateY(-${lift}px)`,opacity:1,offset:.4},
          {transform:'translateY(0)',opacity:1}
        ],{duration:650,delay:distance*1.5,easing:'ease-in-out'});
        animations.add(animation);animation.onfinish=()=>animations.delete(animation);
      }
    };
    const move=e=>{
      if(e.pointerType==='touch')return;
      const rect=svg.getBoundingClientRect();
      // Account for SVG's centered, aspect-preserving viewport.
      const scale=Math.min(rect.width/500,rect.height/340);
      if(!scale)return;
      ripple((e.clientX-rect.left-(rect.width-500*scale)/2)/scale,(e.clientY-rect.top-(rect.height-340*scale)/2)/scale);
    };
    const observer=new IntersectionObserver(entries=>{
      visible=entries[0].isIntersecting;
      if(visible)ripple(250,150);else stop();
    },{threshold:.2});observer.observe(host);
    const motion=matchMedia('(prefers-reduced-motion: reduce)');
    host.addEventListener('pointermove',move);
    document.addEventListener('visibilitychange',stop);
    document.addEventListener('submit',stop,true);
    document.querySelector('#effects-toggle').addEventListener('click',stop);
    motion.addEventListener('change',stop);
    cleanup=()=>{
      stop();observer.disconnect();host.removeEventListener('pointermove',move);
      document.removeEventListener('visibilitychange',stop);document.removeEventListener('submit',stop,true);
      document.querySelector('#effects-toggle').removeEventListener('click',stop);
      motion.removeEventListener('change',stop);svg.remove();
    };
  }
  return {mount,dispose:()=>{cleanup();cleanup=()=>{};}};
})();
