'use strict';
// Visual inspiration: Zepa UI wave-hero, originally by @franky-adl (MIT).
// https://zepa.design/components/wave-hero
// Independent lightweight Canvas 2D implementation; no simulation telemetry.
(()=>{
  let dispose=()=>{};
  const palettes={
    light:{left:'#E4E1DA',right:'#D3CFC6',top:[247,245,241],crest:[200,162,108]},
    dark:{left:'#1C1D20',right:'#131416',top:[40,41,45],crest:[207,172,116]}
  };
  const palette=()=>palettes[document.documentElement.dataset.theme==='dark'?'dark':'light'];
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  function mount(){
    dispose();
    const canvas=document.querySelector('[data-page="overview"] .wave-grid');
    if(!canvas)return;
    const ctx=canvas.getContext('2d');if(!ctx)return;
    const host=canvas.parentElement;
    let timer=0,visible=false,phase=0,lastPointer=0;
    const ripples=[];
    function allowed(){return visible&&!document.hidden&&effectsEnabled&&!reduced.matches&&(!state.job||state.job.state==='finished');}
    function face(points,color){
      ctx.fillStyle=color;ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fill();
    }
    function draw(animated=false){
      ctx.clearRect(0,0,640,360);
      for(let row=0;row<16;row++)for(let col=0;col<16;col++){
        const x=320+(col-row)*18,y=32+(col+row)*8;
        let wave=animated?Math.sin(Math.hypot(col-7.5,row-7.5)*.85-phase*1.4)*4:0;
        for(const ripple of ripples){
          const age=phase-ripple.at,distance=Math.hypot(col-ripple.x,row-ripple.y);
          wave+=Math.exp(-((distance-age*6)**2)/5)*Math.exp(-age)*14;
        }
        const height=9+Math.max(-3,wave),top=y-height;
        // Crests warm from the paper/graphite base toward brand gold as they rise.
        const p=palette(),t=Math.min(1,Math.max(0,(height-7)/12)),mix=i=>Math.round(p.top[i]+(p.crest[i]-p.top[i])*t);
        face([[x-16,top],[x,top+7],[x,y+7],[x-16,y]],p.left);
        face([[x,top+7],[x+16,top],[x+16,y],[x,y+7]],p.right);
        face([[x,top-7],[x+16,top],[x,top+7],[x-16,top]],`rgb(${mix(0)},${mix(1)},${mix(2)})`);
      }
    }
    function frame(){
      timer=0;
      if(!allowed()){draw();return;}
      phase+=1/24;
      while(ripples.length&&phase-ripples[0].at>3)ripples.shift();
      draw(true);timer=setTimeout(frame,1000/24);
    }
    function sync(){clearTimeout(timer);timer=0;ripples.length=0;if(allowed())frame();else draw();}
    function pointer(e){
      if(!allowed()||performance.now()-lastPointer<100)return;
      lastPointer=performance.now();
      const rect=canvas.getBoundingClientRect();
      const x=(e.clientX-rect.left)/rect.width*640-320,y=(e.clientY-rect.top)/rect.height*360-32;
      ripples.push({x:(y/8+x/18)/2,y:(y/8-x/18)/2,at:phase});
      if(ripples.length>6)ripples.shift();
    }
    const observer=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;sync();},{threshold:.1});observer.observe(canvas);
    host.addEventListener('pointermove',pointer,{passive:true});
    document.addEventListener('visibilitychange',sync);
    document.addEventListener('solver-state',sync);
    document.addEventListener('theme-change',sync);
    document.querySelector('#effects-toggle').addEventListener('click',sync);
    reduced.addEventListener('change',sync);
    draw();
    dispose=()=>{
      clearTimeout(timer);observer.disconnect();host.removeEventListener('pointermove',pointer);
      document.removeEventListener('visibilitychange',sync);document.removeEventListener('solver-state',sync);document.removeEventListener('theme-change',sync);
      document.querySelector('#effects-toggle').removeEventListener('click',sync);reduced.removeEventListener('change',sync);
    };
  }
  window.VantageWave={mount,dispose:()=>dispose()};
})();
