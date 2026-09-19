import {Engine,BUILD_ID} from './engine.js';
import {METROS,DEFAULT_METRO,DEFAULT_REGION,regionById,loadRegion} from './metros.js';
const $=id=>document.getElementById(id);
const canvas=$('world'),engine=new Engine(canvas),params=new URLSearchParams(location.search);
const keys=new Set(),pulse=new Float32Array(32);let ready=false,faulted=false,busy=false,hidden=false,dragging=false;
let width=[768,1024,1600,1920].includes(Number(params.get('width')))?Number(params.get('width')):(matchMedia('(pointer:coarse)').matches?768:1600);
let seed=Number(params.get('seed')??1788);if(!Number.isInteger(seed)||seed<0||seed>9999)seed=1788;
let last=0,mouseX=0,mouseY=0,wheel=0,resizeTimer,noticeTimer,polling=false,resizePending=false;
let frameStart=0,completed=0,displayFPS=0,frameWindow=performance.now(),lastPoll=0,lastInfo=null;
const testing=params.has('test');
let metro=METROS[params.get('metro')]||METROS[DEFAULT_METRO],region=regionById(metro,params.get('region'))||regionById(metro,DEFAULT_REGION);
const compileConsole=$('compile-console'),compileCount=$('compile-count'),compilePercent=$('compile-percent'),compileLines=$('compile-lines'),compileEta=$('compile-eta'),compileJobs=new Map();let compileDone=0,compileTotal=11,totalLines=0,finishedLines=0,startupBegan=performance.now(),lastEta=0;
function formatEta(ms){if(!Number.isFinite(ms)||ms<=0)return'Calculating ETA…';const s=Math.ceil(ms/1000);return s<60?`~${s}s remaining`:`~${Math.floor(s/60)}m ${s%60}s remaining`;}
function updateEstimate(){
 const elapsed=performance.now()-startupBegan,progress=totalLines>0?Math.min(.98,finishedLines/totalLines):compileDone/Math.max(1,compileTotal);
 if(progress>.03){const estimate=elapsed*(1-progress)/progress;lastEta=lastEta?lastEta*.7+estimate*.3:estimate;compileEta.textContent=formatEta(lastEta);}
 compileLines.textContent=totalLines?`${totalLines.toLocaleString()} CUDA lines · ${finishedLines.toLocaleString()} processed`:'Scanning CUDA source…';
}
const stateLabel={['compile-start']:'TRANSLATING',['compile-done']:'WGSL READY',['trim-done']:'TRIMMED',['pipeline-start']:'GPU PIPELINE',done:'READY'};
function ensureJob(entry){
 let row=compileJobs.get(entry);if(row)return row;
 row=document.createElement('div');row.className='console-line active';row.dataset.entry=entry;
 const icon=document.createElement('i');icon.className='job-icon';icon.append(document.createElement('span'));
 const copy=document.createElement('div');copy.className='job-copy';const title=document.createElement('b'),sub=document.createElement('small');title.textContent=entry;sub.textContent='Queued';copy.append(title,sub);
 const state=document.createElement('em');state.textContent='QUEUED';row.append(icon,copy,state);compileConsole.append(row);compileJobs.set(entry,row);return row;
}
function startupProgress(event,fallback){
 const e=typeof event==='string'?{message:event,progress:fallback}:event||{},p=Math.max(0,Math.min(1,Number(e.progress??fallback??0)));
 if(e.entry&&e.lines){const row0=compileJobs.get(e.entry);if(!row0||!row0.dataset.lines){totalLines+=Number(e.lines);if(row0)row0.dataset.lines=String(e.lines);}}
 updateEstimate();
 $('progress').style.width=`${Math.round(p*100)}%`;compilePercent.textContent=`${Math.round(p*100)}%`;
 if(e.type==='phase'){
  const m=String(e.message||'Preparing GPU programs'),n=m.match(/Compiling (\d+)/);if(n)compileTotal=Number(n[1]);
  compileCount.textContent=`${compileDone} / ${compileTotal}`;$('status').textContent='Preparing GPU programs';
  const boot=compileConsole.querySelector('[data-entry="boot"]');if(boot){boot.className='console-line done';boot.querySelector('b').textContent='City definition loaded';boot.querySelector('small').textContent='Shape plan + farmed genome';boot.querySelector('em').textContent='READY';}
  return;
 }
 if(e.entry){
  const row=ensureJob(e.entry);if(e.lines&&!row.dataset.lines){row.dataset.lines=String(e.lines);totalLines+=Number(e.lines);}const sub=row.querySelector('small'),state=row.querySelector('em');row.className='console-line '+(e.type==='done'?'done':'active');
  if(e.type==='queued')sub.textContent=`${Number(e.lines||0).toLocaleString()} CUDA lines · queued`;else if(e.type==='compile-start')sub.textContent=`${Number(e.lines||0).toLocaleString()} CUDA lines · CUDA source → portable WGSL`;
  else if(e.type==='compile-done')sub.textContent='Translation complete · optimizing WGSL';
  else if(e.type==='trim-done'){const t=e.trim,cut=t?.beforeBytes?100*(1-t.afterBytes/t.beforeBytes):0;sub.textContent=`WGSL ${Math.round((t?.beforeBytes||0)/1024)}KB → ${Math.round((t?.afterBytes||0)/1024)}KB · -${cut.toFixed(0)}% · ${t?.removedFunctions||0} helpers removed · ${Math.round(t?.trimMs||0)}ms`;}
  else if(e.type==='pipeline-start'){const t=e.trim,cut=t?.beforeBytes?100*(1-t.afterBytes/t.beforeBytes):0;sub.textContent=`${String(e.message||'GPU pipeline').replace(/^GPU pipeline\s*[·:]?\s*/,'')||'Creating WebGPU pipeline'} · WGSL -${cut.toFixed(0)}%`;}
  else if(e.type==='done'){sub.textContent=e.timing?`${Number(e.lines||0).toLocaleString()} lines · ${e.message} · trim ${Math.round(e.timing.trimMs||0)}ms · pipeline ${Math.round(e.timing.pipelineMs)}ms`:String(e.message||'Complete');compileDone++;if(!row.dataset.counted){finishedLines+=Number(e.lines||0);row.dataset.counted='1';}updateEstimate();}
  state.textContent=e.type==='done'&&e.timing?Math.round(e.timing.loadOrTranslateMs+e.timing.pipelineMs)+'ms':(stateLabel[e.type]||'WORKING');
  compileCount.textContent=`${compileDone} / ${compileTotal}`;$('status').textContent=compileDone===compileTotal?'GPU programs ready':`Compiling GPU programs · ${compileDone}/${compileTotal}`;compileConsole.scrollTop=compileConsole.scrollHeight;return;
 }
 if(e.type==='bounds'){
  const row=ensureJob('city bounds');row.className='console-line active';row.querySelector('small').textContent='Exact feature acceleration structure';row.querySelector('em').textContent=Math.round(e.chunk/e.total*100)+'%';$('status').textContent='Building city acceleration';compileConsole.scrollTop=compileConsole.scrollHeight;return;
 }
 if(e.type==='ready'){finishedLines=totalLines;updateEstimate();compileEta.textContent='Complete';const row=compileJobs.get('city bounds');if(row){row.className='console-line done';row.querySelector('small').textContent='Exact feature acceleration structure';row.querySelector('em').textContent='READY';}$('status').textContent='STRATUM CITY ready';compilePercent.textContent='100%';}
}
const views={1:['A city, selected.<br>Not streamed.','Finite Manhattan-inspired plan. One detailed geometry definition.'],2:['Life between towers.','Storefronts, fire escapes and windows with interior depth.'],3:['A second skyline.','Glass towers and stepped masonry above a lower-rise city.'],4:['The shape plan.','Island, waterfront, parks and skyline anchors are explicit constraints.'],5:['At the waterline.','One-bounce city reflections and procedural water.'],6:['Behind the glass.','Seeded interiors change with the viewing angle.']};
function notice(text){$('notice').textContent=text;$('notice').classList.add('show');clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>$('notice').classList.remove('show'),3800);}
function fatal(error){if(faulted)return;faulted=true;console.error(error);$('boot').hidden=false;$('status').textContent='The renderer could not start.';$('detail').textContent=String(error?.message??error);$('retry').hidden=false;document.body.classList.remove('ready');}
$('retry').onclick=()=>location.reload();
function chooseView(id){if(!ready)return;pulse[8]=id;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',Number(b.dataset.view)===id));const v=views[id];if(v){$('place-title').innerHTML=v[0];$('place-sub').textContent=v[1];}}
function toggleNotes(){const p=$('notes');p.classList.toggle('closed');}
function action(slot,text){if(!ready)return;pulse[slot]=1;if(text)notice(text);}
$('quality').value=String(width);$('seed-label').textContent='Loading…';$('build-label').textContent=BUILD_ID;
const regionSelect=$('region-select');for(const r of metro.regions){const o=document.createElement('option');o.value=r.id;o.textContent=r.name;regionSelect.append(o);}regionSelect.value=region.id;
async function switchRegion(id){if(!ready||busy||id===region.id)return;const next=regionById(metro,id);if(!next)return;busy=true;ready=false;regionSelect.disabled=true;$('boot').hidden=false;$('status').textContent='Generating '+next.name;try{const data=await loadRegion(next);await engine.setCity(data.plan,data.genome);region=next;const u=new URL(location.href);u.searchParams.set('metro',DEFAULT_METRO);u.searchParams.set('region',region.id);history.replaceState(null,'',u);$('region-label').textContent=region.name.toUpperCase();clearControls();last=0;ready=true;$('boot').hidden=true;await poll();notice(region.name+' · farmed regional genome loaded');}catch(e){fatal(e);}finally{busy=false;regionSelect.disabled=false;}}
regionSelect.onchange=e=>switchRegion(e.target.value);
$('quality').onchange=e=>{width=Number(e.target.value);queueResize();};
$('info-button').onclick=toggleNotes;$('close-notes').onclick=toggleNotes;
$('fullscreen').onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen().catch(()=>notice('Fullscreen is not available.'));

$('reflection-toggle').onclick=()=>action(20);$('interior-toggle').onclick=()=>action(21);$('debug-button').onclick=()=>action(10);$('shadow-toggle').onclick=()=>action(14);$('tour').onclick=()=>action(16);
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>chooseView(Number(b.dataset.view));
$('export-stats').onclick=()=>{if(lastInfo)downloadBlob(new Blob([JSON.stringify({...lastInfo,seed,capturedAt:new Date().toISOString()},null,2)],{type:'application/json'}),'stratum-measurements.json');};
function downloadBlob(blob,name){const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function screenshot(){if(!ready)return;try{const pixels=await engine.screenshotPixels();const c=document.createElement('canvas');c.width=engine.width;c.height=engine.height;c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels),c.width,c.height),0,0);c.toBlob(blob=>{if(blob)downloadBlob(blob,'stratum-frame.png');});}catch(e){notice(e.message);}}
addEventListener('keydown',e=>{
 if(e.target instanceof HTMLSelectElement)return;
 if(['Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Tab'].includes(e.code))e.preventDefault();
 if(!ready||faulted)return;
 keys.add(e.code);if(e.repeat)return;
 const digits={Digit1:1,Digit2:2,Digit3:3,Digit4:4,Digit5:5,Digit6:6};if(digits[e.code])chooseView(digits[e.code]);
 const actions={KeyJ:14,KeyP:16,KeyV:10,KeyR:20,KeyI:21};if(actions[e.code])action(actions[e.code]);
 if(e.code==='KeyH'||e.code==='Tab')toggleNotes();
 if(e.code==='KeyF')$('fullscreen').click();
 if(e.code==='KeyC'){hidden=!hidden;document.body.classList.toggle('clean',hidden);}
 if(e.code==='KeyK')screenshot();
 if(e.code==='Escape'){dragging=false;keys.clear();$('notes').classList.add('closed');}
});
addEventListener('keyup',e=>keys.delete(e.code));
function clearControls(){keys.clear();mouseX=0;mouseY=0;wheel=0;pulse.fill(0);dragging=false;}
addEventListener('blur',clearControls);addEventListener('visibilitychange',()=>{clearControls();last=0;});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{e.preventDefault();canvas.focus();dragging=true;canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(dragging){mouseX+=e.movementX;mouseY+=e.movementY;}});
canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointercancel',clearControls);
canvas.addEventListener('wheel',e=>{e.preventDefault();wheel+=e.deltaY;},{passive:false});
function fillInput(){
 const i=engine.input;i.fill(0);
 i[0]=Number(keys.has('KeyW')||keys.has('ArrowUp'))-Number(keys.has('KeyS')||keys.has('ArrowDown'));
 i[1]=Number(keys.has('KeyD')||keys.has('ArrowRight'))-Number(keys.has('KeyA')||keys.has('ArrowLeft'));
 i[2]=Number(keys.has('KeyE')||keys.has('Space'))-Number(keys.has('KeyQ'));
 i[3]=mouseX;i[4]=mouseY;i[5]=Number(keys.has('ShiftLeft')||keys.has('ShiftRight'));i[6]=Number(keys.has('ControlLeft')||keys.has('ControlRight'));i[7]=wheel;
 i[9]=Number(keys.has('BracketRight'))-Number(keys.has('BracketLeft'));
 i[13]=Number(keys.has('Equal'))-Number(keys.has('Minus'));
 for(let k=0;k<pulse.length;k++)if(pulse[k])i[k]=pulse[k];pulse.fill(0);mouseX=0;mouseY=0;wheel=0;
}
function dimensions(){return [width,Math.round(width*innerHeight/innerWidth)];}
function queueResize(){clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>resizePending=true,150);}
addEventListener('resize',queueResize);
const fmt=n=>Math.round(n).toLocaleString();const mib=n=>(n/1048576).toFixed(1)+' MiB';
async function poll(){
 if(polling)return;polling=true;
 try{
  const v=await engine.inspect();lastInfo=v;const c=v.camera;
  $('seed-label').textContent=String(v.genomeSeed);$('pages-count').textContent=fmt(v.accelerationLots)+' bound records';
  $('search-count').textContent=v.search?fmt(v.search.evaluated):'Imported';$('search-loss').textContent=v.search?v.search.train.toFixed(4)+' / '+v.search.audit.toFixed(4):'—';
  $('sample-count').textContent=String(Math.round(c[19]))+' / 64';$('cache-memory').textContent=mib(v.accelerationBytes);$('total-memory').textContent=mib(v.allocatedBytes);
  $('reflection-toggle').textContent='R · City reflections '+(c[25]>.5?'ON':'OFF');$('interior-toggle').textContent='I · Parallax interiors '+(c[26]>.5?'ON':'OFF');
  $('camera-height').textContent=`${c[1]<10?c[1].toFixed(2):fmt(c[1])} m`;
  $('render-size').textContent=`${v.width} × ${v.height}`;
  if(v.timings){$('timings').replaceChildren(...v.timings.map(({label,ms})=>{const row=document.createElement('div');row.className='timing-row';const a=document.createElement('span'),b=document.createElement('strong');a.textContent=label;b.textContent=ms.toFixed(2)+' ms';row.append(a,b);return row;}));}
  $('tour').classList.toggle('active',c[22]>.5);
 }catch(e){fatal(e);}finally{polling=false;}
}
async function frame(now){
 requestAnimationFrame(frame);
 if(!ready||faulted||testing||document.hidden)return;
 if(busy)return;
 if(resizePending){resizePending=false;busy=true;try{await engine.resize(...dimensions());}catch(e){fatal(e);}finally{busy=false;last=0;}return;}
 const dt=last?Math.min(.1,(now-last)/1000):1/60;last=now;fillInput();busy=true;frameStart=performance.now();
 try{engine.frame(dt);await engine.device.queue.onSubmittedWorkDone();completed++;}catch(e){fatal(e);}finally{busy=false;}
 if(now-frameWindow>=1000){displayFPS=completed*1000/(now-frameWindow);completed=0;frameWindow=now;$('frame-time').textContent=`${displayFPS.toFixed(1)} FPS`;}
 if(now-lastPoll>650){lastPoll=now;poll();}
}
async function start(){try{
 const [w,h]=dimensions(),initial=await loadRegion(region);$('region-label').textContent=region.name.toUpperCase();await engine.init({width:w,height:h,onProgress:startupProgress,onError:fatal,plan:initial.plan,genome:initial.genome});
 if(faulted)return;ready=true;document.body.classList.add('ready');$('boot').hidden=true;canvas.focus();
 window.stratum={engine,ready:true,chooseView,screenshot,switchRegion,metro,inspect:()=>engine.inspect()};
 engine.frame(1/60);await engine.runtime.idle();await poll();requestAnimationFrame(frame);
 if(matchMedia('(pointer:coarse)').matches)notice('This build uses keyboard flight controls. Drag to look; use the view buttons to explore.');
}catch(e){fatal(e);}}

$('export-genome').onclick=()=>{if(ready)downloadBlob(new Blob([JSON.stringify(engine.genome,null,2)],{type:'application/json'}),'stratum-city-genome.json');};
$('import-genome').onclick=()=>{if(ready&&!busy)$('genome-file').click();};
$('genome-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{const value=JSON.parse(await file.text());while(busy)await new Promise(r=>setTimeout(r,20));busy=true;ready=false;$('boot').hidden=false;await engine.setGenome(value);ready=true;$('boot').hidden=true;clearControls();last=0;await poll();notice('Imported genome · exact city rebuilt');}catch(e){notice(e.message);if(!engine.disposed)ready=true;$('boot').hidden=true;}finally{busy=false;e.target.value='';}};
if(matchMedia('(pointer:coarse)').matches)$('notes').classList.add('closed');
start();
