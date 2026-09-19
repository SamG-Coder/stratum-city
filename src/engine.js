import {GpuRuntime} from '../vendor/cuda-webshader/runtime/runtime.js';import {KernelLoader} from './kernel-loader.js';import {RENDER_SPECS} from './kernel-specs.js';import {encodePlan,validateGenomeFile} from './city-data.js';
export const ABI=Object.freeze({WORLD_SIDE:128,WORLD_LOTS:16384,LOT_FLOATS:16,CLUSTERS:64,GROUP_NODES:128,MAX_FEATURES:512,BUILD_CHUNK:2048,BUILD_CHUNKS:8,RAY_DISTANCE:8000});
export const BUILD_ID='stratum-city-seed-farm-1';const STAGES=['Camera','Visibility','Reflections','Materials','Resolve'];
const fetchJSON=async url=>{const r=await fetch(url,{cache:'no-cache'});if(!r.ok)throw Error('Missing city data: '+url);return r.json();};
export class Engine{
 constructor(canvas){this.canvas=canvas;this.buffers={};this.kernels={};this.input=new Float32Array(32);this.frames=0;this.errors=[];this.disposed=false;this.timings=null;this.timingBusy=false;this.lastRebuildLots=0;}
 async init({width=1600,height=900,onProgress=()=>{},onError=()=>{},device=null,adapter=null,warmup=true,plan=null,genome=null}={}){
  const ownsDevice=!device;this.onProgress=onProgress;
  [this.plan,this.genome]=await Promise.all([plan||fetchJSON(new URL('../cities/manhattan.plan.json',import.meta.url)),genome||fetchJSON(new URL('../cities/manhattan.genome.json',import.meta.url))]);this.planData=encodePlan(this.plan);this.genes=validateGenomeFile(this.genome);
  if(!device){if(!navigator.gpu)throw Error('WebGPU requires a supported browser on localhost or HTTPS.');adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No WebGPU adapter available.');device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[]});}
  this.device=device;this.adapter=adapter;this.runtime=new GpuRuntime(device,{adapter,ownsDevice,uniformCapacity:262144,onError:e=>{this.errors.push(String(e?.message||e));onError(e);}});this.info=this.runtime.describe();this.context=this.canvas?.getContext('webgpu')??null;
  this.loader=new KernelLoader(this.runtime,onProgress);
  // Kernel compilation is independent. Translate/create pipelines concurrently; the runtime
  // and browser are free to schedule native shader compilation in parallel too.
  onProgress({type:'phase',message:'Compiling '+RENDER_SPECS.length+' GPU programs',progress:.03});
  const loaded=await Promise.all(RENDER_SPECS.map((spec,i)=>this.loader.load(spec.entry,.03+.57*(i+1)/RENDER_SPECS.length)));
  for(let i=0;i<RENDER_SPECS.length;i++)this.kernels[RENDER_SPECS[i].entry]=loaded[i];
  const a=ABI,sizes={World:a.WORLD_LOTS*a.LOT_FLOATS,Nodes:a.WORLD_LOTS*a.GROUP_NODES*8,G:12,Plan:320,C:64,I:32,Queue:a.WORLD_LOTS+1,Args:a.BUILD_CHUNKS*21};
  for(const [name,count]of Object.entries(sizes))this.buffers[name]=this.runtime.createBuffer(count*4,{label:'CITY '+name,usage:name==='Args'?GPUBufferUsage.INDIRECT:0});this.runtime.write(this.buffers.G,this.genes);this.runtime.write(this.buffers.Plan,this.planData);
  if(device.features.has('timestamp-query')){this.queries=device.createQuerySet({type:'timestamp',count:10});this.queryResolve=device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});this.queryRead=device.createBuffer({size:80,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});}
  await this.resize(width,height);this.runtime.batch().dispatch(this.bind('initCamera',{seed:this.genes[0],interiorSeed:this.genes[10]}),[1]).submit();await this.runtime.idle();if(warmup)await this.rebuild();onProgress({type:'ready',message:'Ready',progress:1});return this;
 }
 bind(name,scalars={}){const k=this.kernels[name];if(!k)throw Error('Unknown kernel '+name);return k.bind(Object.fromEntries(k.artifact.metadata.bindings.map(b=>{if(!this.buffers[b.name])throw Error('Missing '+name+'.'+b.name);return[b.name,this.buffers[b.name]];})),scalars);}
 async resize(width,height){
  if(!Number.isFinite(width)||!Number.isFinite(height))throw Error('Invalid dimensions');width=Math.max(64,Math.ceil(width/64)*64);height=Math.max(64,Math.ceil(height/8)*8);if(width>this.device.limits.maxTextureDimension2D||height>this.device.limits.maxTextureDimension2D||width*height*16>this.device.limits.maxStorageBufferBindingSize)throw Error('Resolution exceeds GPU buffer limit.');
  if(this.width===width&&this.height===height)return;await this.runtime.idle();for(const n of ['Hit','Surface','Room','Linear','History','Pixels','Reflection'])if(this.buffers[n])this.runtime.destroyBuffer(this.buffers[n]);this.width=width;this.height=height;
  for(const n of ['Hit','Surface','Room','Linear','History'])this.buffers[n]=this.runtime.createBuffer(width*height*16,{label:n});this.buffers.Pixels=this.runtime.createBuffer(width*height*4,{label:'Final RGBA8'});this.buffers.Reflection=this.runtime.createBuffer(Math.ceil(width/2)*Math.ceil(height/2)*16,{label:'Single-bounce reflections'});
  if(this.canvas){this.canvas.width=width;this.canvas.height=height;}if(this.context)this.context.configure({device:this.device,format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT,alphaMode:'opaque'});
  this.calls={};for(const n of ['clearQueue','prepareLots','planBounds'])this.calls[n]=this.bind(n);this.calls.stepCamera=this.bind('stepCamera',{width,height,dt:0});
  for(const n of ['tracePrimary','reflectPixels','shadePixels'])this.calls[n]=this.bind(n,{width,height,rowStart:0,rowCount:32});this.calls.resolveFrame=this.bind('resolveFrame',{width,height});
  this.chunks=[];for(let i=0;i<ABI.BUILD_CHUNKS;i++){const queueBase=i*ABI.BUILD_CHUNK,c=[this.bind('buildGroupBounds',{queueBase})];for(let first=32;first>=1;first/=2)c.push(this.bind('reduceGroupBounds',{first,queueBase}));this.chunks.push(c);}this.resizeDirty=true;
 }
 async rebuild(){
  await this.runtime.idle();this.runtime.batch().dispatch(this.calls.clearQueue,[1]).dispatch(this.calls.prepareLots,[ABI.WORLD_LOTS/64]).dispatch(this.calls.planBounds,[1]).submit();await this.runtime.idle();
  for(let chunk=0;chunk<ABI.BUILD_CHUNKS;chunk++){const b=this.runtime.batch({label:'Bounds chunk '+chunk});for(let level=0;level<7;level++)b.dispatch(this.chunks[chunk][level],[1],{resource:this.buffers.Args,offset:(chunk*21+level*3)*4});b.submit();await this.runtime.idle();this.onProgress({type:'bounds',message:'City bounds '+(chunk+1)+'/'+ABI.BUILD_CHUNKS,progress:.62+.37*(chunk+1)/ABI.BUILD_CHUNKS,chunk:chunk+1,total:ABI.BUILD_CHUNKS});}
  const q=await this.runtime.read(this.buffers.Queue,Uint32Array,4,0);this.lastRebuildLots=q[0];
 }
 stage(i,timed){return this.runtime.batch({label:STAGES[i],...(timed?{timestampWrites:{querySet:this.queries,beginningOfPassWriteIndex:i*2,endOfPassWriteIndex:i*2+1}}:{})});}
 frame(dt=1/60,{draw=true}={}){
  if(this.disposed)throw Error('Renderer disposed');const timed=!!this.queries&&!this.timingBusy&&draw&&this.frames%24===0;this.input[23]=this.resizeDirty?1:0;this.resizeDirty=false;this.runtime.write(this.buffers.I,this.input);this.input[23]=0;
  this.calls.stepCamera.setScalars({width:this.width,height:this.height,dt:Math.max(0,Math.min(.1,dt))});this.stage(0,timed).dispatch(this.calls.stepCamera,[1]).submit();
  if(draw){
   // Each pixel kernel already has a 2-D global invocation guard. Dispatch the full image
   // once instead of slicing it into 32-row bands. The old path emitted ~150 compute
   // dispatches per frame at 1920p and repeatedly rewrote scalar uniforms; that CPU/WebGPU
   // submission overhead was large enough to hide the actual GPU timings.
   for(const [stage,name,scale]of [[1,'tracePrimary',1],[2,'reflectPixels',2],[3,'shadePixels',1]]){
    const b=this.stage(stage,timed),w=Math.ceil(this.width/scale),h=Math.ceil(this.height/scale);
    b.dispatch(this.calls[name].setScalars({rowStart:0,rowCount:h}),[Math.ceil(w/8),Math.ceil(h/8)]).submit();
   }
   const b=this.stage(4,timed);b.dispatch(this.calls.resolveFrame,[Math.ceil(this.width/8),Math.ceil(this.height/8)]).endPass();if(this.context)b.encoder.copyBufferToTexture({buffer:this.buffers.Pixels.gpuBuffer,bytesPerRow:this.width*4,rowsPerImage:this.height},{texture:this.context.getCurrentTexture()},[this.width,this.height]);
   if(timed){b.encoder.resolveQuerySet(this.queries,0,10,this.queryResolve,0);b.encoder.copyBufferToBuffer(this.queryResolve,0,this.queryRead,0,80);}b.submit();if(timed)this.readTimings();
  }this.frames++;
 }
 async readTimings(){this.timingBusy=true;try{await this.queryRead.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(this.queryRead.getMappedRange());this.timings=STAGES.map((label,i)=>({label,ms:Number(t[i*2+1]-t[i*2])/1e6}));this.queryRead.unmap();}catch{this.timings=null;}finally{this.timingBusy=false;}}
 async inspect(){const c=await this.runtime.read(this.buffers.C);return{build:BUILD_ID,mode:'finite-shape-plan',city:this.plan.name,genomeSeed:this.genes[0],search:Number.isFinite(this.genome.search?.train?.loss)&&Number.isFinite(this.genome.search?.audit?.loss)?{evaluated:Number(this.genome.search.evaluated)||0,train:this.genome.search.train.loss,audit:this.genome.search.audit.loss}:null,camera:Array.from(c),geometryModels:1,primitivePageAllocation:0,accelerationLots:ABI.WORLD_LOTS,accelerationBytes:this.buffers.Nodes.byteLength,allocatedBytes:Object.values(this.buffers).reduce((a,b)=>a+b.byteLength,0),width:this.width,height:this.height,frames:this.frames,timings:this.timings,startup:this.loader.timings,adapter:this.info,errors:this.errors};}
 async screenshotPixels(){return this.runtime.read(this.buffers.Pixels,Uint8Array);}
 async setCity(plan,value){const planData=encodePlan(plan),genes=validateGenomeFile(value);if(value.planHash){const bytes=new Uint8Array(await crypto.subtle.digest('SHA-256',planData));const hash=Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');if(hash!==value.planHash)throw Error('This genome was farmed for a different shape plan.');}await this.runtime.idle();this.plan=plan;this.planData=planData;this.genome=value;this.genes=genes;this.input.fill(0);this.runtime.write(this.buffers.Plan,planData);this.runtime.write(this.buffers.G,genes);this.runtime.batch().clear(this.buffers.World).clear(this.buffers.History).clear(this.buffers.Queue).dispatch(this.bind('initCamera',{seed:genes[0],interiorSeed:genes[10]}),[1]).submit();await this.rebuild();}
 async setGenome(value){return this.setCity(this.plan,value);}
 async reset(){return this.setGenome(this.genome);}
 async dispose(){if(this.disposed)return;await this.runtime.idle();this.disposed=true;this.loader.dispose();this.queries?.destroy();this.queryResolve?.destroy();this.queryRead?.destroy();this.runtime.dispose();}
}
