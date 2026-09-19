import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs/promises';import {createHash} from 'node:crypto';
import {compile} from '../vendor/cuda-webshader/compiler/compiler.js';import {SPECS,RENDER_SPECS,compilerOptions} from '../src/kernel-specs.js';import {COMPILER_STAMP} from '../src/compiler-stamp.js';import {Engine,ABI} from '../src/engine.js';import {encodePlan,encodeGenome,validateGenomeFile} from '../src/city-data.js';import {installConstants,mockDevice,mockCanvas} from './mock-device.mjs';
const root=new URL('../',import.meta.url),read=p=>fs.readFile(new URL(p,root),'utf8');
test('all 13 CUDA kernels translate and match their hashed portable artifacts',async()=>{
 assert.equal(SPECS.length,13);const sources=new Map();
 for(const spec of SPECS){for(const n of spec.dependencies)if(!sources.has(n))sources.set(n,await read('kernels/'+n+'.cu'));const source=spec.dependencies.map(n=>sources.get(n)).join('\n');const a=compile(source,compilerOptions(spec));const saved=JSON.parse(await read('generated/'+spec.entry+'.json'));
 assert.equal(a.wgsl,saved.wgsl,spec.entry);assert.equal(a.wgsl,await read('generated/'+spec.entry+'.wgsl'));assert.equal(saved.stratumHash,createHash('sha256').update(COMPILER_STAMP+'|specialize|'+spec.entry+'|'+spec.workgroupSize.join(',')+'|'+source).digest('hex'));assert.ok(a.metadata.bindings.length<=8);assert.ok(a.metadata.workgroupStorageBytes<=16384);}
});
test('plan and genome round-trip; invalid input is rejected before GPU allocation',async()=>{
 const plan=JSON.parse(await read('cities/manhattan.plan.json')),genome=JSON.parse(await read('cities/manhattan.genome.json'));assert.equal(encodePlan(plan).length,320);assert.equal(validateGenomeFile(genome).length,12);assert.throws(()=>encodePlan({...plan,cellSize:0}));assert.throws(()=>encodeGenome([0]));assert.throws(()=>encodeGenome(genome.genes.map((v,i)=>i===4?NaN:v)));assert.equal(genome.search.evaluated,8192);assert.ok(genome.search.train.loss<genome.search.baseline.loss);for(let i=1;i<genome.search.bestTrainingHistory.length;i++)assert.ok(genome.search.bestTrainingHistory[i]<=genome.search.bestTrainingHistory[i-1]+1e-6);
});
test('geometry is shared, finite, non-neural, and reflections query the same scene',async()=>{
 const common=await read('kernels/common.cu'),trace=await read('kernels/city-trace.cu'),reflection=await read('kernels/reflection.cu'),sink=await read('kernels/sink.cu');
 for(const name of ['WORLD_SIDE','WORLD_LOTS','LOT_FLOATS','CLUSTERS','GROUP_NODES','MAX_FEATURES','BUILD_CHUNK','BUILD_CHUNKS'])assert.match(common,new RegExp('#define '+name+' '+ABI[name]+'(?:\\s|$)'));
 assert.equal(ABI.GROUP_NODES,ABI.CLUSTERS*2);assert.doesNotMatch(trace,/macroHit|pageHit|if\(cached\)/);assert.match(trace,/queryLot/);assert.match(reflection,/sceneRay\(World,Nodes,Plan/);assert.match(sink,/featureHit\(f,/);assert.doesNotMatch(sink,/count>=32|n>=PER_CLUSTER/);assert.ok(!RENDER_SPECS.some(s=>s.entry==='probeGrammar'||s.entry==='farmCandidates'));
});
test('host bindings, bounded dispatches, imports and presentation contract (not GPU execution)',async()=>{
 installConstants();const device=mockDevice(),canvas=mockCanvas(),original=globalThis.fetch;
 globalThis.fetch=async url=>{const bytes=await fs.readFile(new URL(url));return{ok:true,json:async()=>JSON.parse(bytes),text:async()=>bytes.toString()};};
 try{const e=await new Engine(canvas).init({device,adapter:{info:{vendor:'API TEST DOUBLE'}},width:128,height:96});e.frame();assert.ok(device.calls.some(c=>c[0]==='present'));assert.ok(device.calls.some(c=>c[0]==='indirect'));assert.equal(e.buffers.Nodes.byteLength,64*1048576);assert.equal(e.buffers.P,undefined);assert.equal(e.buffers.Reflection.byteLength,64*48*16);await e.resize(192,128);e.frame();await e.setGenome(e.genome);await e.dispose();assert.ok(e.disposed);}finally{globalThis.fetch=original;}
});
test('runtime can compile from CUDA without generated/ and exposes driver timing separately',async()=>{
 const loader=await read('src/kernel-loader.js'),pages=await read('tools/pages.mjs');assert.match(loader,/stratumHash===hash/);assert.match(loader,/new Worker/);assert.match(loader,/pipelineMs/);assert.match(pages,/cities/);assert.match(pages,/farm\.html/);
 installConstants();const device=mockDevice(),canvas=mockCanvas(),original=globalThis.fetch;
 globalThis.fetch=async url=>{const u=new URL(url);if(u.pathname.includes('/generated/'))return{ok:false};const bytes=await fs.readFile(u);return{ok:true,json:async()=>JSON.parse(bytes),text:async()=>bytes.toString()};};
 try{const e=await new Engine(canvas).init({device,adapter:{info:{vendor:'API TEST DOUBLE'}},width:64,height:64,warmup:false});assert.equal(e.loader.timings.length,11);assert.ok(e.loader.timings.every(t=>t.source==='runtime'));await e.dispose();}finally{globalThis.fetch=original;}
});
test('app DOM hooks exist and production links target this repository',async()=>{
 const app=await read('src/app.js'),html=await read('index.html');for(const m of app.matchAll(/\$\('([^']+)'\)/g))assert.ok(html.includes('id="'+m[1]+'"'),'Missing #'+m[1]);const pkg=JSON.parse(await read('package.json'));assert.equal(pkg.homepage,'https://samg-coder.github.io/stratum-city/');
});
