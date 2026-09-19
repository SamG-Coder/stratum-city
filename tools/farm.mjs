import fs from 'node:fs/promises';import path from 'node:path';import os from 'node:os';import {spawnSync} from 'node:child_process';import {createHash} from 'node:crypto';import {fileURLToPath} from 'node:url';
import {encodePlan,encodeGenome,GENOME_NAMES,GENE_LIMITS} from '../src/city-data.js';
const root=fileURLToPath(new URL('../',import.meta.url));const args=process.argv.slice(2);const option=(name,def)=>{const i=args.indexOf(name);return i<0?def:args[i+1];};
const planPath=path.resolve(root,option('--plan','cities/manhattan.plan.json')),output=path.resolve(root,option('--out','cities/manhattan.genome.json'));
const candidates=Number(option('--candidates','8192')),searchSeed=Number(option('--seed','20260919'));
if(!Number.isInteger(candidates)||candidates<64||candidates>1000000||!Number.isInteger(searchSeed))throw Error('Invalid search budget or seed.');
const raw=await fs.readFile(planPath,'utf8'),plan=JSON.parse(raw),data=encodePlan(plan),temp=await fs.mkdtemp(path.join(os.tmpdir(),'stratum-farm-'));
try{
 const bin=path.join(temp,process.platform==='win32'?'farm.exe':'farm'),planBin=path.join(temp,'plan.f32');await fs.writeFile(planBin,Buffer.from(data.buffer));
 const cxx=process.env.CXX||'g++';let r=spawnSync(cxx,['-std=c++17','-O2','tools/seed-farm.cpp','-o',bin],{cwd:root,encoding:'utf8'});
 if(r.error)throw Error('Offline seed farming needs a C++17 compiler. Install g++/LLVM and set CXX; ordinary city viewing does not require it.');if(r.status!==0)throw Error(r.stderr||'Seed farmer did not compile.');
 console.log(`Evaluating ${candidates.toLocaleString()} genomes against ${plan.name}...`);r=spawnSync(bin,[planBin,String(candidates),String(searchSeed)],{encoding:'utf8',maxBuffer:16*1024*1024,timeout:300000});if(r.status!==0)throw Error(r.stderr||'Seed farming failed.');const search=JSON.parse(r.stdout);search.genes=search.genes.map((v,i)=>Math.max(GENE_LIMITS[i][0],Math.min(GENE_LIMITS[i][1],v)));encodeGenome(search.genes);
 const digest=createHash('sha256');for(const name of ['kernels/common.cu','kernels/plan.cu','kernels/farm.cu'])digest.update(await fs.readFile(path.join(root,name)));
 const genome={schema:'stratum.city-genome.v1',name:plan.name+' — farmed',planFile:path.basename(planPath),planHash:createHash('sha256').update(Buffer.from(data.buffer)).digest('hex'),descriptorHash:digest.digest('hex'),geneNames:GENOME_NAMES,genes:search.genes,search};
 await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(genome,null,2)+'\n');
 console.log(`Baseline loss ${search.baseline.loss.toFixed(6)} → selected training ${search.train.loss.toFixed(6)}; fresh audit ${search.audit.loss.toFixed(6)}.`);console.log(output);
}finally{await fs.rm(temp,{recursive:true,force:true});}
