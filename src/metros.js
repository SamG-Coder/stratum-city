export const METROS=Object.freeze({
  nyc:Object.freeze({name:'New York Metro',regions:Object.freeze([
    {id:'manhattan',name:'Manhattan',plan:'manhattan.plan.json',genome:'manhattan.genome.json',links:['brooklyn','queens','bronx','jersey-city']},
    {id:'brooklyn',name:'Brooklyn',plan:'brooklyn.plan.json',genome:'brooklyn.genome.json',links:['manhattan','queens']},
    {id:'queens',name:'Queens / LIC',plan:'queens.plan.json',genome:'queens.genome.json',links:['manhattan','brooklyn','bronx']},
    {id:'bronx',name:'Bronx',plan:'bronx.plan.json',genome:'bronx.genome.json',links:['manhattan','queens']},
    {id:'jersey-city',name:'Jersey City / Newark',plan:'jersey-city.plan.json',genome:'jersey-city.genome.json',links:['manhattan']}
  ])})
});
export const DEFAULT_METRO='nyc',DEFAULT_REGION='manhattan';
export const regionById=(metro,id)=>metro.regions.find(r=>r.id===id);
export async function loadRegion(region){
 const [planResponse,genomeResponse]=await Promise.all([
  fetch(new URL('../cities/'+region.plan,import.meta.url),{cache:'no-cache'}),
  fetch(new URL('../cities/'+region.genome,import.meta.url),{cache:'no-cache'})
 ]);
 if(!planResponse.ok||!genomeResponse.ok)throw Error('Missing farmed metro region '+region.name);
 return{plan:await planResponse.json(),genome:await genomeResponse.json()};
}
