import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
const here=path.dirname(fileURLToPath(import.meta.url));
const pdfjs=await import(pathToFileURL(process.argv[2]).href);
const out=path.join(here,'outputs','seed');await fs.mkdir(out,{recursive:true});
const records=[];
const sha=b=>createHash('sha256').update(b).digest('hex');
for (const [edition,first,last] of [['2018',103,146],['2021',112,155]]) {
  const input=await fs.readFile(path.join(here,'inputs','seed',`wmd_${edition}.pdf`));
  const pdf=await pdfjs.getDocument({data:new Uint8Array(input),useSystemFonts:false,disableFontFace:true}).promise;
  const lines=[];
  for(let n=first;n<=last;n++) {
    const page=await pdf.getPage(n);const content=await page.getTextContent();
    content.items.forEach((v,i)=> {if(v.str!==undefined)lines.push(JSON.stringify({record_type:'text',page:n,item_index:i,x:v.transform[4],y:v.transform[5],width:v.width,height:v.height,text:v.str}));});
  }
  const name=`WMD${edition}_coordinates.jsonl.gz`;const data=gzipSync(lines.join('\n')+'\n');
  await fs.writeFile(path.join(out,name),data);
  records.push({edition,source_sha256:sha(input),coordinate_file:name,coordinate_sha256:sha(data),text_items:lines.length});
  await pdf.destroy();console.log('Raw PDF coordinates rebuilt',edition,lines.length);
}
await fs.writeFile(path.join(out,'coordinate_extraction.json'),JSON.stringify({engine:'pdfjs-dist',version:pdfjs.version,script_sha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),records},null,2));
