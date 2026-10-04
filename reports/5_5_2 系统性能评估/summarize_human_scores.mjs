// 仅汇总评阅人已填写的评分；空白或部分填写不计为0分，也不补模型分数。
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const [input, agentPath, output] = process.argv.slice(2);
if (!output) throw new Error('需要人工评分xlsx、原始Agent JSON和新输出JSON三个参数');
try { await fs.access(output); throw new Error('输出已存在，请使用新路径'); }
catch(error) { if(error.code !== 'ENOENT') throw error; }
const agent=JSON.parse(await fs.readFile(agentPath,'utf8'));
const expected=new Set(agent.rows.map(r=>`${r.id}:${r.profile}`));
const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const rows=workbook.worksheets.getItem('人工评分').getRange(`A6:H${agent.rows.length+5}`).values;
const accepted=[],pending=[],seen=new Set();
for(const row of rows) {
  const [id,profile,correctness,completeness,citationAccuracy,comment,reviewer,reviewedAt]=row;
  const key=`${id}:${profile}`;
  if(!expected.has(key) || seen.has(key)) throw new Error(`评分记录题号/配置错误或重复：${key}`);
  seen.add(key);
  const scores=[correctness,completeness,citationAccuracy];
  if(scores.every(v=>v===null || v==='')) {pending.push(key);continue;}
  if(!scores.every(v=>Number.isInteger(v) && v>=0 && v<=4) || !String(comment??'').trim() ||
     !String(reviewer??'').trim() || reviewedAt===null || reviewedAt==='') {
    throw new Error(`评分未填完整或分数不在0～4整数范围：${key}`);
  }
  accepted.push({id,profile,correctness,completeness,citation_accuracy:citationAccuracy,
                 comment,reviewer,reviewed_at:reviewedAt});
}
const summaries={};
for(const profile of Object.keys(agent.profiles)) {
  const part=accepted.filter(r=>r.profile===profile);
  summaries[profile]={reviewed:part.length,expected:agent.rows.filter(r=>r.profile===profile).length};
  for(const key of ['correctness','completeness','citation_accuracy']) {
    summaries[profile][`${key}_mean`]=part.length?part.reduce((sum,r)=>sum+r[key],0)/part.length:null;
  }
}
await fs.writeFile(output,JSON.stringify({status:accepted.length===expected.size?'completed':'awaiting_human_review',
  source_sha256:crypto.createHash('sha256').update(await fs.readFile(input)).digest('hex'),
  scoring_method:'用户提供的评阅人评分；不包含助手审阅或自动匹配分数',
  expected:expected.size,reviewed:accepted.length,pending,summaries,rows:accepted},null,2)+'\n');
console.log(`已记录${accepted.length}/${expected.size}条人工评分，空白保持待评分。`);
