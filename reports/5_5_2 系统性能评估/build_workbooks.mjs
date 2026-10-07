// 从原始评测结果制作可复算性能表与待填写人工评分表，不用助手分数冒充人工分数。
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [retrievalPath, agentPath, outputDir] = process.argv.slice(2);
if (!outputDir) throw new Error('需要检索JSON、Agent JSON和输出目录三个参数');
const retrieval = JSON.parse(await fs.readFile(retrievalPath, 'utf8'));
const agent = JSON.parse(await fs.readFile(agentPath, 'utf8'));
if (retrieval.status !== 'completed' || agent.status !== 'completed') throw new Error('实验尚未完成');
// 结果允许放在按日期归档的子目录；评测集始终从本仓库reports读取。
const reportsRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const questions = JSON.parse(await fs.readFile(path.join(reportsRoot,'评测集.json'), 'utf8'));
const manifest = JSON.parse(await fs.readFile(path.join(reportsRoot,'5_5_1 评测集构建/论文清单.json'), 'utf8'));
const qa = new Map(questions.map(q => [q.id, q]));
const paperHashes = new Map(manifest.papers.map(p => [p.id, p.doc_id]));
const profileLabels = Object.fromEntries(Object.entries(agent.profiles).map(([key, p]) => [key, p.label]));
await fs.mkdir(outputDir, { recursive: true });
for (const name of ['系统性能评测数据.xlsx', '答案质量人工评分表.xlsx']) {
  try { await fs.access(`${outputDir}/${name}`); throw new Error(`输出已存在：${name}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}

function styleTable(sheet, range, header, widths) {
  sheet.showGridLines = false;
  sheet.getRange(range).format.font = { name: 'Arial', size: 10 };
  sheet.getRange(range).format.verticalAlignment = 'center';
  sheet.getRange(range).format.rowHeight = 23;
  sheet.getRange(header).format = { fill: '#243B53', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
                                   rowHeight: 34, wrapText: true, horizontalAlignment: 'center' };
  widths.forEach(([col, width]) => { sheet.getRange(`${col}:${col}`).format.columnWidth = width; });
}

const wb = Workbook.create();
const summary = wb.worksheets.add('性能汇总');
const detail = wb.worksheets.add('检索逐题');
const decisions = wb.worksheets.add('Agent逐题');
detail.getRange('A1:O1').values = [['题号','配置','类型','问题语言','首个命中块排名','标注页数','命中标注页数',
  '标注论文数','命中标注论文数','检索秒数','Hit@5','MRR@5','Recall@5','论文覆盖率@5','全部论文命中@5']];
const rawRetrieval = retrieval.rows.map(r => {
  const q = qa.get(r.id);
  const gold = new Set(q.evidence.map(e => `${paperHashes.get(e.paper_id)}:${e.page_number}`));
  const found = new Set();
  for (const chunk of r.top5) {
    const m = chunk.metadata;
    for (let p = m.page_number; p <= (m.page_end ?? m.page_number); p++) {
      if (gold.has(`${m.doc_id}:${p}`)) found.add(m.doc_id);
    }
  }
  return [r.id,r.profile,r.category,r.language,r.first_relevant_rank,r.gold_pages,r.matched_pages,
          q.paper_ids.length,found.size,r.seconds];
});
detail.getRange(`A2:J${rawRetrieval.length+1}`).values = rawRetrieval;
detail.getRange(`K2:O${rawRetrieval.length+1}`).formulas = rawRetrieval.map((r, i) => {
  const n=i+2; return [`=IF(G${n}>0,1,0)`,`=IF(E${n}="",0,1/E${n})`,`=G${n}/F${n}`,`=I${n}/H${n}`,`=IF(I${n}=H${n},1,0)`];
});
styleTable(detail,`A1:O${rawRetrieval.length+1}`,'A1:O1',[['A',12],['B',16],['C',15],['D',12],['E',19],['F',13],['G',17],['H',15],['I',19],['J',14],['K',14],['L',14],['M',14],['N',18],['O',21]]);
detail.getRange(`J2:J${rawRetrieval.length+1}`).setNumberFormat('0.000');
detail.getRange(`K2:O${rawRetrieval.length+1}`).setNumberFormat('0.000');
detail.freezePanes.freezeRows(1);

decisions.getRange('A1:N1').values = [['题号','配置','类型','问题语言','工具选择正确','ReAct轮次','完整响应秒数',
  '已知输入Token','已知输出Token','缺失用量调用数','实际总Token','模型调用数','自报任务完成','终止原因']];
decisions.getRange(`A2:N${agent.rows.length+1}`).values = agent.rows.map(r => [r.id,r.profile,r.category,r.language,
  Number(r.tool_selection_correct),r.iterations,r.seconds,r.tokens.input_known,r.tokens.output_known,
  r.tokens.unknown_calls,null,r.model_calls,Number(r.task_complete),r.stop_reason]);
decisions.getRange(`K2:K${agent.rows.length+1}`).formulas = agent.rows.map((r,i) => [`=IF(J${i+2}>0,"未知",SUM(H${i+2}:I${i+2}))`]);
styleTable(decisions,`A1:N${agent.rows.length+1}`,'A1:N1',[['A',12],['B',16],['C',15],['D',12],['E',18],['F',14],['G',19],['H',20],['I',20],['J',20],['K',18],['L',17],['M',19],['N',23]]);
decisions.getRange(`G2:G${agent.rows.length+1}`).setNumberFormat('0.000');
decisions.freezePanes.freezeRows(1);

const platformLabel = retrieval.environment.platform.startsWith('Windows') ? 'Windows' : 'Mac';
summary.getRange('A2').values = [[`${platformLabel}视觉Transformer系统性能评测（12篇，60题；${retrieval.started_at.slice(0,10)}）`]];
summary.getRange('A3').values = [['检索按论文物理页判定；MRR/Recall截断至5。原文全部英文，失败题保留。']];
summary.getRange('A5:I5').values = [['检索配置','候选数','题数','Hit@5','MRR@5','Recall@5','论文覆盖率@5','全部论文命中@5','平均延迟(ms)']];
const nr=rawRetrieval.length+1;
const profileKeys=Object.keys(retrieval.profiles);
for (const [i,key] of profileKeys.entries()) {
  const row=i+6,p=retrieval.profiles[key];
  summary.getRange(`A${row}:B${row}`).values = [[p.label,key==='vector'?'不适用':p.candidate_k]];
  const f=(col)=>`AVERAGEIFS('检索逐题'!$${col}$2:$${col}$${nr},'检索逐题'!$B$2:$B$${nr},"${key}")`;
  summary.getRange(`C${row}:I${row}`).formulas = [[`=COUNTIFS('检索逐题'!$B$2:$B$${nr},"${key}")`,
    `=${f('K')}`,`=${f('L')}`,`=${f('M')}`,`=${f('N')}`,`=${f('O')}`,`=${f('J')}*1000`]];
}
summary.getRange('A14:I14').values = [['Agent配置','题数','工具选择准确率','平均轮次','平均完整响应(s)','平均Token(完整请求)','完整请求Token合计','用量不完整题数','自报完成率']];
const na=agent.rows.length+1;
for (const [i,key] of Object.keys(agent.profiles).entries()) {
  const row=i+15,match=`'Agent逐题'!$B$2:$B$${na},"${key}"`;
  const avg=col=>`AVERAGEIFS('Agent逐题'!$${col}$2:$${col}$${na},${match})`;
  summary.getRange(`A${row}`).values = [[profileLabels[key]]];
  summary.getRange(`B${row}:I${row}`).formulas = [[`=COUNTIFS(${match})`,`=${avg('E')}`,`=${avg('F')}`,`=${avg('G')}`,
    `=IF(COUNTIFS(${match},'Agent逐题'!$J$2:$J$${na},0)=0,"未知",AVERAGEIFS('Agent逐题'!$K$2:$K$${na},${match},'Agent逐题'!$J$2:$J$${na},0))`,
    `=SUMIFS('Agent逐题'!$K$2:$K$${na},${match})`,`=COUNTIFS(${match},'Agent逐题'!$J$2:$J$${na},">0")`,`=${avg('M')}`]];
}
summary.getRange('A19').values = [['工具选择准确率和自报完成率均不等于答案正确率；人工评分在另一本工作簿填写。']];
styleTable(summary,'A2:I19','A5:I5',[['A',34],['B',12],['C',23],['D',16],['E',23],['F',27],['G',25],['H',25],['I',23]]);
summary.getRange('A14:I14').format = {fill:'#243B53',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true,horizontalAlignment:'center'};
summary.getRange('A2').format.font = {name:'Arial',size:14,bold:true};
summary.getRange('A3').format.font = {name:'Arial',size:10,italic:true};
summary.getRange('D6:D10').setNumberFormat('0.0%');
summary.getRange('E6:E10').setNumberFormat('0.0000');
summary.getRange('F6:H10').setNumberFormat('0.0%');
summary.getRange('I6:I10').setNumberFormat('0.0');
summary.getRange('C15:C16').setNumberFormat('0.0%');
summary.getRange('D15:F16').setNumberFormat('0.0');
summary.getRange('G15:H16').setNumberFormat('#,##0');
summary.getRange('I15:I16').setNumberFormat('0.0%');
summary.getRange('A22').values = [[`原始结果：${path.relative(path.dirname(reportsRoot), path.dirname(retrievalPath))}/retrieval.json、agent.json。`]];

const human=Workbook.create();
const scores=human.worksheets.add('人工评分');
const answers=human.worksheets.add('答案与依据');
const rubric=human.worksheets.add('评分标准');
const protocol=JSON.parse(await fs.readFile(path.join(reportsRoot,'5_5_2 系统性能评估/评测方案.json'),'utf8'));
rubric.getRange('A1:D1').values=[['分数','正确性','完整性','引用准确性']];
rubric.getRange('A2:D6').values=Array.from({length:5},(_,i)=>[i,
  protocol.human_quality.correctness[i],protocol.human_quality.completeness[i],protocol.human_quality.citation_accuracy[i]]);
styleTable(rubric,'A1:D6','A1:D1',[['A',12],['B',60],['C',60],['D',75]]);
rubric.getRange('B2:D6').format.wrapText=true;
rubric.getRange('A2:D6').format.rowHeight=65;
rubric.getRange('A8').values=[['引用须核对真实论文和物理页；对比/归纳题还需核对多篇来源覆盖。']];
rubric.getRange('A9').values=[['原文：项目data/raw/evaluation_vision_transformers/；固定版本及下载链接见reports/5_5_1 评测集构建/论文清单.json。']];
scores.getRange('A2').values=[['答案质量人工评分：三项均为0–4整数']];
scores.getRange('A3').values=[['请填写分数、理由、评阅人和日期。空白为待评分；助手审阅不得作为人工分数。']];
scores.getRange('A4').values=[['配置：default=默认规则路由；no_rules=关闭规则路由。每题两条实际答案分别评分。']];
scores.getRange('A5:J5').values=[['题号','配置','正确性','完整性','引用准确性','评分理由','评阅人','评分日期','评分完整','问题']];
scores.getRange(`A6:J${agent.rows.length+5}`).values=agent.rows.map(r=>[r.id,r.profile,null,null,null,null,null,null,null,qa.get(r.id).question]);
scores.getRange(`I6:I${agent.rows.length+5}`).formulas=agent.rows.map((r,i)=>{
  const n=i+6;return [`=IF(AND(ISNUMBER(C${n}),C${n}>=0,C${n}<=4,MOD(C${n},1)=0,ISNUMBER(D${n}),D${n}>=0,D${n}<=4,MOD(D${n},1)=0,ISNUMBER(E${n}),E${n}>=0,E${n}<=4,MOD(E${n},1)=0,LEN(TRIM(F${n}))>0,LEN(TRIM(G${n}))>0,H${n}<>""),1,0)`];
});
styleTable(scores,`A2:J${agent.rows.length+5}`,'A5:J5',[['A',12],['B',16],['C',12],['D',12],['E',17],['F',50],['G',20],['H',18],['I',17],['J',70]]);
scores.getRange(`C6:H${agent.rows.length+5}`).format.fill='#FFF4CC';
scores.dataValidations.add({range:`C6:E${agent.rows.length+5}`,rule:{type:'whole',operator:'between',formula1:0,formula2:4}});
scores.getRange(`H6:H${agent.rows.length+5}`).setNumberFormat('yyyy-mm-dd');
scores.getRange(`J6:J${agent.rows.length+5}`).format.wrapText=true;
scores.getRange(`F6:F${agent.rows.length+5}`).format.wrapText=true;
scores.getRange(`A6:J${agent.rows.length+5}`).format.rowHeight=65;
scores.freezePanes.freezeRows(5);
answers.getRange('A1:G1').values=[['题号','配置','问题','参考答案','评分要点','原文依据','实际Agent答案']];
const answerRows=agent.rows.map(r=>{
  const q=qa.get(r.id);return [r.id,r.profile,q.question,q.reference_answer,q.answer_points.join('\n'),
    q.evidence.map(e=>`${e.paper_id}.pdf 第${e.page_number}页 ${e.section}`).join('\n'),r.answer];
});
answers.getRange(`A2:G${agent.rows.length+1}`).values=answerRows;
styleTable(answers,`A1:G${agent.rows.length+1}`,'A1:G1',[['A',12],['B',16],['C',60],['D',60],['E',60],['F',60],['G',90]]);
answers.getRange(`C2:G${agent.rows.length+1}`).format.wrapText=true;
answers.getRange(`A2:G${agent.rows.length+1}`).format.rowHeight=180;
// 按中英文宽度和换行估算行高，让较长实际答案能直接阅读。
answerRows.forEach((row,i)=>{
  const lines=Math.max(...row.slice(2).map((value,j)=>String(value).split('\n').reduce((sum,line)=>{
    const width=[...line].reduce((n,c)=>n+(c.charCodeAt(0)>127?2:1),0);
    return sum+Math.max(1,Math.ceil(width/(j===4?87:57)));
  },0)));
  answers.getRange(`A${i+2}:G${i+2}`).format.rowHeight=Math.min(409,Math.max(180,lines*15+12));
});
answers.freezePanes.freezeRows(1);

// 用内置计算器完成公式重算；输出人工评分缺失状态，不填0分或模型估计分。
const originalScores=scores.getRange('C6:H6').values;
scores.getRange('C6:H6').values=[[0,0,0,'公式验证样例，非评阅记录','临时测试','2026-10-03']];
if(scores.getRange('I6').values[0][0]!==1) throw new Error('0分完整记录被错误当作缺失');
scores.getRange('H6').values=[[null]];
if(scores.getRange('I6').values[0][0]!==0) throw new Error('缺少日期仍被当作评分完整');
scores.getRange('C6:H6').values=originalScores;
for (const [book,name,ranges] of [[wb,'系统性能评测数据',[['性能汇总','A2:I22'],['检索逐题','A1:O6'],['Agent逐题','A1:N6']]],
                                [human,'答案质量人工评分表',[['人工评分','A2:J7'],['答案与依据','A1:G3'],['评分标准','A1:D9']]]]) {
  book.recalculate();
  if(book===wb) {
    const actual=summary.getRange('C6:I10').values;
    for(const [i,key] of profileKeys.entries()) {
      const p=retrieval.profiles[key];
      const expected=[p.questions,p.hit_at_5,p.mrr_at_5,p.recall_at_5,p.paper_coverage_at_5,p.all_papers_hit_at_5,p.latency_mean_ms];
      if(actual[i].some((v,j)=>typeof v!=='number' || Math.abs(v-expected[j])>1e-7)) throw new Error(`检索公式与原始JSON不一致：${key}`);
    }
    const observed=summary.getRange('B15:I16').values;
    for(const [i,key] of Object.keys(agent.profiles).entries()) {
      const p=agent.profiles[key];
      const expected=[p.questions,p.tool_selection_accuracy,p.iterations_mean,p.latency_mean_seconds,p.tokens_mean_known_requests??'未知',p.tokens_total,p.token_unknown_requests,p.task_complete_rate];
      if(observed[i].some((v,j)=>typeof expected[j]==='number'?typeof v!=='number'||Math.abs(v-expected[j])>1e-7:v!==expected[j])) throw new Error(`Agent公式与原始JSON不一致：${key}`);
    }
  }
  const errors=await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',
    options:{useRegex:true,maxResults:20},maxChars:3000});
  await fs.writeFile(`${outputDir}/${name}_公式核验.txt`,errors.ndjson);
  for (const [sheetName,range] of ranges) {
    const preview=await book.render({sheetName,range,scale:1,format:'png'});
    await fs.writeFile(`${outputDir}/${name}_${sheetName}_预览.png`,new Uint8Array(await preview.arrayBuffer()));
  }
  const file=await SpreadsheetFile.exportXlsx(book);
  await file.save(`${outputDir}/${name}.xlsx`);
}
console.log('已导出可复算性能表和空白人工评分表；原始观测与评分输入分开保存。');
