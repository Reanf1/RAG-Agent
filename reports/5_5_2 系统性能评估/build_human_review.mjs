// 从固定版本的180条真实回答制作人工评分表，原120条工作簿保留。
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const outputDir=process.argv[2];
const retrievalPath=path.join(outputDir,'检索五组结果_20261003.json');
const agent=JSON.parse(await fs.readFile(path.join(outputDir,'人工评分180条来源.json'),'utf8'));
const questions=JSON.parse(await fs.readFile(path.join(outputDir,'../评测集.json'),'utf8'));
const qa=new Map(questions.map(q=>[q.id,q]));
function styleTable(sheet,range,header,widths) {
  sheet.showGridLines=false;
  sheet.getRange(range).format.font={name:'Arial',size:10};
  sheet.getRange(range).format.verticalAlignment='center';
  sheet.getRange(range).format.rowHeight=23;
  sheet.getRange(header).format={fill:'#243B53',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:34,wrapText:true,horizontalAlignment:'center'};
  widths.forEach(([col,width])=>sheet.getRange(`${col}:${col}`).format.columnWidth=width);
}
const human=Workbook.create();
const scores=human.worksheets.add('人工评分');
const answers=human.worksheets.add('答案与依据');
const rubric=human.worksheets.add('评分标准');
const protocol=JSON.parse(await fs.readFile(path.join(path.dirname(path.resolve(retrievalPath)),'评测方案.json'),'utf8'));
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
scores.getRange('A4').values=[['范围：default与no_rules为基线120条；optimized为优化后60条。共180条，逐条独立评阅。']];
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


// 验证0分是有效评分；清除临时样例后再保存。
const original=scores.getRange('C6:H6').values;
scores.getRange('C6:H6').values=[[0,0,0,'临时公式验证','临时测试','2026-10-04']];
if(scores.getRange('I6').values[0][0]!==1) throw new Error('0分被误判为缺失');
scores.getRange('H6').values=[[null]];
if(scores.getRange('I6').values[0][0]!==0) throw new Error('缺少日期被误判为完整');
scores.getRange('C6:H6').values=original;
human.recalculate();
if(scores.getRange('I6:I185').values.some(r=>r[0]!==0)) throw new Error('存在不应填写的人工评分');
const errors=await human.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:20},maxChars:3000});
await fs.writeFile(path.join(outputDir,'人工评分180条_公式核验.txt'),errors.ndjson);
for(const [sheetName,range] of [['人工评分','A2:J7'],['答案与依据','A1:G3'],['评分标准','A1:D9']]) {
  const preview=await human.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(outputDir,`人工评分180条_${sheetName}_预览.png`),new Uint8Array(await preview.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(human);
await file.save(path.join(outputDir,'答案质量人工评分表_180条.xlsx'));
console.log('已保存180条待评记录，三项评分保持空白。');
