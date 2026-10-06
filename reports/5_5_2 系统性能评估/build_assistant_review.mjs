// 将逐条阅读后的助手初评制作成审核表；黄色栏仅由用户填写。
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const here = process.argv[2] ? path.resolve(process.argv[2]) : path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const outputDir = path.join(root, 'outputs/quality-review-20261006');
const input = JSON.parse(await fs.readFile(path.join(here, '助手初评180条_20261006.json'), 'utf8'));
const protocol = JSON.parse(await fs.readFile(path.join(here, '评测方案.json'), 'utf8'));
const rows = [...input.rows].sort((a, b) => a.id.localeCompare(b.id) ||
  ['default', 'no_rules', 'optimized'].indexOf(a.profile) - ['default', 'no_rules', 'optimized'].indexOf(b.profile));
const book = Workbook.create();
const review = book.worksheets.add('初评与审核');
const answers = book.worksheets.add('答案与依据');
const rubric = book.worksheets.add('评分标准');

function table(sheet, range, header, widths) {
  sheet.showGridLines = false;
  sheet.getRange(range).format = {font: {name: 'Arial', size: 10}, verticalAlignment: 'top', wrapText: true};
  sheet.getRange(header).format = {fill: '#243B53', font: {name: 'Arial', size: 10, bold: true, color: '#FFFFFF'},
    rowHeight: 32, wrapText: true, verticalAlignment: 'center', horizontalAlignment: 'center'};
  widths.forEach(([col, width]) => { sheet.getRange(`${col}:${col}`).format.columnWidth = width; });
}

// 按实际字宽分段，避免Excel单行409磅上限截断长答案；拼接后仍是原答案。
function chunks(text, width, maxLines = 22) {
  const result = [];
  let current = '', lineWidth = 0, lineCount = 1;
  for (const ch of text) {
    const cost = ch.codePointAt(0) > 127 ? 2 : 1;
    if (ch === '\n') { lineCount += 1; lineWidth = 0; }
    else if (lineWidth + cost > width) { lineCount += 1; lineWidth = cost; }
    else lineWidth += cost;
    if (lineCount > maxLines && current) {
      result.push(current); current = ''; lineCount = 1; lineWidth = ch === '\n' ? 0 : cost;
    }
    current += ch;
  }
  if (current || !result.length) result.push(current);
  if (result.join('') !== text) throw new Error('长答案分段改变了原文');
  return result;
}

const details = [], anchors = [];
for (const row of rows) {
  anchors.push(details.length + 2);
  const cells = [row.question, row.answer_points.join('\n'),
    row.evidence.map(e => `data/raw/evaluation_vision_transformers/${e.paper_id}.pdf 第${e.page_number}物理页；${e.section}`).join('\n'), row.answer];
  const segments = cells.map((value, i) => chunks(value, i === 3 ? 84 : 54));
  for (let part = 0; part < Math.max(...segments.map(s => s.length)); part++) {
    details.push([row.id, row.profile, ...segments.map(s => s[part] ?? ''), `${part + 1}/${Math.max(...segments.map(s => s.length))}`]);
  }
}
answers.getRange('A1:G1').values = [['题号', '配置', '问题', '必要评分要点', '原文依据（物理页）', '历史最终答案（续行不删字）', '分段']];
answers.getRange(`A2:G${details.length + 1}`).values = details;
table(answers, `A1:G${details.length + 1}`, 'A1:G1', [['A', 10], ['B', 14], ['C', 58], ['D', 58], ['E', 58], ['F', 90], ['G', 9]]);
details.forEach((row, i) => {
  const lines = Math.max(...row.slice(2, 6).map((value, j) => String(value).split('\n').reduce((sum, line) =>
    sum + Math.max(1, Math.ceil([...line].reduce((n, c) => n + (c.codePointAt(0) > 127 ? 2 : 1), 0) / (j === 3 ? 84 : 54))), 0)));
  answers.getRange(`A${i + 2}:G${i + 2}`).format.rowHeight = Math.max(90, Math.min(395, lines * 16 + 18));
});
answers.freezePanes.freezeRows(1);

for (const [line, text] of [
  [1, '180条历史答案：助手初评与用户审核'],
  [2, '每项0–4分。助手已逐条初评180/180；用户审核栏保留空白。'],
  [3, '范围为2026-10-03的三组60题历史运行；不能代表41e3f6d代码重新运行的质量。'],
  [4, '可修订黄色G–L列，或在消息中确认接受全部/指定题号。收到明确审核后记录，0分有效。'],
  [11, '右侧N列定位“答案与依据”的首行。原文、原答案与原人工空表保持不变。']]) {
  review.getRange(`A${line}:I${line}`).merge();
  review.getRange(`A${line}`).values = [[text]];
}
review.getRange('A6:I6').values = [['配置', '初评条数', '初评正确性', '初评完整性', '初评引用', '用户已审核', '用户正确性均分', '用户完整性均分', '用户引用均分']];
review.getRange('A12:N12').values = [['题号', '配置', '初评正确性', '初评完整性', '初评引用', '助手逐条判断理由',
  '用户正确性', '用户完整性', '用户引用', '用户审核理由', '评阅人', '审核日期', '审核完整', '答案表首行']];
review.getRange('A13:N192').values = rows.map((r, i) => [r.id, r.profile, r.assistant_scores.correctness,
  r.assistant_scores.completeness, r.assistant_scores.citation_accuracy, r.assistant_reason,
  null, null, null, null, null, null, null, anchors[i]]);
review.getRange('M13:M192').formulas = rows.map((_, i) => {
  const n = i + 13;
  return [`=IF(AND(ISNUMBER(G${n}),G${n}>=0,G${n}<=4,MOD(G${n},1)=0,ISNUMBER(H${n}),H${n}>=0,H${n}<=4,MOD(H${n},1)=0,ISNUMBER(I${n}),I${n}>=0,I${n}<=4,MOD(I${n},1)=0,LEN(TRIM(J${n}))>0,LEN(TRIM(K${n}))>0,L${n}<>""),1,0)`];
});
['default', 'no_rules', 'optimized'].forEach((profile, i) => {
  const n = i + 7;
  review.getRange(`A${n}`).values = [[profile]];
  review.getRange(`B${n}:I${n}`).formulas = [[`=COUNTIF($B$13:$B$192,A${n})`,
    ...['C', 'D', 'E'].map(c => `=AVERAGEIF($B$13:$B$192,A${n},$${c}$13:$${c}$192)`),
    `=SUMIF($B$13:$B$192,A${n},$M$13:$M$192)`,
    ...['G', 'H', 'I'].map(c => `=IF(F${n}=0,"待审核",SUMIFS($${c}$13:$${c}$192,$B$13:$B$192,A${n},$M$13:$M$192,1)/F${n})`)]];
});
table(review, 'A1:N192', 'A12:N12', [['A', 10], ['B', 14], ['C', 13], ['D', 13], ['E', 13], ['F', 75],
  ['G', 17], ['H', 17], ['I', 17], ['J', 55], ['K', 18], ['L', 18], ['M', 12], ['N', 14]]);
review.getRange('A6:I6').format = {fill: '#243B53', font: {color: '#FFFFFF', bold: true}, rowHeight: 35, wrapText: true};
review.getRange('A1:I1').format = {fill: '#243B53', font: {size: 17, color: '#FFFFFF', bold: true}, rowHeight: 35};
review.getRange('A2:I4').format.rowHeight = 26;
review.getRange('A7:I9').format.rowHeight = 27;
review.getRange('C7:E9').setNumberFormat('0.00');
review.getRange('G7:I9').setNumberFormat('0.00');
review.getRange('B7:I9').format.horizontalAlignment = 'center';
review.getRange('C13:E192').format.horizontalAlignment = 'center';
review.getRange('G13:I192').format.horizontalAlignment = 'center';
review.getRange('A13:N192').format.rowHeight = 88;
review.getRange('G13:L192').format.fill = '#FFF4CC';
review.dataValidations.add({range: 'G13:I192', rule: {type: 'whole', operator: 'between', formula1: 0, formula2: 4}});
review.freezePanes.freezeRows(12);

rubric.getRange('A1:D1').values = [['分数', '正确性', '完整性', '引用准确性']];
rubric.getRange('A2:D6').values = Array.from({length: 5}, (_, i) => [i, protocol.human_quality.correctness[i],
  protocol.human_quality.completeness[i], protocol.human_quality.citation_accuracy[i]]);
table(rubric, 'A1:D21', 'A1:D1', [['A', 10], ['B', 58], ['C', 58], ['D', 72]]);
rubric.getRange('A2:D6').format.rowHeight = 62;
input.method.forEach((text, i) => {
  rubric.getRange(`A${i + 8}:D${i + 8}`).merge(); rubric.getRange(`A${i + 8}`).values = [[text]];
  rubric.getRange(`A${i + 8}:D${i + 8}`).format.rowHeight = 34;
});
rubric.getRange('A14:D14').merge(); rubric.getRange('A14').values = [['已核验12份论文指纹及59页原文；完整输入指纹、逐条答案及理由保存在同目录助手初评180条_20261006.json。']];
rubric.getRange('A14:D14').format.rowHeight = 32;
rubric.getRange('A16:D16').merge(); rubric.getRange('A16').values = [['可逐条修订黄色栏，或在消息中明确接受全部/指定题号。收到明确审核后再记录用户确认，当前用户栏保留空白。']];
rubric.getRange('A16:D16').format.rowHeight = 32;
rubric.freezePanes.freezeRows(1);

// 验证0分、缺项与用户均分的动态计算，再清除临时审核记录。
review.getRange('G13:L13').values = [[0, 0, 0, '临时验证', '临时验证', '2026-10-06']];
if (review.getRange('M13').values[0][0] !== 1 || review.getRange('F7').values[0][0] !== 1 || review.getRange('G7').values[0][0] !== 0) throw new Error('0分审核/汇总失败');
review.getRange('L13').values = [[null]];
if (review.getRange('M13').values[0][0] !== 0 || review.getRange('G7').values[0][0] !== '待审核') throw new Error('缺日期审核被误计');
review.getRange('G13:L13').values = [[null, null, null, null, null, null]];
book.recalculate();
if (review.getRange('M13:M192').values.some(r => r[0] !== 0)) throw new Error('用户审核被意外填入');
for (let i = 0; i < 3; i++) {
  const profile = ['default', 'no_rules', 'optimized'][i];
  const expected = Object.values(input.summary[profile].means);
  const actual = review.getRange(`C${i + 7}:E${i + 7}`).values[0];
  if (actual.some((value, j) => Math.abs(value - expected[j]) > 1e-10)) throw new Error('工作簿均值与JSON不一致');
}
const errors = await book.inspect({kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!', options: {useRegex: true, maxResults: 20}, maxChars: 3000});
await fs.writeFile(path.join(here, '助手初评180条_公式核验.txt'), errors.ndjson);
for (const [sheetName, range] of [['初评与审核', 'A1:I15'], ['答案与依据', 'A1:G4'], ['评分标准', 'A1:D16']]) {
  const preview = await book.render({sheetName, range, scale: 1.3, format: 'png'});
  await fs.writeFile(path.join(here, `助手初评180条_${sheetName}_预览.png`), new Uint8Array(await preview.arrayBuffer()));
}
const longRow = details.findIndex(row => row[6].startsWith('1/') && row[6] !== '1/1') + 2;
for (const [label, sheetName, range] of [['审核输入', '初评与审核', 'G12:N15'],
  ['长答案续行', '答案与依据', `A${longRow}:G${longRow + 1}`]]) {
  const preview = await book.render({sheetName, range, scale: 1.3, format: 'png'});
  await fs.writeFile(path.join(here, `助手初评180条_${label}_预览.png`), new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(outputDir, {recursive: true});
const exported = await SpreadsheetFile.exportXlsx(book);
await exported.save(path.join(outputDir, '历史答案助手初评与用户审核_180条.xlsx'));
console.log(JSON.stringify({records: rows.length, detailRows: details.length, userReviewed: 0, outputDir}));
