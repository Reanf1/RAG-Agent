/** 以已归档的真实结果创建可编辑课程演示；未完成输入拒绝导出。 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const { RAG_PROJECT, RAG_RESULTS, RAG_WORKSPACE, RAG_REVISION, SKILL_DIR, RUNTIME_PYTHON } = process.env;
if (![RAG_PROJECT, RAG_RESULTS, RAG_WORKSPACE, SKILL_DIR, RUNTIME_PYTHON].every(v => v && path.isAbsolute(v))) {
  throw new Error('项目、结果、工作目录、技能与Python路径必须为绝对路径');
}
const read = async p => JSON.parse(await fs.readFile(p, 'utf8'));
const retrieval = await read(path.join(RAG_RESULTS, 'retrieval.json'));
const agent = await read(path.join(RAG_RESULTS, 'agent.json'));
const routing = await read(path.join(RAG_RESULTS, 'routing.json'));
const parallel = await read(path.join(RAG_RESULTS, 'parallel.json'));
const quality = await read(path.join(RAG_RESULTS, '助手逐题初评.json'));
const chunking = await read(path.join(RAG_PROJECT, 'reports/5_1_2 文本分块策略/五组分块检索对比_20261004.json'));
const deployRoot = path.join(RAG_PROJECT, 'reports/5_5_4 项目交付/Windows容器离线验收_20261007');
const deploy = await read(path.join(deployRoot, 'container-7.json'));
if (![retrieval, agent, routing, parallel, chunking].every(r => r.status === 'completed') || deploy.status !== 'passed' || !deploy.new_process_reopen) {
  throw new Error('正式实验和容器重开尚未完成');
}
if (quality.rows.length !== 120 || quality.rows.some(r => Object.values(r.user_review).some(v => v !== null))) {
  throw new Error('本轮120条助手初评与空白用户审核记录不完整');
}
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')));
const family = 'Songti SC';
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const navy = '#142735', green = '#397769', gold = '#BB9043';
const build = path.join(RAG_WORKSPACE, 'build-' + RAG_REVISION);
const final = path.join(RAG_WORKSPACE, 'output', '项目演示-' + RAG_REVISION + '.pptx');
await fs.mkdir(build, { recursive: true });
await fs.mkdir(path.dirname(final), { recursive: true });

function text(slide, value, left, top, width, height, size = 26, bold = false, color = navy) {
  const box = slide.shapes.add({ geometry: 'textbox', position: { left, top, width, height }, fill: 'none', line: { fill: 'none', width: 0 } });
  box.text = value;
  box.text.style = { typeface: family, fontSize: size, bold, color, autoFit: 'none' };
  return box;
}
function slide(title, source) {
  const s = deck.slides.add(); s.background.fill = '#FFFFFF';
  text(s, title, 64, 42, 1152, 92, 44, true);
  text(s, '智能科研助理 · 课程实践', 64, 675, 700, 24, 18, false, '#596A74');
  text(s, String(deck.slides.items.length).padStart(2, '0'), 1160, 675, 56, 24, 18);
  s.speakerNotes.textFrame.setText(source);
  return s;
}
async function image(s, relative, left, top, width, height) {
  const file = path.join(RAG_PROJECT, relative);
  s.images.add({ blob: new Uint8Array(await fs.readFile(file)), contentType: 'image/png', alt: path.basename(file), fit: 'contain', position: { left, top, width, height } });
}
function chart(s, title, categories, values, left, top, width, height, percent = false, maximum = null) {
  // 演示图保留六位小数，便于Excel编辑；完整测量精度保留在原始JSON中。
  values = values.map(value => Math.round(value * 1e6) / 1e6);
  const c = s.charts.add('bar', {
    position: { left, top, width, height }, categories,
    series: [{ name: title, values, fill: green,
      dataLabelOverrides: values.map((value, idx) => ({ idx,
        text: percent ? (value * 100).toFixed(1) + '%' : value.toFixed(Math.abs(value) >= 100 ? 0 : 2),
        textStyle: { typeface: family, fontSize: 20, fill: navy } })),
    }],
    title, titleTextStyle: { typeface: family, fontSize: 25, bold: true, fill: navy },
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 90 }, hasLegend: false,
    xAxis: { textStyle: { typeface: family, fontSize: 20 }, majorGridlines: null },
    yAxis: { min: 0, ...(percent ? { max: 1, numberFormatCode: '0%' } : {}), ...(maximum === null ? {} : { max: maximum, majorUnit: 1 }), textStyle: { typeface: family, fontSize: 18 }, majorGridlines: { fill: '#DDE5E6', width: 1, style: 'solid' } },
    dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: family, fontSize: 20, fill: navy } },
  });
  applyPresentationChartFont(c, { fontFamily: family });
  return c;
}
function table(s, values, widths, top = 172, height = 410) {
  const t = s.tables.add({ rows: values.length, columns: values[0].length, left: 64, top, width: 1152, height, columnWidths: widths, values });
  t.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: values[0].length }).assign({ fill: '#FFFFFF', textStyle: { typeface: family, fontSize: 25, color: navy }, margins: { left: 14, right: 14, top: 10, bottom: 10 }, anchor: 'center' });
  t.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: values[0].length }).assign({ fill: '#EAF1EF', textStyle: { typeface: family, fontSize: 25, bold: true, color: navy } });
  t.borders.assign({ style: 'solid', fill: '#CDDAD6', width: 1 });
  return t;
}
const mean = rows => rows.reduce((total, row) => total + row.seconds, 0) / rows.length;
const rp = Object.values(retrieval.profiles), ap = Object.values(agent.profiles);

let s = slide('智能科研助理', '依据：docs/技术设计文档.md、docs/课程报告.md。成员真实资料暂未提供，保留空白。');
text(s, '基于 RAG + Agent 的论文知识库问答系统', 64, 208, 1120, 140, 50, true);
text(s, '本地论文 → 检索证据 → 工具决策 → 可追溯回答', 64, 385, 1120, 80, 32);
text(s, '2026年10月7日 · Windows正式复测与容器离线验收', 64, 520, 1120, 65, 28);
text(s, '成员姓名、班级、学号及真实贡献：待填写', 64, 605, 1120, 40, 24, false, '#596A74');

s = slide('五个课程模块接入同一应用', '依据：docs/技术设计文档.md 第6节验收映射。自动测试通过不等于科研答案全部正确。');
table(s, [['模块', '实现内容', '可核验交付'], ['数据与检索', '多格式、三种分块、RRF与BGE', '12篇论文、60题、排名记录'], ['生成与缓存', '引用、流式、低分确认、语义缓存', '正文、来源、实际Token'], ['Agent与记忆', '有界ReAct、8工具、路由与恢复', '决策轨迹、会话与摘要'], ['集成与前端', 'Streamlit、Chroma、SQLite', '上传、检索、原文、历史'], ['评测与交付', '性能对照、质量初评、部署', '报告、手册、Word与演示']], [230, 510, 412], 155, 470);

s = slide('业务保持单个Streamlit进程', '来源：reports/5_5_4 项目交付/系统架构图.png。TCP入口仅转发字节，业务与Ollama位于内部网络，Chroma和SQLite保持本地文件。');
await image(s, 'reports/5_5_4 项目交付/系统架构图.png', 64, 145, 1152, 430);
text(s, '容器入口转发HTTP与WebSocket；应用和模型服务无法直接连接外网', 64, 594, 1152, 64, 25);

s = slide('分块质量与返回块数共同影响检索', '历史Mac开发实验：reports/5_1_2 文本分块策略/五组分块检索对比_20261004.json；300条，900次独立指标复算。不是Windows耗时。');
const cp = Object.entries(chunking.profiles);
const chunkLabels = { fixed256: '固定256', fixed512: '固定512', fixed1024: '固定1024', recursive: '递归512', semantic: '句段512' };
const retrievalLabels = ['向量', '混合', '重排20', '重排10', '重排40'];
chart(s, 'Hit@5', cp.map(([name]) => chunkLabels[name]), cp.map(([, p]) => p.at_k['5'].hit), 64, 160, 560, 380, true);
chart(s, '页级Recall@5', cp.map(([name]) => chunkLabels[name]), cp.map(([, p]) => p.at_k['5'].recall), 656, 160, 560, 380, true);
text(s, '同12篇论文、60题；五组分块；M3E + 手写RRF Top-20 + BGE', 64, 570, 1152, 80, 26);

s = slide('Windows检索五组配置的实际表现', '本轮正式Windows：' + path.join(RAG_RESULTS, 'retrieval.json') + '。60题×5组，失败与中文查询英文原文均保留；用户要求暂不优化跨语言排名。');
chart(s, 'Hit@5', retrievalLabels, rp.map(p => p.hit_at_5), 64, 160, 560, 390, true);
chart(s, '平均检索耗时 / 毫秒', retrievalLabels, rp.map(p => p.latency_mean_ms), 656, 160, 560, 390);
text(s, '命中按原文标注页计算；同语言容器样例通过不能替代完整评测', 64, 580, 1152, 64, 26);

s = slide('页面返回真实文档块与物理页码', 'I01后镜像页面证据：reports/5_5_4 项目交付/Windows容器离线验收_20261007/I01后镜像页面检索.png。2026-10-07，独立ViT索引167块。');
await image(s, 'reports/5_5_4 项目交付/Windows容器离线验收_20261007/I01后镜像页面检索.png', 64, 143, 1152, 490);

s = slide('Agent公开工具决策与执行结果', '来源：reports/5_5_4 项目交付/工具与轨迹_标注.png；历史真实界面，不是本轮Windows截图。展示公开计划、参数、结果、耗时及终止状态，不要求私有推理过程。');
await image(s, 'reports/5_5_4 项目交付/工具与轨迹_标注.png', 64, 145, 1152, 465);
text(s, '历史界面示例；每轮决策 → 执行 → 观察，上限8轮', 64, 615, 1152, 42, 24);

s = slide('路由质量与Token成本分别计算', '本轮Windows：agent.json默认规则与关闭规则各60题；routing.json 72题×两策略=144请求，固定RAG确实每题检索；预热单列，原始HTTP用量独立复算。');
chart(s, '工具选择准确率', ['默认规则', '关闭规则'], ap.map(p => p.tool_selection_accuracy), 64, 160, 560, 380, true);
chart(s, '平均实际Token / 请求', ['Agent', '固定RAG'], ['agent', 'fixed_rag'].map(p => routing.profiles[p].tokens_mean_known), 656, 160, 560, 380);
text(s, '本轮Agent Token增加531.55%；自报完成与工具成功不等于答案正确', 64, 580, 1152, 64, 25);

s = slide('并行实验保留失败与调度差异', '本轮Windows：parallel.json，两种工具组合各3对，共12请求。独立核验第一次实际工具数量、参数独立性和execution_mode；失败快速返回不能解释为有效加速。');
for (const [i, category] of ['time_keywords', 'two_knowledge'].entries()) {
  chart(s, i === 0 ? '时间与关键词 / 秒' : '两个知识库问题 / 秒', ['串行', '并行'], ['serial', 'parallel'].map(p => mean(parallel.rows.filter(r => r.category === category && r.profile === p))), 64 + i * 592, 160, 560, 380);
}
text(s, '知识库组合下降18.76%；时间／关键词增加2.50%；每组仅3对', 64, 580, 1152, 65, 26, false, gold);

s = slide('缓存和记忆均有明确失效条件', '依据：WindowsSSH集中复测_20261007/WindowsSSH集中复测报告.md。真实TXT RAG exact/semantic命中、语料失效、会话隔离；历史2000Token窗口与阶段性摘要已验证。');
text(s, '语义缓存：相似度阈值0.97；仅保存正常、有来源、无警告结果', 64, 190, 1120, 90, 31);
text(s, '命中跳过RAG内部检索与生成；其余Agent阶段仍可能消耗Token', 64, 315, 1120, 90, 31);
text(s, '记忆：2000Token滑动窗口；10轮触发摘要，保留最近4轮', 64, 440, 1120, 90, 31);
text(s, '语料、配置与会话变化使旧缓存失效；摘要存在信息损失', 64, 565, 1120, 68, 28, false, gold);

s = slide('Windows故障路径已做实际注入', '本轮：reports/5_4_4 端到端联调与测试/Windows故障与离线验收_20261007/faults-5.json、recovery.json。I01后780项两端回归另存；faults-4真实Qwen误规划批量替代的失败保留。');
table(s, [['故障', '验证方式', '观察结果'], ['模型服务断开与恢复', '真实连接拒绝；恢复Qwen算术', '错误明确；9×8=72'], ['流式中途断开', '真实NDJSON连接中止', '保留部分正文，标记错误'], ['超时与线程额度', '真实HTTP延迟；最多1工作线程', '超时后拒绝新增，退出释放'], ['独立调用失败', 'HTTP503与版本查询同批', '失败与成功分别保留'], ['执行失败与重试', '失败替代、一次重试、耗尽替代', '有界恢复，不假报完成']], [330, 490, 332], 155, 440);
text(s, '已运行的同步线程无法强杀；晚到工作退出后才释放额度', 64, 615, 1152, 40, 24, false, gold);

s = slide('助手初评与用户审核分别记录', '本轮Windows助手逐题初评.json，120/120条；用户审核0/120。助手参与参考答案构建，初评不属于独立盲评；历史180条另存。');
// 三项质量分别展示，引用有效不能掩盖答非所问或漏答。
for (const [i, profile] of ['default', 'no_rules'].entries()) {
  const rows = quality.rows.filter(r => r.profile === profile);
  const values = ['correctness', 'completeness', 'citation_accuracy'].map(metric =>
    rows.reduce((total, row) => total + row.assistant_scores[metric], 0) / rows.length);
  chart(s, i === 0 ? '默认规则 / 4分' : '关闭规则 / 4分', ['正确性', '完整性', '引用'], values,
    64 + i * 592, 170, 560, 370, false, 4);
}
text(s, '120条助手初评；用户审核0/120；历史180条另存，未混合统计', 64, 568, 1152, 44, 26);
text(s, '三项均按0～4分；引用有效不等于回答正确；空白与0分严格区分', 64, 618, 1152, 44, 24);

s = slide('最终镜像通过离线业务与重开验证', '本轮容器完整证据：Windows容器离线验收_20261007/container-7.json、environment-3.json。镜像99db29e，Python3.10.10，SQLite3.46.1，Chroma0.5.23；CPU。应用与Ollama TCP外网连接被拒绝，浏览器经固定TCP入口；不等同物理拔网线。');
table(s, [['项目', '真实验证结果'], ['环境与依赖', 'Python3.10.10；pip check通过；SQLite兼容修复'], ['外网隔离', '应用与Ollama的1.1.1.1:443连接均不可达'], ['论文与检索', 'ViT独立167块；混合BGE重排；物理原文页'], ['Agent与持久化', '真实工具回答与引用；重启后索引和会话一致'], ['页面入口', '本机8502验收；发布配置8501；HTTP/WebSocket正常']], [330, 822], 155, 450);
text(s, '镜像和模型提前准备；容器CPU结果与Windows原生GPU性能分开记录', 64, 615, 1152, 44, 24);

s = slide('交付证据与尚待人工完成的事项', '依据：docs/课程报告.md、技术设计文档.md、用户使用手册.md、最终交付核验。成员资料未提供；用户最终审核未完成；跨语言排名依用户要求暂缓。');
text(s, '代码与提交历史、配置和环境说明、部署修复与失败记录', 64, 185, 1152, 80, 32);
text(s, '逐题结果、模型实际用量、独立复算、图表、Bad Case', 64, 295, 1152, 80, 32);
text(s, '课程Word、可编辑演示、技术设计与操作手册', 64, 405, 1152, 80, 32);
text(s, '待成员真实资料、贡献比例与用户质量审核；跨语言排名暂缓', 64, 540, 1152, 100, 29, false, gold);

const candidate = path.join(build, 'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
await finalizePresentation({ workspaceDir: RAG_WORKSPACE, candidatePath: candidate, finalPath: final,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit',
    ...[2, 11, 13].flatMap(number => ['--require-native-table-slide', String(number)])],
  explicitTotalSlideCount: 14, requiredNativeTableOwnerSlides: [2, 11, 13], requiredNativeChartOwnerSlides: [4, 5, 8, 9, 12],
  materializeLiteralChartWorkbooks: true, fontPolicy: { basis: 'design', families: [family] },
  verifyArtifactToolImport: true, receiptPath: path.join(build, 'validation.json'),
});
// 最终包重新导入并渲染全部幻灯片，不把候选预览当最终视觉核验。
const { FileBlob } = await import('@oai/artifact-tool');
const checked = await PresentationFile.importPptx(await FileBlob.load(final));
for (const [i, page] of checked.slides.items.entries()) {
  const preview = await checked.export({ slide: page, format: 'png', scale: 1 });
  await fs.writeFile(path.join(build, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(final);
