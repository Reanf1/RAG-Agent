/** 依据真实报告生成课程答辩PPT；未完成实验只允许输出标明状态的临时初稿。 */
const fs = require('node:fs');
const path = require('node:path');
const PptxGenJS = require('pptxgenjs');
const {imageSize} = require('image-size');
const ROOT = path.resolve(__dirname, '../..');
const target = process.argv[2];
const draft = process.argv.includes('--draft');
if (!target) throw new Error('请传入输出PPTX绝对路径');
const out = path.resolve(target);
const read = rel => JSON.parse(fs.readFileSync(path.join(ROOT, rel), 'utf8'));
const chunkPath = 'reports/5_1_2 文本分块策略/五组分块检索对比_20261004.json';
const routePath = 'reports/5_5_2 系统性能评估/Agent与固定RAG对照_20261004.json';
const parallelPath = 'reports/5_5_2 系统性能评估/独立工具串行并行对照_20261004.json';
const deployPath = 'reports/5_5_4 项目交付/容器完整验证_20261004.json';
const get = rel => fs.existsSync(path.join(ROOT, rel)) ? read(rel) : {};
const chunk=get(chunkPath), route=get(routePath), parallel=get(parallelPath), deploy=get(deployPath);
if (!draft && [chunk,route,parallel].some(d=>d.status !== 'completed')) throw new Error('实验未完成，不能导出最终演示');
if (!draft && deploy.status !== 'passed') throw new Error('完整容器验证未通过，不能写成部署已完成');
const deck = new PptxGenJS();
deck.layout='LAYOUT_WIDE';deck.author='项目课程实践';deck.subject='真实实现、评测与交付';
deck.title='智能科研助理：RAG + Agent';deck.company='本科课程实践';deck.lang='zh-CN';
deck.theme={headFontFace:'Heiti SC',bodyFontFace:'PingFang SC',lang:'zh-CN'};
const C={forest:'224A3E',green:'397769',cream:'F4F6EF',moss:'ADC698',ink:'20342E',muted:'5A6C64',gold:'BB9043',white:'FFFFFF',pale:'E4EBDF'};
function text(s,t,x,y,w,h,size=19,options={}) {s.addText(t,{x,y,w,h,fontFace:'PingFang SC',fontSize:size,color:C.ink,margin:0,breakLine:false,valign:'top',paraSpaceAfterPt:9,...options});}
function rect(s,x,y,w,h,color=C.white){s.addShape(deck.ShapeType.rect,{x,y,w,h,fill:{color},line:{color,width:0}});}
function page(title,sub='',dark=false){const s=deck.addSlide();s.background={color:dark?C.forest:C.cream};const fg=dark?C.white:C.ink;text(s,title,.62,.45,12.05,.66,34,{bold:true,fontFace:'Heiti SC',color:fg});if(sub)text(s,sub,.65,1.2,12,.5,14,{color:dark?C.moss:C.muted});text(s,`智能科研助理 · 本地 RAG + Agent${draft?' · 初稿':''}`,.65,7.06,10,.22,10,{color:dark?C.moss:C.muted});text(s,String(deck._slides.length),12.12,7.05,.5,.25,11,{align:'right',color:dark?C.moss:C.muted});return s;}
function note(s,body,sources=[]){s.addNotes([body,'证据：',...sources.map(p=>path.join(ROOT,p))]);}
function card(s,title,body,x,y,w,h){rect(s,x,y,w,h);text(s,title,x+.24,y+.22,w-.48,.48,23,{bold:true,color:C.green});text(s,body,x+.24,y+.87,w-.48,h-1.06,17);}
function img(s,rel,x,y,w,h){const p=path.join(ROOT,rel);if(!fs.existsSync(p)){if(!draft)throw new Error(`缺少图表 ${p}`);rect(s,x,y,w,h,C.pale);text(s,'真实实验进行中',x+.3,y+h/2-.25,w-.6,.5,22,{align:'center'});return;}
  // 按真实文件格式读取尺寸，部分浏览器截图的扩展名与JPEG编码不同。
  const {width:iw,height:ih}=imageSize(fs.readFileSync(p)), scale=Math.min(w/iw,h/ih);
  s.addImage({path:p,x:x+(w-iw*scale)/2,y:y+(h-ih*scale)/2,w:iw*scale,h:ih*scale});}
let s=page('智能科研助理','论文阅读分散、答案难追溯 · 本科生产实习 · 2026年10月4日',true);
text(s,'让论文知识库参与决策',.7,2.1,8.8,1,40,{bold:true,color:C.white});
text(s,'Agent 统一判断问题与工具\nRAG 提供可追溯的文献证据',.75,3.25,7.4,1.25,25,{color:C.moss});
['上传文献','检索证据','工具决策','引用回答'].forEach((t,i)=>{rect(s,.75+i*3.1,5.3,2.68,.95,i===2?C.gold:C.green);text(s,t,.91+i*3.1,5.6,2.36,.45,23,{color:C.white,bold:true,align:'center'});});
note(s,'开场说明：这是本科生可以解释的单应用系统。所有数值引用实际记录，不把宣传目标当测量结果。',['AGENTS.md','docs/技术设计文档.md']);

s=page('一个应用，五个课程模块','按加载 → 检索 → 生成 → 决策 → 交付逐步构建');
card(s,'01 文档与检索','PDF / Word / 文本\n三种分块与增量索引\n向量 + BM25 + RRF + BGE',.65,1.95,3.84,2.12);
card(s,'02 本地 RAG','上下文预算与来源编号\n流式 / 缓存 / 降级\n完整日志与实际 Token',4.75,1.95,3.84,2.12);
card(s,'03 手写 Agent','有界 ReAct，八个本地工具\n路由 / 并行 / 错误恢复\n窗口与摘要记忆',8.85,1.95,3.84,2.12);
card(s,'04 交互与监测','文档、会话、原文页管理\n公开决策轨迹及运行指标',.65,4.45,5.85,1.89);
card(s,'05 评测与交付','12篇论文、60题、完整优化对照\n180条真实答案待人工评分',6.82,4.45,5.85,1.89);
note(s,'对应课程5.1到5.5；工具和指标不是空壳。人评还没有真实分数。',['docs/课程报告.md']);

s=page('Agent 为决策中枢，RAG 为知识工具','保留 src 模块目录，采用单个 Streamlit 进程');
img(s,'reports/5_5_4 项目交付/系统架构图.png',.65,1.93,7.6,4.83);
card(s,'模型与数据都在本地','Qwen2.5:7b / Ollama\nM3E-base / BGE reranker\nChroma + SQLite',8.65,2.02,4.02,2.5);
text(s,'手写核心：ReAct、RRF\n基础组件：LangChain\n模型准备后可离线运行',8.86,4.91,3.6,1.3,19);
note(s,'固定参考版本的消息、工具和模型调用流程适配到现有目录，不另建HTTP服务。',['docs/技术设计文档.md','config.yaml']);

s=page('分块策略：同一标注集，五组对照','固定 256 / 512 / 1024；递归与语义使用默认 512 / 64');
img(s,'reports/5_1_2 文本分块策略/五组分块检索对比_20261004.png',.65,1.83,12.02,3.99);
text(s,'保留递归默认：Hit@5 66.67%，Recall@5 42.97%。\n固定512的MRR@5更高；返回K=10提高召回，也增加生成上下文。',.83,6.0,11.7,.76,19);
note(s,'300次查询、900组K指标独立复算。共享资源与断点续跑使延迟不适合公平速度排名；仅固定策略改变大小。',[chunkPath,'reports/5_1_2 文本分块策略/五组分块检索对比实验报告_20261004.md']);

s=page('重排提高命中，但没有达到 85% 目标','12篇 / 60题：纯向量 → 混合 → 混合 + 模型重排');
const hit=[40,36.67,66.67];const label=['纯向量','向量 + BM25 / RRF','RRF + BGE候选20'];
hit.forEach((v,i)=>{const x=.7+i*4.14;rect(s,x,2.12,3.82,3.32);text(s,`${v.toFixed(2)}%`,x+.25,2.65,3.32,1,48,{bold:true,color:i===2?C.green:C.ink,align:'center'});text(s,label[i],x+.27,4.03,3.28,.7,21,{align:'center'});});
text(s,'Hit@5 是标注页命中率；页面“候选命中率”只是检索非空，不能代替它。\n默认重排20候选：MRR@5 0.3536，页级Recall@5 42.97%。',.88,5.94,11.75,.9,20);
note(s,'原始三档对照的真实结果。混合低于纯向量也保留，不选择性删除失败题。',['reports/5_5_2 系统性能评估/检索五组结果_20261003.json']);

s=page('流式回答，可以打开引用原文页','正文逐步出现，完成后校验来源；页码明确采用物理页');
card(s,'检索 → 上下文 → 引用','相关性排序与预算截断\n文档名 + 物理页 + 块ID\n点击查看实际PDF页\n按文件指纹核对原文',.65,1.98,4.15,3.65);
img(s,'reports/5_4_3 前端与会话管理/交付补齐_20261004/引用原文物理页.png',5.16,1.97,7.5,4.87);
text(s,'来源有效仍需核验语义；\n实际答案质量由评阅人确认。',.89,5.95,3.65,.82,17,{color:C.muted});
note(s,'真实浏览器打开ViT物理第4页截图。Agent Observation有真实增量正文，暂存内容校验后替换；知识库工具仍先完成内部RAG，不把它描述成嵌套全链路流式。',['src/frontend/app.py','reports/5_4_3 前端与会话管理/交付补齐_20261004/流式核验.json']);

s=page('手写 ReAct：决策、执行、观察、终止','最大轮数、重复计划、超时与异常均有明确边界');
['Thought\n公开决策说明','Action\n实际工具调用','Observation\n继续或完成'].forEach((t,i)=>{rect(s,.8+i*4.15,2.05,3.72,1.5,i===1?C.green:C.forest);text(s,t,1+i*4.15,2.39,3.3,.95,25,{color:C.white,align:'center'});});
const tools=['知识库问答','论文元信息','论文对比','关键词提取','结构化摘要','当前时间','计算器','文献列表'];
tools.forEach((t,i)=>{const x=.8+(i%4)*3.1,y=4.09+Math.floor(i/4)*1.02;rect(s,x,y,2.68,.79,C.pale);text(s,t,x+.12,y+.2,2.44,.45,20,{align:'center'});});
text(s,'联网搜索是额外可选工具，默认关闭；不计为八个本地工具。',.87,6.5,11.7,.37,16,{color:C.muted});
note(s,'公开说明不是模型私有思维。同名工具可用不同参数并行；相同参数重复仍拒绝。',['src/agent/react_loop.py','src/agent/tools.py']);

s=page('智能路由，需要和固定 RAG 真正比较','60题论文 + 12题通用概念 / 算式 / 时间；共144请求');
img(s,'reports/5_5_2 系统性能评估/Agent与固定RAG对照_20261004.png',.65,1.87,12,4.31);
let rnote='实验进行中，最终结果待实际运行完成。';
if(route.status==='completed') {const a=route.profiles.agent,f=route.profiles.fixed_rag;const delta=(a.tokens_mean_known/f.tokens_mean_known-1)*100;rnote=`全部请求平均Token：Agent相对固定RAG ${delta>=0?'增加':'减少'}${Math.abs(delta).toFixed(1)}%。\n固定RAG低相关时会拒答或等待确认；不能用低Token等同高质量。`;}
text(s,rnote,.83,6.23,11.7,.63,18);
note(s,'原默认/关闭规则两组都是Agent，因此新增每次固定调用RAG的真正对照。失败与拒答保留，Token来自实际HTTP。',[routePath]);

s=page('并行执行收益，取决于工具和模型容量','两种真实工具组合，各3对串行 / 并行完整请求');
img(s,'reports/5_5_2 系统性能评估/独立工具串行并行对照_20261004.png',.65,1.9,12.02,4.31);
text(s,'核对实际出现两项独立调用、执行方式和结果。\nOllama 单并发可能使两个RAG生成排队，不能承诺端到端提速。',.84,6.22,11.7,.66,18);
note(s,'时间+给定文本关键词、两篇论文各自RAG；每次模型规划、工具执行和Observation实际运行。失败完整保留。',[parallelPath]);

s=page('会话隔离、摘要记忆与公开运行轨迹','历史保存在 SQLite；窗口预算与阶段摘要控制长对话');
img(s,'reports/5_5_4 项目交付/决策轨迹局部.png',.65,1.92,8.1,4.96);
card(s,'可观测指标','真实输入 / 输出Token\n工具状态与耗时\n完整决策路径\n检索延迟与分数分布',9.13,2.02,3.52,3.8);
text(s,'访客标识用于演示隔离，\n不是登录认证。',9.37,6.18,3.13,.63,16,{color:C.muted});
note(s,'窗口采用本地Qwen词表计数，摘要失败保留原文不覆盖。重放历史不会再调用模型。',['src/agent/memory.py','src/frontend/components/sessions.py']);

s=page('Bad Case：修复后也保留退化结果','四类：检索失败、生成错误、引用错误、Agent决策错误');
card(s,'15 → 0','无效必填论文ID调用\n通过输入约束与论文定位修复',.7,2.04,3.84,2.3);
card(s,'5 → 0','重复计划导致强制终止\n加强计划校验和观察结束条件',4.75,2.04,3.84,2.3);
card(s,'0/22 → 22/22','条件RAG完整来源文本保留\n按实际RAG结果规范回传',8.8,2.04,3.84,2.3);
text(s,'完整60题对照：平均Token下降10.69%，平均延迟增加7.76%。\n来源保留是结构指标；已知语义错答仍存在，180条等待人工评分。',.91,5.05,11.63,1.23,24);
note(s,'一轮优化，不只挑选失败案例复测。语义错误与资源共享边界在报告列出。',['reports/Bad_Case分析报告.md','reports/5_5_3 Bad Case分析与优化/Agent优化后60题_20261003.json']);

s=page('本地部署：应用 + Ollama','模型与镜像先准备，再使用封闭业务网络运行');
rect(s,.75,2.11,5.63,2.56,C.forest);text(s,'Streamlit 应用',1.05,2.47,4.98,.56,28,{color:C.white,bold:true});text(s,'Python 3.10.10\nChroma / SQLite / 本地权重',1.08,3.3,4.92,1,22,{color:C.moss});
rect(s,6.85,2.11,5.72,2.56,C.green);text(s,'Ollama 0.34.0',7.16,2.47,5.02,.56,28,{color:C.white,bold:true});text(s,'Qwen2.5:7b\n通过内部容器网络通信',7.17,3.3,4.91,1,22,{color:C.white});
text(s,'docker compose -f docker/docker-compose.yml up -d --build',.95,5.12,11.47,.61,19,{fontFace:'Menlo'});
text(s,deploy.status==='passed'?'实际镜像、健康检查、文档入库、问答及外网阻断均已核验。\n本次 Linux ARM64 容器使用 CPU；不把本机 Metal 延迟作为容器指标。':'镜像构建与依赖检查已完成；完整容器运行验证进行中。',.95,5.95,11.45,.84,20);
note(s,'开放端口仅127.0.0.1:8501，模型端口不对外开放。初次准备需联网，业务阶段不静默转云端。',['docker/Dockerfile','docker/docker-compose.yml',deployPath]);

s=page('完整交付：代码、报告、演示与评阅数据','原始输入和历史实验保持不覆盖');
card(s,'报告与手册','技术设计文档\n用户使用手册\n课程Word报告 / 本演示PPT',.7,2.08,3.84,3.06);
card(s,'可核验评测','逐题排名 / 原始模型响应\n实际Token / 延迟 / 图表\nBad Case与优化前后对照',4.75,2.08,3.84,3.06);
card(s,'180条人评表','基线120条 + 优化60条\n正确性 / 完整性 / 引用准确性\n待填写，不生成虚假均分',8.8,2.08,3.84,3.06);
text(s,'成员班级、学号、姓名与实际贡献比例，仍须由本人提供。\n教师考核与签名保留空白；没有填写就不声称已完成评分。',.91,5.82,11.63,.93,22);
note(s,'用户已确定180条评分范围，当前0/180已评。填写后的副本回传再用同口径汇总。',['reports/5_5_2 系统性能评估/人工评分180条来源.json','reports/5_5_2 系统性能评估/人工评分180条待填核验_20261004.json']);

s=page('演示：沿一条证据链走完系统','建议 8–10 分钟，现场演示约 3 分钟',true);
['上传PDF并观察进度','确认向量化与文献列表','提出论文问题并看流式答案','打开引用原文物理页','展示轨迹、Token与历史会话'].forEach((t,i)=>{const y=2.02+i*.76;rect(s,.81,y,.54,.54,C.gold);text(s,String(i+1),.86,y+.1,.44,.36,19,{align:'center',bold:true,color:C.white});text(s,t,1.66,y+.08,10.4,.49,26,{color:C.white});});
note(s,'演示建议：使用ViT三种数据集问题；通过calculator证明路由，通过历史切换证明会话管理。事先检查本地模型与索引。不要把模型自报完成当答案正确。',['docs/用户使用手册.md']);
fs.mkdirSync(path.dirname(out),{recursive:true});
deck.writeFile({fileName:out});
