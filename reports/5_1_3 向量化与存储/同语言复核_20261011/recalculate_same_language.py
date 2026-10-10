"""仅复算历史记录中未经改写的同语言查询，不伪造新运行耗时。"""
import json, hashlib, statistics
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
newpath=OUT/'冻结同语言输入.json';sample=json.loads(newpath.read_text());ids={q['id'] for q in sample['queries']}
source=json.loads((ROOT/'reports/5_1_3 向量化与存储/Embedding论文双语对比结果.json').read_text())
old_ids=[q['query_id'] for q in source['models'][0]['queries']];indices=[i for i,key in enumerate(old_ids) if key in ids]
assert len(indices)==48 and all(q['query_language']==q['document_language'] for q in sample['queries'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def quality(rows):
 rows=deepcopy(rows)
 for r in rows:
  assert r['query_id'] in ids
  relevant=set(r['relevant_ids']);top=r['top10_ids'];first=next((i for i,p in enumerate(top,1) if p in relevant),None)
  assert r['hit_at_5']==bool(relevant&set(top[:5]))
  assert abs(r['recall_at_5']-len(relevant&set(top[:5]))/len(relevant))<1e-12
  assert abs(r['reciprocal_rank_at_10']-(1/first if first else 0))<1e-12
 result={'queries':rows,'query_count':len(rows)}
 for out,key in [('hit_at_5','hit_at_5'),('recall_at_5','recall_at_5'),('mrr_at_10','reciprocal_rank_at_10')]:result[out]=statistics.mean(r[key] for r in rows)
 result['groups']={g:quality_group([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})};return result
def quality_group(rows):return {'queries':len(rows),'hit_at_5':statistics.mean(r['hit_at_5'] for r in rows),'recall_at_5':statistics.mean(r['recall_at_5'] for r in rows),'mrr_at_10':statistics.mean(r['reciprocal_rank_at_10'] for r in rows)}
def times(values):
 assert len(values)%96==0
 return [values[run*96+i] for run in range(len(values)//96) for i in indices]
common={'status':'completed','recalculated_at':datetime.now().astimezone().isoformat(),'scope':'只保留既有原始记录中48道同语言问题，英文原文英文题24、中文原文中文题24；候选821块不变。此次只复算，不重新调用模型或数据库；耗时保持原实验测量。','sample_sha256':sha(newpath),'source_sample_sha256':source['sample_sha256'],'queries':sample['queries'],'corpus_count':821}
src=ROOT/'reports/5_1_3 向量化与存储/Embedding论文双语对比结果.json';e=json.loads(src.read_text());result={**common,'source_run_at':e['started_at'],'source_result_path':str(src.relative_to(ROOT)),'source_result_sha256':sha(src),'environment':{k:e[k] for k in ['python','system','device','dtype','cpu_threads','batch_size','max_tokens','runs','package_versions']},'models':[]}
for m in e['models']:
 row=quality([r for r in m['queries'] if r['query_id'] in ids]);row.update({k:deepcopy(m[k]) for k in ['model','revision','dimension','document_seconds_runs','document_seconds_median','load_seconds','document_languages','query_instruction','weight_bytes']})
 measured=times(m['query_seconds_runs']);row['query_seconds_runs']=measured;row['query_ms_median']=statistics.median(measured)*1000;row['query_ms_p95']=float(np.percentile(measured,95))*1000;result['models'].append(row)
save('Embedding同语言对比结果.json',result)
src=ROOT/'reports/5_1_3 向量化与存储/向量数据库对比结果.json';d=json.loads(src.read_text());db={**common,'source_run_at':d['created_at'],'source_result_path':str(src.relative_to(ROOT)),'source_result_sha256':sha(src),'source_vectors_sha256':d['vectors_sha256'],'embedding':d['embedding'],'environment':d['environment'],'chroma_hnsw':d['chroma_hnsw'],'runs':{},'quality':{}}
for b,runs in d['runs'].items():
 db['runs'][b]=[];db['quality'][b]=quality([r for r in d['quality'][b]['queries'] if r['query_id'] in ids])
 for run in runs:
  r=deepcopy(run);r['rankings']=[r['rankings'][i] for i in indices];r['query_ms']=times(r['query_ms']);r['query_p50_ms']=statistics.median(r['query_ms']);r['query_p95_ms']=float(np.percentile(r['query_ms'],95));db['runs'][b].append(r)
 for i in range(48):assert all(run['rankings'][i]['ids']==db['runs']['chroma'][0]['rankings'][i]['ids'] for run in db['runs'][b])
db['top5_overlap_with_exact']=statistics.mean(len(set(a['ids'][:5])&set(b['ids'][:5]))/5 for a,b in zip(db['runs']['chroma'][0]['rankings'],db['runs']['faiss'][0]['rankings']))
save('向量数据库同语言对比结果.json',db)
save('冻结同语言输入.json',sample)
print('Embedding',[(r['model'],r['hit_at_5'],r['recall_at_5'],r['mrr_at_10'],r['query_ms_median']) for r in result['models']]);print('DB',{b:{k:v for k,v in x.items() if k not in ['queries','groups']} for b,x in db['quality'].items()})
