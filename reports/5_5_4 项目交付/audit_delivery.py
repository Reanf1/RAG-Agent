"""最终交付文件核验：原件、真实实验、测试、模板版式与待填评分分别检查。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-pdf', type=Path, required=True)
    parser.add_argument('--presentation-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('最终核验不得覆盖历史报告')
    evidence=ROOT/'reports/5_5_4 项目交付'
    benchmark=ROOT/'reports/5_5_2 系统性能评估/Windows正式性能评测_20261007'
    basis=json.loads((evidence/'构建与运行依据_20261004.json').read_text(encoding='utf-8'))
    for name,digest in basis['original_docs_sha256'].items():
        assert hashlib.sha256((ROOT/'docs'/name).read_bytes()).hexdigest()==digest
    for path,expected in [(ROOT/'reports/5_1_2 文本分块策略/五组分块检索对比_20261004.json',300),
                          *[(benchmark/(name+'.json'),count) for name,count in
                            [('retrieval',300),('agent',120),('routing',144),('parallel',12)]]]:
        data=json.loads(path.read_text(encoding='utf-8'))
        assert data['status']=='completed' and len(data['rows'])==expected, path
    run=json.loads((benchmark/'run.json').read_text(encoding='utf-8'))
    assert run['status']=='completed' and len(run['stages'])==4
    assert all(stage['exit_code']==0 for stage in run['stages'])
    metrics=json.loads((benchmark/'系统指标独立复核.json').read_text(encoding='utf-8'))
    assert metrics['passed'] and metrics['source_and_input_hashes_unchanged']
    routing=json.loads((benchmark/'routing_独立复核.json').read_text(encoding='utf-8'))
    parallel=json.loads((benchmark/'parallel_独立复核.json').read_text(encoding='utf-8'))
    assert routing['status']=='passed' and routing['rows']==144
    # 有调度失败的完整实验可以归档，但不能将其核验状态写成全成功。
    assert parallel['status'] in ('passed','completed_with_schedule_failures') and parallel['rows']==12
    # 四项正式实验必须来自同一冻结源码与题集，修复后的单次实调用单列。
    retrieval_data=json.loads((benchmark/'retrieval.json').read_text(encoding='utf-8'))
    agent_data=json.loads((benchmark/'agent.json').read_text(encoding='utf-8'))
    assert retrieval_data['inputs']==agent_data['inputs']
    normalize=lambda hashes: {name.replace('\\','/'):digest for name,digest in hashes.items()}
    frozen_sources=normalize(retrieval_data['inputs']['source_sha256'])
    for name in ('routing','parallel'):
        experiment=json.loads((benchmark/(name+'.json')).read_text(encoding='utf-8'))
        assert normalize(experiment['source_sha256'])==frozen_sources,name
        fingerprints=normalize(experiment['inputs_sha256'])
        assert fingerprints['config.yaml']==retrieval_data['inputs']['config_sha256']
        assert fingerprints['reports/评测集.json']==retrieval_data['inputs']['dataset_sha256']
        assert fingerprints['reports/5_5_1 评测集构建/论文清单.json']==retrieval_data['inputs']['manifest_sha256']
    deploy=json.loads((evidence/'Windows容器离线验收_20261007/container-7.json').read_text(encoding='utf-8'))
    assert deploy['status']=='passed' and deploy['new_process_reopen']['chunks']==deploy['chunks']
    assert deploy['external_tcp_blocked'] and deploy['health_after']['status']=='ok'
    original_index=json.loads((evidence/'Windows容器离线验收_20261007/原库最终只读核验.json').read_text(encoding='utf-8'))
    assert original_index['passed'] and original_index['chunks']==278 and original_index['documents']==6
    assert len(original_index['files'])==6 and all(f['unchanged'] for f in original_index['files'])
    resources=json.loads((benchmark/'资源采样汇总.json').read_text(encoding='utf-8'))
    assert resources['status']=='completed' and resources['overall']['samples']==1655
    assert sum(s['samples'] for s in resources['stages'].values())==1655
    tests=benchmark
    for name in ('I01_本地全量回归_UTF8修复','I01_Windows全量回归_2'):
        result=json.loads((tests/(name+'.json')).read_text(encoding='utf-8'))
        assert result['passed'] and result['run']==780
        assert all(not result[k] for k in ('errors','failures','skipped'))
    native_source=json.loads((benchmark/'I01_Windows源码核对.json').read_text(encoding='utf-8'))
    assert native_source['passed'] and native_source['files']==38 and not native_source['differences']
    for name,digest in native_source['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    test_sources=json.loads((benchmark/'I01_UTF8测试实际源码.json').read_text(encoding='utf-8'))
    assert len(test_sources)==7
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in test_sources.items())
    environment=json.loads((evidence/'Windows容器离线验收_20261007/environment-3.json').read_text(encoding='utf-8'))
    for name,digest in environment['source_sha256'].items():
        raw=(ROOT/name.removeprefix('/app/')).read_bytes()
        # 接受明确的LF/CRLF换行等价口径，保留容器实际原始字节指纹。
        assert digest in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\n',b'\r\n')).hexdigest()),name
    replay=json.loads((benchmark/'I01_Windows实际请求结果.json').read_text(encoding='utf-8'))
    assert replay['status']=='passed' and replay['http_status']==200
    assert replay['estimated_tokens']<=replay['input_budget']
    assert replay['source_sha256']==hashlib.sha256((ROOT/'src/agent/react_loop.py').read_bytes()).hexdigest()
    assert replay['response_sha256']==hashlib.sha256((benchmark/'I01_Windows实际请求结果.response.jsonl').read_bytes()).hexdigest()
    docpath=ROOT/'docs/课程报告.docx'
    with ZipFile(docpath) as z, ZipFile(ROOT/'docs/项目交付模板.docx') as original:
        body=ET.fromstring(z.read('word/document.xml'))
        old=ET.fromstring(original.read('word/document.xml'))
        for tag in ('pgSz','pgMar','docGrid'):
            assert body.find('.//w:sectPr/w:'+tag,NS).attrib==old.find('.//w:sectPr/w:'+tag,NS).attrib
        content=''.join(t.text or '' for t in body.findall('.//w:t',NS))
        assert '待核定' not in content and 'DELIVERY_' not in content
        for key in ['一 项目概述','二 团队分工说明','三 模块交付物清单','四 技术实现记录','五 技术设计文档','六 总结与展望','七 附录']:
            assert key in content
        tables=body.findall('.//w:tbl',NS)
        completion=sum([('功能' in ''.join(t.itertext()) and '已实现' in ''.join(t.itertext()) and '未实现' in ''.join(t.itertext())) for t in tables])
        assert completion==5
        images=len(body.findall('.//w:drawing',NS))
        assert images>=10
        # 原模板关系文件和样式保留，新增页脚/标题书签允许改变相关部分。
        assert 'word/styles.xml' in z.namelist() and 'word/theme/theme1.xml' in z.namelist()
    from pypdf import PdfReader
    rendered=PdfReader(args.render_pdf)
    pages=len(rendered.pages)
    toc=json.loads((evidence/'课程报告目录页号_20261007.json').read_text(encoding='utf-8'))
    for title,page in toc.items():
        assert re.sub(r'\s+','',title) in re.sub(r'\s+','',rendered.pages[page-1].extract_text())
    ppt=ROOT/'docs/项目演示.pptx'
    receipt=json.loads(args.presentation_receipt.read_text(encoding='utf-8'))
    assert receipt['finalSha256']==hashlib.sha256(ppt.read_bytes()).hexdigest()
    assert receipt['firstPartyImport']['passed'] and receipt['nativeChartValidation']['passed']
    assert receipt['fontSelection']['passed'] and receipt['presentationLayout']['exitCode']==0
    visual=json.loads((evidence/'交付文件视觉核验_20261007.json').read_text(encoding='utf-8'))
    assert visual['status']=='passed' and visual['docx']['pages_visually_checked']==pages
    assert visual['docx']['sha256']==hashlib.sha256(docpath.read_bytes()).hexdigest()
    assert visual['pptx']['sha256']==receipt['finalSha256'] and visual['pptx']['slides_visually_checked']==14
    with ZipFile(ppt) as z:
        slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        assert len(slides)==14
        slide_text=''.join(''.join(ET.fromstring(z.read(n)).itertext()) for n in slides)
        assert '实验进行中' not in slide_text and '初稿' not in slide_text
        assert len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==14
        for owner in (2,11,13):
            assert ET.fromstring(z.read(f'ppt/slides/slide{owner}.xml')).find('.//a:tbl',NS) is not None
        assert receipt['nativeChartValidation']['detectedOwnerSlides']==[4,5,8,9,12]
    review=json.loads((ROOT/'reports/5_5_2 系统性能评估/助手初评180条_交付核验.json').read_text(encoding='utf-8'))
    assert review['records']==180 and review['assistant_scores']==540 and review['user_reviewed']==0
    current_review=json.loads((benchmark/'助手逐题初评.json').read_text(encoding='utf-8'))
    assert len(current_review['rows'])==120
    answers=json.loads((benchmark/'agent.json').read_text(encoding='utf-8'))
    original_answers={(r['id'],r['profile']):r['answer'] for r in answers['rows']}
    assert {(r['id'],r['profile']) for r in current_review['rows']}==set(original_answers)
    for row in current_review['rows']:
        assert row['answer']==original_answers[(row['id'],row['profile'])]
        assert row['assistant_reason'] and all(type(v) is int and 0<=v<=4 for v in row['assistant_scores'].values())
        assert all(v is None for v in row['user_review'].values())
    workbook=ROOT/'outputs/quality-review-20261007/Windows答案助手初评与用户审核_120条.xlsx'
    ordered=sorted(current_review['rows'],key=lambda r:(r['id'],['default','no_rules'].index(r['profile'])))
    with ZipFile(workbook) as z:
        xns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        values={c.attrib['r']:c.findtext('s:v',default='',namespaces=xns) for c in sheet.findall('.//s:c',xns)}
        # 直接检查导出的数值和空白字段，不依赖生成脚本声称已清空。
        for number,row in enumerate(ordered,13):
            for column,key in zip('CDE',('correctness','completeness','citation_accuracy')):
                assert float(values[column+str(number)])==row['assistant_scores'][key]
            assert all(values.get(c+str(number),'')=='' for c in 'GHIJKL')
            assert values['M'+str(number)]=='0'
        for number,profile in enumerate(('default','no_rules'),7):
            assert float(values['B'+str(number)])==60 and values['F'+str(number)]=='0'
            for column,key in zip('CDE',('correctness','completeness','citation_accuracy')):
                expected=sum(r['assistant_scores'][key] for r in ordered if r['profile']==profile)/60
                assert abs(float(values[column+str(number)])-expected)<1e-12
        assert not any(c.attrib.get('t')=='e' for c in sheet.findall('.//s:c',xns))
        strings=ET.fromstring(z.read('xl/sharedStrings.xml')) if 'xl/sharedStrings.xml' in z.namelist() else []
        shared=[''.join(t.text or '' for t in item.findall('.//s:t',xns)) for item in strings]
        detail=ET.fromstring(z.read('xl/worksheets/sheet2.xml'))
        answer_cells={}
        for cell in detail.findall('.//s:c',xns):
            if not cell.attrib['r'].startswith('F'): continue
            value=cell.findtext('s:v',default='',namespaces=xns)
            answer_cells[int(cell.attrib['r'][1:])]=(shared[int(value)] if cell.attrib.get('t')=='s' else
                ''.join(t.text or '' for t in cell.findall('.//s:t',xns)) if cell.attrib.get('t')=='inlineStr' else value)
        anchors=[int(values['N'+str(number)]) for number in range(13,133)]
        for index,row in enumerate(ordered):
            end=anchors[index+1] if index+1<len(anchors) else max(answer_cells)+1
            assert ''.join(answer_cells.get(number,'') for number in range(anchors[index],end))==row['answer']
    links=[]
    checkpaths=[ROOT/'README.md',ROOT/'reports/README.md',ROOT/'tests/README.md',ROOT/'docs/课程报告.md',ROOT/'docs/技术设计文档.md',ROOT/'docs/用户使用手册.md',ROOT/'docs/QA/5.5.4 项目交付.md',evidence/'README.md']
    for path in checkpaths:
        for target in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)',path.read_text(encoding='utf-8')):
            url=urlparse(target.strip('<>'))
            if url.scheme or not url.path:continue
            destination=(path.parent/unquote(url.path)).resolve()
            # 本次核验报告在全部检查结束后生成，允许清单引用这个确切输出。
            assert destination.exists() or destination==args.output.resolve(), (path,target)
            links.append(str(destination))
    out={'status':'passed','original_docs_unchanged':True,'all_tests_each_platform':780,
         'frozen_experiments_same_source_and_inputs':True,'I01_native_and_container_source_verified':True,
         'chunking_rows':300,'chunking_metric_checks':900,'routing_requests':144,'parallel_requests':12,
         'parallel_validation_status':parallel['status'],'parallel_schedule_failures':parallel['schedule_failures'],
         'original_index_unchanged':{'chunks':278,'documents':6,'content_sha256':original_index['content_sha256'],'all_raw_files_unchanged':True},
         'resource_samples':resources['overall']['samples'],
         'docx_pages':pages,'completion_tables':completion,'docx_images':images,'pptx_slides':14,
         'visual_checks':{'docx_pages':pages,'pptx_slides':14,'quality_workbook_previews':5,'performance_workbook_previews':3},
         'local_links_checked':len(links),'human_review':{'historical_assistant':180,'windows_assistant':120,'user_reviewed':0,'status':'awaiting_user_review'},
         'members':'真实身份/实际贡献未提供，保留待填','artifact_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [docpath,ppt,workbook,ROOT/'outputs/performance-20261007/系统性能评测数据.xlsx',ROOT/'reports/5_5_2 系统性能评估/答案质量人工评分表_180条.xlsx']},
         'boundary':'技术交付与真实证据核验通过不等于用户质量审核、成员资料或教师考核已完成。'}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
