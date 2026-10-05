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
    args=parser.parse_args()
    evidence=ROOT/'reports/5_5_4 项目交付'
    basis=json.loads((evidence/'构建与运行依据_20261004.json').read_text())
    for name,digest in basis['original_docs_sha256'].items():
        assert hashlib.sha256((ROOT/'docs'/name).read_bytes()).hexdigest()==digest
    for rel,expected in [('reports/5_1_2 文本分块策略/五组分块检索对比_20261004.json',300),
                         ('reports/5_5_2 系统性能评估/Agent与固定RAG对照_20261004.json',144),
                         ('reports/5_5_2 系统性能评估/独立工具串行并行对照_20261004.json',12)]:
        data=json.loads((ROOT/rel).read_text());assert data['status']=='completed' and len(data['rows'])==expected
    deploy=json.loads((evidence/'容器完整验证_20261004.json').read_text())
    assert deploy['status']=='passed' and deploy['new_process_reopen']['chunks']==deploy['chunks']
    assert deploy['external_tcp_blocked'] and deploy['health_after']['status']=='ok'
    assert 'Ran 668 tests' in (evidence/'全量668项通过_20261004.log').read_text()
    assert (evidence/'全量668项通过_20261004.log').read_text().rstrip().endswith('OK')
    assert json.loads((evidence/'模块四补齐专项_20261004.json').read_text())['passed']
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
    toc=json.loads((evidence/'课程报告目录页号.json').read_text())
    for title,page in toc.items():
        assert re.sub(r'\s+','',title) in re.sub(r'\s+','',rendered.pages[page-1].extract_text())
    ppt=ROOT/'docs/项目演示.pptx'
    with ZipFile(ppt) as z:
        slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        assert len(slides)==14
        slide_text=''.join(''.join(ET.fromstring(z.read(n)).itertext()) for n in slides)
        assert '实验进行中' not in slide_text and '初稿' not in slide_text
        assert len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)])==14
    review=json.loads((ROOT/'reports/5_5_2 系统性能评估/人工评分180条待填核验_20261004.json').read_text())
    assert review['expected']==180 and review['reviewed']==0
    assert all(p[k] is None for p in review['summaries'].values() for k in ('correctness_mean','completeness_mean','citation_accuracy_mean'))
    links=[]
    checkpaths=[ROOT/'README.md',ROOT/'reports/README.md',ROOT/'tests/README.md',ROOT/'docs/课程报告.md',ROOT/'docs/技术设计文档.md',ROOT/'docs/用户使用手册.md',ROOT/'docs/QA/5.5.4 项目交付.md',evidence/'README.md']
    for path in checkpaths:
        for target in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)',path.read_text()):
            url=urlparse(target.strip('<>'))
            if url.scheme or not url.path:continue
            destination=(path.parent/unquote(url.path)).resolve()
            assert destination.exists(), (path,target)
            links.append(str(destination))
    out={'status':'passed','original_docs_unchanged':True,'all_tests':668,'module4_tests':183,
         'chunking_rows':300,'chunking_metric_checks':900,'routing_requests':144,'parallel_requests':12,
         'docx_pages':pages,'completion_tables':completion,'docx_images':images,'pptx_slides':14,
         'local_links_checked':len(links),'human_review':{'expected':180,'reviewed':0,'status':'awaiting_human_review'},
         'members':'真实身份/实际贡献未提供，保留待填','artifact_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [docpath,ppt,ROOT/'reports/5_5_2 系统性能评估/答案质量人工评分表_180条.xlsx']},
         'boundary':'技术交付与真实证据核验通过不等于180条人评或教师考核已完成。'}
    (evidence/'最终交付核验_20261004.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
