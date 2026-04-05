import os
import json
from datetime import datetime

def generate_report(results, summary, output_path='coderot_report.html'):
    """Generate a standalone HTML report from analysis results."""

    lang_counts = {}
    for r in results:
        lang = r.get('language', 'Unknown')
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    lang_tags = ''.join([
        f'<span class="lang-tag">{l} <b>{c}</b></span>'
        for l, c in sorted(lang_counts.items(), key=lambda x: -x[1])
    ])

    file_cards = ''
    for r in results:
        is_def   = r['rf_pred'] == 1
        color    = '#f77' if r['risk_score'] > 70 else '#fa7' if r['risk_score'] > 40 else '#7f7'
        verdict  = 'At risk' if is_def else 'Healthy'
        badge_cls= 'defective' if is_def else 'clean'
        metrics  = ''.join([
            f'<div class="mi"><div class="mk">{k}</div><div class="mv">{v:.3f if isinstance(v,float) else v}</div></div>'
            for k, v in r['metrics'].items()
        ])
        agree_text = ('Both models agree — defect likely' if r['agree'] and is_def
                      else 'Both models agree — looks healthy' if r['agree']
                      else 'Models disagree — manual review recommended')
        agree_cls  = ('ag-def' if r['agree'] and is_def
                      else 'ag-clean' if r['agree'] else 'ag-dis')

        file_cards += f'''
        <div class="fc">
          <div class="fh">
            <div>
              <div class="fn">{r["filename"]}</div>
              <div class="fp">{r["filepath"]}</div>
              <span class="lb">{r["language"]}</span>
              <div class="rb"><div class="rbi" style="width:{r["risk_score"]}%;background:{color}"></div></div>
            </div>
            <div style="display:flex;align-items:center;gap:10px">
              <span style="color:#888;font-size:.82rem">Risk: {r["risk_score"]}%</span>
              <span class="badge {badge_cls}">{verdict}</span>
            </div>
          </div>
          <div class="fd">
            <div class="ab {agree_cls}">{agree_text}</div>
            <div class="mr">
              <div class="mb">
                <div style="color:#888;font-size:.75rem">Random Forest</div>
                <div style="font-weight:700;color:{'#f77' if r['rf_pred']==1 else '#7f7'}">
                  {'At risk' if r['rf_pred']==1 else 'Healthy'}</div>
                <div style="color:#888;font-size:.78rem">{r['rf_conf']}% confidence</div>
              </div>
              <div class="mb">
                <div style="color:#888;font-size:.75rem">SVM</div>
                <div style="font-weight:700;color:{'#f77' if r['svm_pred']==1 else '#7f7'}">
                  {'At risk' if r['svm_pred']==1 else 'Healthy'}</div>
                <div style="color:#888;font-size:.78rem">{r['svm_conf']}% confidence</div>
              </div>
            </div>
            <div class="mg">{metrics}</div>
          </div>
        </div>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Code Decay Report — {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;padding:30px 20px}}
    .wrap{{max-width:960px;margin:0 auto}}
    h1{{font-size:2rem;color:#fff;margin-bottom:6px}}
    h1 span{{color:#4f8ef7}}
    .meta{{color:#555;font-size:.85rem;margin-bottom:28px}}
    .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}}
    .st{{background:#1a1d27;border:1px solid #2a2d3e;border-radius:12px;padding:18px;text-align:center}}
    .st .n{{font-size:2rem;font-weight:700}}
    .st .l{{color:#888;font-size:.78rem;margin-top:4px}}
    .langs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}}
    .lang-tag{{background:#1a1d27;border:1px solid #2a2d3e;border-radius:8px;padding:5px 12px;font-size:.82rem}}
    .lang-tag b{{color:#4f8ef7;margin-left:4px}}
    .fc{{background:#1a1d27;border:1px solid #2a2d3e;border-radius:12px;padding:18px;margin-bottom:10px}}
    .fh{{display:flex;justify-content:space-between;cursor:pointer}}
    .fn{{font-weight:600;font-size:.95rem}}
    .fp{{color:#555;font-size:.75rem;margin-top:2px}}
    .lb{{display:inline-block;background:#1a2535;color:#4f8ef7;border:1px solid #1e3a5a;border-radius:4px;font-size:.7rem;padding:1px 7px;margin-top:4px}}
    .rb{{margin-top:8px;width:220px;background:#0f1117;border-radius:6px;height:5px;overflow:hidden}}
    .rbi{{height:100%;border-radius:6px}}
    .badge{{padding:4px 12px;border-radius:20px;font-size:.78rem;font-weight:600}}
    .defective{{background:#2a1a1a;color:#f77;border:1px solid #f774}}
    .clean{{background:#1a2a1a;color:#7f7;border:1px solid #7f74}}
    .fd{{display:none;margin-top:14px;border-top:1px solid #2a2d3e;padding-top:14px}}
    .fd.open{{display:block}}
    .ab{{text-align:center;padding:8px;border-radius:8px;font-size:.82rem;margin-bottom:12px}}
    .ag-def{{background:#2a1a1a;color:#f77}}
    .ag-clean{{background:#1a2a1a;color:#7f7}}
    .ag-dis{{background:#2a2a1a;color:#fa7}}
    .mr{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}}
    .mb{{background:#0f1117;border-radius:8px;padding:12px;text-align:center}}
    .mg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}}
    .mi{{background:#0f1117;border-radius:8px;padding:10px 12px}}
    .mk{{color:#888;font-size:.72rem}}
    .mv{{font-size:.95rem;font-weight:600;margin-top:2px}}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Code <span>Decay</span> Report</h1>
  <div class="meta">Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")} · {summary["total_files"]} files scanned</div>
  <div class="stats">
    <div class="st"><div class="n" style="color:#4f8ef7">{summary["total_files"]}</div><div class="l">Files scanned</div></div>
    <div class="st"><div class="n" style="color:#f77">{summary["defective"]}</div><div class="l">At risk</div></div>
    <div class="st"><div class="n" style="color:#7f7">{summary["clean"]}</div><div class="l">Healthy</div></div>
    <div class="st"><div class="n" style="color:#fa7">{summary["avg_risk"]}%</div><div class="l">Avg risk score</div></div>
  </div>
  <div class="langs">{lang_tags}</div>
  {file_cards}
</div>
<script>
document.querySelectorAll('.fh').forEach((h,i)=>{{
  h.addEventListener('click',()=>{{
    h.nextElementSibling.classList.toggle('open');
  }});
}});
</script>
</body></html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path