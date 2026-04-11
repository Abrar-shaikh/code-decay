from flask import Flask, render_template, request, jsonify
import joblib
import os
import shutil
import tempfile
from coderot.analyzer import analyze_codebase, extract_zip, LANGUAGE_MAP
from data_collector import save_batch, get_dataset_stats

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Load saved models
rf_model      = joblib.load('saved_models/random_forest_model.pkl')
svm_model     = joblib.load('saved_models/svm_model.pkl')
scaler        = joblib.load('saved_models/scaler.pkl')
feature_names = joblib.load('saved_models/feature_names.pkl')

FEATURE_ORDER = [
    'lines_of_code', 'cyclomatic_complexity', 'num_functions', 'num_classes',
    'comment_density', 'code_churn', 'developer_experience_years',
    'num_developers', 'commit_frequency', 'bug_fix_commits', 'past_defects',
    'test_coverage', 'duplication_percentage', 'avg_function_length',
    'depth_of_inheritance', 'response_for_class', 'coupling_between_objects',
    'lack_of_cohesion', 'build_failures', 'static_analysis_warnings',
    'security_vulnerabilities', 'performance_issues'
]


@app.route('/')
def home():
    supported = sorted(set(LANGUAGE_MAP.values()))
    stats     = get_dataset_stats()
    return render_template('index.html',
                           supported_languages=supported,
                           dataset_rows=stats['rows'])


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'project_zip' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    zip_file = request.files['project_zip']
    if not zip_file.filename.endswith('.zip'):
        return jsonify({'error': 'Please upload a .zip file.'})

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    tmp_dir = None

    try:
        zip_file.save(tmp_zip.name)
        tmp_zip.close()

        tmp_dir    = extract_zip(tmp_zip.name)
        files_data = analyze_codebase(tmp_dir)

        if not files_data:
            return jsonify({
                'error': 'No supported source files found in the ZIP.'
            })

        results     = []
        lang_counts = {}

        for info in files_data:
            lang = info.get('language', 'Unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            vec    = [info.get(f, 0) for f in FEATURE_ORDER]
            scaled = scaler.transform([vec])

            rf_pred  = int(rf_model.predict(scaled)[0])
            rf_conf  = float(
                rf_model.predict_proba(scaled)[0][rf_pred] * 100)
            svm_pred = int(svm_model.predict(scaled)[0])
            svm_conf = float(
                svm_model.predict_proba(scaled)[0][svm_pred] * 100)

            risk = round(
                (rf_conf  if rf_pred  == 1 else 100 - rf_conf)  * 0.6 +
                (svm_conf if svm_pred == 1 else 100 - svm_conf) * 0.4, 1)

            display_path = info['filepath'].replace(
                tmp_dir, '').lstrip('/\\')

            results.append({
                'filename'  : info['filename'],
                'filepath'  : display_path,
                'language'  : lang,
                'rf_pred'   : rf_pred,
                'rf_conf'   : round(rf_conf, 1),
                'svm_pred'  : svm_pred,
                'svm_conf'  : round(svm_conf, 1),
                'agree'     : rf_pred == svm_pred,
                'risk_score': risk,
                'metrics': {
                    'Lines of code'        : info.get('lines_of_code', 0),
                    'Cyclomatic complexity': info.get('cyclomatic_complexity', 0),
                    'Functions'            : info.get('num_functions', 0),
                    'Classes'              : info.get('num_classes', 0),
                    'Comment density'      : info.get('comment_density', 0),
                    'Static warnings'      : info.get('static_analysis_warnings', 0),
                    'Security issues'      : info.get('security_vulnerabilities', 0),
                    'Duplication %'        : info.get('duplication_percentage', 0),
                    'Test coverage est.'   : info.get('test_coverage', 0),
                    'Coupling'             : info.get('coupling_between_objects', 0),
                }
            })

        results.sort(key=lambda x: x['risk_score'], reverse=True)

        # ── Save metrics to community dataset in background ──────────────────
        # User's actual code is already deleted below in the finally block.
        # Only the 26 extracted numbers per file are saved — no code, no paths.
        save_batch(results, files_data)

        summary = {
            'total_files': len(results),
            'defective'  : sum(1 for r in results if r['rf_pred'] == 1),
            'clean'      : sum(1 for r in results if r['rf_pred'] == 0),
            'avg_risk'   : round(
                sum(r['risk_score'] for r in results) / len(results), 1),
            'lang_counts': lang_counts,
        }

        return jsonify({'results': results, 'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
        # ── Always delete user's code immediately ────────────────────────────
        try:
            os.unlink(tmp_zip.name)
        except:
            pass
        try:
            if tmp_dir:
                shutil.rmtree(tmp_dir)
        except:
            pass


@app.route('/dataset-stats')
def dataset_stats():
    """Returns live stats about the community dataset for the counter on the page."""
    return jsonify(get_dataset_stats())


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)