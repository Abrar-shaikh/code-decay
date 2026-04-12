from flask import Flask, render_template, request, jsonify
import joblib
import os
import shutil
import tempfile
from coderot.analyzer import analyze_codebase, extract_zip, LANGUAGE_MAP
from data_collector import save_batch, get_dataset_stats

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Load all 5 models
rf_model  = joblib.load('saved_models/random_forest_model.pkl')
svm_model = joblib.load('saved_models/svm_model.pkl')
ada_model = joblib.load('saved_models/adaboost_model.pkl')
gb_model  = joblib.load('saved_models/gradient_boost_model.pkl')
xgb_model = joblib.load('saved_models/xgboost_model.pkl')
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


def get_prediction(model, scaled):
    """Get prediction and confidence from any model."""
    pred = int(model.predict(scaled)[0])
    conf = float(model.predict_proba(scaled)[0][pred] * 100)
    return pred, round(conf, 1)


def calculate_risk(predictions):
    """
    Calculate overall risk score from all 5 models.
    Weighted: GB and XGBoost get more weight as top performers.
    """
    weights = {
        'rf' : 0.15,
        'svm': 0.15,
        'ada': 0.10,
        'gb' : 0.35,   # highest weight — 100% accuracy
        'xgb': 0.25,   # second highest weight
    }
    score = 0
    for key, weight in weights.items():
        pred, conf = predictions[key]
        score += (conf if pred == 1 else 100 - conf) * weight
    return round(score, 1)


def majority_vote(predictions):
    """Returns True if majority of models say defective."""
    votes = sum(1 for pred, _ in predictions.values() if pred == 1)
    return votes >= 3  # 3 or more out of 5 say defective


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

            # Get predictions from all 5 models
            predictions = {
                'rf' : get_prediction(rf_model,  scaled),
                'svm': get_prediction(svm_model, scaled),
                'ada': get_prediction(ada_model, scaled),
                'gb' : get_prediction(gb_model,  scaled),
                'xgb': get_prediction(xgb_model, scaled),
            }

            risk        = calculate_risk(predictions)
            is_defective = majority_vote(predictions)
            votes_defective = sum(
                1 for pred, _ in predictions.values() if pred == 1)

            display_path = info['filepath'].replace(
                tmp_dir, '').lstrip('/\\')

            results.append({
                'filename'       : info['filename'],
                'filepath'       : display_path,
                'language'       : lang,
                'is_defective'   : is_defective,
                'votes_defective': votes_defective,
                'risk_score'     : risk,
                'agree'          : votes_defective == 5 or votes_defective == 0,
                'models': {
                    'Random Forest'     : {'pred': predictions['rf'][0],  'conf': predictions['rf'][1]},
                    'SVM'               : {'pred': predictions['svm'][0], 'conf': predictions['svm'][1]},
                    'AdaBoost'          : {'pred': predictions['ada'][0], 'conf': predictions['ada'][1]},
                    'Gradient Boosting' : {'pred': predictions['gb'][0],  'conf': predictions['gb'][1]},
                    'XGBoost'           : {'pred': predictions['xgb'][0], 'conf': predictions['xgb'][1]},
                },
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

        # Save to community dataset in background
        save_batch(results, files_data)

        summary = {
            'total_files': len(results),
            'defective'  : sum(1 for r in results if r['is_defective']),
            'clean'      : sum(1 for r in results if not r['is_defective']),
            'avg_risk'   : round(
                sum(r['risk_score'] for r in results) / len(results), 1),
            'lang_counts': lang_counts,
        }

        return jsonify({'results': results, 'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
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
    return jsonify(get_dataset_stats())


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)