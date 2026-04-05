from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from coderot.analyzer import analyze_codebase

app = Flask(__name__)

# Load saved models
rf_model       = joblib.load('saved_models/random_forest_model.pkl')
svm_model      = joblib.load('saved_models/svm_model.pkl')
scaler         = joblib.load('saved_models/scaler.pkl')
feature_names  = joblib.load('saved_models/feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    folder = request.form.get('folder_path', '').strip()

    if not folder or not os.path.isdir(folder):
        return jsonify({'error': 'Invalid folder path. Please enter a valid directory.'})

    try:
        files_data = analyze_codebase(folder)

        if not files_data:
            return jsonify({'error': 'No Python files found in that folder.'})

        results = []
        for file_info in files_data:
            # Build feature vector in correct order
            feature_vector = [file_info.get(f, 0) for f in feature_names]
            scaled = scaler.transform([feature_vector])

            rf_pred  = int(rf_model.predict(scaled)[0])
            rf_conf  = float(rf_model.predict_proba(scaled)[0][rf_pred] * 100)
            svm_pred = int(svm_model.predict(scaled)[0])
            svm_conf = float(svm_model.predict_proba(scaled)[0][svm_pred] * 100)

            # Risk score: average confidence weighted toward defective
            risk_score = round((rf_conf if rf_pred == 1 else 100 - rf_conf) * 0.6 +
                               (svm_conf if svm_pred == 1 else 100 - svm_conf) * 0.4, 1)

            results.append({
                'filename'   : file_info['filename'],
                'filepath'   : file_info['filepath'],
                'rf_pred'    : rf_pred,
                'rf_conf'    : round(rf_conf, 1),
                'svm_pred'   : svm_pred,
                'svm_conf'   : round(svm_conf, 1),
                'agree'      : rf_pred == svm_pred,
                'risk_score' : risk_score,
                'metrics'    : {
                    'Lines of code'          : file_info.get('lines_of_code', 0),
                    'Cyclomatic complexity'  : file_info.get('cyclomatic_complexity', 0),
                    'Functions'              : file_info.get('num_functions', 0),
                    'Classes'                : file_info.get('num_classes', 0),
                    'Static warnings'        : file_info.get('static_analysis_warnings', 0),
                    'Security issues'        : file_info.get('security_vulnerabilities', 0),
                    'Duplication %'          : file_info.get('duplication_percentage', 0),
                    'Test coverage est.'     : file_info.get('test_coverage', 0),
                    'AST nodes'              : file_info.get('ast_num_nodes', 0),
                    'AST depth'              : file_info.get('ast_max_depth', 0),
                }
            })

        # Sort by risk score descending
        results.sort(key=lambda x: x['risk_score'], reverse=True)

        summary = {
            'total_files'   : len(results),
            'defective'     : sum(1 for r in results if r['rf_pred'] == 1),
            'clean'         : sum(1 for r in results if r['rf_pred'] == 0),
            'avg_risk'      : round(sum(r['risk_score'] for r in results) / len(results), 1)
        }

        return jsonify({'results': results, 'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)