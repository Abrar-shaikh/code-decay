from flask import Flask, render_template, request, jsonify
import joblib
import os
import shutil
import tempfile
from coderot.analyzer import analyze_codebase, extract_zip, LANGUAGE_MAP
from data_collector import save_batch, get_dataset_stats

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

FEATURE_ORDER = [
    'lines_of_code', 'cyclomatic_complexity', 'num_functions', 'num_classes',
    'comment_density', 'code_churn', 'developer_experience_years',
    'num_developers', 'commit_frequency', 'bug_fix_commits', 'past_defects',
    'test_coverage', 'duplication_percentage', 'avg_function_length',
    'depth_of_inheritance', 'response_for_class', 'coupling_between_objects',
    'lack_of_cohesion', 'build_failures', 'static_analysis_warnings',
    'security_vulnerabilities', 'performance_issues'
]

# ── Lazy model loading ────────────────────────────────────────────────────────
# Models are loaded once on first request, not at startup.
# This prevents Render free tier from timing out during boot.
_models = {}


def get_models():
    """Load models once and cache them."""
    if not _models:
        print("[app] Loading models...")
        _models['rf']     = joblib.load('saved_models/random_forest_model.pkl')
        _models['svm']    = joblib.load('saved_models/svm_model.pkl')
        _models['ada']    = joblib.load('saved_models/adaboost_model.pkl')
        _models['gb']     = joblib.load('saved_models/gradient_boost_model.pkl')
        _models['xgb']    = joblib.load('saved_models/xgboost_model.pkl')
        _models['scaler'] = joblib.load('saved_models/scaler.pkl')
        print("[app] All models loaded!")
    return _models


def get_prediction(model, scaled):
    pred = int(model.predict(scaled)[0])
    conf = float(model.predict_proba(scaled)[0][pred] * 100)
    return pred, round(conf, 1)


def calculate_risk(predictions):
    weights = {
        'rf': 0.15, 'svm': 0.15, 'ada': 0.10, 'gb': 0.35, 'xgb': 0.25
    }
    score = 0
    for key, weight in weights.items():
        pred, conf = predictions[key]
        score += (conf if pred == 1 else 100 - conf) * weight
    return round(score, 1)


def majority_vote(predictions):
    votes = sum(1 for pred, _ in predictions.values() if pred == 1)
    return votes >= 3


def generate_fixes(info, risk_score):
    """Return a list of actionable fix suggestions based on measured metrics.

    Each suggestion is a dict:
        severity  : 'critical' | 'warning' | 'info'
        metric    : human-readable metric name
        value     : the measured value
        problem   : one-sentence explanation of why it's a problem
        fix       : concrete steps the developer can take
        example   : short before/after code snippet (plain text), or None
    """
    fixes = []

    loc  = info.get('lines_of_code', 0)
    cc   = info.get('cyclomatic_complexity', 0)
    fns  = info.get('num_functions', 0)
    afl  = info.get('avg_function_length', 0)
    cd   = info.get('comment_density', 0)
    saw  = info.get('static_analysis_warnings', 0)
    sec  = info.get('security_vulnerabilities', 0)
    dup  = info.get('duplication_percentage', 0)
    tc   = info.get('test_coverage', 0)
    coup = info.get('coupling_between_objects', 0)
    lang = info.get('language', '')

    # ── Security issues (always critical, show first) ──────────────────────
    if sec >= 1:
        fixes.append({
            'severity': 'critical',
            'metric':   'Security Issues',
            'value':    sec,
            'problem':  f'{sec} dangerous pattern(s) detected (e.g. eval(), '
                        'shell=True, pickle.loads, verify=False, innerHTML).',
            'fix':      (
                'Search the file for each flagged pattern. '
                'Replace eval()/exec() with ast.literal_eval() for safe parsing. '
                'Replace subprocess(shell=True) with a list of arguments '
                '(e.g. subprocess.run(["ls", "-la"])). '
                'Replace pickle.loads with json.loads or a safe serialiser. '
                'Replace verify=False with a proper CA bundle path. '
                'Replace innerHTML assignments with textContent or '
                'DOM createElement() to prevent XSS.'
            ),
            'example': (
                '# BEFORE (dangerous)\n'
                'result = eval(user_input)\n'
                'subprocess.call(cmd, shell=True)\n\n'
                '# AFTER (safe)\n'
                'result = ast.literal_eval(user_input)\n'
                'subprocess.run(["cmd", arg1], shell=False)'
            )
        })

    # ── Static warnings ────────────────────────────────────────────────────
    if saw >= 4:
        fixes.append({
            'severity': 'critical',
            'metric':   'Static Warnings',
            'value':    saw,
            'problem':  f'{saw} risky code markers found (TODO/FIXME/HACK, '
                        'bare except, os.system, etc.).',
            'fix':      (
                'Resolve each TODO/FIXME — if they can\'t be resolved now, '
                'convert them into tracked issues in your issue tracker and remove '
                'the marker from the code. '
                'Replace bare "except: pass" blocks with specific exception types '
                'and proper logging. '
                'Replace os.system() calls with subprocess.run() with a list of args.'
            ),
            'example': (
                '# BEFORE\n'
                'try:\n'
                '    do_something()\n'
                'except:\n'
                '    pass  # TODO fix this later\n\n'
                '# AFTER\n'
                'try:\n'
                '    do_something()\n'
                'except ValueError as e:\n'
                '    logger.error("do_something failed: %s", e)'
            )
        })
    elif saw >= 1:
        fixes.append({
            'severity': 'warning',
            'metric':   'Static Warnings',
            'value':    saw,
            'problem':  f'{saw} risky marker(s) found (TODO, FIXME, bare except, etc.).',
            'fix':      (
                'Search the file for TODO, FIXME, HACK, XXX comments and either '
                'resolve them or move them to your issue tracker. '
                'Replace any bare "except: pass" with a specific exception type.'
            ),
            'example': None
        })

    # ── Cyclomatic complexity ──────────────────────────────────────────────
    if cc > 20:
        fixes.append({
            'severity': 'critical',
            'metric':   'Cyclomatic Complexity',
            'value':    cc,
            'problem':  f'Complexity of {cc} means {cc} independent paths through '
                        'this file — very hard to test exhaustively.',
            'fix':      (
                'Extract long if/else chains into separate functions with descriptive names. '
                'Replace deep nested conditions with early returns (guard clauses). '
                'Consider the Strategy or Command pattern to replace large switch/case blocks. '
                'Aim for each function to have complexity ≤ 10.'
            ),
            'example': (
                '# BEFORE — deeply nested\n'
                'def process(x):\n'
                '    if x is not None:\n'
                '        if x > 0:\n'
                '            if x < 100:\n'
                '                return x * 2\n\n'
                '# AFTER — guard clauses (early return)\n'
                'def process(x):\n'
                '    if x is None: return None\n'
                '    if x <= 0:    return None\n'
                '    if x >= 100:  return None\n'
                '    return x * 2'
            )
        })
    elif cc > 10:
        fixes.append({
            'severity': 'warning',
            'metric':   'Cyclomatic Complexity',
            'value':    cc,
            'problem':  f'Complexity of {cc} — moderate, but each branch needs its own test.',
            'fix':      (
                'Look for functions with 3+ nested levels and extract inner blocks '
                'into helper functions. Use guard clauses (early returns) to flatten nesting.'
            ),
            'example': None
        })

    # ── Test coverage ──────────────────────────────────────────────────────
    if tc < 0.10:
        fixes.append({
            'severity': 'critical' if risk_score > 50 else 'warning',
            'metric':   'Test Coverage Est.',
            'value':    round(tc, 3),
            'problem':  'No test keywords detected in this file — it appears untested.',
            'fix':      (
                'Create a corresponding test file (e.g. test_filename.py for Python, '
                'filename.test.js for JS). '
                'For Python, write functions starting with "def test_" using pytest '
                'and assert statements. '
                'Start with the highest-risk functions (those with high cyclomatic complexity). '
                'Aim for at least one test per public function covering the happy path '
                'and one edge/error case.'
            ),
            'example': (
                '# Python example (pytest)\n'
                '# File: test_my_module.py\n\n'
                'from my_module import calculate_risk\n\n'
                'def test_calculate_risk_returns_float():\n'
                '    result = calculate_risk({"rf": (1, 80.0)})\n'
                '    assert isinstance(result, float)\n\n'
                'def test_calculate_risk_zero_for_clean():\n'
                '    result = calculate_risk({"rf": (0, 95.0)})\n'
                '    assert result < 20'
            )
        })
    elif tc < 0.50:
        fixes.append({
            'severity': 'info',
            'metric':   'Test Coverage Est.',
            'value':    round(tc, 3),
            'problem':  'Some tests detected but coverage appears low.',
            'fix':      (
                'Identify functions that have no corresponding test_ function. '
                'Add tests for error paths and edge cases, not just happy paths. '
                'Use "pytest --cov" to measure real line coverage.'
            ),
            'example': None
        })

    # ── Lines of code / file size ──────────────────────────────────────────
    if loc > 500:
        fixes.append({
            'severity': 'warning',
            'metric':   'Lines of Code',
            'value':    loc,
            'problem':  f'File has {loc} lines — large files are harder to navigate, '
                        'review, and test.',
            'fix':      (
                'Split the file by responsibility using the Single Responsibility Principle: '
                'one file should do one thing. '
                'Group related functions into their own module. '
                'For Python Flask apps: split routes, models, utilities into separate files '
                'and use Blueprints.'
            ),
            'example': (
                '# BEFORE: everything in app.py\n'
                '# AFTER: separate modules\n'
                '# app.py          — Flask app + routes only\n'
                '# analyzer.py     — analysis logic only\n'
                '# data_collector.py — data saving only\n'
                '# models.py       — ML model loading only'
            )
        })

    # ── Duplication ────────────────────────────────────────────────────────
    if dup > 0.30:
        fixes.append({
            'severity': 'warning',
            'metric':   'Duplication %',
            'value':    round(dup, 3),
            'problem':  f'{round(dup * 100, 1)}% of lines are duplicates within this file.',
            'fix':      (
                'Extract repeated code blocks into a single reusable function. '
                'If the same logic appears in multiple files, move it to a shared '
                'utility module. '
                'Apply the DRY principle (Don\'t Repeat Yourself): '
                'every piece of knowledge should have one authoritative location.'
            ),
            'example': (
                '# BEFORE — repeated logic\n'
                'risk_a = (conf_a if pred_a == 1 else 100 - conf_a) * 0.35\n'
                'risk_b = (conf_b if pred_b == 1 else 100 - conf_b) * 0.25\n\n'
                '# AFTER — extracted function\n'
                'def weighted_risk(pred, conf, weight):\n'
                '    return (conf if pred == 1 else 100 - conf) * weight\n\n'
                'risk_a = weighted_risk(pred_a, conf_a, 0.35)\n'
                'risk_b = weighted_risk(pred_b, conf_b, 0.25)'
            )
        })

    # ── Comment density ────────────────────────────────────────────────────
    if cd < 0.03:
        fixes.append({
            'severity': 'info',
            'metric':   'Comment Density',
            'value':    cd,
            'problem':  'Almost no comments — future developers (including you) '
                        'will struggle to understand the intent.',
            'fix':      (
                'Add a one-line docstring to every function explaining what it does, '
                'not how (the code shows how). '
                'Add inline comments only where the logic is non-obvious. '
                f'For {lang}, use the standard docstring format '
                '(triple-quotes for Python, JSDoc for JS/TS, Javadoc for Java).'
            ),
            'example': (
                '# BEFORE\n'
                'def calc(p):\n'
                '    return sum((c if v==1 else 100-c)*w '
                'for (v,c),(w) in zip(p.values(), WEIGHTS))\n\n'
                '# AFTER\n'
                'def calculate_risk(predictions: dict) -> float:\n'
                '    """Compute weighted defect probability (0–100).\n'
                '    Higher = more likely defective.\n'
                '    """\n'
                '    return sum(\n'
                '        (conf if pred == 1 else 100 - conf) * weight\n'
                '        for (pred, conf), weight in zip(\n'
                '            predictions.values(), WEIGHTS.values())\n'
                '    )'
            )
        })

    # ── Coupling ───────────────────────────────────────────────────────────
    if coup > 15:
        fixes.append({
            'severity': 'warning',
            'metric':   'Coupling',
            'value':    coup,
            'problem':  f'{coup} import statements — highly coupled to external modules, '
                        'making isolated testing and maintenance harder.',
            'fix':      (
                'Group imports by type (stdlib, third-party, local). '
                'Remove unused imports. '
                'If the file needs this many dependencies, consider splitting it: '
                'a file that needs 15+ imports is likely doing too many things. '
                'Use dependency injection to pass dependencies in rather than '
                'importing them at the top of every module.'
            ),
            'example': None
        })

    # ── If everything is fine ──────────────────────────────────────────────
    if not fixes:
        fixes.append({
            'severity': 'info',
            'metric':   'Overall',
            'value':    round(risk_score, 1),
            'problem':  'No specific metric exceeded warning thresholds.',
            'fix':      (
                'This file looks healthy based on static metrics. '
                'Continue to add tests as the file grows, keep functions short, '
                'and resolve any TODO/FIXME comments promptly.'
            ),
            'example': None
        })

    # Sort: critical first, then warning, then info
    order = {'critical': 0, 'warning': 1, 'info': 2}
    fixes.sort(key=lambda x: order[x['severity']])
    return fixes


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
        models = get_models()

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
            scaled = models['scaler'].transform([vec])

            predictions = {
                'rf' : get_prediction(models['rf'],  scaled),
                'svm': get_prediction(models['svm'], scaled),
                'ada': get_prediction(models['ada'], scaled),
                'gb' : get_prediction(models['gb'],  scaled),
                'xgb': get_prediction(models['xgb'], scaled),
            }

            risk         = calculate_risk(predictions)
            is_defective = majority_vote(predictions)
            votes_def    = sum(
                1 for pred, _ in predictions.values() if pred == 1)

            display_path = info['filepath'].replace(
                tmp_dir, '').lstrip('/\\')

            results.append({
                'filename'       : info['filename'],
                'filepath'       : display_path,
                'language'       : lang,
                'is_defective'   : is_defective,
                'votes_defective': votes_def,
                'risk_score'     : risk,
                'agree'          : votes_def == 5 or votes_def == 0,
                'fixes'          : generate_fixes(info, risk),
                'models': {
                    'Random Forest'    : {'pred': predictions['rf'][0],  'conf': predictions['rf'][1]},
                    'SVM'              : {'pred': predictions['svm'][0], 'conf': predictions['svm'][1]},
                    'AdaBoost'         : {'pred': predictions['ada'][0], 'conf': predictions['ada'][1]},
                    'Gradient Boosting': {'pred': predictions['gb'][0],  'conf': predictions['gb'][1]},
                    'XGBoost'          : {'pred': predictions['xgb'][0], 'conf': predictions['xgb'][1]},
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
        import traceback
        print("ERROR:", traceback.format_exc())
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