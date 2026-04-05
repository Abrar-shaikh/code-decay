import os
import sys
import argparse
import joblib
import time

def get_models_path():
    """Find the saved_models folder wherever the package is installed."""
    # First check same directory as this file
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, 'saved_models')
    if os.path.exists(candidate):
        return candidate
    # Then check current working directory
    cwd_candidate = os.path.join(os.getcwd(), 'saved_models')
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return None

def load_models():
    models_path = get_models_path()
    if not models_path:
        print("ERROR: Could not find saved_models folder.")
        print("Make sure saved_models/ is in the same folder as coderot/")
        sys.exit(1)
    rf     = joblib.load(os.path.join(models_path, 'random_forest_model.pkl'))
    svm    = joblib.load(os.path.join(models_path, 'svm_model.pkl'))
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
    return rf, svm, scaler

FEATURE_ORDER = [
    'lines_of_code','cyclomatic_complexity','num_functions','num_classes',
    'comment_density','code_churn','developer_experience_years','num_developers',
    'commit_frequency','bug_fix_commits','past_defects','test_coverage',
    'duplication_percentage','avg_function_length','depth_of_inheritance',
    'response_for_class','coupling_between_objects','lack_of_cohesion',
    'build_failures','static_analysis_warnings','security_vulnerabilities',
    'performance_issues'
]

def run_scan(target_path, output=None, no_browser=False):
    from .analyzer import analyze_codebase
    from .report   import generate_report

    print(f"\n  Code Decay Detector")
    print(f"  Scanning: {os.path.abspath(target_path)}\n")

    if not os.path.exists(target_path):
        print(f"ERROR: Path does not exist: {target_path}")
        sys.exit(1)

    print("  Loading models...", end=' ', flush=True)
    rf, svm, scaler = load_models()
    print("done")

    print("  Scanning files...", end=' ', flush=True)
    start      = time.time()
    files_data = analyze_codebase(target_path)
    print(f"found {len(files_data)} files ({round(time.time()-start,1)}s)")

    if not files_data:
        print("\n  No supported source files found.")
        sys.exit(0)

    print("  Running predictions...", end=' ', flush=True)
    results = []
    for info in files_data:
        vec    = [info.get(f, 0) for f in FEATURE_ORDER]
        scaled = scaler.transform([vec])

        rf_pred  = int(rf.predict(scaled)[0])
        rf_conf  = float(rf.predict_proba(scaled)[0][rf_pred] * 100)
        svm_pred = int(svm.predict(scaled)[0])
        svm_conf = float(svm.predict_proba(scaled)[0][svm_pred] * 100)
        risk     = round((rf_conf  if rf_pred  == 1 else 100 - rf_conf)  * 0.6 +
                         (svm_conf if svm_pred == 1 else 100 - svm_conf) * 0.4, 1)

        display_path = os.path.relpath(info['filepath'], target_path)
        results.append({
            'filename'  : info['filename'],
            'filepath'  : display_path,
            'language'  : info.get('language','Unknown'),
            'rf_pred'   : rf_pred,   'rf_conf'  : round(rf_conf,1),
            'svm_pred'  : svm_pred,  'svm_conf' : round(svm_conf,1),
            'agree'     : rf_pred == svm_pred,
            'risk_score': risk,
            'metrics'   : {
                'Lines of code'        : info.get('lines_of_code',0),
                'Cyclomatic complexity': info.get('cyclomatic_complexity',0),
                'Functions'            : info.get('num_functions',0),
                'Classes'              : info.get('num_classes',0),
                'Static warnings'      : info.get('static_analysis_warnings',0),
                'Security issues'      : info.get('security_vulnerabilities',0),
                'Duplication %'        : info.get('duplication_percentage',0),
                'Test coverage est.'   : info.get('test_coverage',0),
            }
        })

    results.sort(key=lambda x: x['risk_score'], reverse=True)
    defective = sum(1 for r in results if r['rf_pred'] == 1)
    avg_risk  = round(sum(r['risk_score'] for r in results) / len(results), 1)
    summary   = {
        'total_files': len(results),
        'defective'  : defective,
        'clean'      : len(results) - defective,
        'avg_risk'   : avg_risk
    }
    print("done\n")

    # Print terminal summary
    print("  " + "─" * 50)
    print(f"  FILES SCANNED  : {summary['total_files']}")
    print(f"  AT RISK        : {summary['defective']}")
    print(f"  HEALTHY        : {summary['clean']}")
    print(f"  AVG RISK SCORE : {summary['avg_risk']}%")
    print("  " + "─" * 50)

    print("\n  Top 5 riskiest files:")
    for r in results[:5]:
        icon = "⚠" if r['rf_pred'] == 1 else "✓"
        print(f"  {icon} [{r['risk_score']:>5.1f}%] {r['filepath']} ({r['language']})")

    # Generate HTML report
    report_path = output or os.path.join(os.getcwd(), 'coderot_report.html')
    generate_report(results, summary, report_path)
    print(f"\n  Full report saved to: {report_path}")

    if not no_browser:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(report_path)}')
        print("  Opening in browser...")

    print()

def main():
    parser = argparse.ArgumentParser(
        prog='coderot',
        description='Code Decay Detector — predict software rot in any codebase'
    )
    subparsers = parser.add_subparsers(dest='command')

    # coderot scan <path>
    scan = subparsers.add_parser('scan', help='Scan a project folder')
    scan.add_argument('path', nargs='?', default='.',
                      help='Path to scan (default: current directory)')
    scan.add_argument('--output', '-o', help='Output HTML report path')
    scan.add_argument('--no-browser', action='store_true',
                      help='Do not open browser after scan')

    # coderot version
    subparsers.add_parser('version', help='Show version')

    args = parser.parse_args()

    if args.command == 'scan':
        run_scan(args.path, args.output, args.no_browser)
    elif args.command == 'version':
        from . import __version__
        print(f"coderot v{__version__}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()