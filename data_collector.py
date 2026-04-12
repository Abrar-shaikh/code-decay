import os
import threading
import hashlib
from datetime import datetime

# ── Try to use Google Sheets, fall back to CSV if not configured ──────────
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

FEATURE_COLUMNS = [
    'lines_of_code', 'cyclomatic_complexity', 'num_functions', 'num_classes',
    'comment_density', 'code_churn', 'developer_experience_years',
    'num_developers', 'commit_frequency', 'bug_fix_commits', 'past_defects',
    'test_coverage', 'duplication_percentage', 'avg_function_length',
    'depth_of_inheritance', 'response_for_class', 'coupling_between_objects',
    'lack_of_cohesion', 'build_failures', 'static_analysis_warnings',
    'security_vulnerabilities', 'performance_issues'
]

HEADER = (
    ['timestamp', 'language', 'filename_hash'] +
    FEATURE_COLUMNS +
    ['rf_prediction', 'svm_prediction', 'risk_score']
)

_lock       = threading.Lock()
_sheet      = None
_sheet_init = False


def _get_sheet():
    """Connect to Google Sheet — only runs once."""
    global _sheet, _sheet_init
    if _sheet_init:
        return _sheet

    _sheet_init = True

    if not SHEETS_AVAILABLE:
        return None

    try:
        # Get credentials from environment variable
        import json
        # Try secret file first (Render), then environment variable (local)
        secret_path = '/etc/secrets/google_credentials.json'
        if os.path.exists(secret_path):
            with open(secret_path, 'r') as f:
                creds_dict = json.load(f)
        elif os.environ.get('GOOGLE_CREDENTIALS'):
            creds_dict = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
        else:
            print("[data_collector] No Google credentials found")
            return None
        scopes     = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        sheet_name = os.environ.get('SHEET_NAME', 'coderot_community_dataset')
        spreadsheet = client.open(sheet_name)
        worksheet   = spreadsheet.sheet1

        # Add header row if sheet is empty
        if worksheet.row_count == 0 or not worksheet.row_values(1):
            worksheet.append_row(HEADER)

        _sheet = worksheet
        print("[data_collector] Connected to Google Sheets successfully!")
        return _sheet

    except Exception as e:
        print(f"[data_collector] Google Sheets connection failed: {e}")
        return None


def _hash_filename(filename):
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def _build_row(file_info, rf_pred, svm_pred, risk_score):
    return (
        [
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            file_info.get('language', 'Unknown'),
            _hash_filename(file_info.get('filename', 'unknown')),
        ] +
        [file_info.get(col, 0) for col in FEATURE_COLUMNS] +
        [rf_pred, svm_pred, round(risk_score, 2)]
    )


def _save_to_sheet(rows):
    """Append multiple rows to Google Sheet at once."""
    try:
        sheet = _get_sheet()
        if sheet is None:
            return
        with _lock:
            sheet.append_rows(rows, value_input_option='RAW')
    except Exception as e:
        print(f"[data_collector] Failed to save to sheet: {e}")


def _save_to_csv_fallback(rows):
    """Local CSV fallback when Google Sheets isn't configured."""
    try:
        import csv
        os.makedirs('collected_data', exist_ok=True)
        path = 'collected_data/community_dataset.csv'
        file_exists = os.path.exists(path)
        with _lock:
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(HEADER)
                writer.writerows(rows)
    except Exception as e:
        print(f"[data_collector] CSV fallback failed: {e}")


def save_batch(results_list, files_data):
    """Save all metrics from one session — runs in background thread."""
    def _save():
        rows = []
        for result, info in zip(results_list, files_data):
            row = _build_row(
                info,
                result['rf_pred'],
                result['svm_pred'],
                result['risk_score']
            )
            rows.append(row)

        if not rows:
            return

        # Try Google Sheets first, fall back to CSV
        sheet = _get_sheet()
        if sheet:
            _save_to_sheet(rows)
        else:
            _save_to_csv_fallback(rows)

    thread = threading.Thread(target=_save, daemon=True)
    thread.start()


def get_dataset_stats():
    """Return how many rows have been collected."""
    try:
        sheet = _get_sheet()
        if sheet:
            # Subtract 1 for header row
            count = max(sheet.row_count - 1, 0)
            return {'rows': count, 'source': 'sheets'}
    except:
        pass

    # CSV fallback count
    try:
        path = 'collected_data/community_dataset.csv'
        if os.path.exists(path):
            with open(path, 'r') as f:
                count = sum(1 for _ in f) - 1
            return {'rows': max(count, 0), 'source': 'csv'}
    except:
        pass

    return {'rows': 0, 'source': 'none'}