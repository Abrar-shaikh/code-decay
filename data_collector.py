import csv
import os
import threading
from datetime import datetime

# Path where the growing dataset will be saved
DATASET_PATH = 'collected_data/community_dataset.csv'

# Column order — must match FEATURE_ORDER in app.py
FEATURE_COLUMNS = [
    'lines_of_code', 'cyclomatic_complexity', 'num_functions', 'num_classes',
    'comment_density', 'code_churn', 'developer_experience_years',
    'num_developers', 'commit_frequency', 'bug_fix_commits', 'past_defects',
    'test_coverage', 'duplication_percentage', 'avg_function_length',
    'depth_of_inheritance', 'response_for_class', 'coupling_between_objects',
    'lack_of_cohesion', 'build_failures', 'static_analysis_warnings',
    'security_vulnerabilities', 'performance_issues'
]

# Full header including metadata and predictions
HEADER = (
    ['timestamp', 'language', 'filename_hash'] +
    FEATURE_COLUMNS +
    ['rf_prediction', 'svm_prediction', 'risk_score']
)

# Thread lock so multiple users don't corrupt the file simultaneously
_lock = threading.Lock()


def _ensure_file_exists():
    """Create the dataset folder and CSV file with header if they don't exist."""
    os.makedirs('collected_data', exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def _hash_filename(filename):
    """
    Hash the filename so we store no identifiable info.
    e.g. 'my_secret_project.py' becomes 'a3f9b2...'
    """
    import hashlib
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def save_file_metrics(file_info, rf_pred, svm_pred, risk_score):
    """
    Save one file's metrics to the community dataset.
    Called after every file is analyzed — runs in background thread.
    No actual code or filenames are stored.
    """
    try:
        _ensure_file_exists()

        row = (
            [
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                file_info.get('language', 'Unknown'),
                _hash_filename(file_info.get('filename', 'unknown')),
            ] +
            [file_info.get(col, 0) for col in FEATURE_COLUMNS] +
            [rf_pred, svm_pred, round(risk_score, 2)]
        )

        with _lock:
            with open(DATASET_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    except Exception as e:
        # Never let data collection crash the main app
        print(f"[data_collector] Failed to save row: {e}")


def save_batch(results_list, files_data):
    """
    Save all files from one analysis session in a background thread.
    This way the user gets their results immediately without waiting.
    """
    def _save():
        for result, info in zip(results_list, files_data):
            save_file_metrics(
                info,
                result['rf_pred'],
                result['svm_pred'],
                result['risk_score']
            )

    thread = threading.Thread(target=_save, daemon=True)
    thread.start()


def get_dataset_stats():
    """Return basic stats about how much data has been collected so far."""
    try:
        if not os.path.exists(DATASET_PATH):
            return {'rows': 0, 'size_mb': 0}

        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            rows = sum(1 for _ in f) - 1  # subtract header

        size_mb = round(os.path.getsize(DATASET_PATH) / 1024 / 1024, 2)
        return {'rows': max(rows, 0), 'size_mb': size_mb}

    except Exception:
        return {'rows': 0, 'size_mb': 0}