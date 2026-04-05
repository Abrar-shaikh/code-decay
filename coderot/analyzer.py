import ast
import os
import math
import networkx as nx
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze

def analyze_file(filepath):
    """Analyze a single Python file and extract all metrics automatically."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
    except Exception as e:
        return None

    metrics = {}

    # --- Basic raw metrics ---
    try:
        raw = analyze(source)
        metrics['lines_of_code']       = raw.lloc
        metrics['comment_density']     = round(raw.comments / max(raw.lloc, 1), 4)
        metrics['avg_function_length'] = raw.lloc
    except:
        metrics['lines_of_code']       = 0
        metrics['comment_density']     = 0
        metrics['avg_function_length'] = 0

    # --- AST based metrics ---
    try:
        tree = ast.parse(source)

        functions   = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes     = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        all_nodes   = list(ast.walk(tree))

        metrics['num_functions'] = len(functions)
        metrics['num_classes']   = len(classes)

        # Average function length
        func_lengths = []
        for func in functions:
            start = func.lineno
            end   = max((getattr(n, 'lineno', start) for n in ast.walk(func)), default=start)
            func_lengths.append(end - start + 1)
        metrics['avg_function_length'] = int(sum(func_lengths) / max(len(func_lengths), 1))

        # Depth of inheritance
        max_depth = 0
        for cls in classes:
            max_depth = max(max_depth, len(cls.bases))
        metrics['depth_of_inheritance'] = max_depth

        # Coupling — number of unique imports
        imports = [n for n in all_nodes if isinstance(n, (ast.Import, ast.ImportFrom))]
        metrics['coupling_between_objects'] = len(imports)

        # Duplication estimate — repeated lines ratio
        lines = [l.strip() for l in source.splitlines() if l.strip()]
        metrics['duplication_percentage'] = round(
            1 - len(set(lines)) / max(len(lines), 1), 4)

        # Lack of cohesion — ratio of functions with no docstring
        no_doc = sum(1 for f in functions if not ast.get_docstring(f))
        metrics['lack_of_cohesion'] = round(no_doc / max(len(functions), 1), 4)

        # Response for class — total methods across all classes
        metrics['response_for_class'] = sum(
            len([n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)])
            for cls in classes)

    except:
        metrics.update({
            'num_functions': 0, 'num_classes': 0,
            'avg_function_length': 0, 'depth_of_inheritance': 0,
            'coupling_between_objects': 0, 'duplication_percentage': 0,
            'lack_of_cohesion': 0, 'response_for_class': 0
        })

    # --- Cyclomatic complexity via radon ---
    try:
        blocks = cc_visit(source)
        complexities = [b.complexity for b in blocks]
        metrics['cyclomatic_complexity'] = int(sum(complexities) / max(len(complexities), 1))
    except:
        metrics['cyclomatic_complexity'] = 1

    # --- AST graph metrics ---
    try:
        G = build_ast_graph(source)
        metrics['ast_num_nodes'] = G.number_of_nodes()
        metrics['ast_max_depth'] = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else 0
        metrics['ast_density']   = round(nx.density(G), 6)
    except:
        metrics['ast_num_nodes'] = 0
        metrics['ast_max_depth'] = 0
        metrics['ast_density']   = 0

    # --- Static analysis warnings (count risky patterns) ---
    warnings = 0
    risky_patterns = ['eval(', 'exec(', 'os.system(', '__import__(',
                      'except:', 'pass', 'global ', 'TODO', 'FIXME', 'HACK']
    for pattern in risky_patterns:
        warnings += source.count(pattern)
    metrics['static_analysis_warnings'] = warnings

    # --- Security vulnerabilities (basic patterns) ---
    sec_patterns = ['eval(', 'exec(', 'os.system(', 'subprocess.call(',
                    'pickle.loads(', 'yaml.load(', '__import__(']
    metrics['security_vulnerabilities'] = sum(source.count(p) for p in sec_patterns)

    # --- Performance issues (basic patterns) ---
    perf_patterns = [' + ' , 'time.sleep(', 'while True:', '.append(']
    metrics['performance_issues'] = sum(source.count(p) for p in perf_patterns)

    # --- Defaults for git-based metrics (set to 0 if no git) ---
    metrics.setdefault('code_churn', 0)
    metrics.setdefault('developer_experience_years', 0)
    metrics.setdefault('num_developers', 1)
    metrics.setdefault('commit_frequency', 0)
    metrics.setdefault('bug_fix_commits', 0)
    metrics.setdefault('past_defects', 0)
    metrics.setdefault('test_coverage', estimate_test_coverage(source))
    metrics.setdefault('build_failures', 0)

    return metrics


def build_ast_graph(source):
    """Convert Python source into an AST graph."""
    tree = ast.parse(source)
    G = nx.DiGraph()
    node_id = [0]

    def visit(node, parent_id=None):
        current_id = node_id[0]
        node_id[0] += 1
        G.add_node(current_id, label=type(node).__name__)
        if parent_id is not None:
            G.add_edge(parent_id, current_id)
        for child in ast.iter_child_nodes(node):
            visit(child, current_id)

    visit(tree)
    return G


def estimate_test_coverage(source):
    """Estimate test coverage by checking ratio of assert/test patterns."""
    lines = source.splitlines()
    test_lines = sum(1 for l in lines if 'assert ' in l or 'def test_' in l)
    return round(min(test_lines / max(len(lines), 1) * 5, 1.0), 4)


def analyze_codebase(folder_path):
    """Scan an entire folder and analyze every Python file."""
    results = []
    for root, dirs, files in os.walk(folder_path):
        # Skip common non-source folders
        dirs[:] = [d for d in dirs if d not in
                   ['__pycache__', '.git', 'venv', 'env', 'node_modules', '.venv']]
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                metrics  = analyze_file(filepath)
                if metrics:
                    metrics['filepath'] = filepath
                    metrics['filename'] = file
                    results.append(metrics)
    return results