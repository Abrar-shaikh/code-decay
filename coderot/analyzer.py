import os
import re
import ast
import math
import zipfile
import tempfile
import networkx as nx

LANGUAGE_MAP = {
    '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
    '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
    '.rb': 'Ruby', '.go': 'Go', '.php': 'PHP', '.rs': 'Rust',
    '.kt': 'Kotlin', '.swift': 'Swift', '.r': 'R', '.m': 'MATLAB',
    '.scala': 'Scala', '.sh': 'Shell', '.html': 'HTML',
    '.css': 'CSS', '.sql': 'SQL', '.dart': 'Dart',
    '.lua': 'Lua', '.pl': 'Perl',
}

COMMENT_PATTERNS = {
    'Python': r'#.*', 'JavaScript': r'//.*|/\*[\s\S]*?\*/',
    'TypeScript': r'//.*|/\*[\s\S]*?\*/', 'Java': r'//.*|/\*[\s\S]*?\*/',
    'C++': r'//.*|/\*[\s\S]*?\*/', 'C': r'//.*|/\*[\s\S]*?\*/',
    'C#': r'//.*|/\*[\s\S]*?\*/', 'Ruby': r'#.*',
    'Go': r'//.*|/\*[\s\S]*?\*/', 'PHP': r'//.*|#.*|/\*[\s\S]*?\*/',
    'Rust': r'//.*', 'Kotlin': r'//.*|/\*[\s\S]*?\*/',
    'Swift': r'//.*|/\*[\s\S]*?\*/', 'Shell': r'#.*',
    'SQL': r'--.*|/\*[\s\S]*?\*/',
}

FUNCTION_PATTERNS = {
    'Python': r'^\s*def\s+\w+',
    'JavaScript': r'function\s+\w+|=>\s*{',
    'TypeScript': r'function\s+\w+|=>\s*{',
    'Java': r'(public|private|protected|static).*\w+\s*\(',
    'C++': r'\w+\s+\w+\s*\([^)]*\)\s*\{',
    'C': r'\w+\s+\w+\s*\([^)]*\)\s*\{',
    'C#': r'(public|private|protected|static).*\w+\s*\(',
    'Ruby': r'^\s*def\s+\w+', 'Go': r'^\s*func\s+\w+',
    'PHP': r'function\s+\w+', 'Rust': r'^\s*fn\s+\w+',
    'Kotlin': r'^\s*fun\s+\w+', 'Swift': r'^\s*func\s+\w+',
}

CLASS_PATTERNS = {
    'Python': r'^\s*class\s+\w+', 'JavaScript': r'^\s*class\s+\w+',
    'TypeScript': r'^\s*class\s+\w+',
    'Java': r'^\s*(public|private)?\s*class\s+\w+',
    'C++': r'^\s*class\s+\w+',
    'C#': r'^\s*(public|private)?\s*class\s+\w+',
    'Ruby': r'^\s*class\s+\w+', 'Kotlin': r'^\s*(data\s+)?class\s+\w+',
    'Swift': r'^\s*class\s+\w+', 'PHP': r'^\s*class\s+\w+',
}

RISKY_PATTERNS = [
    'eval(', 'exec(', 'TODO', 'FIXME', 'HACK', 'XXX',
    'System.exit', 'os.system', 'subprocess', 'catch(Exception)',
    'catch (Exception)', 'ignore', 'pass', 'delete ', 'free(', 'unsafe',
]

SECURITY_PATTERNS = [
    'eval(', 'exec(', 'os.system(', 'subprocess.call(',
    'pickle.loads(', 'yaml.load(', '__import__(',
    'Runtime.exec', 'ProcessBuilder', 'innerHTML',
    'document.write', 'dangerouslySetInnerHTML',
]

SKIP_DIRS = {
    '__pycache__', '.git', 'venv', 'env', 'node_modules',
    '.venv', 'dist', 'build', '.idea', '.vscode', 'vendor',
    '.ipynb_checkpoints'
}


def detect_language(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return LANGUAGE_MAP.get(ext, None)


def build_ast_graph(source):
    tree = ast.parse(source)
    G = nx.DiGraph()
    node_id = [0]

    def visit(node, parent_id=None):
        cid = node_id[0]
        node_id[0] += 1
        G.add_node(cid, label=type(node).__name__)
        if parent_id is not None:
            G.add_edge(parent_id, cid)
        for child in ast.iter_child_nodes(node):
            visit(child, cid)

    visit(tree)
    return G


def analyze_file(filepath):
    lang = detect_language(filepath)
    if not lang:
        return None

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
    except:
        return None

    if not source.strip():
        return None

    lines = source.splitlines()
    total_lines = max(len(lines), 1)
    metrics = {}

    # Lines of code
    code_lines = [l for l in lines if l.strip() and
                  not l.strip().startswith(('#', '//', '/*', '*', '--'))]
    metrics['lines_of_code'] = len(code_lines)

    # Comment density
    comment_pat = COMMENT_PATTERNS.get(lang, r'#.*|//.*')
    comments = re.findall(comment_pat, source)
    metrics['comment_density'] = round(len(comments) / total_lines, 4)

    # Functions and classes
    func_pat = FUNCTION_PATTERNS.get(lang, r'def\s+\w+|function\s+\w+')
    metrics['num_functions'] = len(re.findall(func_pat, source, re.MULTILINE))
    cls_pat = CLASS_PATTERNS.get(lang, r'class\s+\w+')
    metrics['num_classes'] = len(re.findall(cls_pat, source, re.MULTILINE))

    # Average function length
    metrics['avg_function_length'] = int(
        total_lines / max(metrics['num_functions'], 1))

    # Cyclomatic complexity
    decision_keywords = len(re.findall(
        r'\b(if|else|elif|for|while|case|switch|catch|except|and|or|&&|\|\|)\b',
        source))
    metrics['cyclomatic_complexity'] = max(1, decision_keywords)

    # Duplication estimate
    stripped = [l.strip() for l in lines if l.strip()]
    metrics['duplication_percentage'] = round(
        1 - len(set(stripped)) / max(len(stripped), 1), 4)

    # Coupling
    imports = len(re.findall(
        r'^\s*(import|from|require|include|using|#include|use)\s+',
        source, re.MULTILINE))
    metrics['coupling_between_objects'] = imports

    # Depth of inheritance
    inheritance = re.findall(
        r'extends|implements|:\s*\w+|<\s*\w+\s*>', source)
    metrics['depth_of_inheritance'] = min(len(inheritance), 10)

    # Lack of cohesion
    metrics['lack_of_cohesion'] = round(
        min(metrics['avg_function_length'], 100) / 100, 4)

    # Response for class
    metrics['response_for_class'] = metrics['num_functions'] + imports

    # Static warnings
    metrics['static_analysis_warnings'] = sum(
        source.count(p) for p in RISKY_PATTERNS)

    # Security vulnerabilities
    metrics['security_vulnerabilities'] = sum(
        source.count(p) for p in SECURITY_PATTERNS)

    # Performance issues
    perf = len(re.findall(
        r'\b(sleep|wait|delay|blocking|synchronized|lock)\b', source))
    metrics['performance_issues'] = perf

    # Test coverage estimate
    test_lines = len(re.findall(
        r'\b(assert|test_|_test|expect|should|describe|it\()\b', source))
    metrics['test_coverage'] = round(
        min(test_lines / total_lines * 5, 1.0), 4)

    # AST graph metrics (Python only)
    if lang == 'Python':
        try:
            G = build_ast_graph(source)
            metrics['ast_num_nodes'] = G.number_of_nodes()
            metrics['ast_max_depth'] = (
                nx.dag_longest_path_length(G)
                if nx.is_directed_acyclic_graph(G) else 0)
            metrics['ast_density'] = round(nx.density(G), 6)
        except:
            metrics['ast_num_nodes'] = total_lines
            metrics['ast_max_depth'] = int(math.log(max(total_lines, 1)))
            metrics['ast_density'] = 0.1
    else:
        metrics['ast_num_nodes'] = total_lines
        metrics['ast_max_depth'] = int(math.log(max(total_lines, 1)))
        metrics['ast_density'] = 0.1

    # Git-based metrics (default 0 for uploaded files)
    metrics['code_churn'] = 0
    metrics['developer_experience_years'] = 0
    metrics['num_developers'] = 1
    metrics['commit_frequency'] = 0
    metrics['bug_fix_commits'] = 0
    metrics['past_defects'] = 0
    metrics['build_failures'] = 0

    metrics['filepath'] = filepath
    metrics['filename'] = os.path.basename(filepath)
    metrics['language'] = lang

    return metrics


def extract_zip(zip_path):
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_dir)
    return tmp_dir


def analyze_codebase(folder_path):
    results = []
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            filepath = os.path.join(root, file)
            metrics = analyze_file(filepath)
            if metrics:
                results.append(metrics)
    return results