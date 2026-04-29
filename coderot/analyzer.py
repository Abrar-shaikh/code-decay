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

# ── FIX: Greatly expanded class patterns per language.
# Old patterns missed: abstract/final/sealed classes, interfaces,
# enums, structs, traits, protocols — all of which ARE class-like structures.
CLASS_PATTERNS = {
    'Python':     r'^\s*class\s+\w+',
    'JavaScript': r'^\s*class\s+\w+',
    'TypeScript': r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+\w+'
                  r'|^\s*(?:export\s+)?interface\s+\w+',
    # Java: catches public/private/abstract/final + class/interface/enum
    'Java':       r'^\s*(?:(?:public|private|protected|abstract|final|static)\s+)*'
                  r'(?:class|interface|enum)\s+\w+',
    # C++: struct is a class with public-by-default members
    'C++':        r'^\s*(?:class|struct)\s+\w+',
    # C#: abstract/sealed/static/partial + class/interface/struct/enum
    'C#':         r'^\s*(?:(?:public|private|protected|internal|abstract|sealed|static|partial)\s+)*'
                  r'(?:class|interface|struct|enum)\s+\w+',
    'Ruby':       r'^\s*class\s+\w+',
    # Kotlin: data/sealed/abstract/open class + interface
    'Kotlin':     r'^\s*(?:(?:data|sealed|abstract|open|inner)\s+)?class\s+\w+'
                  r'|^\s*(?:fun\s+)?interface\s+\w+',
    # Swift: class/struct/enum/protocol/actor are all type definitions
    'Swift':      r'^\s*(?:class|struct|enum|protocol|actor)\s+\w+',
    'PHP':        r'^\s*(?:(?:abstract|final)\s+)?class\s+\w+'
                  r'|^\s*interface\s+\w+',
    # Go uses structs — there are no classes
    'Go':         r'^\s*type\s+\w+\s+struct',
    # Rust: struct/enum/trait are all class-equivalents
    'Rust':       r'^\s*(?:pub\s+)?(?:struct|enum|trait)\s+\w+',
    # Scala: class/case class/object/trait
    'Scala':      r'^\s*(?:case\s+)?class\s+\w+'
                  r'|^\s*object\s+\w+|^\s*trait\s+\w+',
}

RISKY_PATTERNS = [
    'eval(', 'exec(', 'TODO', 'FIXME', 'HACK', 'XXX',
    'System.exit', 'os.system', 'subprocess', 'catch(Exception)',
    'catch (Exception)', 'ignore', 'pass', 'delete ', 'free(', 'unsafe',
]

# ── FIX: Expanded security patterns — 7 new real-world dangerous patterns added.
# Root cause of showing 0: the original 12 patterns are rare in typical
# Python Flask projects. These 7 additions cover the most common actual
# vulnerabilities in web applications.
SECURITY_PATTERNS = [
    # --- Original patterns ---
    'eval(',                    # Arbitrary code execution
    'exec(',                    # Arbitrary code execution
    'os.system(',               # Shell command injection
    'subprocess.call(',         # Process execution
    'pickle.loads(',            # Unsafe deserialisation — can run arbitrary code
    'yaml.load(',               # Use yaml.safe_load() instead
    '__import__(',              # Dynamic module loading — can load anything
    'Runtime.exec',             # Java: shell execution
    'ProcessBuilder',           # Java: process spawning
    'innerHTML',                # XSS via DOM — any assignment is a risk
    'document.write',           # XSS via legacy DOM write
    'dangerouslySetInnerHTML',  # React: intentional XSS bypass
    # --- NEW patterns ---
    'shell=True',               # subprocess(shell=True) → command injection
    'verify=False',             # requests.get(verify=False) → SSL stripped
    'hashlib.md5(',             # MD5 is cryptographically broken (collisions)
    'hashlib.sha1(',            # SHA-1 is broken — use SHA-256 minimum
    'subprocess.Popen(',        # Direct process spawning without validation
    'os.popen(',                # Deprecated shell execution
    'tempfile.mktemp(',         # Race condition between mktemp and open
]

SKIP_DIRS = {
    '__pycache__', '.git', 'venv', 'env', 'node_modules',
    '.venv', 'dist', 'build', '.idea', '.vscode', 'vendor',
    '.ipynb_checkpoints'
}

# ── FIX: Language-specific test patterns.
#
# ROOT CAUSE of test_coverage always being 0:
#   The original regex was:
#       r'\b(assert|test_|_test|expect|should|describe|it\()\b'
#
#   \b is a word-boundary — it fires between a word-char (\w = [a-zA-Z0-9_])
#   and a non-word char. Since underscore (_) IS a word character:
#
#   • \btest_\b  needs a NON-word char right after '_'.
#     In "def test_login():", the char after '_' is 'l' → word char → NO MATCH.
#     Every Python test function was silently skipped.
#
#   • \b_test\b  needs a NON-word char right before '_'.
#     In "module_test", 'e' before '_' → word char → NO MATCH.
#
#   FIX: Replace with language-specific patterns that explicitly match
#   real test idioms for each language, with no \b around underscores.
TEST_PATTERNS = {
    # Python: def test_*, class Test*, assert statement, unittest, @pytest
    'Python':     r'(?:def\s+test_\w+'
                  r'|class\s+Test\w+'
                  r'|\bassert\s+'
                  r'|\bunittest\b'
                  r'|@pytest)',

    # JavaScript / TypeScript: describe(), it(), test(), expect(), jest, mocha
    'JavaScript': r'(?:describe\s*\('
                  r'|(?<![a-zA-Z])it\s*\('
                  r'|(?<![a-zA-Z])test\s*\('
                  r'|expect\s*\('
                  r'|\.toBe\s*\('
                  r'|beforeEach\s*\('
                  r'|afterEach\s*\('
                  r'|jest\.'
                  r'|mocha)',
    'TypeScript': r'(?:describe\s*\('
                  r'|(?<![a-zA-Z])it\s*\('
                  r'|(?<![a-zA-Z])test\s*\('
                  r'|expect\s*\('
                  r'|\.toBe\s*\('
                  r'|beforeEach\s*\('
                  r'|afterEach\s*\()',

    # Java: JUnit 4 + 5 annotations, assertion methods
    'Java':    r'(?:@Test\b'
               r'|assertEquals\s*\('
               r'|assertTrue\s*\('
               r'|assertFalse\s*\('
               r'|assertThat\s*\('
               r'|@Before\b|@After\b'
               r'|@BeforeEach\b|@AfterEach\b)',

    # Kotlin: JUnit + Kotest
    'Kotlin':  r'(?:@Test\b'
               r'|assertEquals\s*\('
               r'|assertTrue\s*\('
               r'|shouldBe\b'
               r'|kotest)',

    # C#: xUnit, NUnit, MSTest
    'C#':   r'(?:\[Test\]|\[TestMethod\]|\[Fact\]|\[Theory\]'
            r'|Assert\.'
            r'|NUnit|xUnit)',

    # Ruby: RSpec, Minitest
    'Ruby': r'(?:describe\s+'
            r'|(?:^|\s)it\s+["\']'
            r'|expect\s*\('
            r'|RSpec'
            r'|\bshould\b'
            r'|assert_equal)',

    # Go: test functions must start with Test, use *testing.T
    'Go':   r'(?:func\s+Test\w+\s*\('
            r'|\bt\.Error\b|\bt\.Fatal\b'
            r'|\bt\.Log\b'
            r'|testing\.T)',

    # PHP: PHPUnit
    'PHP':  r'(?:assertEquals\s*\('
            r'|assertTrue\s*\('
            r'|@test\b'
            r'|PHPUnit'
            r'|assertSame\s*\()',

    # Rust: #[test], assert_eq!, assert!, test module
    'Rust': r'(?:#\[test\]'
            r'|assert_eq!\s*\('
            r'|assert!\s*\('
            r'|#\[cfg\(test\)\])',

    # Swift: XCTest framework
    'Swift': r'(?:XCTest'
             r'|func\s+test\w+'
             r'|XCTAssert'
             r'|XCTEqual'
             r'|setUp\(\)|tearDown\(\))',

    # Scala: ScalaTest / specs2
    'Scala': r'(?:extends\s+(?:FlatSpec|WordSpec|FunSuite|Spec)'
             r'|\bshouldBe\b'
             r'|\bassert\s*\()',

    # Fallback for Shell, SQL, HTML, CSS etc.
    'default': r'(?:\bassert\b|def\s+test_\w+|class\s+Test\w+)',
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

    # Security vulnerabilities — uses expanded SECURITY_PATTERNS
    metrics['security_vulnerabilities'] = sum(
        source.count(p) for p in SECURITY_PATTERNS)

    # Performance issues
    perf = len(re.findall(
        r'\b(sleep|wait|delay|blocking|synchronized|lock)\b', source))
    metrics['performance_issues'] = perf

    # ── FIX: Language-specific test coverage using correct patterns.
    # Falls back to 'default' for unmapped languages (HTML, CSS, SQL, etc.)
    test_pat = TEST_PATTERNS.get(lang, TEST_PATTERNS['default'])
    test_hits = len(re.findall(test_pat, source, re.MULTILINE))
    metrics['test_coverage'] = round(min(test_hits / total_lines * 5, 1.0), 4)

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