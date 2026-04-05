# coderot — Code Decay Detector

Predict software rot in any codebase using ML models trained on 77,000+ real-world samples.

## Installation
```bash
pip install -e .
```

## Usage
```bash
# Scan current folder
coderot scan .

# Scan any project
coderot scan C:\Users\you\myproject

# Save report to specific location
coderot scan . --output my_report.html

# Run without opening browser
coderot scan . --no-browser
```

## Supported Languages
Python, JavaScript, TypeScript, Java, C++, C, C#, Ruby, Go, PHP, Rust, Kotlin, Swift, and more.