from setuptools import setup, find_packages

setup(
    name='coderot',
    version='1.0.0',
    description='Code Decay & Software Rot Predictor — ML-powered static analysis',
    author='Your Name',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy',
        'joblib',
        'networkx',
        'radon',
        'imbalanced-learn',
    ],
    entry_points={
        'console_scripts': [
            'coderot=coderot.cli:main',
        ],
    },
    python_requires='>=3.8',
)
