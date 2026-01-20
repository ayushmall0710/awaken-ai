"""
Setup configuration for Awaken AI EEG Data Pipeline.

This allows the package to be installed in development mode:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
readme_path = Path(__file__).parent / 'README.md'
with open(readme_path) as f:
    long_description = f.read()

setup(
    name='awaken-ai',
    version='0.1.0',
    description='EEG Prognostic Data Pipeline for Awaken AI Capstone Project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Awaken AI Team',
    python_requires='>=3.8',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
        ]
    },
    entry_points={
        'console_scripts': [
            # Entry point for timestamp alignment demo (optional)
            # 'awaken-align=examples.timestamp_alignment_demo:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
