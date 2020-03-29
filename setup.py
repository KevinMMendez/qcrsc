from setuptools import setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


setup(
    name="qcrsc",
    version="0.1.0",
    description="quality control-robust spline correction",
    long_description=readme(),
    long_description_content_type='text/markdown',
    license="MIT License",
    url="https://github.com/kevinmmendez/qcrsc",
    packages=["smooth"],
    python_requires='>=3.5',
    install_requires=[
        "bokeh",
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "tqdm",
        "openpyxl"],
    author='Kevin Mendez, David Broadhurst',
    author_email='k.mendez@ecu.edu.au, d.broadhurst@ecu.edu.au',
)
