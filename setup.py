from setuptools import setup, find_packages

setup(
    name="ivi_thickness",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'pyyaml>=5.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
    ],
    python_requires='>=3.8',
    author="",
    author_email="",
    description="IVI Time-Thickness package for analyzing time dilation effects",
    url="",
)
