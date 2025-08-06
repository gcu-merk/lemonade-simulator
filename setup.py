from setuptools import setup, find_packages

setup(
    name="lemonade-simulator",
    version="1.0.0",
    description="A Lemonade Stand Simulator with Bayesian MEU AI",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0"
    ],
    python_requires=">=3.8",
)
