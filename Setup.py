from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="ml-trading-backtester",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Профессиональная система для бэктестинга и ML на финансовых рынках",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-trading-backtester",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ml-backtest=run_pipeline:main",
            "ml-train=main:main",
            "ml-bot=dynamic_level_bot:main",
        ],
    },
)