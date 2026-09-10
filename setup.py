from setuptools import setup, find_packages

setup(
    name="upframe",
    version="1.0.0",
    description="Production-Grade Video Upframing Pipeline for Football Broadcast Video",
    author="Upframe Team",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "upframe = upframe.cli.upframe:main",
        ],
    },
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "pyyaml",
        "tqdm",
        "opencv-python-headless",
    ],
    python_requires=">=3.9",
)
