# setup.py
from setuptools import setup, find_packages

setup(
    name="rl_bipedal_walking",
    version="0.1.0",
    description="Reinforcement Learning for Bipedal Robot Walking",
    author="darsh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "torch>=1.11.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "rclpy>=3.3.0",
        "tf-transformations",
        "scipy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },
)

