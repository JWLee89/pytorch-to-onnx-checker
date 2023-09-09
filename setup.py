from setuptools import find_packages, setup

__VERSION__ = "0.0.1"
__REQUIREMENTS__ = ["base", "check", "docs"]

requirements = dict()
for req in __REQUIREMENTS__:
    req_path = f"requirements/{req}.txt"
    with open(req_path) as f:
        requirements[req] = f.read().splitlines()

setup(
    name="pytorch-to-onnx-checker",
    version=__VERSION__,
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements["base"],
    python_requires=">=3.8",
    extras_require={
        "all": sum(requirements.values(), []),
        **requirements,
    },
)
