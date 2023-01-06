from skbuild import setup

setup(
    name="hg64",
    version="0.0.1",
    description="minimal bindings for hg64 histogram sketching library",
    license="MPL-2.0",
    packages=["hg64"],
    package_dir={"": "src"},
    cmake_install_dir="src/hg64",
    python_requires=">=3.7",
)
