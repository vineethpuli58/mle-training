import setuptools

setuptools.setup(
    include_package_data=True,
    name="mle_training",
    version="0.3.0",
    author="vineeth puli",
    package_dir={
        "": "src",
    },
    packages=setuptools.find_packages(where="src"),
    description="mle package ",
    long_description=open("README.md").read(),
    install_requires=[],
)
