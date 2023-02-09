from setuptools import setup

setup(
    name="house-diffusion",
    py_modules=["house_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
