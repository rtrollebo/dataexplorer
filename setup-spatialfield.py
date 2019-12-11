from setuptools import setup, find_packages
import pyspatialfield
setup(
    name="pyspatialfield",
    version=pyspatialfield.__version__,
    packages=find_packages(exclude=("test",)),
    install_requires=[
        'numpy>=1.16.1',
        'scipy>=1.2.1',
        'matplotlib>=3.0.3',
        'opencv-python>=4.0.1.24',
        'requests>=2.21.0',
        'ipython'],
    package_data={
        '': ['*.txt', '*.rst']
    },
    author="Robert Trollebø",
    author_email="rtrollebo@gmail.com",
    description="Calculate spatial moments and manage spatial features",
    keywords="spatial moments",
    url="http://www.github.com/rtrollebo/pyspatialfield",
    project_urls={
        "Project": "http://www.github.com/rtrollebo/pyspatialfield"
    }
)