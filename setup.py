from setuptools import setup
 
setup(
    name="PyGS", 
    version="1.0.0", 
    author="Yu Chen, Qiang Hu, and Jinlei Zheng", 
    author_email="yc0020@uah.edu, qh0001@uah.edu", 
    description="Application of the Grad-Shafranov-based techniques",
    url="https://github.com/PyGSDR/PyGS",
    packages=['PyGS'],
    classifiers=[
        'Development Status :: 5 - Stable',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    'numpy',
    'pandas',
    'scipy',
    'spacepy',
    'matplotlib',
    'ai.cdas',
    ],
    python_requires='>=3',
)
