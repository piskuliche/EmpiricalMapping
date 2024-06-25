from setuptools import setup, find_packages

setup(
    name='empirical-mapper',
    version='0.1',
    packages=find_packages(),
    url='http://github.com/piskuliche/empirical-mapper',
    license='MIT',
    author='Zeke Piskulich',
    author_email='piskulichz@gmail.com',
    description='This is a package to generate and calculate empirical maps for molecular dynamics simulations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'matplotlib',
        'MDAnalysis'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
