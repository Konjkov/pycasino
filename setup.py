from setuptools import setup

setup(
    name='casino',
    version='0.2.0',
    description='Quantum Monte Carlo Python package',
    data_files=[('', ['README.md'])],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Konjkov/pycasino',
    author='Vladimir Konkov',
    author_email='Konjkov.VV@gmail.com',
    license='MIT',
    packages=['casino', 'casino.readers'],
    # https://scientific-python.org/specs/spec-0000/
    install_requires=[
        'numba>=0.59.0',
        'mpi4py>=3.1.0',
        'pyblock>=0.6.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'sympy>=1.12.0',
        'icc-rt',  # Intel(R) Compiler Runtime
        'tbb',  # threading layer backed by Intel TBB
        'pyyaml>=6.0.0',  # for gjastrow reader
        'matplotlib>=3.6.0',
        'pandas>=2.0.2'   # for pyblock
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ],
    platforms=['Linux', 'Unix'],
    python_requires='>=3.10',
    # https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html
    entry_points = {
        'console_scripts': ['pycasino=casino.pycasino:main'],
    }
)
