from setuptools import setup

setup(
    name='casino',
    version='0.1.0',
    description='Quantum Monte Carlo Python package',
    url='https://github.com/Konjkov/pycasino',
    author='Vladimir Konkov',
    author_email='Konjkov.VV@gmail.com',
    license='MIT',
    packages=['pycasino', 'pycasino.readers', 'sympy_utils'],
    install_requires=[
        'numba==0.57.1',
        'numba-mpi>=0.34',
        'pyblock>=0.6.0',
        'numpy==1.24.3',
        'scipy==1.10.1',
        'sympy==1.12.0',
        'pyyaml==6.0.0',  # for gjastrow loader
        'matplotlib==3.5.2',
        'pandas==2.0.2'   # for pyblock
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
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
        'console_scripts': ['pycasino=pycasino.casino:main'],
    }
)
