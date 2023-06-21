from setuptools import setup

setup(
    name='pycasino',
    version='0.1.0',
    description='Quantum Monte Carlo Python package',
    url='https://github.com/Konjkov/pycasino',
    author='Vladimir Konkov',
    author_email='Konjkov.VV@gmail.com',
    license='BSD 2-clause',
    packages=['pycasino', 'pycasino.readers', 'sympy_utils'],
    install_requires=[
        'numba>=0.57.0',
        'numba-mpi==0.30',
        'pyblock==0.6.0',
        'scipy==1.10.1',
        'sympy==1.12.0',
        'pyyaml==6.0.0',  # for gjastrow loader
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
