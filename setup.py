from setuptools import setup

setup(
    name='pycasino',
    version='0.1.0',
    description='Quantum Monte Carlo Python package',
    url='https://github.com/Konjkov/pycasino',
    author='Vladimir Konkov',
    author_email='Konjkov.VV@gmail.com',
    license='BSD 2-clause',
    packages=['pycasino'],
    install_requires=[
        'pyblock==0.6.0',
        'numba==0.56.4',
        'numba-mpi==0.30',
        'pandas==1.4.3',  # for pyblock
        'scipy==1.10.1',
        'sympy==1.11.1',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
