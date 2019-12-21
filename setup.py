from setuptools import setup, find_packages

setup(
    name='bop_toolkit',
    version='0.0.1',
    packages=find_packages(exclude=('docs')),
    author='Tomas Hodan, Martin Sundermeyer',
    author_email='hodantom@cmp.felk.cvut.cz, Martin.Sundermeyer@dlr.de',
    license='MIT',
    package_data={'scripts': ['meshlab_scripts/*']}
)