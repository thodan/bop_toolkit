from setuptools import setup, find_packages

setup(
    name='bop_toolkit',
    version='1.0',
    packages=find_packages(exclude=('docs')),
    install_requires=[],
    author='Tomas Hodan',
    author_email='Martin.Sundermeyer@dlr.de',
    license='MIT license',
    package_data={'bop_toolkit_lib':['*']},
    # include_package_data=True
)