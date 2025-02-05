from setuptools import setup, find_packages

package_name = 'bop_toolkit_lib'

setup(
    name=package_name,
    version="1.0",
    packages=find_packages(exclude=("docs")),
    install_requires=["pytz", "vispy>=0.6.5", "PyOpenGL==3.1.0", "pypng", "cython"],
    author="Tomas Hodan, Martin Sundermeyer",
    author_email="tom.hodan@gmail.com, Martin.Sundermeyer@dlr.de",
    license="MIT license",
    package_data={"bop_toolkit_lib": ["*"]},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
)
