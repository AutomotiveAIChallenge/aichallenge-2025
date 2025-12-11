from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'bag_manager_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A package for managing bag files in ROS2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bag_manager_node = bag_manager_py.bag_manager_node:main',
        ],
    },
)