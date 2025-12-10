from setuptools import setup, find_packages
from pathlib import Path

package_name = 'tiny_lidar_net_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (str(Path('share') / package_name), ['package.xml']),
        (str(Path('share') / package_name / 'launch'), 
         [str(p) for p in Path('launch').glob('*.launch.xml')]),
        (str(Path('share') / package_name / 'config'), 
         [str(p) for p in Path('config').glob('*.yaml')]),
        (str(Path('share') / package_name / 'ckpt'), 
         [str(p) for p in Path('ckpt').glob('*')]),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Arata Tanaka',
    maintainer_email='tanaka.arata.y5@s.gifu-u.ac.jp',
    description='A package for Tiny Lidar Net controller',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'tiny_lidar_net_controller_node = tiny_lidar_net_controller.tiny_lidar_net_controller_node:main',
        ],
    },
)