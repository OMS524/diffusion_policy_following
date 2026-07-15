import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dp_arm_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='airo',
    maintainer_email='msd030428@gmail.com',
    description='Read-only CAN-to-ROS2 telemetry bridge for the dual Piper arms.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_bridge_node = dp_arm_bridge.arm_bridge_node:main',
            'dp_eef_bridge_node = dp_arm_bridge.dp_eef_bridge_node:main',
        ],
    },
)
