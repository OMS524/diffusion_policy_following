from glob import glob

from setuptools import find_packages, setup

package_name = "aloha_bag_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="airo",
    maintainer_email="vontte21@hanyang.ac.kr",
    description="Episode-wise rosbag2 recorder for the airo / Piper ALOHA setup.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "recorder_node = aloha_bag_recorder.recorder_node:main",
            "dp_recorder_node = aloha_bag_recorder.dp_recorder_node:main",
            "bag_to_video = aloha_bag_recorder.bag_to_video:main",
        ],
    },
)
