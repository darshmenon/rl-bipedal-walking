from setuptools import find_packages, setup

package_name = 'policy_runner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/policy_runner.launch.py']),
        ('share/' + package_name + '/config', ['config/xbot_joints.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='darshmenon',
    maintainer_email='darshmenon02@gmail.com',
    description='Trained locomotion policy inference node for ros2_control.',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'policy_inference = policy_runner.policy_inference_node:main',
        ],
    },
)
