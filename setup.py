from setuptools import setup, find_packages

setup(name='gymMDT',
      version='0.0.1',
      description='Markov Decision Task',
      author='Younghoon Kim',
      author_email='aktivhoon@snu.ac.kr',
      packages=find_packages(),
      include_package_data=True,
      install_requires=['gym', 'numpy-stl', 'pyglet']
)