from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='sleep-utils',
      version='1.16',
      description='A collection of tools for sleep research',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/skjerns/sleep-utils',
      author='skjerns',
      author_email='nomail@nomail.com',
      license='GNU 2.0',
      packages=['sleep_utils'],
      install_requires=['mne', 'matplotlib', 'pandas', 'seaborn', 'scipy', 'lspopt'],
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],)

