from setuptools import setup

setup(name='sleep-utils',
      version='1.11',
      description='A collection of tools for sleep research',
      url='http://github.com/skjerns/sleep-utils',
      author='skjerns',
      author_email='nomail@nomail.com',
      license='GNU 2.0',
      packages=['sleep_utils'],
      install_requires=['mne', 'matplotlib', 'pandas', 'seaborn', 'scipy', 'lspopt'],
      zip_safe=False)

