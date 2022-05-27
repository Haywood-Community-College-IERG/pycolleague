from setuptools import setup

setup(name='pycolleague',
      version='0.1.5',
      description="A module for accessing data exported from Colleague SIS. Includes the ability to work with data exported as CSVs or data imported into the SAS Data Mart or CCDW database",
      url='https://www.haywood.edu/',
      author='David Onder',
      author_email='donder@haywood.edu',
      license='GNU GENERAL PUBLIC LICENSE v3.0',
      packages=['pycolleague'],
      install_requires=[
            'sqlalchemy',
            'pyodbc',
            'pyyaml',
            'python-dotenv',
            'pandas',
            'pydantic',
      ],
      zip_safe=False)