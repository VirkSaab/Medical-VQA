from setuptools import setup, find_packages

VERSION = '0.0.1'

f = open('README.md', 'r')
LONG_DESCRIPTION = f.read()
f.close()
    
setup(
    name=f"mvqag",
    version=VERSION,
    description='Medical Visual Question-Answering and Generation research project in collaboration with IIT Ropar, Punjab, India and IIAI, Abu Dhabi, UAE',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Jitender Singh Virk',
    author_email='jitender.virk@iitrpr.ac.in',
    url="https://github.com/VirkSaab/VQAMixUp.git",
    packages=find_packages(exclude=['ez_setup', 'tests', '.github']),
    include_package_data=True,
    entry_points=f"""
        [console_scripts]
        mvqag=mvqag.main:cli
    """,
)
