from setuptools import setup


def _parse_requirements(requirements_txt_path):
    with open(requirements_txt_path) as fp:
        return filter(lambda x: x[0] not in "#-", fp.read().splitlines())


setup(
    name='nnvpy',
    version='0.0.1',
    # Contained modules and scripts.
    install_requires=_parse_requirements('requirements.txt'),
    tests_require=_parse_requirements('requirements-test.txt'),
    requires_python='>=3.6',
    packages=['nnvpy'],
)
