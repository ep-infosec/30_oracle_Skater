# Reference: http://python-packaging.readthedocs.io/en/latest/dependencies.html
import distutils.command.install as orig
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
import io
import sys
import contextlib


class InvokingShScript(install):
    """Custom install setup to help run shell commands (outside shell) before installation"""

    # reference:
    # Syntax (long option, short option, description).
    user_options = install.user_options + [
        ('ostype=', None, 'The OS type of the box(linux-ubuntu/mac)'),
        ('rl=', None, 'Enable rule based learning')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.ostype = 'linux-ubuntu'
        self.rl = False

    def finalize_options(self):
        if self.ostype is None:
            raise Exception("specify os type ...")
        if self.rl is None:
            raise Exception(" should based learning be enabled? ...")
        install.finalize_options(self)

    def run(self):
        # install rule based learners when asked
        if self.rl:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            shell_script_path = os.path.join(dir_path, 'setup.sh')

            subprocess.check_output(['bash', shell_script_path, self.ostype])
        orig.install.run(self)


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    with chdir(root):
        with io.open(os.path.join(root, 'skater', 'about.py'), encoding='utf8') as f:
            about = {}
            exec(f.read(), about)

        with io.open(os.path.join(root, 'README.md'), encoding='utf8') as f:
            readme = f.read()

    setup(
        name=about['__title__'],
        packages=find_packages(),
        description=about['__summary__'],
        long_description=readme,
        author=about['__author__'],
        author_email=about['__email__'],
        version=about['__version__'],
        url=about['__uri__'],
        license=about['__license__'],
        cmdclass={'install': InvokingShScript},
        include_package_data=True,
        install_requires=[
            'scikit-learn>=0.19',
            'scikit-image>=0.14',
            'pandas>=0.22.0',
            'lime>=0.1.1.21',
            'requests',
            'multiprocess',
            'dill>=0.2.6',
            'joblib>=0.11',
            'Jinja2>=2.10.1',
            'pydotplus>=2.0.2',
            'plotly>=3.2.0',
            'bs4',
            'six'],
        extras_require={
            'all': 'matplotlib'},
        zip_safe=False)


if __name__ == '__main__':
    setup_package()
