"""Run all tests"""

import unittest
import sys


def run_tests():
    """Run all tests in current directory with signature test*.py"""
    testsuite = unittest.TestLoader().discover('', pattern="test*.py")
    run = unittest.TextTestRunner(verbosity=2).run(testsuite)
    return run.wasSuccessful()


if __name__ == '__main__':
    successful = run_tests()
    sys.exit(not successful)
