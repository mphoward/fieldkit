import unittest
import os

def run():
    """ Simple runner for all tests in the directory.

    """
    testdir = os.path.dirname(os.path.abspath(__file__))
    tests = unittest.TestLoader().discover(testdir)
    result = unittest.TextTestRunner().run(tests)
