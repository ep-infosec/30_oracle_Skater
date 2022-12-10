import unittest


class TestValidation(unittest.TestCase):
    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestValidation)
    unittest.TextTestRunner(verbosity=2).run(suite)
