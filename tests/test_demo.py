import unittest
from src.demo import say_hello


class TestDemo(unittest.TestCase):

    def setUp(self):

        # defining input shared across tests
        self.dummy_name = 'dummy'

    def test_say_hello(self):

        # get output of say_hello
        output = say_hello(name=self.dummy_name)

        # check output
        self.assertEqual(output, 'hello dummy')

    def tearDown(self):

        # define actions that are carried out after each test
        pass
