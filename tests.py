import myCalc
import unittest

class MyCalcTest(unittest.TestCase):

	def test_sum(self):
		c = myCalc.sum(20,10)
		self.assertsEqual(c,30)

	def test_sub(self):
		c = myCalc.sub(20,10)
		self.assertEqual(c,10)

if __name__ == "__main__":
	unittest.main()