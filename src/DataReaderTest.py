from __future__ import division
from DataReader import DataReader, DataParserOpt
from DataSampler import DataSampler
import numpy as np


def test():
    opt = DataParserOpt()
    # print type(opt)
    myOpt = DataParserOpt(only_left=True, only_right=False)
    myDataReader = DataReader(
        './src/mapping.txt', './samples/sample_b_1.json', myOpt)
    a, y = myDataReader.extract_features()
    np.savetxt('testdata.csv', a, delimiter=',')
    np.save('testdata', a)
    print a.shape, len(y)

    # Test sampler
    myDataSampler = DataSampler(a, 'uniform_sampling', 'rate', 20)
    print myDataSampler.__call__.shape
if __name__ == '__main__':
    test()
