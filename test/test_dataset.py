# Testing script to ensure the Dataset class works as expected

from utils.dataset import Dataset


def test_quac_preprocessing():
    dataset = Dataset("data/QuAC")

def test_coqa_preprocessing():
    dataset = Dataset("data/CoQA")

if __name__ == "__main__":
    # test_quac_preprocessing()
    test_coqa_preprocessing()
