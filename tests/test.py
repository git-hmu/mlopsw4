import unittest
import joblib
import numpy as np
from train import load_and_split_data   # make sure train.py is in the repo root


class TestIrisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure you've already run `python train.py` so model.joblib exists
        cls.model = joblib.load("model.joblib")

    def test_model_accuracy(self):
        _, X_test, _, y_test = load_and_split_data()
        score = self.model.score(X_test, y_test)
        self.assertGreater(score, 0.85, "Model accuracy is too low")

    def test_predict_sample(self):
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = self.model.predict(sample)
        self.assertEqual(prediction[0], "setosa", "Expected class 'setosa'")

if __name__ == "__main__":
    unittest.main()
