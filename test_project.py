"""실행: python -m unittest -v test_project.py"""
import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest
from ml import BASE, validate_data, make_pipeline, predict_batch


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = validate_data(pd.read_csv(BASE / 'data/train.csv'), training=True)
        cls.test = pd.read_csv(BASE / 'data/test.csv')
        cls.model = make_pipeline(LogisticRegression(max_iter=2000)).fit(cls.train.drop(columns='Survived'), cls.train.Survived)

    def test_schema_errors(self):
        for bad in [self.test.drop(columns='Age'), self.test.assign(Fare=-1), self.test.assign(PassengerId=1), self.test.assign(Age='bad')]:
            with self.assertRaises(ValueError):
                validate_data(bad, require_id=True)

    def test_missing_and_unseen_features(self):
        data = self.test.head(3).drop(columns=['Name', 'Cabin']).assign(Age=np.nan, Fare=np.nan, Embarked=np.nan, Sex=np.nan)
        submission, details = predict_batch(self.model, data)
        self.assertTrue(details.SurvivalProbability.between(0, 1).all())
        self.assertEqual(submission.PassengerId.tolist(), data.PassengerId.tolist())

    def test_submission_and_label_isolation(self):
        submission, details = predict_batch(self.model, self.test)
        self.assertEqual(submission.columns.tolist(), ['PassengerId', 'Survived'])
        self.assertEqual(len(submission), len(self.test))
        self.assertTrue(submission.Survived.isin([0, 1]).all())
        _, other = predict_batch(self.model, self.test.assign(Survived=1))
        np.testing.assert_array_equal(details.SurvivalProbability, other.SurvivalProbability)

    def test_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(self.model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(restored.predict(self.test), self.model.predict(self.test))

    def test_app_workflow(self):
        app = AppTest.from_file(str(BASE / 'app.py'), default_timeout=180).run()
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label == '모델 학습 시작').click().run(timeout=180)
        self.assertFalse(app.exception)
        self.assertIn('training_result', app.session_state)
        next(button for button in app.button if button.label == '생존 가능성 예측').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(metric.label == '모델이 추정한 생존 확률' for metric in app.metric))
        app.slider[0].set_value(0.7).run()
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
