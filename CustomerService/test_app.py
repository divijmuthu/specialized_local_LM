# Test cases for Customer Feedback Analysis Tool
import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import preprocess_text, analyze_sentiment, extract_topics, get_key_phrases, analyze_text_for_actions, generate_insights

class TestFeedbackAnalysis(unittest.TestCase):
    def setUp(self):
        self.sample_text = "The product is good, but the pricing is too high."
        self.sample_df = pd.DataFrame([
            {"text": "I love your product! It's amazing.", "date": "2023-01-01"},
            {"text": "Customer service was terrible. They didn't respond to my issue.", "date": "2023-01-02"},
            {"text": "The product is good, but the pricing is too high compared to competitors.", "date": "2023-01-03"},
            {"text": "Just had the worst experience with your customer service. Never buying from you again!", "date": "2023-02-15"},
            {"text": "The new update is fantastic! Much better than before.", "date": "2023-03-01"},
        ])

    def test_preprocess_text(self):
        processed = preprocess_text(self.sample_text)
        self.assertIsInstance(processed, str)
        self.assertNotIn('.', processed)
        self.assertTrue(len(processed) > 0)

    def test_analyze_sentiment(self):
        sentiment, score = analyze_sentiment(self.sample_text)
        self.assertIn(sentiment.lower(), ['positive', 'negative', 'neutral'])
        self.assertTrue(0.0 <= score <= 1.0)

    def test_extract_topics(self):
        topics, labels = extract_topics(self.sample_df['text'].tolist(), n_topics=2)
        self.assertIsInstance(topics, list)
        self.assertTrue(len(topics) > 0 or len(self.sample_df) < 2)  # Accept empty if not enough data
        self.assertIsInstance(labels, (list, np.ndarray))

    def test_get_key_phrases(self):
        phrases = get_key_phrases(self.sample_text)
        self.assertIsInstance(phrases, list)

    def test_analyze_text_for_actions(self):
        actions = analyze_text_for_actions(self.sample_text)
        self.assertIsInstance(actions, list)
        # Should detect pricing related action
        self.assertTrue(any(a[0] == 'pricing' for a in actions))

    def test_generate_insights(self):
        insights, enriched_df = generate_insights(self.sample_df)
        self.assertIsInstance(insights, dict)
        self.assertIsInstance(enriched_df, pd.DataFrame)
        self.assertIn('sentiment_distribution', insights)
        self.assertIn('topic_distribution', insights)
        self.assertIn('common_key_phrases', insights)
        self.assertIn('actionable_items', insights)

    def test_empty_dataframe(self):
        df = pd.DataFrame([])
        insights, enriched_df = generate_insights(df)
        self.assertEqual(insights, {})
        self.assertTrue(enriched_df.empty)

    def test_missing_columns(self):
        df = pd.DataFrame([{"foo": "bar"}])
        with self.assertRaises(Exception):
            generate_insights(df)

    def test_file_upload_allowed(self):
        from app import allowed_file
        self.assertTrue(allowed_file("test.csv"))
        self.assertTrue(allowed_file("test.xlsx"))
        self.assertFalse(allowed_file("test.txt"))

    def test_actionable_items_extraction(self):
        text = "There is a bug and the documentation is poor."
        actions = analyze_text_for_actions(text)
        categories = [a[0] for a in actions]
        self.assertIn('bug', categories)
        self.assertIn('documentation', categories)

    def test_sentiment_trends(self):
        df = self.sample_df.copy()
        insights, _ = generate_insights(df)
        if 'sentiment_over_time' in insights:
            self.assertIsInstance(insights['sentiment_over_time'], list)

    def test_topic_trends(self):
        df = self.sample_df.copy()
        insights, _ = generate_insights(df)
        if 'topic_trends' in insights:
            self.assertIsInstance(insights['topic_trends'], dict)

    def test_top_actionable_items(self):
        df = self.sample_df.copy()
        insights, _ = generate_insights(df)
        if 'top_actionable_items' in insights:
            self.assertIsInstance(insights['top_actionable_items'], list)

    def test_model_training(self):
        from app import train_model
        df = self.sample_df.copy()
        result = train_model(df)
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)

if __name__ == "__main__":
    unittest.main()
