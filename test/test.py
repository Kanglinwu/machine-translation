import unittest
import requests

class TestTranslationAPI(unittest.TestCase):
    BASE_URL = 'http://10.10.10.48:2486/translate'

    def test_translate_english_to_chinese(self):
        payload = {
            "msg": "How old are you?",
            "target_lang": "zh"
        }
        response = requests.post(self.BASE_URL, json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn('target_msg', response.json())
        self.assertIn('source_lang', response.json())
        self.assertTrue(response.json()['is_trans'])

    def test_translate_english_to_vietnamese(self):
        payload = {
            "msg": "Where are you from?",
            "target_lang": "vi"
        }
        response = requests.post(self.BASE_URL, json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn('target_msg', response.json())
        self.assertIn('source_lang', response.json())
        self.assertTrue(response.json()['is_trans'])

    def test_invalid_language(self):
        payload = {
            "msg": "This is a test.",
            "target_lang": "invalid_lang"
        }
        response = requests.post(self.BASE_URL, json=payload)
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.json())

    def test_no_input_text(self):
        payload = {
            "msg": "",
            "target_lang": "zh"
        }
        response = requests.post(self.BASE_URL, json=payload)
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.json())

    def test_missing_target_language(self):
        payload = {
            "msg": "Hello World",
        }
        response = requests.post(self.BASE_URL, json=payload)
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.json())

if __name__ == '__main__':
    unittest.main()
