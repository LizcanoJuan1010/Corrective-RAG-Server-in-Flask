import unittest
from unittest.mock import patch
from app.db_utils import get_db_engine

class TestDbUtils(unittest.TestCase):

    @patch('app.db_utils.create_engine')
    @patch('os.getenv')
    def test_get_db_engine(self, mock_getenv, mock_create_engine):
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "DB_USER": "testuser",
            "DB_PASSWORD": "testpassword",
            "DB_HOST": "testhost",
            "DB_PORT": "5432",
            "DB_NAME": "testdb",
        }[key]

        # Call the function
        get_db_engine()

        # Assert that create_engine was called with the correct URL
        mock_create_engine.assert_called_once()
        args, _ = mock_create_engine.call_args
        url = args[0]
        self.assertEqual(url.drivername, "postgresql+psycopg2")
        self.assertEqual(url.username, "testuser")
        self.assertEqual(url.password, "testpassword")
        self.assertEqual(url.host, "testhost")
        self.assertEqual(url.port, 5432)
        self.assertEqual(url.database, "testdb")

if __name__ == '__main__':
    unittest.main()
