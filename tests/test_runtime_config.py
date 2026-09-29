import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from raybundle.runtime_config import private_root


class RuntimeConfigTests(unittest.TestCase):
    def test_configuration_required(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):
                private_root()

    def test_existing_environment_and_owned_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ,dict(RAYBUNDLE_ROOT=folder,RAYBUNDLE_ENV_PREFIX=sys.prefix)):
                self.assertEqual(private_root(),Path(folder).resolve())

    def test_other_environment_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ,dict(RAYBUNDLE_ROOT=folder,RAYBUNDLE_ENV_PREFIX=folder)):
                with self.assertRaises(RuntimeError):
                    private_root()


if __name__=='__main__':
    unittest.main()
