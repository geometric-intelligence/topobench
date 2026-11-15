"""Tests for the io_utils module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from topobench.data.utils.io_utils import *


def test_get_file_id_from_url():
    """Test get_file_id_from_url."""
    url_1 = "https://docs.google.com/file/d/SOME-FILE-ID-1"
    url_2 = "https://docs.google.com/?id=SOME-FILE-ID-2"
    url_3 = "https://docs.google.com/?arg1=9&id=SOME-FILE-ID-3"
    url_4 = "https://docs.google.com/file/d/idSOME-TRICKY-FILE"
    url_wrong = "https://docs.google.com/?arg1=9&idSOME-FILE-ID"

    assert get_file_id_from_url(url_1) == "SOME-FILE-ID-1"
    assert get_file_id_from_url(url_2) == "SOME-FILE-ID-2"
    assert get_file_id_from_url(url_3) == "SOME-FILE-ID-3"
    assert get_file_id_from_url(url_4) == "idSOME-TRICKY-FILE"

    with pytest.raises(ValueError):
        get_file_id_from_url(url_wrong)


class TestDownloadFileFromLink:
    """Test suite for download_file_from_link function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs.

        Returns
        -------
        str
            Path to temporary directory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_response(self):
        """Create mock response object.

        Returns
        -------
        MagicMock
            Mock response object with status code and headers.
        """
        response = MagicMock()
        response.status_code = 200
        response.headers = {"content-length": "5242880"}  # 5 MB
        response.elapsed.total_seconds.return_value = 1.0
        return response

    def test_download_success_with_progress(self, temp_dir, mock_response):
        """Test successful download with progress reporting.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        mock_response : MagicMock
            Mock response object.
        """
        # Setup mock chunks (5MB total in 1MB chunks)
        chunk_data = [b"x" * (1024 * 1024) for _ in range(5)]
        mock_response.iter_content.return_value = chunk_data

        with patch("requests.get", return_value=mock_response):
            download_file_from_link(
                file_link="http://example.com/dataset.tar.gz",
                path_to_save=temp_dir,
                dataset_name="test_dataset",
                file_format="tar.gz",
                timeout=60,
                retries=1,
            )

        # Verify file was created and has correct size
        output_file = os.path.join(temp_dir, "test_dataset.tar.gz")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) == 5 * 1024 * 1024

    def test_download_creates_directory_if_not_exists(self, temp_dir):
        """Test that download creates directory structure.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        """
        nested_dir = os.path.join(temp_dir, "nested", "path")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.iter_content.return_value = [b"x" * 1024]

        with patch("requests.get", return_value=mock_response):
            download_file_from_link(
                file_link="http://example.com/dataset.tar.gz",
                path_to_save=nested_dir,
                dataset_name="test_dataset",
                file_format="tar.gz",
                timeout=60,
                retries=1,
            )

        output_file = os.path.join(nested_dir, "test_dataset.tar.gz")
        assert os.path.exists(output_file)
        assert os.path.isdir(nested_dir)

    def test_download_http_error(self, temp_dir):
        """Test handling of HTTP error responses.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        """
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            download_file_from_link(
                file_link="http://example.com/nonexistent.tar.gz",
                path_to_save=temp_dir,
                dataset_name="test_dataset",
                file_format="tar.gz",
                timeout=60,
                retries=1,
            )

        # File should not be created on HTTP error
        output_file = os.path.join(temp_dir, "test_dataset.tar.gz")
        assert not os.path.exists(output_file)

    def test_download_timeout_retry(self, temp_dir):
        """Test retry logic on timeout.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        """
        import requests

        with patch("requests.get") as mock_get:
            # First call times out, second succeeds
            mock_response_success = MagicMock()
            mock_response_success.status_code = 200
            mock_response_success.headers = {"content-length": "1024"}
            mock_response_success.elapsed.total_seconds.return_value = 0.5
            mock_response_success.iter_content.return_value = [b"x" * 1024]

            mock_get.side_effect = [
                requests.exceptions.Timeout("Connection timed out"),
                mock_response_success,
            ]

            with patch("time.sleep"):  # Mock sleep to speed up test
                download_file_from_link(
                    file_link="http://example.com/dataset.tar.gz",
                    path_to_save=temp_dir,
                    dataset_name="test_dataset",
                    file_format="tar.gz",
                    timeout=60,
                    retries=3,
                )

        # File should be created on successful retry
        output_file = os.path.join(temp_dir, "test_dataset.tar.gz")
        assert os.path.exists(output_file)
        assert mock_get.call_count == 2

    def test_download_exhausts_retries(self, temp_dir):
        """Test that exception is raised after all retries exhausted.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        """
        import requests

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout(
                "Connection timed out"
            )

            with patch("time.sleep"):
                with pytest.raises(requests.exceptions.Timeout):
                    download_file_from_link(
                        file_link="http://example.com/dataset.tar.gz",
                        path_to_save=temp_dir,
                        dataset_name="test_dataset",
                        file_format="tar.gz",
                        timeout=60,
                        retries=2,
                    )

        # Verify retries were attempted
        assert mock_get.call_count == 2

    def test_download_with_different_formats(self, temp_dir, mock_response):
        """Test download with different file formats.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        mock_response : MagicMock
            Mock response object.
        """
        mock_response.iter_content.return_value = [b"test content"]

        formats = ["zip", "tar", "tar.gz"]

        with patch("requests.get", return_value=mock_response):
            for fmt in formats:
                download_file_from_link(
                    file_link="http://example.com/dataset",
                    path_to_save=temp_dir,
                    dataset_name=f"test_dataset_{fmt.replace('.', '_')}",
                    file_format=fmt,
                    timeout=60,
                    retries=1,
                )

        # Verify all files were created with correct extensions
        for fmt in formats:
            output_file = os.path.join(
                temp_dir, f"test_dataset_{fmt.replace('.', '_')}.{fmt}"
            )
            assert os.path.exists(output_file)

    def test_download_empty_chunks(self, temp_dir):
        """Test handling of empty chunks in response.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.elapsed.total_seconds.return_value = 1.0
        # Include empty chunks (should be skipped)
        mock_response.iter_content.return_value = [
            b"x" * 512,
            b"",  # Empty chunk
            b"y" * 512,
            b"",  # Another empty chunk
        ]

        with patch("requests.get", return_value=mock_response):
            download_file_from_link(
                file_link="http://example.com/dataset.tar.gz",
                path_to_save=temp_dir,
                dataset_name="test_dataset",
                file_format="tar.gz",
                timeout=60,
                retries=1,
            )

        # File should contain only non-empty chunks
        output_file = os.path.join(temp_dir, "test_dataset.tar.gz")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) == 1024

    def test_download_unknown_size(self, temp_dir):
        """Test download when content-length header is missing.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}  # No content-length header
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.iter_content.return_value = [b"x" * 1024]

        with patch("requests.get", return_value=mock_response):
            download_file_from_link(
                file_link="http://example.com/dataset.tar.gz",
                path_to_save=temp_dir,
                dataset_name="test_dataset",
                file_format="tar.gz",
                timeout=60,
                retries=1,
            )

        output_file = os.path.join(temp_dir, "test_dataset.tar.gz")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) == 1024

    def test_download_ssl_verification_disabled(self, temp_dir, mock_response):
        """Test that SSL verification can be disabled.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        mock_response : MagicMock
            Mock response object.
        """
        mock_response.iter_content.return_value = [b"test content"]

        with patch("requests.get", return_value=mock_response) as mock_get:
            download_file_from_link(
                file_link="https://example.com/dataset.tar.gz",
                path_to_save=temp_dir,
                dataset_name="test_dataset",
                file_format="tar.gz",
                verify=False,
                timeout=60,
                retries=1,
            )

        # Verify requests.get was called with verify=False
        mock_get.assert_called_once()
        assert mock_get.call_args[1]["verify"] is False

    def test_download_custom_timeout(self, temp_dir, mock_response):
        """Test that custom timeout is used.

        Parameters
        ----------
        temp_dir : str
            Temporary directory path.
        mock_response : MagicMock
            Mock response object.
        """
        mock_response.iter_content.return_value = [b"test content"]

        with patch("requests.get", return_value=mock_response) as mock_get:
            custom_timeout = 120  # 2 minutes per chunk
            download_file_from_link(
                file_link="https://github.com/aidos-lab/mantra/releases/download/{version}/2_manifolds.json.gz",
                path_to_save=temp_dir,
                dataset_name="test_dataset",
                file_format="tar.gz",
                timeout=custom_timeout,
                retries=1,
            )

        # Verify requests.get was called with correct timeout
        mock_get.assert_called_once()
        assert mock_get.call_args[1]["timeout"] == (30, custom_timeout)
