import json
from pathlib import Path
from unittest.mock import patch

import pytest

from extract_requirements import CLIArgsNS, main, parse_args, run


@pytest.fixture
def temp_txt_file(tmp_path: Path) -> Path:
    """
    Fixture to create a temporary text file for testing.

    Args:
        tmp_path: Temporary path provided by pytest.

    Returns:
        Path to the temporary text file.
    """
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test file.\nThis is another test.")
    return file_path


@pytest.fixture
def temp_output_path(tmp_path: Path) -> Path:
    """
    Fixture to create a temporary output path for testing.

    Args:
        tmp_path: Temporary path provided by pytest.

    Returns:
        Path to the temporary output file.
    """
    return tmp_path / "output.json"


def test_cli_run(temp_txt_file: Path, temp_output_path: Path) -> None:
    """
    Test running the CLI with valid arguments.

    Args:
        temp_txt_file: Path to the temporary text file.
        temp_output_path: Path to the temporary output file.
    """
    test_args = parse_args(
        [
            str(temp_txt_file),
            str(temp_output_path),
            "--overwrite",
            "--chunk_method",
            "new_line",
        ]
    )

    result = run(
        input_path=test_args.input_path,
        output_path=test_args.output_path,
        overwrite=test_args.overwrite,
        chunk_method=test_args.chunk_method,
        semantic_similarity_threshold=test_args.semantic_similarity_threshold,
    )

    assert result is True
    assert temp_output_path.exists()

    with temp_output_path.open() as file:
        data = json.load(file)
        assert len(data) == 2
        assert data[0]["original_text"] == "This is a test file."
        assert data[1]["original_text"] == "This is another test."


@pytest.mark.parametrize(
    "chunk_method, expected_length, raw_text",
    [
        (
            "semantic",
            6,
            """The neural network processes input data through layers. 
                    Training data is essential for model performance.
                    GPUs accelerate neural network computations significantly.
                    Quality training data improves model accuracy.
                    TPUs provide specialized hardware for deep learning.
                    Data preprocessing is a crucial step in training.""",
        ),
        ("new_line", 1, "Short text."),
    ],
)
def test_cli_chunk_methods(
    tmp_path,
    temp_output_path: Path,
    chunk_method: str,
    expected_length: int,
    raw_text: str,
) -> None:
    """
    Test running the CLI with different chunk methods.

    Args:
        temp_txt_file: Path to the temporary text file.
        temp_output_path: Path to the temporary output file.
        chunk_method: Chunk method to use.
        expected_length: Expected number of chunks.
    """
    # setup
    temp_txt_file = tmp_path / "input_text.txt"
    temp_txt_file.write_text(raw_text)

    # run test
    test_args = parse_args(
        [
            str(temp_txt_file),
            str(temp_output_path),
            "--overwrite",
            "--chunk_method",
            chunk_method,
        ]
    )

    result = run(
        input_path=test_args.input_path,
        output_path=test_args.output_path,
        overwrite=test_args.overwrite,
        chunk_method=test_args.chunk_method,
        semantic_similarity_threshold=1.0,
    )

    assert result is True
    assert temp_output_path.exists()

    with temp_output_path.open() as file:
        data = json.load(file)
        # remove meta_data, as it is random
        for d in data:
            d["meta_data"] = {}
        assert len(data) == expected_length


def test_main_mocked() -> None:
    """
    Test the main function with mocked arguments and run function.
    """
    mock_args = CLIArgsNS(
        input_path="input.txt",
        output_path="output.json",
        overwrite=True,
        chunk_method="new_line",
        semantic_similarity_threshold=None,
    )
    with (
        patch("extract_requirements.run") as mock_run,
        patch("extract_requirements.parse_args", return_value=mock_args),
    ):
        main()
        mock_run.assert_called_once_with(
            input_path="input.txt",
            output_path="output.json",
            overwrite=True,
            chunk_method="new_line",
            semantic_similarity_threshold=None,
        )
