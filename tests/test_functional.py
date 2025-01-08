import json
from pathlib import Path

import pytest

from extract_requirements import (
    SectionSummaryResult,
    chunk_new_line,
    chunk_semantic,
    read_txt_file,
    run,
    simulate_llm_summary,
    write_results_to_json,
)


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
    file_path.write_text("This is a test file.")
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


@pytest.fixture
def sample_results() -> list[SectionSummaryResult]:
    """
    Fixture to create sample results for testing.

    Returns:
        List of SectionSummaryResult objects.
    """
    return [
        SectionSummaryResult(
            section_number=1,
            original_text="Original text",
            summarised_requirements="Summarised requirements",
            meta_data={"key": "value"},
        )
    ]


class TestReadTxtFile:
    def test_read_txt_file(self, temp_txt_file: Path) -> None:
        """
        Test reading a text file.

        Args:
            temp_txt_file: Path to the temporary text file.
        """
        content = read_txt_file(temp_txt_file)
        assert content == "This is a test file."


class TestWriteResultsToJson:
    def test_write_results_to_json(
        self, temp_output_path: Path, sample_results: list[SectionSummaryResult]
    ) -> None:
        """
        Test writing results to a JSON file.

        Args:
            temp_output_path: Path to the temporary output file.
            sample_results: Sample results to write.
        """
        write_results_to_json(sample_results, temp_output_path)
        assert temp_output_path.exists()

        with temp_output_path.open() as file:
            data = json.load(file)
            assert data == [
                {
                    "section_number": 1,
                    "original_text": "Original text",
                    "summarised_requirements": "Summarised requirements",
                    "meta_data": {"key": "value"},
                }
            ]


class TestChunkSemantic:
    @pytest.mark.parametrize(
        "text, expected_length",
        [
            (
                """The neural network processes input data through layers.
Training data is essential for model performance.
GPUs accelerate neural network computations significantly.
Quality training data improves model accuracy.
TPUs provide specialized hardware for deep learning.
Data preprocessing is a crucial step in training.""",
                6,
            ),
            ("Short text.", 1),
        ],
    )
    def test_chunk_semantic(self, text: str, expected_length: int) -> None:
        """
        Test chunking text using semantic chunking.

        Args:
            text: Text to be chunked.
            expected_length: Expected number of chunks.
        """
        chunks = chunk_semantic(text, threshold=1.0)
        assert isinstance(chunks, list)
        assert len(chunks) == expected_length


class TestChunkNewLine:
    @pytest.mark.parametrize(
        "text, expected_chunks",
        [
            ("Line 1\n\nLine 2\nLine 3\n\n", ["Line 1", "Line 2", "Line 3"]),
            ("Single line", ["Single line"]),
        ],
    )
    def test_chunk_new_line(self, text: str, expected_chunks: list[str]) -> None:
        """
        Test chunking text by new lines.

        Args:
            text: Text to be chunked.
            expected_chunks: Expected list of chunks.
        """
        chunks = chunk_new_line(text)
        assert chunks == expected_chunks

    def test_chunk_new_line___not_string(self) -> None:
        """
        Test chunking text when input is not a string.
        """
        with pytest.raises(ValueError):
            chunk_new_line(123)


class TestSimulateLLMSummary:
    @pytest.mark.parametrize(
        "text_section, expected_summary",
        [
            ("This is a test", "sihT si a tset"),
            ("Another test", "rehtonA tset"),
        ],
    )
    def test_simulate_llm_summary(
        self, text_section: str, expected_summary: str
    ) -> None:
        """
        Test simulating a summary of the text section.

        Args:
            text_section: Text section to be summarised.
            expected_summary: Expected summary.
        """
        summary = simulate_llm_summary(text_section)
        assert summary == expected_summary


class TestRun:
    def test_run(self, tmp_path: Path) -> None:
        """
        Test running the extraction and summarisation process.

        Args:
            tmp_path: Temporary path provided by pytest.
        """
        input_path = tmp_path / "input.txt"
        input_path.write_text("This is a test.\nThis is another test.")
        output_path = tmp_path / "output.json"

        result = run(
            str(input_path), str(output_path), overwrite=True, chunk_method="new_line"
        )
        assert result is True
        assert output_path.exists()

        with output_path.open() as file:
            data = json.load(file)
            assert len(data) == 2
            assert data[0]["original_text"] == "This is a test."
            assert data[1]["original_text"] == "This is another test."

    def test_run___input_not_found(self, tmp_path: Path) -> None:
        """
        Test running the process when input file is not found.

        Args:
            tmp_path: Temporary path provided by pytest.
        """
        input_path = tmp_path / "input.txt"
        output_path = tmp_path / "output.json"

        with pytest.raises(FileNotFoundError):
            run(
                str(input_path),
                str(output_path),
                overwrite=True,
                chunk_method="new_line",
            )

    def test_run___output_exists_and_no_overwrite(self, tmp_path: Path) -> None:
        """
        Test running the process when output file exists and overwrite is False.

        Args:
            tmp_path: Temporary path provided by pytest.
        """
        input_path = tmp_path / "input.txt"
        input_path.write_text("This is a test.\nThis is another test.")
        output_path = tmp_path / "output.json"
        output_path.write_text("Existing content")

        with pytest.raises(FileExistsError):
            run(
                str(input_path),
                str(output_path),
                overwrite=False,
                chunk_method="new_line",
            )
        assert output_path.read_text() == "Existing content"

    def test_run___invalid_chunk_method(self, tmp_path: Path) -> None:
        """
        Test running the process with an invalid chunk method.

        Args:
            tmp_path: Temporary path provided by pytest.
        """
        input_path = tmp_path / "input.txt"
        input_path.write_text("This is a test.\nThis is another test.")
        output_path = tmp_path / "output.json"

        with pytest.raises(ValueError):
            run(
                str(input_path),
                str(output_path),
                overwrite=True,
                chunk_method="invalid_method",
            )
