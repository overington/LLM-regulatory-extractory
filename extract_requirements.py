"""
Module for extracting and summarising requirements from text files.
"""

import json
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from random import random as rand
from typing import Literal, Optional

from chonkie import SDPMChunker

logger = getLogger(__name__)


class CLIArgsNS(Namespace):
    """
    Namespace for command-line arguments.
    """

    input_path: str
    output_path: str
    overwrite: bool
    chunk_method: str
    semantic_similarity_threshold: float | int | Literal["auto"]


@dataclass
class SectionSummaryResult:
    """
    Data class for storing section summary results.
    """

    section_number: int
    original_text: str
    summarised_requirements: str
    # assuming metadata is a dictionary of strings, ints, or floats
    meta_data: Optional[dict[str, str | int | float]] = None

    def to_json(self) -> dict[str, int | str | dict[str, str | int | float]]:
        """
        Convert the object to a JSON-serializable dictionary using hard-coded types.
        """

        # Assuming self.meta_data is a dictionary of strings, ints, or floats or None
        return {
            "section_number": int(self.section_number),
            "original_text": str(self.original_text),
            "summarised_requirements": str(self.summarised_requirements),
            "meta_data": self.meta_data if self.meta_data is not None else {},
        }


def read_txt_file(file_path: Path) -> str:
    """
    Read a text file and return its content as a string.

    Args:
        file_path (Path): Path to the text file.

    Returns:
        str: Content of the text file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with file_path.open(mode="r") as file:
        return file.read()


def write_results_to_json(
    results: list[SectionSummaryResult], output_path: Path, overwrite: bool = False
) -> None:
    """
    Write the results to a JSON file.

    Args:
        results (list[SectionSummaryResult]): List of section summary results.
        output_path (Path): Path to the output JSON file.
        overwrite (bool): Whether to overwrite the file if it exists. Default is True.
    """
    # Convert results to JSON and write to file, creating parent directories if necessary
    text_summaries = [result.to_json() for result in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overwrite = "w" if overwrite else "x"
    with output_path.open(mode=overwrite) as file:
        json.dump(text_summaries, file, indent=2)


def chunk_semantic(text: str, threshold: int | float | str = "auto") -> list[str]:
    """
    Chunk text using semantic chunking.

    Args:
        text (str): Text to be chunked.

    Returns:
        list[str]: List of text chunks.
    """
    chunker = SDPMChunker(
        embedding_model="minishlab/potion-base-8M",
        threshold=threshold,
        chunk_size=512,
        min_sentences=1,
        skip_window=1,
    )
    chunks = chunker.chunk(text)
    return [chunk.text.rstrip(" \n") for chunk in chunks]


def chunk_new_line(text: str) -> list[str]:
    """
    Chunk text by new lines.

    Args:
        text (str): Text to be chunked.

    Returns:
        list[str]: List of text chunks.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    if "\n" not in text:
        return [text]
    # Remove empty lines
    return list(filter(lambda s: s.rstrip() != "", text.split("\n")))


def simulate_llm_summary(text_section: str) -> str:
    """
    Simulate a summary of the text section.

    Args:
        text_section (str): Text section to be summarised.

    Returns:
        str: Simulated summary of the text section.
    """
    return " ".join([word[::-1] for word in text_section.split()])


def run(
    input_path: str,
    output_path: str,
    overwrite: bool,
    chunk_method: str = "new_line",
    semantic_similarity_threshold: Optional[float | Literal["auto"]] = None,
) -> None:
    """
    Run the extraction and summarisation process.

    Args:
        input_path (str): Path to the input text file.
        output_path (str): Path to the output JSON file.
        overwrite (bool): Whether to overwrite the output file if it exists. Default is False.
        chunk_method (str): Method to chunk the text. Default is "new_line".
        semantic_similarity_threshold (Optional[float | Literal["auto"]]): Threshold for semantic similarity.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Catch IO errors early
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise FileNotFoundError(f"Input file not found: {input_path=}")
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            logger.error(
                "Output file already exists: %s, and overwrite is False", output_path
            )
            raise FileExistsError(
                f"Output file already exists: {output_path=}, and overwrite is False"
            )

    text = read_txt_file(input_path)

    # Choose chunking method based on the provided argument
    match chunk_method:
        case "semantic":
            # If semantic chunking is selected, use the provided threshold or default to "auto"
            threshold_value = (
                semantic_similarity_threshold
                if isinstance(semantic_similarity_threshold, (int, float))
                or semantic_similarity_threshold == "auto"
                else "auto"
            )
            chunks = chunk_semantic(text, threshold=threshold_value)
        case "new_line":
            chunks = chunk_new_line(text)
        case _:
            logger.error("Invalid chunk method: %s", chunk_method)
            raise ValueError(f"Invalid chunk method: {chunk_method=}")

    results: list[SectionSummaryResult] = []

    # Process each chunk, simulate LLM summary, and collect results as a list of SectionSummaryResult
    for idx, text_section in enumerate(chunks):
        original_text = text_section
        summarised_requirements = simulate_llm_summary(text_section)
        metadata = {"random_number": rand()} if rand() > 0.5 else None
        results.append(
            SectionSummaryResult(
                section_number=idx,
                original_text=original_text,
                summarised_requirements=summarised_requirements,
                meta_data=metadata,
            )
        )
    write_results_to_json(results, output_path, overwrite=overwrite)
    return True


def parse_args(args) -> CLIArgsNS:
    """
    Parse command-line arguments.

    Args:
        args: Command-line arguments.

    Returns:
        CLIArgsNS: Parsed command-line arguments.
    """
    parser = ArgumentParser(
        prog="extract_requirements",
        description="Extract requirements from a text file and summarise them",
    )

    # Positional arguments for input / output files
    parser.add_argument("input_path", type=str, help="Path to the input text file")
    parser.add_argument("output_path", type=str, help="Path to the output JSON file")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--chunk_method", type=str, default="new_line", choices=["semantic", "new_line"]
    )
    parser.add_argument(
        "--semantic_similarity_threshold",
        default=None,
        help=(
            "Threshold for semantic similarity. Will be ignored if chunk_method is not 'semantic'. "
            "Options are 'auto', a float value between 0 and 1 or an integer value between 0-100 "
            "which will be inferred as a percentage. Default is 'auto'."
        ),
    )
    return parser.parse_args(args)


def main(args=None):
    """
    Main function to execute the script.
    """
    args: CLIArgsNS = parse_args(args)
    run(
        input_path=args.input_path,
        output_path=args.output_path,
        overwrite=args.overwrite,
        chunk_method=args.chunk_method,
        semantic_similarity_threshold=args.semantic_similarity_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
