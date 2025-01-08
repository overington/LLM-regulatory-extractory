# LLM Regulatory Extractor

The script reads a text file and extracts regulatory requirements from it. The input document is chunked into smaller parts with two available `chunk_method` options: `semantic` and `new_line`.

### `new_line` Chunking

The text chunking function is a simpple implementation that splits on new lines, and removes blank lines. It is implemented using python standard new line character, and without external libraries.

### `semantic` Chunking

Split text using _Semantic Double-Pass Merging_ for improved context preservation

The text chunking function is implemented using the [`chonkie`](https://docs.chonkie.ai/getting-started/introduction) library - a lightweight, easy-to-use, and fast library for text chunking. The chunking function uses the SDPMChunker (from the [docs](https://docs.chonkie.ai/chunkers/sdpm-chunker)):

> The `SDPMChunker` extends semantic chunking by using a double-pass merging approach. It first groups content by semantic similarity, then merges similar groups within a skip window, allowing it to connect related content that may not be consecutive in the text. This technique is particularly useful for documents with recurring themes or concepts spread apart.


The `semantic` chunking method requires the `chonkie` library to be installed which is included in the `requirements.txt` file - see Setup below for script installation and usage.

There is also a `--semantic_similarity_threshold` argument that can be used to set the threshold for semantic similarity. The threshold can be set to `auto`, a float value between 0 and 1, or an integer value between 0-100. This denotes the similarity threshold to consider sentences similar.


## Setup

0. **Prerequisites: Python 3.10+**

This script uses modern Python features - like match case, which are available in Python 3.10 and above. Please ensure you use Python 3.10 or above to run the script. An easy way to install Python 3.10 is to use [pyenv](https://github.com/pyenv/pyenv).


1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd LLM-regulatory-extractory
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the script, use the following command (ensure you are using Python 3.10+):

```sh
python extract_requirements.py <input_path> <output_path> [--overwrite] [--chunk_method <method>] [--semantic_similarity_threshold <threshold>]
```

### Arguments

- `<input_path>`: Path to the input text file.
- `<output_path>`: Path to the output JSON file.
- `--overwrite`: Optional flag to overwrite the output file if it exists.
- `--chunk_method`: Method to chunk the text. Options are `semantic` or `new_line`. Default is `new_line`.
- `--semantic_similarity_threshold`: Threshold for semantic similarity. Options are `auto`, a float value between 0 and 1, or an integer value between 0-100. Default is `auto`.

### Example

```sh
python extract_requirements.py regulations.txt output.json --overwrite --chunk_method semantic --semantic_similarity_threshold 0.5
```

## Testing

To run the tests, use the following command (NB using `python -m pytest` to ensure `pytest` can find the script as a module):

```sh
python -m pytest --cov=extract_requirements --cov-report=term-missing 
```