from chonkie import RecursiveChunker, RecursiveRules


def read_txt_file(file_path):
    with open(file_path, mode="r") as file:
        return file.read()


chunker = RecursiveChunker(
    tokenizer="gpt2",
    chunk_size=512,
    rules=RecursiveRules(),  # Default rules
    min_characters_per_chunk=12,
)

text = read_txt_file("regulations.txt")
chunks = chunker.chunk(text)

for chunk in chunks:
    print(f"Chunk text: {chunk.text}")
    print(f"Token count: {chunk.token_count}")
