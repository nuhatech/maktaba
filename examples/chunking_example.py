"""
Example: Document Chunking with UnstructuredChunker

Demonstrates how to chunk documents using Maktaba's integrated chunking module.
This replaces the need for partition-api HTTP calls.
"""

import asyncio
from pathlib import Path

from maktaba.chunking import UnstructuredChunker


async def main():
    print("=" * 60)
    print("Document Chunking Example")
    print("=" * 60)

    # Initialize chunker with defaults
    chunker = UnstructuredChunker(
        strategy="auto",  # auto, fast, hi_res, ocr_only
        chunking_strategy="basic",  # basic, by_title
    )

    # Example 1: Chunk text
    print("\n1. Chunking raw text:")
    print("-" * 60)

    text = """
    In the name of Allah, the Most Gracious, the Most Merciful.

    All praise is due to Allah, Lord of all the worlds.
    The Most Gracious, the Most Merciful.
    Master of the Day of Judgment.

    You alone we worship, and You alone we ask for help.
    Guide us along the Straight Path.
    The path of those You have blessed—not those You are displeased with,
    or those who are astray.
    """

    result = await chunker.chunk_text(
        text=text,
        filename="surah_al_fatiha.txt",
        extra_metadata={"surah": 1, "category": "quran"},
    )

    print(f"✓ Created {result.total_chunks} chunk(s)")
    print(f"✓ Total characters: {result.total_characters}")
    print(f"✓ File type: {result.metadata.filetype}")
    print(f"✓ Size: {result.metadata.size_in_bytes} bytes")

    print("\nFirst chunk:")
    print(f"  Text: {result.documents[0].text[:100]}...")
    print(f"  Metadata: {result.documents[0].metadata}")

    # Example 2: Chunk local file
    print("\n2. Chunking local file:")
    print("-" * 60)

    # Create a sample file
    sample_file = Path("sample_document.txt")
    sample_file.write_text(
        "This is a sample document for testing.\n"
        "It contains multiple lines.\n"
        "Each line will be processed by the chunker."
    )

    try:
        result = await chunker.chunk_file(
            file_path=sample_file,
            extra_metadata={"source": "local", "category": "test"},
        )

        print(f"✓ Created {result.total_chunks} chunk(s)")
        print(f"✓ Filename: {result.metadata.filename}")
        print(f"✓ Total characters: {result.total_characters}")

    finally:
        # Clean up
        sample_file.unlink()

    # Example 3: Chunk from URL
    print("\n3. Chunking from URL:")
    print("-" * 60)

    try:
        result = await chunker.chunk_url(
            url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            filename="dummy.pdf",
            extra_metadata={"source": "web", "category": "test"},
        )

        print(f"✓ Created {result.total_chunks} chunk(s)")
        print(f"✓ File type: {result.metadata.filetype}")
        print(f"✓ Total pages: {result.total_pages or 'N/A'}")
        print(f"✓ Size: {result.metadata.size_in_bytes} bytes")

    except Exception as e:
        print(f"✗ Failed to chunk URL: {e}")

    # Example 4: Advanced chunking with custom options
    print("\n4. Advanced chunking (hi_res strategy):")
    print("-" * 60)

    text = "Advanced document processing with high-resolution parsing."

    result = await chunker.chunk_text(
        text=text,
        filename="advanced.txt",
        strategy="hi_res",  # Override default strategy
        chunking_strategy="by_title",  # Override default chunking
        extra_metadata={"quality": "high"},
    )

    print(f"✓ Created {result.total_chunks} chunk(s)")
    print("✓ Strategy: hi_res")
    print("✓ Chunking: by_title")

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
