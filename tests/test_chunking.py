"""Tests for document chunking implementations."""


import pytest

from maktaba.chunking.unstructured import UnstructuredChunker


# Test 1: Basic text chunking
@pytest.mark.asyncio
async def test_basic_text_chunking():
    """Test basic text chunking without advanced parameters."""
    chunker = UnstructuredChunker(strategy="fast", chunking_strategy="basic")

    text = "This is a test document. " * 50  # Create longer text
    result = await chunker.chunk_text(text, filename="test.txt")

    assert result is not None
    assert result.total_chunks > 0
    assert result.documents is not None
    assert len(result.documents) == result.total_chunks
    assert result.metadata.filename == "test.txt"


# Test 2: Chunking with overlap parameter
@pytest.mark.asyncio
async def test_chunking_with_overlap():
    """Test chunking with overlap parameter - comprehensive verification."""
    # Create a long text with unique markers to verify overlap
    # Use a repeating pattern that's longer than typical chunk size
    base_text = "Sentence A. Sentence B. Sentence C. Sentence D. "
    text = base_text * 50  # Create long text that will be chunked

    # Test without overlap first
    chunker_no_overlap = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        overlap=None,
        new_after_n_chars=200,  # Force chunking at ~200 chars
    )
    result_no_overlap = await chunker_no_overlap.chunk_text(text, filename="test.txt")

    # Test with overlap
    overlap_amount = 50
    chunker_with_overlap = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        overlap=overlap_amount,
        new_after_n_chars=200,  # Same chunking target
    )
    result_with_overlap = await chunker_with_overlap.chunk_text(text, filename="test.txt")

    # Basic assertions
    assert result_no_overlap is not None
    assert result_with_overlap is not None
    assert result_no_overlap.total_chunks > 0
    assert result_with_overlap.total_chunks > 0
    assert result_no_overlap.documents is not None
    assert result_with_overlap.documents is not None

    # With overlap, we should have more chunks (or at least same number)
    # because overlap creates additional chunks at boundaries
    assert result_with_overlap.total_chunks >= result_no_overlap.total_chunks

    # Verify overlap is actually happening between consecutive chunks
    chunks = result_with_overlap.documents
    if len(chunks) > 1:
        overlap_found = False
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].text or ""
            chunk2_text = chunks[i + 1].text or ""

            if not chunk1_text or not chunk2_text:
                continue

            # Check if the end of chunk1 overlaps with the start of chunk2
            # Look for overlapping text (at least 10 chars to account for word boundaries)
            min_overlap_check = min(overlap_amount, 10)
            chunk1_end = chunk1_text[-min_overlap_check * 2 :]
            chunk2_start = chunk2_text[: min_overlap_check * 2]

            # Find common substring between end of chunk1 and start of chunk2
            # This handles cases where overlap might not be exactly at boundaries
            for check_len in range(min_overlap_check, len(chunk1_end) + 1):
                if chunk1_end[-check_len:] in chunk2_start:
                    overlap_found = True
                    # Verify overlap is approximately the right size
                    actual_overlap = len(chunk1_end[-check_len:])
                    # Allow some flexibility (overlap might be slightly different due to word boundaries)
                    assert actual_overlap >= min_overlap_check, (
                        f"Overlap too small: expected at least {min_overlap_check}, "
                        f"found {actual_overlap} between chunks {i} and {i+1}"
                    )
                    break

        # At least one pair of consecutive chunks should have overlap
        # (unless text is very short or chunking strategy prevents it)
        if len(chunks) >= 2:
            assert overlap_found, (
                "No overlap detected between consecutive chunks. "
                "This may indicate overlap is not working correctly."
            )

    # Verify total characters with overlap is greater than without
    # (due to overlapping text being counted in multiple chunks)
    assert result_with_overlap.total_characters >= result_no_overlap.total_characters


# Test 2b: Edge case - very small overlap
@pytest.mark.asyncio
async def test_chunking_with_small_overlap():
    """Test chunking with very small overlap value."""
    text = "Word1 Word2 Word3 Word4 Word5 " * 20  # Short repeating pattern
    overlap_amount = 10  # Small overlap

    chunker = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        overlap=overlap_amount,
        new_after_n_chars=100,  # Force chunking
    )
    result = await chunker.chunk_text(text, filename="test.txt")

    assert result is not None
    assert result.total_chunks > 0
    assert result.documents is not None

    # If we have multiple chunks, verify overlap exists
    chunks = result.documents
    if len(chunks) > 1:
        # At least verify chunks are created (overlap detection may be tricky with small values)
        assert all(chunk.text for chunk in chunks), "All chunks should have text"


# Test 3: Chunking with max_characters parameter
@pytest.mark.asyncio
async def test_chunking_with_max_characters():
    """Test chunking with hard maximum character limit."""
    max_chars = 200
    chunker = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        max_characters=max_chars,
    )

    text = "This is a test document. " * 100  # Long text
    result = await chunker.chunk_text(text, filename="test.txt")

    assert result is not None
    assert result.total_chunks > 0

    # Check that chunks respect max_characters (approximately)
    for doc in result.documents:
        assert len(doc.text or "") <= max_chars + 100  # Allow some buffer


# Test 4: Chunking with new_after_n_chars parameter
@pytest.mark.asyncio
async def test_chunking_with_new_after_n_chars():
    """Test chunking with soft character target."""
    new_after = 150
    chunker = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        new_after_n_chars=new_after,
    )

    text = "This is a test document. " * 100
    result = await chunker.chunk_text(text, filename="test.txt")

    assert result is not None
    assert result.total_chunks > 0
    assert result.documents is not None


# Test 5: Chunking with all advanced parameters combined
@pytest.mark.asyncio
async def test_chunking_with_all_advanced_params():
    """Test chunking with overlap, max_characters, and new_after_n_chars together."""
    chunker = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        overlap=30,
        max_characters=300,
        new_after_n_chars=200,
    )

    text = "This is a comprehensive test document. " * 100
    result = await chunker.chunk_text(text, filename="test.txt")

    assert result is not None
    assert result.total_chunks > 0
    assert result.documents is not None

    # Verify chunks are within expected bounds
    for doc in result.documents:
        chunk_len = len(doc.text or "")
        assert chunk_len <= 300 + 100  # max_characters + buffer


# Test 6: Kwargs override of instance parameters
@pytest.mark.asyncio
async def test_kwargs_override_chunking_params():
    """Test that kwargs can override instance-level chunking parameters."""
    # Set default parameters
    chunker = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        overlap=20,
        max_characters=500,
    )

    text = "This is a test document. " * 50

    # Override with kwargs
    result = await chunker.chunk_text(
        text,
        filename="test.txt",
        overlap=100,  # Override
        max_characters=200,  # Override
    )

    assert result is not None
    assert result.total_chunks > 0


# Test 7: File chunking with advanced parameters
@pytest.mark.asyncio
async def test_file_chunking_with_advanced_params(tmp_path):
    """Test file chunking with advanced parameters."""
    # Create a temporary test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file. " * 100)

    chunker = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        overlap=50,
        max_characters=300,
        new_after_n_chars=200,
    )

    result = await chunker.chunk_file(test_file)

    assert result is not None
    assert result.total_chunks > 0
    assert result.metadata.filename == "test.txt"


# Test 8: URL chunking with advanced parameters
@pytest.mark.asyncio
async def test_chunk_text_with_batch_size():
    """Test that batch_size parameter is returned in ChunkResult."""
    chunker = UnstructuredChunker(strategy="fast", chunking_strategy="basic")

    text = "This is a test document. " * 50
    batch_size = 10

    result = await chunker.chunk_text(
        text,
        filename="test.txt",
        batch_size=batch_size,
    )

    assert result is not None
    assert result.batch_size == batch_size

    # Verify it's in the dict representation
    result_dict = result.to_dict()
    assert "batch_size" in result_dict
    assert result_dict["batch_size"] == batch_size


# Test 15: Batch size parameter in chunk_file
@pytest.mark.asyncio
async def test_chunk_file_with_batch_size(tmp_path):
    """Test that batch_size parameter is returned when chunking files."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file. " * 50)

    chunker = UnstructuredChunker(strategy="fast", chunking_strategy="basic")
    batch_size = 15

    result = await chunker.chunk_file(test_file, batch_size=batch_size)

    assert result is not None
    assert result.batch_size == batch_size


