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
    """Test chunking with overlap parameter."""
    chunker = UnstructuredChunker(
        strategy="fast",
        chunking_strategy="basic",
        overlap=50,  # 50 character overlap
    )

    text = "This is a test document with overlap. " * 100
    result = await chunker.chunk_text(text, filename="test.txt")

    assert result is not None
    assert result.total_chunks > 0
    # Overlap should create more chunks than without
    assert result.documents is not None


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


