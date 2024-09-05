import pytest
from mem0.configs.vector_stores.milvus import MilvusConfig

@pytest.fixture
def valid_config():
    """Fixture for a valid Milvus configuration."""
    return MilvusConfig(
        cluster_endpoint="localhost:19530",
        token="your_token",
        dimension=128,
        collection_name="test_collection"
    )

def test_milvus_config_valid(valid_config):
    """Test that a valid configuration initializes correctly."""
    assert valid_config.cluster_endpoint == "localhost:19530"
    assert valid_config.token == "your_token"
    assert valid_config.dimension == 128
    assert valid_config.collection_name == "test_collection"
    assert valid_config.metric_type == "IP"  # Default value
    assert valid_config.auto_id is True  # Default value

def test_milvus_config_invalid_dimension():
    """Test that an invalid dimension raises a ValueError."""
    with pytest.raises(ValueError, match="Dimension must be greater than 0."):
        MilvusConfig(
            cluster_endpoint="localhost:19530",
            token="your_token",
            dimension=0,
            collection_name="test_collection"
        )

def test_milvus_config_missing_endpoint():
    """Test that missing cluster_endpoint raises a ValueError."""
    with pytest.raises(ValueError, match="Both 'cluster_endpoint' and 'token' must be provided."):
        MilvusConfig(
            token="your_token",
            dimension=128,
            collection_name="test_collection"
        )

def test_milvus_config_missing_token():
    """Test that missing token raises a ValueError."""
    with pytest.raises(ValueError, match="Both 'cluster_endpoint' and 'token' must be provided."):
        MilvusConfig(
            cluster_endpoint="localhost:19530",
            dimension=128,
            collection_name="test_collection"
        )

def test_milvus_config_empty_collection_name():
    """Test that an empty collection name raises a ValueError."""
    with pytest.raises(ValueError, match="Collection name must not be empty."):
        MilvusConfig(
            cluster_endpoint="localhost:19530",
            token="your_token",
            dimension=128,
            collection_name=""
        )

def test_milvus_config_default_values(valid_config):
    """Test that default values are set correctly."""
    assert valid_config.enable_dynamic_field is True  # Default value
    assert valid_config.index_type == "AUTOINDEX"  # Default value

def test_milvus_config_invalid_token():
    """Test that an invalid token raises a ValueError."""
    with pytest.raises(ValueError, match="Both 'cluster_endpoint' and 'token' must be provided."):
        MilvusConfig(
            cluster_endpoint="localhost:19530",
            token="",  # Invalid token
            dimension=128,
            collection_name="test_collection"
        )

def test_milvus_config_edge_case_large_dimension():
    """Test that a large dimension value initializes correctly."""
    config = MilvusConfig(
        cluster_endpoint="localhost:19530",
        token="your_token",
        dimension=10000,  # Large dimension
        collection_name="test_collection"
    )
    assert config.dimension == 10000

def test_milvus_config_edge_case_special_characters():
    """Test that special characters in collection name are handled."""
    config = MilvusConfig(
        cluster_endpoint="localhost:19530",
        token="your_token",
        dimension=128,
        collection_name="test_collection_!@#"
    )
    assert config.collection_name == "test_collection_!@#"

# Add more tests as necessary to cover additional edge cases or requirements.