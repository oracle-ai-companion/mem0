import subprocess
import sys
from typing import Optional, ClassVar, Dict, Any
from pydantic import BaseModel, Field, model_validator

# Import DataType as a ClassVar
try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    user_input: Any = input("The 'pymilvus' library is required. Install it now? [y/N]: ")
    if user_input.lower() == 'y':
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pymilvus"])
            from pymilvus import MilvusClient, DataType
        except subprocess.CalledProcessError:
            print("Failed to install 'pymilvus'. Please install it manually using 'pip install pymilvus'.")
            sys.exit(1)
    else:
        print("The required 'pymilvus' library is not installed.")
        sys.exit(1)

class MilvusConfig(BaseModel):
    MilvusClient: ClassVar[type] = MilvusClient
    DataType: ClassVar[type] = DataType  # Annotate DataType as ClassVar

    cluster_endpoint: str = Field(..., description="The endpoint for the Milvus/Zilliz cluster")
    token: str = Field(..., description="The token for authenticating with the Milvus/Zilliz cluster")
    collection_name: str = Field("default_collection", description="The name of the collection in Milvus/Zilliz")
    dimension: int = Field(..., description="The dimension of the vectors to be stored (must be > 0)")
    metric_type: str = Field("IP", description="The metric type for vector similarity (e.g., 'IP', 'L2', 'COSINE')")
    enable_dynamic_field: bool = Field(True, description="Enable dynamic fields for additional metadata")
    auto_id: bool = Field(True, description="Automatically generate IDs for inserted vectors")
    index_type: str = Field("AUTOINDEX", description="Index type for Zilliz; use 'AUTOINDEX' for Zilliz and other types for Milvus")
    
    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        dimension = values.get("dimension")
        if dimension <= 0:
            raise ValueError("Dimension must be greater than 0.")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_cluster_endpoint_and_token(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = values.get("cluster_endpoint")
        token = values.get("token")
        if not endpoint or not token:
            raise ValueError("Both 'cluster_endpoint' and 'token' must be provided.")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_collection_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        collection_name = values.get("collection_name")
        if not collection_name:
            raise ValueError("Collection name must not be empty.")
        return values

    model_config = {
        "arbitrary_types_allowed": True,
    }
