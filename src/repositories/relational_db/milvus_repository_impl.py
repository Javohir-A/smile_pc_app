from typing import List, Optional, Dict, Any
import uuid
import json
import logging
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType
from src.repositories.face_repository import FaceRepository
from src.models.face_embedding import FaceEmbedding, SearchResult
from src.config.external import MilvusConfig, MilvusIndexConfig

logger = logging.getLogger(__name__)

class MilvusFaceRepository(FaceRepository):
    def __init__(self, config: MilvusConfig):
        self.collection_name = config.collection
        self.embedding_dim = config.embedding_dim
        self.host = config.host
        self.port = config.port
        self._collection = None
        self._index_name = config.index.name
        self._metric_type = config.index.metric_type
        self._index_type = config.index.index_type
        self._params = config.index.params
        
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize connection to Milvus and create collection if needed"""
        connections.connect("default", host=self.host, port=self.port)
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        else:
            self._collection = Collection(self.collection_name)

            # Ensure index exists
            indexes = self._collection.indexes
            if not indexes:  # If index list is empty
                self._collection.create_index(
                    field_name="face_embedding",
                    index_params={
                        "metric_type": self._metric_type,
                        "index_type": self._index_type,
                        "params": self._params
                    }
                )
                
        self._collection.load()
        logger.info(f"Collection {self.collection_name} loaded successfully")

    def _create_collection(self):
        """Create the face embeddings collection in Milvus"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="human_guid", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="face_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(name="human_type", dtype=DataType.VARCHAR, max_length=30)
        ]
        
        schema = CollectionSchema(fields, "Face embeddings collection")
        self._collection = Collection(self.collection_name, schema)
        
        # Create index for vector search
        index_params = {
            "metric_type": self._metric_type,
            "index_type": self._index_type,
            "params": self._params
        }
        self._collection.create_index("face_embedding", index_params)
        logger.info(f"Collection {self.collection_name} created successfully")
    
    def _serialize_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Serialize metadata dict to JSON string"""
        if metadata is None:
            return "{}"
        return json.dumps(metadata)
    
    def _deserialize_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Deserialize JSON string to metadata dict"""
        if not metadata_str:
            return {}
        try:
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def create(self, human_guid: str, name: str, human_type: str, face_embedding: List[float], 
               metadata: Optional[Dict[str, Any]] = None) -> FaceEmbedding:
        
        if len(face_embedding) != self.embedding_dim:
            raise ValueError(f"Face embedding must have {self.embedding_dim} dimensions")
        
        face_emb = FaceEmbedding.create_new(human_guid, name, human_type, face_embedding, metadata)
        
        # FIXED: Use proper column format for Milvus insert
        # Schema fields order: id (auto), human_guid, name, face_embedding, created_at, metadata, human_type
        insert_data = [
            [face_emb.human_guid],                                # human_guid column
            [face_emb.name],                                     # name column  
            [face_emb.face_embedding],                           # face_embedding column
            [face_emb.created_at],                               # created_at column
            [self._serialize_metadata(face_emb.metadata)],       # metadata column
            [face_emb.human_type]                                # human_type column
        ]
        
        logger.debug(f"Inserting data for {human_guid}: name={name}, type={human_type}")
        
        try:
            result = self._collection.insert(insert_data)
            self._collection.flush()
            
            # Get the auto-generated ID from the insert result
            if result.primary_keys:
                face_emb.id = str(result.primary_keys[0])  # Convert to string for consistency
                logger.info(f"Successfully inserted face with ID: {face_emb.id}")
            else:
                logger.error("No primary key returned from insert")
                
        except Exception as e:
            logger.error(f"Error inserting face data: {e}")
            raise
        
        return face_emb
    
    def get_by_id(self, record_id: int) -> Optional[FaceEmbedding]:
        """Get face embedding by ID with improved error handling"""
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            expr = f'id == {record_id}'
            logger.debug(f"Querying with expression: {expr}")
            
            results = self._collection.query(
                expr=expr,
                output_fields=["id", "human_guid", "name", "human_type", "face_embedding", "created_at", "metadata"]
            )
            
            logger.debug(f"Query results for ID {record_id}: {len(results) if results else 0} records found")
            
            if not results:
                logger.warning(f"No face found with ID: {record_id}")
                return None
            
            record = results[0]
            logger.debug(f"Found record: {record.keys() if hasattr(record, 'keys') else type(record)}")
            
            return FaceEmbedding(
                id=str(record["id"]),  # Convert to string for consistency
                human_guid=record["human_guid"],
                name=record["name"],
                human_type=record["human_type"],
                face_embedding=[float(x) for x in record["face_embedding"]],
                created_at=record["created_at"],
                metadata=self._deserialize_metadata(record.get("metadata", "{}"))
            )
            
        except Exception as e:
            logger.error(f"Error getting face by ID {record_id}: {e}")
            return None
    
    def get_by_human_guid(self, human_guid: str) -> List[FaceEmbedding]:
        """Get all faces for a specific human_guid"""
        try:
            expr = f'human_guid == "{human_guid}"'
            logger.debug(f"Querying faces for human_guid: {human_guid}")
            
            results = self._collection.query(
                expr=expr,
                output_fields=["id", "human_guid", "name", "human_type", "face_embedding", "created_at", "metadata"]
            )
            
            logger.debug(f"Found {len(results)} faces for human_guid: {human_guid}")
            
            return [
                FaceEmbedding(
                    id=str(record["id"]),
                    human_guid=record["human_guid"],
                    name=record["name"],
                    human_type=record["human_type"],
                    face_embedding=[float(x) for x in record["face_embedding"]],
                    created_at=record["created_at"],
                    metadata=self._deserialize_metadata(record.get("metadata", "{}"))
                )
                for record in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting faces for human_guid {human_guid}: {e}")
            return []
    
    def update(self, record_id: int, name: Optional[str] = None, human_type: Optional[str] = None, 
               face_embedding: Optional[List[float]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Optional[FaceEmbedding]:
        
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            # Get existing record
            existing = self.get_by_id(record_id)
            if not existing:
                logger.warning(f"No existing face found with ID: {record_id}")
                return None
            
            # Update fields
            updated_name = name if name is not None else existing.name
            updated_embedding = face_embedding if face_embedding is not None else existing.face_embedding
            updated_human_type = human_type if human_type is not None else existing.human_type
            updated_metadata = metadata if metadata is not None else existing.metadata
            
            if face_embedding and len(face_embedding) != self.embedding_dim:
                raise ValueError(f"Face embedding must have {self.embedding_dim} dimensions")
            
            # Delete existing record
            expr = f'id == {record_id}'
            self._collection.delete(expr)
            self._collection.flush()
            
            logger.debug(f"Deleted existing record with ID: {record_id}")
            
            # Insert new record using list format in correct order
            insert_data = [
                [existing.human_guid],                               # human_guid
                [updated_name],                                      # name
                [updated_embedding],                                 # face_embedding  
                [existing.created_at],                               # created_at
                [self._serialize_metadata(updated_metadata)],        # metadata
                [updated_human_type]                                 # human_type
            ]
            
            result = self._collection.insert(insert_data)
            self._collection.flush()
            
            # Get the new auto-generated ID
            new_id = result.primary_keys[0] if result.primary_keys else None
            
            logger.info(f"Updated face, new ID: {new_id}")
            
            return FaceEmbedding(
                id=str(new_id),
                human_guid=existing.human_guid,
                name=updated_name,
                human_type=updated_human_type,
                face_embedding=updated_embedding,
                created_at=existing.created_at,
                metadata=updated_metadata
            )
            
        except Exception as e:
            logger.error(f"Error updating face with ID {record_id}: {e}")
            return None
    
    def delete(self, record_id: int) -> bool:
        """Delete face by ID with improved expression handling"""
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            # Check if record exists first
            if not self.exists(record_id):
                logger.warning(f"Cannot delete: No face found with ID {record_id}")
                return False
            
            # Use proper expression format for large integers
            expr = f'id in [{record_id}]'  # Use 'in' instead of '=='
            logger.debug(f"Deleting with expression: {expr}")
            
            try:
                self._collection.delete(expr)
                self._collection.flush()
                
                # Verify deletion
                if not self.exists(record_id):
                    logger.info(f"Successfully deleted face with ID: {record_id}")
                    return True
                else:
                    logger.error(f"Delete operation completed but record still exists: {record_id}")
                    return False
                    
            except Exception as delete_error:
                logger.error(f"Milvus delete error for ID {record_id}: {delete_error}")
                
                # Try alternative delete method using string expression
                try:
                    str_expr = f'id == "{record_id}"'
                    logger.debug(f"Trying alternative expression: {str_expr}")
                    self._collection.delete(str_expr)
                    self._collection.flush()
                    
                    if not self.exists(record_id):
                        logger.info(f"Successfully deleted face with alternative method: {record_id}")
                        return True
                        
                except Exception as alt_error:
                    logger.error(f"Alternative delete method also failed: {alt_error}")
                    return False
            
        except Exception as e:
            logger.error(f"Error deleting face with ID {record_id}: {e}")
            return False
    
    def get_by_id(self, record_id: int) -> Optional[FaceEmbedding]:
        """Get face embedding by ID with improved query handling"""
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            # Use 'in' operator for better compatibility with large integers
            expr = f'id in [{record_id}]'
            logger.debug(f"Querying with expression: {expr}")
            
            results = self._collection.query(
                expr=expr,
                output_fields=["id", "human_guid", "name", "human_type", "face_embedding", "created_at", "metadata"]
            )
            
            logger.debug(f"Query results for ID {record_id}: {len(results) if results else 0} records found")
            
            if not results:
                logger.warning(f"No face found with ID: {record_id}")
                return None
            
            record = results[0]
            logger.debug(f"Found record: {record.keys() if hasattr(record, 'keys') else type(record)}")
            
            return FaceEmbedding(
                id=str(record["id"]),  # Convert to string for consistency
                human_guid=record["human_guid"],
                name=record["name"],
                human_type=record["human_type"],
                face_embedding=[float(x) for x in record["face_embedding"]],
                created_at=record["created_at"],
                metadata=self._deserialize_metadata(record.get("metadata", "{}"))
            )
            
        except Exception as e:
            logger.error(f"Error getting face by ID {record_id}: {e}")
            return None
    
    def update(self, record_id: int, name: Optional[str] = None, human_type: Optional[str] = None, 
               face_embedding: Optional[List[float]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Optional[FaceEmbedding]:
        
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            # Get existing record
            existing = self.get_by_id(record_id)
            if not existing:
                logger.warning(f"No existing face found with ID: {record_id}")
                return None
            
            # Update fields
            updated_name = name if name is not None else existing.name
            updated_embedding = face_embedding if face_embedding is not None else existing.face_embedding
            updated_human_type = human_type if human_type is not None else existing.human_type
            updated_metadata = metadata if metadata is not None else existing.metadata
            
            if face_embedding and len(face_embedding) != self.embedding_dim:
                raise ValueError(f"Face embedding must have {self.embedding_dim} dimensions")
            
            # Delete existing record using improved expression
            expr = f'id in [{record_id}]'
            logger.debug(f"Deleting existing record with expression: {expr}")
            
            try:
                self._collection.delete(expr)
                self._collection.flush()
                logger.debug(f"Deleted existing record with ID: {record_id}")
            except Exception as delete_error:
                logger.error(f"Error deleting record for update: {delete_error}")
                # Try alternative method
                try:
                    alt_expr = f'id == "{record_id}"'
                    self._collection.delete(alt_expr)
                    self._collection.flush()
                    logger.debug(f"Deleted existing record with alternative method")
                except Exception as alt_error:
                    logger.error(f"Alternative delete for update also failed: {alt_error}")
                    return None
            
            # Insert new record using list format in correct order
            insert_data = [
                [existing.human_guid],                               # human_guid
                [updated_name],                                      # name
                [updated_embedding],                                 # face_embedding  
                [existing.created_at],                               # created_at
                [self._serialize_metadata(updated_metadata)],        # metadata
                [updated_human_type]                                 # human_type
            ]
            
            result = self._collection.insert(insert_data)
            self._collection.flush()
            
            # Get the new auto-generated ID
            new_id = result.primary_keys[0] if result.primary_keys else None
            
            logger.info(f"Updated face, new ID: {new_id}")
            
            return FaceEmbedding(
                id=str(new_id),
                human_guid=existing.human_guid,
                name=updated_name,
                human_type=updated_human_type,
                face_embedding=updated_embedding,
                created_at=existing.created_at,
                metadata=updated_metadata
            )
            
        except Exception as e:
            logger.error(f"Error updating face with ID {record_id}: {e}")
            return None
    
    def exists(self, record_id: int) -> bool:
        """Check if a face exists by ID with improved query"""
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            # Use 'in' operator for better compatibility
            expr = f'id in [{record_id}]'
            results = self._collection.query(expr=expr, output_fields=["id"])
            exists = len(results) > 0
            
            logger.debug(f"Face ID {record_id} exists: {exists}")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking if face exists (ID: {record_id}): {e}")
            return False
    
    def get_embedding_by_id(self, record_id: int) -> Optional[List[float]]:
        """Get only the face embedding vector by ID with improved query"""
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            # Use 'in' operator for better compatibility
            expr = f'id in [{record_id}]'
            results = self._collection.query(
                expr=expr,
                output_fields=["face_embedding"]
            )
            
            if not results:
                logger.warning(f"No embedding found for ID: {record_id}")
                return None
            
            embedding = results[0]["face_embedding"]
            logger.debug(f"Retrieved embedding for ID {record_id}: {len(embedding)} dimensions")
            
            return [float(x) for x in embedding]
            
        except Exception as e:
            logger.error(f"Error getting embedding for ID {record_id}: {e}")
            return None

    def delete_by_human_guid(self, human_guid: str) -> int:
        """Delete all faces for a specific human_guid - useful for cleanup"""
        try:
            # First get all faces for this human_guid
            faces = self.get_by_human_guid(human_guid)
            
            if not faces:
                logger.info(f"No faces found for human_guid: {human_guid}")
                return 0
            
            deleted_count = 0
            for face in faces:
                if self.delete(int(face.id)):
                    deleted_count += 1
                else:
                    logger.warning(f"Failed to delete face ID: {face.id}")
            
            logger.info(f"Deleted {deleted_count}/{len(faces)} faces for human_guid: {human_guid}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting faces for human_guid {human_guid}: {e}")
            return 0

    def force_delete_with_pk_list(self, record_ids: List[int]) -> bool:
        """Force delete using primary key list - alternative method for problematic deletes"""
        try:
            if not record_ids:
                return True
                
            # Convert all to int
            int_ids = [int(rid) for rid in record_ids]
            
            # Use pk list format for delete
            expr = f'id in {int_ids}'
            logger.debug(f"Force deleting with expression: {expr}")
            
            self._collection.delete(expr)
            self._collection.flush()
            
            # Verify deletion
            remaining = 0
            for rid in int_ids:
                if self.exists(rid):
                    remaining += 1
            
            success = remaining == 0
            logger.info(f"Force delete result: {len(int_ids) - remaining}/{len(int_ids)} deleted")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in force delete: {e}")
            return False
    
    def list_all(self, limit: int = 100, offset: int = 0) -> List[FaceEmbedding]:
        """List all faces with pagination"""
        try:
            results = self._collection.query(
                expr='id > 0',
                output_fields=["id", "human_guid", "name", "human_type", "face_embedding", "created_at", "metadata"],
                limit=limit + offset
            )
            
            # Apply offset manually since Milvus doesn't support offset in query
            paginated_results = results[offset:offset+limit] if len(results) > offset else []
            
            logger.debug(f"Listed {len(paginated_results)} faces (offset: {offset}, limit: {limit})")
            
            return [
                FaceEmbedding(
                    id=str(record["id"]),
                    human_guid=record["human_guid"],
                    name=record["name"],
                    human_type=record["human_type"],
                    face_embedding=[float(x) for x in record["face_embedding"]],                
                    created_at=record["created_at"],
                    metadata=self._deserialize_metadata(record.get("metadata", "{}"))
                )
                for record in paginated_results
            ]
            
        except Exception as e:
            logger.error(f"Error listing faces: {e}")
            return []
        
    def search_similar(self, face_embedding: List[float], limit: int = 10, 
                    threshold: float = 50) -> List[SearchResult]:
        """Search for similar faces"""
        
        if len(face_embedding) != self.embedding_dim:
            raise ValueError(f"Face embedding must have {self.embedding_dim} dimensions")
        
        search_params = {
            "metric_type": "L2",  
            "params": {"nprobe": 10}
        }
        
        try:
            logger.debug(f"Searching for similar faces with threshold: {threshold}")
            
            results = self._collection.search(
                data=[face_embedding],
                anns_field="face_embedding",
                param=search_params,
                limit=limit,
                output_fields=["id", "human_guid", "name", "human_type", "metadata"]
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    if hit.score <= threshold:  # L2 distance threshold
                        entity = hit.entity
                        
                        # Improved entity data extraction
                        try:
                            # Try direct attribute access first
                            entity_id = str(getattr(entity, 'id', None))
                            entity_human_guid = getattr(entity, 'human_guid', None)
                            entity_name = getattr(entity, 'name', None)
                            entity_human_type = getattr(entity, 'human_type', None)
                            entity_metadata = getattr(entity, 'metadata', "{}")
                            
                        except AttributeError:
                            # Fallback to dictionary access
                            try:
                                entity_id = str(entity.get("id", None))
                                entity_human_guid = entity.get("human_guid", None)
                                entity_name = entity.get("name", None)
                                entity_human_type = entity.get("human_type", None)
                                entity_metadata = entity.get("metadata", "{}")
                            except:
                                logger.warning(f"Could not extract entity data from search result")
                                continue
                        
                        # Calculate recognition confidence (1 - normalized distance)
                        recognition_confidence = max(0.0, 1.0 - (hit.distance / 2.0))  # Normalize L2 distance
                        
                        search_results.append(SearchResult(
                            id=entity_id,
                            human_guid=entity_human_guid,
                            name=entity_name,
                            human_type=entity_human_type,
                            similarity_score=hit.score,
                            distance=hit.distance,
                            recognition_confidence=recognition_confidence,
                            metadata=self._deserialize_metadata(entity_metadata)
                        ))
            
            logger.debug(f"Found {len(search_results)} similar faces")
            return search_results

        except Exception as e:
            logger.error(f"Error searching for similar faces: {e}")
            raise Exception(f"Failed to search faces: {str(e)}")
    
    def exists(self, record_id: int) -> bool:
        """Check if a face exists by ID"""
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            expr = f'id == {record_id}'
            results = self._collection.query(expr=expr, output_fields=["id"])
            exists = len(results) > 0
            
            logger.debug(f"Face ID {record_id} exists: {exists}")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking if face exists (ID: {record_id}): {e}")
            return False
    
    def count_by_human_guid(self, human_guid: str) -> int:
        """Count faces for a specific human_guid"""
        try:
            expr = f'human_guid == "{human_guid}"'
            results = self._collection.query(expr=expr, output_fields=["id"])
            count = len(results)
            
            logger.debug(f"Found {count} faces for human_guid: {human_guid}")
            return count
            
        except Exception as e:
            logger.error(f"Error counting faces for human_guid {human_guid}: {e}")
            return 0
    
    def get_embedding_by_id(self, record_id: int) -> Optional[List[float]]:
        """Get only the face embedding vector by ID"""
        try:
            # Convert to int if string is passed
            if isinstance(record_id, str):
                record_id = int(record_id)
                
            expr = f'id == {record_id}'
            results = self._collection.query(
                expr=expr,
                output_fields=["face_embedding"]
            )
            
            if not results:
                logger.warning(f"No embedding found for ID: {record_id}")
                return None
            
            embedding = results[0]["face_embedding"]
            logger.debug(f"Retrieved embedding for ID {record_id}: {len(embedding)} dimensions")
            
            return [float(x) for x in embedding]
            
        except Exception as e:
            logger.error(f"Error getting embedding for ID {record_id}: {e}")
            return None

    def debug_collection_info(self):
        """Debug method to check collection info"""
        try:
            logger.info(f"Collection name: {self.collection_name}")
            logger.info(f"Collection schema: {self._collection.schema}")
            
            # Get collection statistics
            stats = self._collection.get_stats()
            logger.info(f"Collection stats: {stats}")
            
            # Count total records
            results = self._collection.query(expr='id > 0', output_fields=["id"])
            logger.info(f"Total records in collection: {len(results)}")
            
            # Show first few records
            if results:
                first_few = results[:3]
                logger.info(f"First few record IDs: {[r['id'] for r in first_few]}")
                
        except Exception as e:
            logger.error(f"Error getting collection debug info: {e}")