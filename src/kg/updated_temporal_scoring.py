"""
Updated temporal scoring with embedding-based similarity blending
"""
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
import logging
from dataclasses import dataclass

from .temporal_reliability_scorer import TemporalReliabilityScorer, TemporalReliabilityMetrics
from .temporal_scoring import TemporalWeightingFunction
from ..embeddings.embedding_integration import EmbeddingIntegration
from ..embeddings.temporal_embeddings import TemporalEmbeddings

logger = logging.getLogger(__name__)


@dataclass 
class TemporalScoringMetrics:
    """Metrics that combine reliability and embedding-based scores"""
    reliability_score: float
    embedding_score: float
    combined_score: float
    confidence: float
    reliability_metrics: TemporalReliabilityMetrics


class UpdatedTemporalScorer:
    """
    Updated scorer that blends temporal reliability with learned embeddings
    """
    
    def __init__(self, 
                 graph,
                 temporal_weighting: TemporalWeightingFunction,
                 embedding_integration: Optional[EmbeddingIntegration] = None,
                 temporal_adapter_path: Optional[str] = None,
                 use_gpu: bool = False,
                 reliability_weight: float = 0.6,
                 embedding_weight: float = 0.4,
                 confidence_threshold: float = 0.7):
        """
        Initialise updated temporal scorer
        """
        self.graph = graph
        self.reliability_scorer = TemporalReliabilityScorer(temporal_weighting)
        self.embedding_integration = embedding_integration
        self.temporal_adapter_path = temporal_adapter_path
        self.use_gpu = use_gpu
        
        # Base weights (will be adjusted dynamically based on confidence)
        self.base_reliability_weight = reliability_weight
        self.base_embedding_weight = embedding_weight
        self.confidence_threshold = confidence_threshold
        
        # Initialise trained embeddings if adapter path provided
        self.trained_embeddings = None
        if temporal_adapter_path:
            try:
                # Load trained temporal adapter/embeddings
                logger.info(f"Loading trained temporal adapter from {temporal_adapter_path}")
                # This will take the loaded model, but for now, we'll use the base embeddings
                pass
            except Exception as e:
                logger.warning(f"Failed to load temporal adapter: {e}. Using base embeddings.")
    
    def batch_score_paths(self, 
                         paths: List,
                         query_time: str = None,
                         query_context: Optional[Dict] = None) -> List[Tuple]:
        """
        Score paths using both reliability and embedding-based scoring - main 
        entry point that implements the blending strategy
        """
        # Extract query text from context
        query = query_context.get('query_text', '') if query_context else ''
        
        if not self.embedding_integration:
            # Fallback to pure reliability scoring
            logger.info("No embedding integration available, using reliability scoring only")
            return self.reliability_scorer.rank_paths_by_reliability(
                paths, query_time, query_context or {'query_text': query}
            )
        
        scored_paths = []
        
        # Compute embedding scores for all paths
        try:
            embedding_scores = self.compute_embedding_scores(paths, query)
            embedding_available = True
        except Exception as e:
            logger.warning(f"Failed to compute embedding scores: {e}. Using reliability only.")
            embedding_scores = [0.0] * len(paths)
            embedding_available = False
        
        for i, path in enumerate(paths):
            # Get reliability score
            reliability_metrics = self.reliability_scorer.score_path_reliability(
                path, query_time, query_context or {'query_text': query}
            )
            reliability_score = reliability_metrics.overall_reliability
            
            # Get embedding score
            embedding_score = embedding_scores[i] if embedding_available else 0.0
            
            # Compute confidence based on various factors
            confidence = self.compute_confidence(
                path, reliability_metrics, embedding_score, embedding_available
            )
            
            # Dynamic weight adjustment based on confidence
            weights = self.adjust_weights_by_confidence(confidence, embedding_available)
            
            # Compute combined score
            combined_score = (
                weights['reliability'] * reliability_score +
                weights['embedding'] * embedding_score
            )
            
            # Create updated metrics
            updated_metrics = TemporalScoringMetrics(
                reliability_score=reliability_score,
                embedding_score=embedding_score,
                combined_score=combined_score,
                confidence=confidence,
                reliability_metrics=reliability_metrics
            )
            
            scored_paths.append((path, updated_metrics))
        
        # Sort by combined score
        scored_paths.sort(key=lambda x: x[1].combined_score, reverse=True)
        
        # Log scoring details for debugging
        if logger.isEnabledFor(logging.DEBUG):
            for i, (path, metrics) in enumerate(scored_paths[:5]):
                logger.debug(
                    f"Path {i}: combined={metrics.combined_score:.3f}, "
                    f"reliability={metrics.reliability_score:.3f}, "
                    f"embedding={metrics.embedding_score:.3f}, "
                    f"confidence={metrics.confidence:.3f}"
                )
        
        return scored_paths
    
    def compute_embedding_scores(self, paths: List, query: str) -> List[float]:
        """
        Compute embedding-based similarity scores for all paths - returns normalised scores 
        in [0, 1] range
        """
        # Get raw similarity scores
        raw_scores = self.embedding_integration.batch_compute_path_similarities(
            paths, query=query
        )
        
        # Normalise scores to [0, 1] range
        if not raw_scores:
            return []
        
        # Handle edge cases
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        
        if max_score == min_score:
            # All scores are the same
            return [0.5] * len(raw_scores)
        
        # Min-max normalisation
        normalised_scores = [
            (score - min_score) / (max_score - min_score)
            for score in raw_scores
        ]
        
        return normalised_scores
    
    def compute_confidence(self,
                          path,
                          reliability_metrics: TemporalReliabilityMetrics,
                          embedding_score: float,
                          embedding_available: bool) -> float:
        """
        Compute confidence score for the prediction
        
        Confidence is based on:
        - Whether embeddings are available
        - Path length (shorter paths are more confident)
        - Temporal consistency
        - Number of supporting sources
        """
        confidence_factors = []
        
        # Embedding availability
        if embedding_available:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
        
        # Path length factor (shorter is more confident)
        path_length = len(path.edges)
        if path_length <= 2:
            confidence_factors.append(1.0)
        elif path_length <= 4:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # Temporal consistency from reliability metrics
        if hasattr(reliability_metrics, 'temporal_consistency'):
            confidence_factors.append(reliability_metrics.temporal_consistency)
        
        # High embedding score indicates confidence
        if embedding_available and embedding_score > 0.8:
            confidence_factors.append(1.0)
        elif embedding_available and embedding_score > 0.6:
            confidence_factors.append(0.8)
        
        # Average confidence
        return np.mean(confidence_factors)
    
    def adjust_weights_by_confidence(self,
                                    confidence: float,
                                    embedding_available: bool) -> Dict[str, float]:
        """
        Dynamically adjust scoring weights based on confidence
        
        High confidence: Trust embeddings more
        Low confidence: Trust reliability scoring more
        No embeddings: Use reliability only
        """
        if not embedding_available:
            return {'reliability': 1.0, 'embedding': 0.0}
        
        if confidence >= self.confidence_threshold:
            # High confidence - trust embeddings more
            reliability_weight = self.base_reliability_weight * 0.8
            embedding_weight = self.base_embedding_weight * 1.2
        else:
            # Low confidence - trust reliability more
            reliability_weight = self.base_reliability_weight * 1.2
            embedding_weight = self.base_embedding_weight * 0.8
        
        # Normalise to sum to 1
        total = reliability_weight + embedding_weight
        return {
            'reliability': reliability_weight / total,
            'embedding': embedding_weight / total
        }
    




def create_updated_temporal_scorer(temporal_weighting: TemporalWeightingFunction,
                                  embedding_integration: Optional[EmbeddingIntegration] = None,
                                  temporal_adapter_path: Optional[str] = None,
                                  use_gpu: bool = False) -> Optional[UpdatedTemporalScorer]:
    """
    Creates the updated scorer that blends learned embeddings
    with manual temporal reliability scoring
    """
    try:
        scorer = UpdatedTemporalScorer(
            graph=temporal_weighting.graph if hasattr(temporal_weighting, 'graph') else None,
            temporal_weighting=temporal_weighting,
            embedding_integration=embedding_integration,
            temporal_adapter_path=temporal_adapter_path,
            use_gpu=use_gpu
        )
        logger.info("Successfully created updated temporal scorer")
        return scorer
    except Exception as e:
        logger.error(f"Failed to create updated temporal scorer: {e}")
        # Fallback to None, which will use standard scoring
        return None