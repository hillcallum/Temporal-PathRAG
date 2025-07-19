"""
Answer extraction for Temporal PathRAG evaluation
"""

import re
import spacy
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AnswerExtractor:
    """
    Answer extractor that handles PathRAG responses
    """
    
    def __init__(self):
        """Initialise the answer extractor"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            logger.warning("SpaCy not available - using pattern-based extraction")
            self.nlp = None
            self.use_spacy = False
        
        # Common patterns that indicate "no answer found"
        self.no_answer_patterns = [
            r"no (?:relevant )?(?:information|data|evidence|answer)",
            r"(?:could not|cannot|unable to) (?:find|determine|identify)",
            r"insufficient (?:information|evidence|data)",
            r"not (?:found|available|present) in (?:the )?(?:knowledge graph|data)",
            r"no (?:paths?|connections?|relationships?) (?:were )?found",
            r"empty (?:result|response|answer)",
            r"(?:sorry|unfortunately)",
            r"I (?:don't|do not) (?:have|know)",
        ]
        
        # Answer indicator patterns with improved capture groups
        self.answer_patterns = {
            'entity': [
                r"(?:the answer is|answer:|is|was|were)\s+([A-Z][^.,;!?\n]+)",
                r"(?:specifically|namely|particularly),?\s+([A-Z][^.,;!?\n]+)",
                r"(?:identified|found|discovered)\s+([A-Z][^.,;!?\n]+)",
                r"(?:entity|person|organization|place):\s*([^.,;!?\n]+)",
                r"^([A-Z][^.,;!?\n]+?)(?:\s+(?:is|was|were))",
            ],
            'time': [
                r"(?:in|during|on|at)\s+(\d{4})",
                r"(?:year|date|time)(?:\s+is)?:?\s*(\d{4})",
                r"(\d{4})(?:\s+(?:is|was|were))",
                r"(?:happened|occurred|took place)\s+(?:in|on|during)\s+(\d{4})",
            ],
            'value': [
                r"(?:count|number|amount|value)(?:\s+is)?:?\s*([\d,]+\.?\d*)",
                r"(?:total|sum)(?:\s+is)?:?\s*([\d,]+\.?\d*)",
                r"(\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:items?|entities|objects?)",
            ]
        }
        
    def extract_answers_from_response(self, 
                                    text: str, 
                                    question: str,
                                    answer_type: str = "auto") -> List[str]:
        """
        Extract answers from PathRAG response 
        """
        if not text:
            return []
        
        # Check if this is a "no answer" response
        if self.is_no_answer_response(text):
            logger.debug("Detected 'no answer found' response")
            return []
        
        # Determine answer type if auto
        if answer_type == "auto":
            answer_type = self.infer_answer_type(question)
            logger.debug(f"Inferred answer type: {answer_type}")
        
        # Extract based on type
        answers = []
        
        if answer_type == "entity":
            answers = self.extract_entity_answers(text, question)
        elif answer_type == "time":
            answers = self.extract_temporal_answers(text, question)
        elif answer_type == "value":
            answers = self.extract_value_answers(text, question)
        else:
            # Try all methods
            answers.extend(self.extract_entity_answers(text, question))
            answers.extend(self.extract_temporal_answers(text, question))
            answers.extend(self.extract_value_answers(text, question))
        
        # Remove duplicates while preserving order
        unique_answers = []
        seen = set()
        for ans in answers:
            normalised = self._normalise_answer(ans)
            if normalised not in seen:
                seen.add(normalised)
                unique_answers.append(ans)
        
        # If still no answers, try more aggressive extraction
        if not unique_answers:
            unique_answers = self.extract_fallback_answers(text, answer_type)
        
        return unique_answers[:10]  # Limit to top 10
    
    def is_no_answer_response(self, text: str) -> bool:
        """Check if response indicates no answer was found"""
        text_lower = text.lower()
        
        for pattern in self.no_answer_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for very short responses that might indicate failure
        if len(text.strip()) < 20:
            return True
        
        return False
    
    def infer_answer_type(self, question: str) -> str:
        """Infer expected answer type from question"""
        q_lower = question.lower()
        
        # Time questions
        time_keywords = ['when', 'what year', 'what date', 'which year', 'what time', 'how long ago']
        if any(kw in q_lower for kw in time_keywords):
            return "time"
        
        # Value questions
        value_keywords = ['how many', 'how much', 'count', 'number', 'amount', 'total']
        if any(kw in q_lower for kw in value_keywords):
            return "value"
        
        # Default to entity
        return "entity"
    
    def extract_entity_answers(self, text: str, question: str) -> List[str]:
        """Extract entity answers with improved patterns"""
        entities = []
        
        # Use answer patterns
        for pattern in self.answer_patterns['entity']:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if self._is_valid_entity(cleaned):
                    entities.append(cleaned)
        
        # Use SpaCy if available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"]:
                    if self._is_valid_entity(ent.text):
                        entities.append(ent.text)
        
        # Look for capitalised sequences
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(cap_pattern, text)
        for match in matches:
            if self._is_valid_entity(match) and len(match.split()) <= 5:
                entities.append(match)
        
        return entities
    
    def extract_temporal_answers(self, text: str, question: str) -> List[str]:
        """Extract temporal answers with improved patterns"""
        times = []
        
        # Year patterns
        year_patterns = [
            r'\b(1\d{3}|2\d{3})\b',  # 1000-2999
            r'(?:year|in|during)\s+(\d{4})',
            r'(\d{4})(?:\s+(?:CE|AD|BC|BCE))?',
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        
        # Use answer patterns
        for pattern in self.answer_patterns['time']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        
        # SpaCy dates
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    # Extract year from date if present
                    year_match = re.search(r'\b(\d{4})\b', ent.text)
                    if year_match:
                        times.append(year_match.group(1))
                    else:
                        times.append(ent.text)
        
        # Clean and validate
        valid_times = []
        for t in times:
            if re.match(r'^\d{4}$', str(t)):
                year = int(t)
                if 1000 <= year <= 2100:  # Reasonable year range
                    valid_times.append(str(year))
        
        return valid_times
    
    def extract_value_answers(self, text: str, question: str) -> List[str]:
        """Extract numeric value answers"""
        values = []
        
        # Number patterns
        number_patterns = [
            r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b',
            r'(?:total|count|number)(?:\s+is)?:?\s*(\d+)',
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            values.extend(matches)
        
        # Use answer patterns
        for pattern in self.answer_patterns['value']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            values.extend(matches)
        
        # Clean values
        clean_values = []
        for v in values:
            # Remove commas for consistency
            clean_v = v.replace(',', '')
            if clean_v.replace('.', '').isdigit():
                clean_values.append(v)  # Keep original formatting
        
        return clean_values
    
    def extract_fallback_answers(self, text: str, answer_type: str) -> List[str]:
        """Fallback extraction when other methods fail"""
        answers = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Look for short phrases that might be answers
            if answer_type == "entity":
                # Look for capitalised words at sentence start
                words = sent.split()
                if words and words[0][0].isupper():
                    candidate = []
                    for word in words:
                        if word[0].isupper() or word.lower() in ['of', 'the', 'and']:
                            candidate.append(word)
                        else:
                            break
                    if candidate and len(candidate) <= 5:
                        answers.append(' '.join(candidate))
            
            elif answer_type == "time":
                # Extract any 4-digit numbers
                year_matches = re.findall(r'\b(\d{4})\b', sent)
                answers.extend(year_matches)
            
            elif answer_type == "value":
                # Extract any numbers
                num_matches = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', sent)
                answers.extend(num_matches)
        
        return answers
    
    def is_valid_entity(self, entity: str) -> bool:
        """Check if extracted entity is valid"""
        if not entity or len(entity) < 2:
            return False
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their',
            'answer', 'question', 'query', 'based', 'evidence', 'information'
        }
        
        entity_lower = entity.lower()
        
        # Check if it's entirely stop words
        words = entity_lower.split()
        if all(w in stop_words for w in words):
            return False
        
        # Must have at least one capitalised word (for entities)
        if not any(w[0].isupper() for w in entity.split() if w):
            return False
        
        # Reasonable length
        if len(entity) > 100:
            return False
        
        return True
    
    def normalise_answer(self, answer: str) -> str:
        """Normalise answer for deduplication"""
        # Basic normalisation
        normalised = answer.strip().lower()
        
        # Remove articles
        normalised = re.sub(r'^(the|a|an)\s+', '', normalised)
        
        # Remove extra whitespace
        normalised = ' '.join(normalised.split())
        
        return normalised
    
    def check_answer_match(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold answer"""
        pred_lower = predicted.lower()
        gold_lower = gold.lower()
        
        # Exact match
        if pred_lower == gold_lower:
            return True
        
        # Substring match
        if gold_lower in pred_lower or pred_lower in gold_lower:
            return True
        
        # Handle year matching (allow close years)
        if re.match(r'^\d{4}$', predicted) and re.match(r'^\d{4}$', gold):
            pred_year = int(predicted)
            gold_year = int(gold)
            if abs(pred_year - gold_year) <= 1:  # Allow 1 year difference
                return True
        
        # Handle last name matching
        pred_parts = pred_lower.split()
        gold_parts = gold_lower.split()
        
        if len(pred_parts) > 1 and len(gold_parts) == 1:
            if gold_parts[0] == pred_parts[-1]:  # Last name match
                return True
        elif len(gold_parts) > 1 and len(pred_parts) == 1:
            if pred_parts[0] == gold_parts[-1]:  # Last name match
                return True
        
        return False