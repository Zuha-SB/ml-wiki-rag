"""
Chunking Strategies for RAG Pipeline

This module implements multiple chunking strategies to demonstrate
different approaches and their trade-offs for retrieval.

Strategies:
1. FixedSizeChunker - Split by word/character count with overlap
2. SentenceChunker - Split by sentence boundaries
3. SemanticChunker - Split by semantic coherence (paragraphs/sections)
4. RecursiveChunker - Hierarchical splitting with configurable separators
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import hashlib


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    source_page: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    strategy: Optional[str] = None
    
    def __hash__(self):
        return hash(self.chunk_id)
    
    def __eq__(self, other):
        if isinstance(other, Chunk):
            return self.chunk_id == other.chunk_id
        return False


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def chunk(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{self.name}_{index}_{content_hash}"
    
    def chunk_document(self, pages: List[str]) -> List[Chunk]:
        """Chunk an entire document (list of page texts)."""
        all_chunks = []
        for page_num, page_text in enumerate(pages, start=1):
            chunks = self.chunk(page_text, page_num=page_num)
            all_chunks.extend(chunks)
        return all_chunks


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking with overlap.
    
    Splits text into chunks of approximately equal size with configurable
    overlap to maintain context across chunk boundaries.
    
    Pros: Simple, predictable chunk sizes, good for uniform content
    Cons: May split mid-sentence, loses semantic coherence
    """
    
    def __init__(self, chunk_size: int = 200, overlap: int = 50, unit: str = "words"):
        super().__init__(f"fixed_{chunk_size}_{overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.unit = unit  # "words" or "chars"
    
    def chunk(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        
        text = self._clean_text(text)
        
        if self.unit == "words":
            return self._chunk_by_words(text, page_num)
        else:
            return self._chunk_by_chars(text, page_num)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _chunk_by_words(self, text: str, page_num: Optional[int]) -> List[Chunk]:
        words = text.split()
        chunks = []
        
        i = 0
        chunk_idx = 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=self._generate_chunk_id(chunk_text, chunk_idx),
                source_page=page_num,
                strategy=self.name
            ))
            
            # Move forward by (chunk_size - overlap)
            i += self.chunk_size - self.overlap
            chunk_idx += 1
            
            if i >= len(words):
                break
        
        return chunks
    
    def _chunk_by_chars(self, text: str, page_num: Optional[int]) -> List[Chunk]:
        chunks = []
        i = 0
        chunk_idx = 0
        
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end]
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=self._generate_chunk_id(chunk_text, chunk_idx),
                source_page=page_num,
                start_char=i,
                end_char=end,
                strategy=self.name
            ))
            
            i += self.chunk_size - self.overlap
            chunk_idx += 1
        
        return chunks


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking.
    
    Groups sentences until reaching a target size, respecting sentence boundaries.
    
    Pros: Preserves complete sentences, more coherent chunks
    Cons: Variable chunk sizes, may create very small/large chunks
    """
    
    def __init__(self, max_sentences: int = 5, min_words: int = 50, max_words: int = 300):
        super().__init__(f"sentence_{max_sentences}")
        self.max_sentences = max_sentences
        self.min_words = min_words
        self.max_words = max_words
    
    def chunk(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_sentences = []
        current_word_count = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            word_count = len(sentence.split())
            
            # Check if adding this sentence would exceed limits
            if current_sentences:
                would_exceed_sentences = len(current_sentences) >= self.max_sentences
                would_exceed_words = current_word_count + word_count > self.max_words
                
                if would_exceed_sentences or would_exceed_words:
                    # Save current chunk
                    chunk_text = ' '.join(current_sentences)
                    if len(chunk_text.split()) >= self.min_words:
                        chunks.append(Chunk(
                            text=chunk_text,
                            chunk_id=self._generate_chunk_id(chunk_text, chunk_idx),
                            source_page=page_num,
                            strategy=self.name
                        ))
                        chunk_idx += 1
                    
                    current_sentences = []
                    current_word_count = 0
            
            current_sentences.append(sentence)
            current_word_count += word_count
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text.split()) >= self.min_words // 2:  # Relaxed for final chunk
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=self._generate_chunk_id(chunk_text, chunk_idx),
                    source_page=page_num,
                    strategy=self.name
                ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations and edge cases
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence-ending punctuation followed by space and capital letter
        # or followed by end of string
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        sentences = re.split(sentence_pattern, text)
        
        return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on natural text boundaries.
    
    Identifies semantic boundaries like paragraphs, sections, and topic shifts.
    Uses multiple signals: blank lines, headers, bullet points, etc.
    
    Pros: Preserves semantic coherence, natural boundaries
    Cons: May create very uneven chunk sizes
    """
    
    def __init__(self, min_chunk_words: int = 50, max_chunk_words: int = 400):
        super().__init__("semantic")
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
    
    def chunk(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        
        # Split by semantic boundaries
        segments = self._identify_segments(text)
        
        # Merge small segments and split large ones
        chunks = self._balance_segments(segments, page_num)
        
        return chunks
    
    def _identify_segments(self, text: str) -> List[str]:
        """Identify natural text segments."""
        # Split on paragraph boundaries (multiple newlines or clear breaks)
        # Also split on section headers (lines that look like titles)
        
        # First, normalize whitespace but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this looks like a section header
            lines = para.split('\n')
            if len(lines) == 1 and len(para.split()) < 10 and para.isupper():
                # Likely a header - could mark it specially
                segments.append(para)
            else:
                # Regular paragraph
                segments.append(re.sub(r'\s+', ' ', para))
        
        return segments
    
    def _balance_segments(self, segments: List[str], page_num: Optional[int]) -> List[Chunk]:
        """Merge small segments and split large ones."""
        chunks = []
        current_text = ""
        current_word_count = 0
        chunk_idx = 0
        
        for segment in segments:
            segment_words = len(segment.split())
            
            # If segment alone is too large, split it
            if segment_words > self.max_chunk_words:
                # First, save any accumulated text
                if current_text and current_word_count >= self.min_chunk_words:
                    chunks.append(Chunk(
                        text=current_text.strip(),
                        chunk_id=self._generate_chunk_id(current_text, chunk_idx),
                        source_page=page_num,
                        strategy=self.name
                    ))
                    chunk_idx += 1
                    current_text = ""
                    current_word_count = 0
                
                # Split the large segment
                sub_chunks = self._split_large_segment(segment, page_num, chunk_idx)
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                continue
            
            # Check if adding this segment would exceed max
            if current_word_count + segment_words > self.max_chunk_words:
                if current_word_count >= self.min_chunk_words:
                    chunks.append(Chunk(
                        text=current_text.strip(),
                        chunk_id=self._generate_chunk_id(current_text, chunk_idx),
                        source_page=page_num,
                        strategy=self.name
                    ))
                    chunk_idx += 1
                    current_text = ""
                    current_word_count = 0
            
            # Add segment to current chunk
            if current_text:
                current_text += " " + segment
            else:
                current_text = segment
            current_word_count += segment_words
        
        # Final chunk
        if current_text and current_word_count >= self.min_chunk_words // 2:
            chunks.append(Chunk(
                text=current_text.strip(),
                chunk_id=self._generate_chunk_id(current_text, chunk_idx),
                source_page=page_num,
                strategy=self.name
            ))
        
        return chunks
    
    def _split_large_segment(self, segment: str, page_num: Optional[int], start_idx: int) -> List[Chunk]:
        """Split a segment that's too large."""
        # Use sentence-based splitting for large segments
        sentences = re.split(r'(?<=[.!?])\s+', segment)
        chunks = []
        current = []
        current_count = 0
        chunk_idx = start_idx
        
        for sentence in sentences:
            word_count = len(sentence.split())
            if current_count + word_count > self.max_chunk_words and current:
                chunk_text = ' '.join(current)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=self._generate_chunk_id(chunk_text, chunk_idx),
                    source_page=page_num,
                    strategy=self.name
                ))
                chunk_idx += 1
                current = []
                current_count = 0
            
            current.append(sentence)
            current_count += word_count
        
        if current:
            chunk_text = ' '.join(current)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=self._generate_chunk_id(chunk_text, chunk_idx),
                source_page=page_num,
                strategy=self.name
            ))
        
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking with hierarchical separators.
    
    Tries to split by the most significant separator first (e.g., paragraphs),
    then recursively splits by smaller separators if chunks are still too large.
    
    Inspired by LangChain's RecursiveCharacterTextSplitter.
    
    Pros: Adaptive to content structure, good balance of coherence and size
    Cons: More complex, may be slower
    """
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        super().__init__(f"recursive_{chunk_size}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    
    def chunk(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
        
        chunks = self._split_text(text, self.separators)
        
        # Convert to Chunk objects
        result = []
        for idx, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                result.append(Chunk(
                    text=chunk_text.strip(),
                    chunk_id=self._generate_chunk_id(chunk_text, idx),
                    source_page=page_num,
                    strategy=self.name
                ))
        
        return result
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the separator hierarchy."""
        if not separators:
            # No more separators, just split by character count
            return self._split_by_char_count(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator back (except for the first piece)
            piece = split if not current_chunk else separator + split
            
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                # Current chunk is full
                if current_chunk:
                    if len(current_chunk) > self.chunk_size:
                        # Recursively split with finer separators
                        sub_chunks = self._split_text(current_chunk, remaining_separators)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current_chunk)
                
                current_chunk = split  # Start new chunk without separator prefix
        
        # Don't forget the last chunk
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._split_text(current_chunk, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)
        
        # Apply overlap
        return self._merge_with_overlap(chunks)
    
    def _split_by_char_count(self, text: str) -> List[str]:
        """Final fallback: split by character count."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks for context continuity."""
        if not chunks or len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Add overlap from previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Find a clean break point (word boundary)
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            
            result.append(overlap_text + " " + curr_chunk)
        
        return result


# Factory function to get chunker by name
def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """
    Factory function to get a chunker by strategy name.
    
    Available strategies:
    - "fixed": FixedSizeChunker
    - "sentence": SentenceChunker  
    - "semantic": SemanticChunker
    - "recursive": RecursiveChunker (default)
    """
    strategies = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
        "recursive": RecursiveChunker
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    
    return strategies[strategy](**kwargs)


def compare_chunking_strategies(text: str) -> dict:
    """
    Compare all chunking strategies on the same text.
    Returns statistics about each strategy's output.
    """
    strategies = {
        "fixed_200_50": FixedSizeChunker(chunk_size=200, overlap=50),
        "fixed_300_75": FixedSizeChunker(chunk_size=300, overlap=75),
        "sentence_5": SentenceChunker(max_sentences=5),
        "sentence_3": SentenceChunker(max_sentences=3),
        "semantic": SemanticChunker(),
        "recursive_300": RecursiveChunker(chunk_size=300),
        "recursive_500": RecursiveChunker(chunk_size=500),
    }
    
    results = {}
    for name, chunker in strategies.items():
        chunks = chunker.chunk(text)
        word_counts = [len(c.text.split()) for c in chunks]
        
        results[name] = {
            "num_chunks": len(chunks),
            "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "total_words": sum(word_counts),
        }
    
    return results


