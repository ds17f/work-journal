"""Entity registry for managing collaborators, projects, and tags across journal entries."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher


@dataclass
class EntityMatch:
    """Represents a potential entity match with confidence scoring."""
    entity_id: str
    canonical_name: str
    matched_text: str
    confidence: float  # 0.0 to 1.0
    match_type: str  # exact, alias, fuzzy, partial


@dataclass
class Entity:
    """Represents a tracked entity (collaborator, project, or tag)."""
    id: str
    canonical_name: str
    aliases: Set[str]
    type: str  # collaborator, project, tag
    usage_count: int = 0
    last_used: Optional[str] = None  # ISO date string
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.aliases, list):
            self.aliases = set(self.aliases)


class EntityRegistry:
    """Centralized registry for managing and matching entities across journal entries."""
    
    def __init__(self, storage_path: Path):
        """Initialize entity registry with storage path."""
        self.storage_path = storage_path
        self.entities: Dict[str, Entity] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # text -> entity_ids
        self.load_entities()
    
    def load_entities(self):
        """Load entities from storage files."""
        entity_files = {
            'collaborator': self.storage_path / 'collaborators.json',
            'project': self.storage_path / 'projects.json', 
            'tag': self.storage_path / 'tags.json'
        }
        
        for entity_type, file_path in entity_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        entities_data = json.load(f)
                        for entity_data in entities_data:
                            entity = Entity(
                                id=entity_data['id'],
                                canonical_name=entity_data['canonical_name'],
                                aliases=set(entity_data.get('aliases', [])),
                                type=entity_type,
                                usage_count=entity_data.get('usage_count', 0),
                                last_used=entity_data.get('last_used'),
                                metadata=entity_data.get('metadata', {})
                            )
                            self.entities[entity.id] = entity
                            self._update_index(entity)
                except Exception as e:
                    print(f"Error loading {entity_type} entities: {e}")
    
    def save_entities(self):
        """Save entities to storage files."""
        entities_by_type = defaultdict(list)
        
        for entity in self.entities.values():
            entities_by_type[entity.type].append({
                'id': entity.id,
                'canonical_name': entity.canonical_name,
                'aliases': list(entity.aliases),
                'usage_count': entity.usage_count,
                'last_used': entity.last_used,
                'metadata': entity.metadata
            })
        
        entity_files = {
            'collaborator': self.storage_path / 'collaborators.json',
            'project': self.storage_path / 'projects.json',
            'tag': self.storage_path / 'tags.json'
        }
        
        for entity_type, file_path in entity_files.items():
            try:
                with open(file_path, 'w') as f:
                    json.dump(entities_by_type[entity_type], f, indent=2)
            except Exception as e:
                print(f"Error saving {entity_type} entities: {e}")
    
    def _update_index(self, entity: Entity):
        """Update the search index with entity names and aliases."""
        # Index canonical name
        self.entity_index[entity.canonical_name.lower()].add(entity.id)
        
        # Index all aliases
        for alias in entity.aliases:
            self.entity_index[alias.lower()].add(entity.id)
            
        # Index partial matches (first name, last name, etc.)
        words = entity.canonical_name.lower().split()
        for word in words:
            if word and len(word) > 2:  # Avoid indexing very short words
                self.entity_index[word].add(entity.id)
    
    def extract_potential_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entity mentions from text without interpretation."""
        potential_entities = {
            'collaborators': [],
            'projects': [],
            'tags': []
        }
        
        # Extract person names - be more conservative
        # Look for patterns that are more likely to be actual names
        person_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last (keep this, it's good)
            r'@[a-zA-Z][a-zA-Z0-9_]*',  # @username (keep this)
            r'\bwith ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "with John" patterns
            r'\band ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "and Sarah" patterns
            # Remove the overly broad single capitalized word pattern
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                match = match.strip().replace('@', '')
                if match and len(match) > 1:
                    potential_entities['collaborators'].append(match)
        
        # Extract project names - be much more conservative
        project_patterns = [
            r'"([^"]+)"',  # Quoted strings (keep this, it's explicit)
            r'\b([\w\s-]+\s+(?:project|initiative|migration|rollout|deployment|system|platform))\b',  # Explicit project patterns
            r'\bthe\s+([A-Z][a-zA-Z0-9\s-]+(?:Project|Initiative|System|Platform))\b',  # "the Something Project" patterns
            # Remove broad capitalized word patterns - they're too aggressive
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                match = match.strip()
                if match and len(match) > 2:
                    potential_entities['projects'].append(match)
        
        # Extract hashtags and common tag patterns
        tag_patterns = [
            r'#(\w+)',  # #hashtags
            r'\b(backend|frontend|mobile|web|api|database|security|performance|testing|deployment)\b',  # Common tech tags
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.lower().strip()
                if match and len(match) > 1:
                    potential_entities['tags'].append(match)
        
        # Filter out common words that are unlikely to be entities
        common_words = {
            'work', 'journal', 'work journal', 'talked', 'discussed', 'meeting', 'call', 'email', 'the', 'and', 
            'or', 'but', 'with', 'about', 'from', 'to', 'in', 'on', 'at', 'by', 'for', 'of',
            'is', 'was', 'are', 'were', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'my', 'our', 'your', 'their', 'his', 'her', 'its',
            'new', 'old', 'good', 'bad', 'great', 'small', 'big', 'high', 'low', 'first', 'last',
            'next', 'previous', 'current', 'future', 'past', 'today', 'yesterday', 'tomorrow',
            'show', 'showcase', 'demo', 'presentation', 'update', 'status', 'review', 'feedback',
            'ai showcase', 'local llms', 'opencode'
        }
        
        # Remove duplicates and filter common words
        for entity_type in potential_entities:
            seen = set()
            unique_entities = []
            for item in potential_entities[entity_type]:
                item_lower = item.lower()
                if (item_lower not in seen and 
                    item_lower not in common_words and 
                    len(item) > 2):  # Minimum length filter
                    seen.add(item_lower)
                    unique_entities.append(item)
            potential_entities[entity_type] = unique_entities
        
        return potential_entities
    
    def find_entity_matches(self, text: str, entity_type: str) -> List[EntityMatch]:
        """Find potential matches for a text string within a specific entity type."""
        matches = []
        text_lower = text.lower()
        
        # Exact matches
        if text_lower in self.entity_index:
            for entity_id in self.entity_index[text_lower]:
                entity = self.entities[entity_id]
                if entity.type == entity_type:
                    matches.append(EntityMatch(
                        entity_id=entity_id,
                        canonical_name=entity.canonical_name,
                        matched_text=text,
                        confidence=1.0,
                        match_type='exact'
                    ))
        
        # Alias matches
        for entity in self.entities.values():
            if entity.type != entity_type:
                continue
                
            for alias in entity.aliases:
                if alias.lower() == text_lower:
                    matches.append(EntityMatch(
                        entity_id=entity.id,
                        canonical_name=entity.canonical_name,
                        matched_text=text,
                        confidence=0.9,
                        match_type='alias'
                    ))
        
        # Fuzzy matching for high-confidence partial matches
        for entity in self.entities.values():
            if entity.type != entity_type:
                continue
            
            # Check fuzzy match with canonical name
            similarity = SequenceMatcher(None, text_lower, entity.canonical_name.lower()).ratio()
            if similarity > 0.8:  # High confidence threshold
                matches.append(EntityMatch(
                    entity_id=entity.id,
                    canonical_name=entity.canonical_name,
                    matched_text=text,
                    confidence=similarity * 0.8,  # Reduce confidence for fuzzy matches
                    match_type='fuzzy'
                ))
        
        # Sort by confidence (highest first) and remove duplicates
        matches.sort(key=lambda m: m.confidence, reverse=True)
        seen_entities = set()
        unique_matches = []
        for match in matches:
            if match.entity_id not in seen_entities:
                seen_entities.add(match.entity_id)
                unique_matches.append(match)
        
        return unique_matches[:3]  # Return top 3 matches
    
    def match_entities(self, potential_entities: Dict[str, List[str]]) -> Dict[str, List[EntityMatch]]:
        """Match potential entity mentions to known entities."""
        matched_entities = {
            'collaborators': [],
            'projects': [],
            'tags': []
        }
        
        entity_type_mapping = {
            'collaborators': 'collaborator',
            'projects': 'project',
            'tags': 'tag'
        }
        
        for entity_group, entity_list in potential_entities.items():
            entity_type = entity_type_mapping[entity_group]
            for potential_entity in entity_list:
                matches = self.find_entity_matches(potential_entity, entity_type)
                matched_entities[entity_group].extend(matches)
        
        return matched_entities
    
    def add_or_update_entity(self, name: str, entity_type: str, aliases: List[str] = None) -> str:
        """Add a new entity or update an existing one. Returns entity ID."""
        if aliases is None:
            aliases = []
        
        # Check if entity already exists
        existing_matches = self.find_entity_matches(name, entity_type)
        if existing_matches and existing_matches[0].confidence > 0.9:
            # Update existing entity
            entity_id = existing_matches[0].entity_id
            entity = self.entities[entity_id]
            entity.aliases.update(aliases)
            entity.usage_count += 1
            entity.last_used = datetime.now().strftime('%Y-%m-%d')
            self._update_index(entity)
            return entity_id
        
        # Create new entity
        entity_id = f"{entity_type}_{len([e for e in self.entities.values() if e.type == entity_type]) + 1:04d}"
        entity = Entity(
            id=entity_id,
            canonical_name=name,
            aliases=set(aliases),
            type=entity_type,
            usage_count=1,
            last_used=datetime.now().strftime('%Y-%m-%d')
        )
        
        self.entities[entity_id] = entity
        self._update_index(entity)
        return entity_id
    
    def get_entity_suggestions(self, entity_type: str, limit: int = 5) -> List[Entity]:
        """Get suggested entities based on usage frequency and recency."""
        entities_of_type = [e for e in self.entities.values() if e.type == entity_type]
        
        # Sort by usage count and recency
        def sort_key(entity):
            usage_score = entity.usage_count
            recency_score = 0
            if entity.last_used:
                try:
                    last_used_date = datetime.fromisoformat(entity.last_used.replace('Z', '+00:00'))
                    days_ago = (datetime.now() - last_used_date).days
                    recency_score = max(0, 30 - days_ago) / 30  # 0 to 1 based on how recent
                except:
                    pass
            return usage_score + recency_score
        
        entities_of_type.sort(key=sort_key, reverse=True)
        return entities_of_type[:limit]
    
    def cleanup_entities(self, min_usage: int = 1):
        """Remove entities with very low usage to keep registry clean."""
        entities_to_remove = []
        for entity_id, entity in self.entities.items():
            if entity.usage_count < min_usage:
                entities_to_remove.append(entity_id)
        
        for entity_id in entities_to_remove:
            del self.entities[entity_id]
        
        # Rebuild index
        self.entity_index.clear()
        for entity in self.entities.values():
            self._update_index(entity)
    
    def update_entity(self, entity_id: str, canonical_name: str = None, aliases: List[str] = None) -> bool:
        """Update an existing entity's name and/or aliases."""
        if entity_id not in self.entities:
            return False
        
        entity = self.entities[entity_id]
        
        # Update canonical name if provided
        if canonical_name is not None:
            entity.canonical_name = canonical_name
        
        # Update aliases if provided
        if aliases is not None:
            entity.aliases = set(aliases)
        
        # Rebuild index for this entity
        self.entity_index.clear()
        for e in self.entities.values():
            self._update_index(e)
        
        return True
    
    def merge_entities(self, primary_entity_id: str, secondary_entity_id: str) -> bool:
        """Merge two entities, combining their usage counts and aliases."""
        if primary_entity_id not in self.entities or secondary_entity_id not in self.entities:
            return False
        
        primary = self.entities[primary_entity_id]
        secondary = self.entities[secondary_entity_id]
        
        # Must be same type
        if primary.type != secondary.type:
            return False
        
        # Merge aliases
        primary.aliases.update(secondary.aliases)
        primary.aliases.add(secondary.canonical_name)  # Add secondary name as alias
        
        # Merge usage counts
        primary.usage_count += secondary.usage_count
        
        # Use most recent last_used date
        if secondary.last_used and (not primary.last_used or secondary.last_used > primary.last_used):
            primary.last_used = secondary.last_used
        
        # Remove secondary entity
        del self.entities[secondary_entity_id]
        
        # Rebuild index
        self.entity_index.clear()
        for entity in self.entities.values():
            self._update_index(entity)
        
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from the registry."""
        if entity_id not in self.entities:
            return False
        
        del self.entities[entity_id]
        
        # Rebuild index
        self.entity_index.clear()
        for entity in self.entities.values():
            self._update_index(entity)
        
        return True
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by its ID."""
        return self.entities.get(entity_id)