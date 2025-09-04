"""Journal entry processing with LLM to structure accomplishments."""

import json
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime

from .models import JournalEntry, ProcessedEntry, Impact
from .llm import LLMClient
from .entity_registry import EntityMatch
from .logging_config import get_logger

logger = get_logger(__name__)


class EntryProcessor:
    """Processes raw journal entries into structured accomplishments."""
    
    def __init__(self, llm_client: LLMClient = None, storage=None):
        """Initialize with LLM client and optional storage for context."""
        self.llm_client = llm_client or LLMClient()
        self.storage = storage
    
    def process_entry(self, raw_input: str, date: str) -> JournalEntry:
        """
        Process raw user input into a structured journal entry.
        
        Args:
            raw_input: User's description of their work
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Complete JournalEntry with structured data
        """
        # Get initial processing from LLM
        processed_data = self._extract_structure(raw_input)
        
        # Create the complete entry
        entry = JournalEntry(
            date=date,
            raw_input=raw_input,
            processed=processed_data,
            metadata={
                "workflowConfig": self._get_current_workflow_config(),
                "processingTime": datetime.now().isoformat()
            }
        )
        
        return entry
    
    def refine_entry(self, entry: JournalEntry, refinement_instruction: str) -> JournalEntry:
        """Refine an existing entry based on user feedback."""
        
        # Create refinement prompt
        current_processed = entry.processed
        
        system_prompt = """You are helping refine a journal entry based on user feedback. 
Your job is to improve the existing structured entry according to the user's instructions.

Keep the same JSON structure but modify the content based on the feedback.
Be precise and maintain professional tone suitable for promotion reviews."""

        user_prompt = f"""Current entry:
{json.dumps(current_processed.model_dump(), indent=2)}

Original raw input:
{entry.raw_input}

User feedback: {refinement_instruction}

Please refine the entry based on this feedback. Respond ONLY with valid JSON in the same format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Use processing workflow for refinement
            logger.debug(f"Calling LLM for refinement with {len(messages)} messages")
            logger.debug(f"System prompt length: {len(messages[0]['content']) if messages else 0}")
            logger.debug(f"User prompt length: {len(messages[1]['content']) if len(messages) > 1 else 0}")
            
            response = self.llm_client.call_llm("processing", messages, max_tokens=1000, temperature=0.3)
            
            # Debug: Check if response is empty or invalid
            logger.debug(f"LLM response type: {type(response)}")
            logger.debug(f"LLM response length: {len(response) if response else 0}")
            if response:
                logger.debug(f"LLM response preview: {response[:100]}...")
            
            if not response or not response.strip():
                logger.error("LLM returned empty response for refinement")
                raise Exception(f"LLM returned empty response")
            
            # Parse JSON response
            refined_data = json.loads(response)
            
            # Create new ProcessedEntry
            refined_processed = ProcessedEntry(
                summary=refined_data["summary"],
                work=refined_data["work"],
                collaborators=refined_data.get("collaborators", []),
                projects=refined_data.get("projects", []),
                impact=Impact(**refined_data["impact"]),
                technical_details=refined_data.get("technical_details", []),
                tags=refined_data.get("tags", [])
            )
            
            # Update the entry
            entry.processed = refined_processed
            entry.metadata["refinement_count"] = entry.metadata.get("refinement_count", 0) + 1
            entry.metadata["last_refined"] = datetime.now().isoformat()
            
            return entry
            
        except json.JSONDecodeError as e:
            # Show the actual response content for debugging
            response_preview = response[:200] if response else "None"
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response was: {response_preview}")
            raise Exception(f"Failed to parse LLM response as JSON: {e}. Response was: {response_preview}")
        except Exception as e:
            logger.error(f"Failed to refine entry: {e}")
            raise Exception(f"Failed to refine entry: {e}")
    
    def _extract_structure(self, raw_input: str) -> ProcessedEntry:
        """Extract structured data from raw input using two-phase entity extraction."""
        
        # Phase 1: Extract potential entities from raw text
        entity_registry = self.storage.entity_registry if self.storage else None
        potential_entities = {}
        matched_entities = {}
        
        if entity_registry:
            potential_entities = entity_registry.extract_potential_entities(raw_input)
            matched_entities = entity_registry.match_entities(potential_entities)
        
        # Phase 2: Get structured content with matched entities as suggestions
        processed_entry = self._llm_extract_with_entities(raw_input, potential_entities, matched_entities)
        
        return processed_entry
    
    def _llm_extract_with_entities(self, raw_input: str, potential_entities: Dict, matched_entities: Dict) -> ProcessedEntry:
        """Extract structured data using LLM with entity suggestions to prevent hallucination."""
        
        # Build entity suggestions for the prompt
        entity_suggestions = self._build_entity_suggestions(potential_entities, matched_entities)
        
        # Updated system prompt that focuses on extraction, not invention
        system_prompt = """You are a professional career development assistant. Your job is to help structure work accomplishments for promotion reviews.

CRITICAL INSTRUCTIONS FOR ENTITY EXTRACTION:
- ONLY extract collaborators, projects, and tags that are explicitly mentioned in the raw text
- DO NOT infer, assume, or add entities that are not directly stated
- If entity suggestions are provided, ONLY use them if they match something in the raw text
- For collaborators: Only include people explicitly named in the text
- For projects: Only include project names explicitly mentioned in the text
- For tags: Only include concepts explicitly discussed in the text

For projects, be very selective. Only include work that contributes to a specific project with concrete deliverables. Do NOT include:
- Meetings, discussions, or conversations
- General processes like "onboarding", "code reviews", "planning"
- Strategy discussions or high-level planning
- Training or learning activities
- General team activities

DO include projects like:
- Software implementations or migrations
- Product features or components  
- Infrastructure deployments
- Specific initiatives with deliverables
- Tool adoptions or rollouts
- System redesigns or refactors

1. **Summary**: A concise, impactful summary (1-2 sentences)
2. **Work**: Detailed description of what was actually done
3. **Collaborators**: List of people involved (ONLY extract names explicitly mentioned in text)
4. **Projects**: List of specific project names this work belongs to (ONLY if explicitly mentioned)
5. **Impact**: Assess the scope (individual/team/organization), significance (1-5 scale), and quantifiable metrics
6. **Technical Details**: Tools, technologies, approaches used
7. **Tags**: Relevant categorization tags (ONLY for concepts explicitly discussed)

Respond ONLY with valid JSON in this exact format:
{
    "summary": "Brief impactful summary",
    "work": "Detailed description of work performed", 
    "collaborators": ["Name 1", "Name 2"],
    "projects": ["Project Name 1", "Project Name 2"],
    "impact": {
        "scope": "individual|team|organization",
        "significance": 3,
        "metrics": ["Specific quantifiable impact"],
        "business_value": "Business value delivered"
    },
    "technical_details": ["Tool/approach 1", "Tool/approach 2"],
    "tags": ["tag1", "tag2", "tag3"]
}"""

        # Build the user prompt with entity suggestions
        user_prompt = f"Extract and structure this work accomplishment:\n\n{raw_input}"
        
        if entity_suggestions:
            user_prompt += f"\n\n{entity_suggestions}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Use processing workflow for this step
            logger.debug(f"Calling LLM for entry extraction with {len(messages)} messages")
            response = self.llm_client.call_llm("processing", messages, max_tokens=1000, temperature=0.3)
            
            # Debug: Check if response is empty or invalid
            logger.debug(f"LLM extraction response length: {len(response) if response else 0}")
            if not response or not response.strip():
                logger.error("LLM returned empty response for entry extraction")
                raise Exception(f"LLM returned empty response")
            
            # Parse JSON response
            structured_data = json.loads(response)
            
            # Create ProcessedEntry with proper nested objects
            impact = Impact(
                scope=structured_data["impact"]["scope"],
                metrics=structured_data["impact"]["metrics"],
                business_value=structured_data["impact"]["business_value"]
            )
            
            processed_entry = ProcessedEntry(
                summary=structured_data["summary"],
                work=structured_data["work"],
                collaborators=structured_data["collaborators"],
                projects=structured_data.get("projects", []),
                impact=impact,
                technical_details=structured_data["technical_details"],
                tags=structured_data["tags"]
            )
            
            return processed_entry
            
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return self._create_fallback_entry(raw_input)
        except Exception as e:
            # Fallback if LLM call fails
            logger.warning(f"LLM processing failed, using fallback: {e}")
            return self._create_fallback_entry(raw_input)
    
    def _build_entity_suggestions(self, potential_entities: Dict, matched_entities: Dict) -> str:
        """Build entity suggestions text for the LLM prompt."""
        if not potential_entities and not matched_entities:
            return ""
        
        suggestions = []
        
        # Add potential entities found in text
        if potential_entities.get('collaborators'):
            suggestions.append(f"POTENTIAL COLLABORATORS mentioned in text: {', '.join(potential_entities['collaborators'])}")
        
        if potential_entities.get('projects'):
            suggestions.append(f"POTENTIAL PROJECTS mentioned in text: {', '.join(potential_entities['projects'])}")
        
        if potential_entities.get('tags'):
            suggestions.append(f"POTENTIAL TAGS mentioned in text: {', '.join(potential_entities['tags'])}")
        
        # Add matched entities (known entities that might be relevant)
        if matched_entities.get('collaborators'):
            known_collaborators = [match.canonical_name for match in matched_entities['collaborators'] if match.confidence > 0.7]
            if known_collaborators:
                suggestions.append(f"KNOWN COLLABORATORS that might match: {', '.join(known_collaborators)}")
        
        if matched_entities.get('projects'):
            known_projects = [match.canonical_name for match in matched_entities['projects'] if match.confidence > 0.7]
            if known_projects:
                suggestions.append(f"KNOWN PROJECTS that might match: {', '.join(known_projects)}")
        
        if matched_entities.get('tags'):
            known_tags = [match.canonical_name for match in matched_entities['tags'] if match.confidence > 0.7]
            if known_tags:
                suggestions.append(f"KNOWN TAGS that might match: {', '.join(known_tags)}")
        
        if suggestions:
            return "ENTITY EXTRACTION GUIDANCE:\n" + "\n".join(f"- {s}" for s in suggestions) + "\n\nIMPORTANT: Only use these suggestions if they match entities explicitly mentioned in the raw text above."
        
        return ""
    
    def _create_fallback_entry(self, raw_input: str) -> ProcessedEntry:
        """Create a basic fallback entry if LLM processing fails."""
        return ProcessedEntry(
            summary=raw_input[:100] + "..." if len(raw_input) > 100 else raw_input,
            work=raw_input,
            collaborators=[],
            impact=Impact(
                scope="individual",
                metrics=[],
                business_value="Work completed"
            ),
            technical_details=[],
            tags=["general"]
        )
    
    def _get_recent_context(self, limit: int = 5) -> str:
        """Get recent entries for context awareness."""
        if not self.storage:
            return ""
        
        try:
            recent_entries = self.storage.load_recent_entries(limit)
            if not recent_entries:
                return ""
            
            # Extract projects and collaborators from recent entries
            recent_projects = set()
            recent_collaborators = set()
            
            for entry in recent_entries:
                if hasattr(entry.processed, 'projects'):
                    recent_projects.update(entry.processed.projects)
                if hasattr(entry.processed, 'collaborators'):
                    recent_collaborators.update(entry.processed.collaborators)
            
            context_parts = []
            if recent_projects:
                context_parts.append(f"Recent projects: {', '.join(sorted(recent_projects))}")
            if recent_collaborators:
                context_parts.append(f"Recent collaborators: {', '.join(sorted(recent_collaborators))}")
            
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception:
            # If context retrieval fails, continue without context
            return ""
    
    def _get_current_workflow_config(self) -> Dict[str, Any]:
        """Get current workflow configuration for metadata."""
        try:
            if self.llm_client.settings.current_config and self.llm_client.settings.current_config in self.llm_client.settings.configurations:
                config = self.llm_client.settings.configurations[self.llm_client.settings.current_config]
                result = {}
                if hasattr(config, 'processing'):
                    result['processing'] = {
                        "provider": config.processing.provider,
                        "model": config.processing.model
                    }
                if hasattr(config, 'conversation'):
                    result['conversation'] = {
                        "provider": config.conversation.provider,
                        "model": config.conversation.model
                    }
                if hasattr(config, 'jira_matching'):
                    result['jira_matching'] = {
                        "provider": config.jira_matching.provider,
                        "model": config.jira_matching.model
                    }
                return result
            return {}
        except Exception:
            return {}
    
    def conversational_refinement(self, entry: JournalEntry) -> JournalEntry:
        """
        Engage in follow-up conversation to refine the entry.
        
        This would ask follow-up questions to get more details about:
        - Impact metrics
        - Collaborators
        - Technical approaches
        - Business context
        
        For now, returns the entry as-is. Future implementation would
        use the "conversation" workflow to ask intelligent follow-ups.
        """
        # TODO: Implement conversational refinement
        # This would use the "conversation" workflow to ask follow-up questions
        return entry