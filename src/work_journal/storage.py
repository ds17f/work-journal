"""Data storage layer for journal entries and configuration."""

import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import JournalEntry, Settings
from .entity_registry import EntityRegistry


class Storage:
    """Manages local JSON storage for journal entries and configuration."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize storage with base directory."""
        if base_path is None:
            base_path = os.path.expanduser("~/.work-journal")
        
        self.base_path = Path(base_path)
        self.entries_path = self.base_path / "entries"
        self.config_path = self.base_path / "config.json"
        self.exports_path = self.base_path / "exports"
        self.entities_path = self.base_path / "entities"
        self.backups_path = self.base_path / "backups"
        
        # Create directories if they don't exist
        self.entries_path.mkdir(parents=True, exist_ok=True)
        self.exports_path.mkdir(parents=True, exist_ok=True)
        self.entities_path.mkdir(parents=True, exist_ok=True)
        self.backups_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize entity registry
        self.entity_registry = EntityRegistry(self.entities_path)
    
    def save_entry(self, entry: JournalEntry) -> None:
        """Save a journal entry to the appropriate daily file."""
        daily_file = self.entries_path / f"{entry.date}.json"
        
        # Load existing entries for the day or create empty list
        entries = []
        if daily_file.exists():
            with open(daily_file, 'r') as f:
                entries_data = json.load(f)
                entries = [JournalEntry.model_validate(data) for data in entries_data]
        
        # Update entity registry with entities from this entry
        if hasattr(entry.processed, 'collaborators'):
            for collaborator in entry.processed.collaborators:
                self.entity_registry.add_or_update_entity(collaborator, 'collaborator')
        
        if hasattr(entry.processed, 'projects'):
            for project in entry.processed.projects:
                self.entity_registry.add_or_update_entity(project, 'project')
        
        if hasattr(entry.processed, 'tags'):
            for tag in entry.processed.tags:
                self.entity_registry.add_or_update_entity(tag, 'tag')
        
        # Save entity registry
        self.entity_registry.save_entities()
        
        # Add new entry
        entries.append(entry)
        
        # Save back to file
        with open(daily_file, 'w') as f:
            json.dump([entry.model_dump(mode='json') for entry in entries], f, indent=2, default=str)
    
    def load_entries_for_date(self, date: str) -> List[JournalEntry]:
        """Load all entries for a specific date."""
        daily_file = self.entries_path / f"{date}.json"
        
        if not daily_file.exists():
            return []
        
        with open(daily_file, 'r') as f:
            entries_data = json.load(f)
            return [JournalEntry.model_validate(data) for data in entries_data]
    
    def load_recent_entries(self, days: int = 7) -> List[JournalEntry]:
        """Load recent entries from the last N days, excluding tombstoned entries."""
        entries = []
        
        # Get all entry files and sort by date
        entry_files = list(self.entries_path.glob("*.json"))
        entry_files.sort(reverse=True)  # Most recent first
        
        for entry_file in entry_files[:days]:
            with open(entry_file, 'r') as f:
                entries_data = json.load(f)
                daily_entries = [JournalEntry.model_validate(data) for data in entries_data 
                               if not data.get('tombstoned', False)]
                entries.extend(daily_entries)
        
        return entries
    
    def tombstone_entry(self, entry_id: str) -> bool:
        """Soft delete an entry by marking it as tombstoned. Returns True if tombstoned, False if not found."""
        from uuid import UUID
        from datetime import datetime
        
        try:
            target_uuid = UUID(entry_id)
        except ValueError:
            return False
        
        # Search through all entry files
        entry_files = list(self.entries_path.glob("*.json"))
        
        for entry_file in entry_files:
            with open(entry_file, 'r') as f:
                entries_data = json.load(f)
            
            # Find and tombstone the entry
            modified = False
            for entry_data in entries_data:
                if entry_data.get('id') == str(target_uuid):
                    entry_data['tombstoned'] = True
                    entry_data['tombstoned_at'] = datetime.now().isoformat()
                    modified = True
                    break
            
            if modified:
                # Write back the modified entries
                with open(entry_file, 'w') as f:
                    json.dump(entries_data, f, indent=2)
                return True
        
        return False
    
    def delete_entry(self, entry_id: str) -> bool:
        """Legacy method - now uses tombstoning."""
        return self.tombstone_entry(entry_id)
    
    def update_entry(self, updated_entry: JournalEntry) -> bool:
        """Update an existing entry. Returns True if updated, False if not found."""
        from uuid import UUID
        
        try:
            target_uuid = UUID(str(updated_entry.id))
        except ValueError:
            return False
        
        # Search through all entry files
        entry_files = list(self.entries_path.glob("*.json"))
        
        for entry_file in entry_files:
            with open(entry_file, 'r') as f:
                entries_data = json.load(f)
            
            # Find and update the entry
            modified = False
            for i, entry_data in enumerate(entries_data):
                if entry_data.get('id') == str(target_uuid):
                    # Replace with updated entry data
                    entries_data[i] = updated_entry.model_dump(mode='json')
                    modified = True
                    break
            
            if modified:
                # Write back the modified entries
                with open(entry_file, 'w') as f:
                    json.dump(entries_data, f, indent=2)
                return True
        
        return False
    
    def load_all_entries(self, limit: int = 50) -> List[JournalEntry]:
        """Load all entries with optional limit for pagination, excluding tombstoned entries."""
        entries = []
        
        # Get all entry files and sort by date
        entry_files = list(self.entries_path.glob("*.json"))
        entry_files.sort(reverse=True)  # Most recent first
        
        for entry_file in entry_files:
            if len(entries) >= limit:
                break
                
            with open(entry_file, 'r') as f:
                entries_data = json.load(f)
                daily_entries = [JournalEntry.model_validate(data) for data in entries_data 
                               if not data.get('tombstoned', False)]
                entries.extend(daily_entries)
        
        return entries[:limit]
    
    def load_tombstoned_entries(self) -> List[Dict[str, Any]]:
        """Load all tombstoned entries for recycle bin management."""
        tombstoned_entries = []
        
        # Get all entry files
        entry_files = list(self.entries_path.glob("*.json"))
        entry_files.sort(reverse=True)  # Most recent first
        
        for entry_file in entry_files:
            with open(entry_file, 'r') as f:
                entries_data = json.load(f)
                for entry_data in entries_data:
                    if entry_data.get('tombstoned', False):
                        tombstoned_entries.append(entry_data)
        
        return tombstoned_entries
    
    def empty_recycle_bin(self) -> Dict[str, int]:
        """Permanently delete all tombstoned entries. Returns count of deleted entries."""
        deleted_count = 0
        files_modified = 0
        
        # Get all entry files
        entry_files = list(self.entries_path.glob("*.json"))
        
        for entry_file in entry_files:
            with open(entry_file, 'r') as f:
                entries_data = json.load(f)
            
            # Filter out tombstoned entries
            original_count = len(entries_data)
            entries_data = [entry for entry in entries_data if not entry.get('tombstoned', False)]
            new_count = len(entries_data)
            
            if original_count != new_count:
                # File had tombstoned entries - rewrite it
                deleted_count += (original_count - new_count)
                files_modified += 1
                
                if entries_data:
                    # Still has entries - rewrite file
                    with open(entry_file, 'w') as f:
                        json.dump(entries_data, f, indent=2)
                else:
                    # No entries left - delete the file
                    entry_file.unlink()
        
        return {
            "deleted_entries": deleted_count,
            "files_modified": files_modified
        }
    
    def get_recycle_bin_stats(self) -> Dict[str, Any]:
        """Get statistics about tombstoned entries in recycle bin."""
        tombstoned_entries = self.load_tombstoned_entries()
        
        if not tombstoned_entries:
            return {
                "total_count": 0,
                "oldest_date": None,
                "newest_date": None,
                "total_size_estimate": 0
            }
        
        # Calculate statistics
        dates = []
        total_size = 0
        
        for entry in tombstoned_entries:
            if 'date' in entry:
                dates.append(entry['date'])
            # Estimate size (rough calculation)
            total_size += len(str(entry))
        
        return {
            "total_count": len(tombstoned_entries),
            "oldest_date": min(dates) if dates else None,
            "newest_date": max(dates) if dates else None,
            "total_size_estimate": total_size
        }
    
    def save_settings(self, settings: Settings) -> None:
        """Save settings to file."""
        with open(self.config_path, 'w') as f:
            json.dump(settings.model_dump(mode='json'), f, indent=2)
    
    def load_settings(self) -> Settings:
        """Load settings from file, return empty settings if file doesn't exist."""
        if not self.config_path.exists():
            return Settings()
        
        try:
            with open(self.config_path, 'r') as f:
                settings_data = json.load(f)
                return Settings.model_validate(settings_data)
        except Exception:
            # If there's any error loading, return empty settings
            return Settings()
    
    def ensure_env_file(self) -> Path:
        """Ensure .env file exists and return its path."""
        env_file = self.base_path / ".env"
        if not env_file.exists():
            env_file.write_text("# Add your API keys and other secrets here\n")
        return env_file
    
    def copy_default_providers_template(self) -> Path:
        """Copy default providers template to user's config directory."""
        template_source = Path(__file__).parent.parent.parent / "default_providers.json"
        user_template = self.base_path / "default_providers.json"
        
        if template_source.exists() and not user_template.exists():
            import shutil
            shutil.copy2(template_source, user_template)
            
        return user_template
    
    # Data Management and Backup Methods
    
    def create_backup(self, backup_name: str = None) -> str:
        """Create a complete backup of all data including entries, entities, and config."""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.backups_path / f"{backup_name}.tar.gz"
        
        # Create temporary directory for backup staging
        temp_backup_dir = self.backups_path / f"temp_{backup_name}"
        temp_backup_dir.mkdir(exist_ok=True)
        
        try:
            # Copy all data directories and files
            if self.entries_path.exists():
                shutil.copytree(self.entries_path, temp_backup_dir / "entries")
            
            if self.entities_path.exists():
                shutil.copytree(self.entities_path, temp_backup_dir / "entities")
            
            if self.config_path.exists():
                shutil.copy2(self.config_path, temp_backup_dir / "config.json")
            
            if self.exports_path.exists():
                shutil.copytree(self.exports_path, temp_backup_dir / "exports")
            
            # Create backup metadata
            metadata = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "backup_type": "full",
                "total_entries": len(list(self.entries_path.glob("*.json"))) if self.entries_path.exists() else 0,
                "total_entities": len(self.entity_registry.entities),
                "version": "1.0"
            }
            
            with open(temp_backup_dir / "backup_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create compressed archive
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(temp_backup_dir, arcname=backup_name)
            
            # Clean up temporary directory
            shutil.rmtree(temp_backup_dir)
            
            return str(backup_path)
            
        except Exception as e:
            # Clean up on failure
            if temp_backup_dir.exists():
                shutil.rmtree(temp_backup_dir)
            if backup_path.exists():
                backup_path.unlink()
            raise Exception(f"Backup creation failed: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata."""
        backups = []
        
        for backup_file in self.backups_path.glob("*.tar.gz"):
            try:
                with tarfile.open(backup_file, "r:gz") as tar:
                    # Try to extract metadata
                    metadata_path = None
                    for member in tar.getmembers():
                        if member.name.endswith("/backup_metadata.json"):
                            metadata_path = member
                            break
                    
                    if metadata_path:
                        metadata_content = tar.extractfile(metadata_path).read()
                        metadata = json.loads(metadata_content)
                    else:
                        # Fallback metadata from filename and file stats
                        metadata = {
                            "backup_name": backup_file.stem,
                            "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                            "backup_type": "full",
                            "total_entries": "unknown",
                            "total_entities": "unknown",
                            "version": "unknown"
                        }
                    
                    metadata["file_path"] = str(backup_file)
                    metadata["file_size"] = backup_file.stat().st_size
                    backups.append(metadata)
                    
            except Exception as e:
                # Skip corrupted backups but note them
                backups.append({
                    "backup_name": backup_file.stem,
                    "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    "backup_type": "corrupted",
                    "error": str(e),
                    "file_path": str(backup_file),
                    "file_size": backup_file.stat().st_size
                })
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups
    
    def restore_from_backup(self, backup_name: str, confirm: bool = False) -> bool:
        """Restore data from a backup. DESTRUCTIVE operation!"""
        if not confirm:
            raise ValueError("Must confirm restore operation - this will overwrite current data!")
        
        backup_file = self.backups_path / f"{backup_name}.tar.gz"
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        # Create safety backup of current state before restore
        safety_backup_name = f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.create_backup(safety_backup_name)
        
        try:
            # Clear existing data
            if self.entries_path.exists():
                shutil.rmtree(self.entries_path)
            if self.entities_path.exists():
                shutil.rmtree(self.entities_path)
            if self.config_path.exists():
                self.config_path.unlink()
            if self.exports_path.exists():
                shutil.rmtree(self.exports_path)
            
            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(self.base_path)
                
                # Move extracted files to correct locations
                extracted_dir = self.base_path / backup_name
                
                if (extracted_dir / "entries").exists():
                    shutil.move(extracted_dir / "entries", self.entries_path)
                
                if (extracted_dir / "entities").exists():
                    shutil.move(extracted_dir / "entities", self.entities_path)
                
                if (extracted_dir / "config.json").exists():
                    shutil.move(extracted_dir / "config.json", self.config_path)
                
                if (extracted_dir / "exports").exists():
                    shutil.move(extracted_dir / "exports", self.exports_path)
                
                # Clean up extracted directory
                if extracted_dir.exists():
                    shutil.rmtree(extracted_dir)
            
            # Recreate directories and reinitialize entity registry
            self.entries_path.mkdir(exist_ok=True)
            self.entities_path.mkdir(exist_ok=True)
            self.exports_path.mkdir(exist_ok=True)
            self.entity_registry = EntityRegistry(self.entities_path)
            
            return True
            
        except Exception as e:
            raise Exception(f"Restore failed: {e}. Safety backup created as: {safety_backup_name}")
    
    def reprocess_entries(self, entry_filter: str = "all", batch_size: int = 10) -> Dict[str, Any]:
        """Reprocess entries with current LLM processing logic."""
        from .entry_processor import EntryProcessor
        
        # Load entries based on filter
        if entry_filter == "all":
            entries = self.load_all_entries(limit=1000)  # Process up to 1000 entries
        elif entry_filter == "recent":
            entries = self.load_recent_entries(days=30)  # Last 30 days
        else:
            # Assume it's a specific date
            entries = self.load_entries_for_date(entry_filter)
        
        if not entries:
            return {"status": "no_entries", "processed": 0, "errors": 0}
        
        processor = EntryProcessor(storage=self)
        processed_count = 0
        error_count = 0
        errors = []
        
        # Process in batches
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            
            for entry in batch:
                try:
                    # Create a fresh processing of the raw input
                    new_processed = processor._extract_structure(entry.raw_input)
                    
                    # Update the entry with new processing
                    entry.processed = new_processed
                    entry.metadata["reprocessed_at"] = datetime.now().isoformat()
                    entry.metadata["reprocessing_version"] = "2.0"
                    
                    # Update the entry in storage
                    self.update_entry(entry)
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append({
                        "entry_id": str(entry.id),
                        "date": entry.date,
                        "error": str(e)
                    })
        
        # Update entity registry with reprocessed entities
        self.entity_registry.save_entities()
        
        return {
            "status": "completed",
            "total_entries": len(entries),
            "processed": processed_count,
            "errors": error_count,
            "error_details": errors,
            "filter_used": entry_filter
        }
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored data."""
        stats = {
            "entries": {
                "total_files": len(list(self.entries_path.glob("*.json"))) if self.entries_path.exists() else 0,
                "total_entries": 0,
                "date_range": {"earliest": None, "latest": None}
            },
            "entities": {
                "collaborators": len([e for e in self.entity_registry.entities.values() if e.type == "collaborator"]),
                "projects": len([e for e in self.entity_registry.entities.values() if e.type == "project"]),
                "tags": len([e for e in self.entity_registry.entities.values() if e.type == "tag"])
            },
            "backups": {
                "total_backups": len(list(self.backups_path.glob("*.tar.gz"))),
                "latest_backup": None
            },
            "storage": {
                "base_path": str(self.base_path),
                "total_size_mb": 0
            }
        }
        
        # Calculate total entries and date range
        all_entries = self.load_all_entries(limit=5000)
        stats["entries"]["total_entries"] = len(all_entries)
        
        if all_entries:
            dates = [entry.date for entry in all_entries]
            stats["entries"]["date_range"]["earliest"] = min(dates)
            stats["entries"]["date_range"]["latest"] = max(dates)
        
        # Get latest backup info
        backups = self.list_backups()
        if backups:
            stats["backups"]["latest_backup"] = backups[0]["backup_name"]
        
        # Calculate total storage size
        def get_dir_size(path):
            total = 0
            if path.exists():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        total += file_path.stat().st_size
            return total
        
        total_bytes = (
            get_dir_size(self.entries_path) +
            get_dir_size(self.entities_path) +
            get_dir_size(self.backups_path) +
            (self.config_path.stat().st_size if self.config_path.exists() else 0)
        )
        
        stats["storage"]["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
        
        return stats