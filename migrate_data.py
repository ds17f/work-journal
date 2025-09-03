#!/usr/bin/env python3
"""
Data migration script from promo-journal to work-journal format.

This script migrates:
1. Configuration files (.json and .env)
2. Entry data files (daily JSON files)
3. Directory structure from ~/.promo-journal to ~/.work-journal

Usage:
    python migrate_data.py [--dry-run] [--backup]
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class PromoToWorkJournalMigrator:
    """Migrates data from promo-journal to work-journal format."""
    
    def __init__(self, dry_run: bool = False, create_backup: bool = True):
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.old_dir = Path.home() / ".promo-journal"
        self.new_dir = Path.home() / ".work-journal"
        self.backup_dir = Path.home() / f"promo-journal-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = "[DRY RUN] " if self.dry_run else ""
        print(f"{prefix}[{timestamp}] {level}: {message}")
    
    def backup_existing_data(self):
        """Create a backup of existing promo-journal data."""
        if not self.old_dir.exists():
            self.log("No existing promo-journal directory found to backup")
            return
            
        if self.create_backup and not self.dry_run:
            self.log(f"Creating backup at {self.backup_dir}")
            shutil.copytree(self.old_dir, self.backup_dir)
            self.log("Backup completed successfully")
        else:
            self.log(f"Would create backup at {self.backup_dir}")
    
    def migrate_config_file(self):
        """Migrate config.json file."""
        old_config = self.old_dir / "config.json"
        new_config = self.new_dir / "config.json"
        
        if not old_config.exists():
            self.log("No config.json found to migrate")
            return
            
        self.log("Migrating config.json...")
        
        if not self.dry_run:
            # Read old config
            with open(old_config, 'r') as f:
                config_data = json.load(f)
            
            # Config format should be compatible, just copy it
            self.new_dir.mkdir(parents=True, exist_ok=True)
            with open(new_config, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.log("Config file migrated successfully")
        else:
            self.log(f"Would migrate {old_config} -> {new_config}")
    
    def migrate_env_file(self):
        """Migrate .env file."""
        old_env = self.old_dir / ".env"
        new_env = self.new_dir / ".env"
        
        if not old_env.exists():
            self.log("No .env file found to migrate")
            return
            
        self.log("Migrating .env file...")
        
        if not self.dry_run:
            # Read old env file
            with open(old_env, 'r') as f:
                env_content = f.read()
            
            # Update any references to PROMO_JOURNAL_HOME
            updated_content = env_content.replace(
                "PROMO_JOURNAL_HOME", 
                "WORK_JOURNAL_HOME"
            )
            
            # Ensure new directory exists
            self.new_dir.mkdir(parents=True, exist_ok=True)
            
            # Write updated env file
            with open(new_env, 'w') as f:
                f.write(updated_content)
            
            self.log(".env file migrated successfully")
        else:
            self.log(f"Would migrate {old_env} -> {new_env}")
    
    def migrate_entry_files(self):
        """Migrate all entry JSON files."""
        old_entries = self.old_dir / "entries"
        new_entries = self.new_dir / "entries"
        
        if not old_entries.exists():
            self.log("No entries directory found to migrate")
            return
            
        entry_files = list(old_entries.glob("*.json"))
        if not entry_files:
            self.log("No entry files found to migrate")
            return
            
        self.log(f"Migrating {len(entry_files)} entry files...")
        
        if not self.dry_run:
            new_entries.mkdir(parents=True, exist_ok=True)
            
            for entry_file in entry_files:
                self.log(f"Migrating {entry_file.name}")
                
                # Read old entry file
                with open(entry_file, 'r') as f:
                    entries_data = json.load(f)
                
                # Process each entry in the file
                migrated_entries = []
                for entry_data in entries_data:
                    migrated_entry = self.migrate_single_entry(entry_data)
                    migrated_entries.append(migrated_entry)
                
                # Write migrated entries
                new_entry_file = new_entries / entry_file.name
                with open(new_entry_file, 'w') as f:
                    json.dump(migrated_entries, f, indent=2, default=str)
                
                self.log(f"Migrated {entry_file.name} -> {new_entry_file.name}")
        else:
            for entry_file in entry_files:
                self.log(f"Would migrate {entry_file} -> {new_entries / entry_file.name}")
    
    def migrate_single_entry(self, entry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a single entry data structure."""
        # The entry data structure is already compatible between promo and work journal
        # No changes needed to the data format itself
        
        # However, we might want to update any metadata references
        if "metadata" in entry_data:
            metadata = entry_data["metadata"]
            
            # Update any tool references in metadata
            if "workflowConfig" in metadata:
                # This is already in the correct format
                pass
            
            # Add migration metadata
            metadata["migrated_from"] = "promo-journal"
            metadata["migration_date"] = datetime.now().isoformat()
        
        return entry_data
    
    def migrate_exports_directory(self):
        """Migrate exports directory if it exists."""
        old_exports = self.old_dir / "exports"
        new_exports = self.new_dir / "exports"
        
        if not old_exports.exists():
            self.log("No exports directory found to migrate")
            return
            
        export_files = list(old_exports.rglob("*"))
        if not export_files:
            self.log("No export files found to migrate")
            return
            
        self.log(f"Migrating exports directory with {len(export_files)} files...")
        
        if not self.dry_run:
            if old_exports.is_dir():
                shutil.copytree(old_exports, new_exports, dirs_exist_ok=True)
                self.log("Exports directory migrated successfully")
        else:
            self.log(f"Would migrate {old_exports} -> {new_exports}")
    
    def verify_migration(self):
        """Verify the migration was successful."""
        if self.dry_run:
            self.log("Dry run completed successfully - ready for actual migration!")
            return True
            
        self.log("Verifying migration...")
        
        # Check if new directory exists
        if not self.new_dir.exists():
            self.log("ERROR: New work-journal directory was not created", "ERROR")
            return False
        
        # Check config file
        if (self.old_dir / "config.json").exists():
            if not (self.new_dir / "config.json").exists():
                self.log("ERROR: config.json was not migrated", "ERROR")
                return False
        
        # Check .env file
        if (self.old_dir / ".env").exists():
            if not (self.new_dir / ".env").exists():
                self.log("ERROR: .env file was not migrated", "ERROR")
                return False
        
        # Check entries
        old_entries = self.old_dir / "entries"
        new_entries = self.new_dir / "entries"
        if old_entries.exists():
            if not new_entries.exists():
                self.log("ERROR: entries directory was not migrated", "ERROR")
                return False
            
            old_files = len(list(old_entries.glob("*.json")))
            new_files = len(list(new_entries.glob("*.json")))
            if old_files != new_files:
                self.log(f"ERROR: Entry file count mismatch. Old: {old_files}, New: {new_files}", "ERROR")
                return False
        
        self.log("Migration verification completed successfully")
        return True
    
    def cleanup_old_data(self):
        """Remove old promo-journal directory after successful migration."""
        if self.dry_run:
            self.log("Would remove old promo-journal directory")
            return
            
        if not self.old_dir.exists():
            self.log("Old directory already removed or doesn't exist")
            return
            
        self.log("Removing old promo-journal directory...")
        shutil.rmtree(self.old_dir)
        self.log("Old directory removed successfully")
    
    def run_migration(self, cleanup: bool = False):
        """Run the complete migration process."""
        self.log("Starting promo-journal to work-journal migration")
        
        if not self.old_dir.exists():
            self.log("No existing promo-journal installation found. Nothing to migrate.")
            self.log("This is normal if you haven't used the old promo-journal version.")
            return True
        
        if self.new_dir.exists() and not self.dry_run:
            response = input(f"work-journal directory already exists at {self.new_dir}. Continue? (y/N): ")
            if response.lower() != 'y':
                self.log("Migration cancelled by user")
                return
        
        try:
            # Step 1: Backup existing data
            self.backup_existing_data()
            
            # Step 2: Migrate configuration files
            self.migrate_config_file()
            self.migrate_env_file()
            
            # Step 3: Migrate entry data
            self.migrate_entry_files()
            
            # Step 4: Migrate exports
            self.migrate_exports_directory()
            
            # Step 5: Verify migration
            if self.verify_migration():
                self.log("Migration completed successfully!")
                
                if cleanup and not self.dry_run:
                    response = input("Remove old promo-journal directory? (y/N): ")
                    if response.lower() == 'y':
                        self.cleanup_old_data()
                else:
                    self.log(f"Old data preserved at: {self.old_dir}")
                    if self.create_backup:
                        self.log(f"Backup created at: {self.backup_dir}")
                return True
            else:
                self.log("Migration verification failed!", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Migration failed with error: {e}", "ERROR")
            return False

def main():
    parser = argparse.ArgumentParser(description="Migrate from promo-journal to work-journal")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip creating backup of original data")
    parser.add_argument("--cleanup", action="store_true",
                       help="Remove old promo-journal directory after successful migration")
    
    args = parser.parse_args()
    
    migrator = PromoToWorkJournalMigrator(
        dry_run=args.dry_run,
        create_backup=not args.no_backup
    )
    
    migrator.run_migration(cleanup=args.cleanup)

if __name__ == "__main__":
    main()