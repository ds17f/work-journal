"""Logging configuration for Work Journal."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for Work Journal.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses ./work-journal.log in current directory
        console: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("work_journal")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Also set the root logging level to ensure all child loggers inherit properly
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up file logging
    if log_file is None:
        # Default to ./work-journal.log in current working directory
        log_file = Path.cwd() / "work-journal.log"
    
    # Use RotatingFileHandler to prevent log files from getting too large
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Also capture LiteLLM logging
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(logging.DEBUG)
    litellm_logger.addHandler(file_handler)
    
    # Capture httpx logging (used by LiteLLM for HTTP requests)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.DEBUG)
    httpx_logger.addHandler(file_handler)
    
    # Set up console logging if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Simpler format for console
        console_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "work_journal") -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name, typically module name
        
    Returns:
        Logger instance
    """
    # Ensure logging is initialized
    init_logging()
    
    # Create a child logger that inherits from the main work_journal logger
    if name != "work_journal" and not name.startswith("work_journal."):
        name = f"work_journal.{name}"
    
    logger = logging.getLogger(name)
    # Don't add handlers to child loggers - they should inherit from parent
    return logger


# Initialize default logging on import
_default_logger = None

def init_logging(log_level: str = None) -> logging.Logger:
    """
    Initialize logging with default configuration.
    Call this once at application startup.
    
    Args:
        log_level: Override default log level from environment
        
    Returns:
        Main logger instance
    """
    global _default_logger
    
    if _default_logger is None:
        # Get log level from environment or default to INFO
        if log_level is None:
            log_level = os.getenv("WORK_JOURNAL_LOG_LEVEL", "INFO")
        
        # Check if we're in a TUI environment (might want to suppress console logging)
        suppress_console = os.getenv("WORK_JOURNAL_SUPPRESS_CONSOLE_LOG", "false").lower() == "true"
        
        _default_logger = setup_logging(
            log_level=log_level,
            console=not suppress_console
        )
        
        _default_logger.info(f"Work Journal logging initialized - Level: {log_level}")
        _default_logger.info(f"Log file: {Path.cwd() / 'work-journal.log'}")
    
    return _default_logger


# Initialize logging immediately when this module is imported
# This ensures all other modules that import get_logger will have proper logging set up
init_logging()