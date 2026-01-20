"""
Custom logging configuration.

Responsibilities:
- Setup structured logging
- Configure log levels and formats
- Output logs to console and file
"""

import logging
import sys
from pathlib import Path

def setup_logger(name: str = "glimpse3d", level: int = logging.INFO) -> logging.Logger:
    """
    Configures the application logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create console handler with UTF-8 encoding for Unicode support (emojis, etc.)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Force UTF-8 encoding on Windows to support Unicode characters
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional: Add file handler with UTF-8 encoding
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / "glimpse3d.log",
            encoding='utf-8'  # UTF-8 encoding for file as well
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If we can't create log file, just use console
        pass
    
    return logger


# Create default logger instance
logger = setup_logger()
