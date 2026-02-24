"""Logging utilities for experiment tracking."""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from src.utils import paths


def setup_logger(name: str, 
                log_dir: str, 
                level: int = logging.INFO,
                console: bool = True) -> logging.Logger:
    """Setup logger with file and optional console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    # Allow callers to pass None or the literal 'logs' to use configured logs path
    if not log_dir or str(log_dir) == 'logs':
        log_dir = paths.LOGS
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def save_experiment_config(config: Dict[Any, Any], 
                          save_dir: str,
                          filename: str = "experiment_config.json") -> None:
    """Save experiment configuration for reproducibility.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save config
        filename: Name of config file
    """
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    config_with_meta = {
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    with open(save_path, 'w') as f:
        json.dump(config_with_meta, f, indent=2)
    
    print(f"✓ Config saved to {save_path}")
