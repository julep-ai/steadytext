# AIDEV-SECTION: SECURITY
"""
security.py - Security and input validation for pg_steadytext

AIDEV-NOTE: This module provides input validation, sanitization, and security
features for the PostgreSQL extension to prevent injection attacks and abuse.
"""

import re
import hashlib
import logging
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# AIDEV-NOTE: Security constants - adjust based on your requirements
MAX_PROMPT_LENGTH = 10000  # Maximum characters in a prompt
MAX_TOKENS_LIMIT = 4096    # Maximum tokens that can be requested
MIN_TOKENS_LIMIT = 1       # Minimum tokens
MAX_BATCH_SIZE = 100       # Maximum items in batch operations

# Regex patterns for validation
SAFE_TEXT_PATTERN = re.compile(r'^[\w\s\-.,!?;:\'"\(\)\[\]{}/@#$%^&*+=<>|~`₹€£¥§¶†‡¿¡«»""''—–…\n\r\t]*$', re.UNICODE)


class SecurityValidator:
    """
    Validates and sanitizes inputs for pg_steadytext.
    
    AIDEV-NOTE: This class provides methods to validate various inputs
    to prevent security issues like SQL injection, command injection,
    and resource exhaustion attacks.
    """
    
    @staticmethod
    def validate_prompt(prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate a text prompt for safety and length.
        
        AIDEV-NOTE: Checks prompt length and content to prevent abuse.
        Does NOT restrict special characters as they may be needed for
        legitimate prompts, but logs suspicious patterns.
        
        Args:
            prompt: Input prompt to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt:
            return False, "Prompt cannot be empty"
        
        if not isinstance(prompt, str):
            return False, "Prompt must be a string"
        
        if len(prompt) > MAX_PROMPT_LENGTH:
            return False, f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters"
        
        # Check for potential command injection patterns
        dangerous_patterns = [
            r';\s*DROP\s+',  # SQL injection
            r';\s*DELETE\s+', # SQL injection
            r';\s*UPDATE\s+', # SQL injection
            r';\s*INSERT\s+', # SQL injection
            r'<script',       # XSS
            r'javascript:',   # XSS
            r'\x00',         # Null bytes
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes that might bypass filters
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected in prompt: {pattern}")
                # AIDEV-NOTE: We log but don't block - adjust based on security requirements
        
        return True, None
    
    @staticmethod
    def validate_max_tokens(max_tokens: Any) -> tuple[bool, Optional[str]]:
        """
        Validate max_tokens parameter.
        
        AIDEV-NOTE: Ensures max_tokens is within reasonable bounds to
        prevent resource exhaustion.
        
        Args:
            max_tokens: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if max_tokens is None:
            return True, None  # Will use default
        
        try:
            tokens = int(max_tokens)
        except (ValueError, TypeError):
            return False, "max_tokens must be an integer"
        
        if tokens < MIN_TOKENS_LIMIT:
            return False, f"max_tokens must be at least {MIN_TOKENS_LIMIT}"
        
        if tokens > MAX_TOKENS_LIMIT:
            return False, f"max_tokens cannot exceed {MAX_TOKENS_LIMIT}"
        
        return True, None
    
    @staticmethod
    def validate_model_name(model_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate model name parameter.
        
        AIDEV-NOTE: Ensures model name is from allowed list to prevent
        arbitrary model loading.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # AIDEV-NOTE: Extend this list as more models are supported
        ALLOWED_MODELS = {
            'qwen3-1.7b',
            'qwen3-0.6b',
            'qwen3-embedding',
            'small',
            'medium',
            'large'
        }
        
        if not model_name:
            return True, None  # Will use default
        
        if not isinstance(model_name, str):
            return False, "Model name must be a string"
        
        if model_name.lower() not in ALLOWED_MODELS:
            return False, f"Model '{model_name}' is not in allowed list"
        
        return True, None
    
    @staticmethod
    def sanitize_cache_key(key: str) -> str:
        """
        Sanitize a cache key for safe storage.
        
        AIDEV-NOTE: Ensures cache keys don't contain characters that
        could cause issues in PostgreSQL or file systems.
        
        Args:
            key: Raw cache key
            
        Returns:
            Sanitized cache key
        """
        # If key is already a hash, return as-is
        if re.match(r'^[a-f0-9]{32}$', key):
            return key
        
        # Otherwise, hash it to ensure safety
        return hashlib.md5(key.encode()).hexdigest()
    
    @staticmethod
    def validate_batch_size(batch_items: list) -> tuple[bool, Optional[str]]:
        """
        Validate batch operation size.
        
        AIDEV-NOTE: Prevents resource exhaustion from overly large batches.
        
        Args:
            batch_items: List of items in batch
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(batch_items, list):
            return False, "Batch must be a list"
        
        if len(batch_items) == 0:
            return False, "Batch cannot be empty"
        
        if len(batch_items) > MAX_BATCH_SIZE:
            return False, f"Batch size cannot exceed {MAX_BATCH_SIZE} items"
        
        return True, None
    
    @staticmethod
    def validate_json_params(params: Any) -> tuple[bool, Optional[str]]:
        """
        Validate JSON parameters.
        
        AIDEV-NOTE: Ensures params are safe JSON-serializable dict.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if params is None:
            return True, None
        
        if not isinstance(params, dict):
            return False, "Parameters must be a dictionary"
        
        # Check for dangerous keys
        dangerous_keys = ['__proto__', 'constructor', 'prototype']
        for key in params:
            if key in dangerous_keys:
                return False, f"Parameter key '{key}' is not allowed"
        
        # Validate common parameter values
        if 'temperature' in params:
            try:
                temp = float(params['temperature'])
                if temp < 0 or temp > 2:
                    return False, "Temperature must be between 0 and 2"
            except (ValueError, TypeError):
                return False, "Temperature must be a number"
        
        return True, None


class RateLimiter:
    """
    Rate limiting implementation for pg_steadytext.
    
    AIDEV-NOTE: This is a placeholder for rate limiting logic.
    In production, this would integrate with PostgreSQL's rate
    limiting tables to track and enforce limits per user.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
    
    def check_rate_limit(self) -> tuple[bool, Optional[str]]:
        """
        Check if user is within rate limits.
        
        AIDEV-TODO: Implement actual rate limiting logic that:
        1. Queries steadytext_rate_limits table
        2. Updates counters
        3. Resets counters based on time windows
        4. Returns whether request is allowed
        
        Returns:
            Tuple of (is_allowed, error_message)
        """
        # AIDEV-NOTE: Placeholder - always allow for now
        return True, None


def validate_generation_request(prompt: str,
                               max_tokens: Optional[int] = None,
                               model_name: Optional[str] = None,
                               params: Optional[Dict[str, Any]] = None,
                               user_id: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    Validate a complete generation request.
    
    AIDEV-NOTE: Comprehensive validation for text generation requests.
    Use this in PostgreSQL functions before processing.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        model_name: Model to use
        params: Additional parameters
        user_id: User identifier for rate limiting
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = SecurityValidator()
    
    # Validate prompt
    valid, error = validator.validate_prompt(prompt)
    if not valid:
        return False, error
    
    # Validate max_tokens
    valid, error = validator.validate_max_tokens(max_tokens)
    if not valid:
        return False, error
    
    # Validate model name
    if model_name:
        valid, error = validator.validate_model_name(model_name)
        if not valid:
            return False, error
    
    # Validate parameters
    if params:
        valid, error = validator.validate_json_params(params)
        if not valid:
            return False, error
    
    # Check rate limits
    if user_id:
        limiter = RateLimiter(user_id)
        allowed, error = limiter.check_rate_limit()
        if not allowed:
            return False, error
    
    return True, None


# AIDEV-NOTE: Export key functions for PostgreSQL integration
__all__ = [
    'SecurityValidator',
    'RateLimiter',
    'validate_generation_request',
    'MAX_PROMPT_LENGTH',
    'MAX_TOKENS_LIMIT'
]