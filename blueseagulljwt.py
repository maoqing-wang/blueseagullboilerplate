#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
JWT Utility Module for Authentication & Authorization

Provides essential utilities to encode, decode, and verify JWT tokens for both 
user and service-to-service authentication. Designed to be stateless, secure, 
and compatible with any async-compatible FastAPI-based microservice.
"""

import os
import jwt
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Union
from enum import Enum


# Configure logging
logger = logging.getLogger(__name__)


class TokenError(Exception):
    """Base exception for JWT token-related errors"""
    pass


class InvalidTokenError(TokenError):
    """Raised when token is invalid or malformed"""
    pass


class ExpiredTokenError(TokenError):
    """Raised when token has expired"""
    pass


class InvalidSignatureError(TokenError):
    """Raised when token signature is invalid"""
    pass


class MissingClaimError(TokenError):
    """Raised when required claim is missing from token"""
    pass


class Algorithm(Enum):
    """Supported JWT algorithms"""
    HS256 = "HS256"  # HMAC with SHA-256 (symmetric)
    RS256 = "RS256"  # RSA with SHA-256 (asymmetric)


class JWTHandler:
    """
    JWT token handler providing encoding, decoding, and validation utilities.
    
    Supports both HMAC (HS256) and RSA (RS256) signing algorithms.
    Handles token expiry, signature validation, and claim extraction.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        private_key: Optional[str] = None,
        public_key: Optional[str] = None,
        algorithm: Algorithm = Algorithm.HS256,
        default_expiry_minutes: int = 60,
        issuer: Optional[str] = None,
        audience: Optional[str] = None
    ):
        """
        Initialize JWT handler with configuration.
        
        Args:
            secret_key: Secret key for HMAC algorithms (HS256)
            private_key: Private key for RSA algorithms (RS256)
            public_key: Public key for RSA algorithms (RS256)
            algorithm: JWT signing algorithm
            default_expiry_minutes: Default token expiry in minutes
            issuer: Token issuer claim (iss)
            audience: Token audience claim (aud)
        """
        self.algorithm = algorithm
        self.default_expiry_minutes = default_expiry_minutes
        self.issuer = issuer or os.getenv("JWT_ISSUER")
        self.audience = audience or os.getenv("JWT_AUDIENCE")
        
        # Set up keys based on algorithm
        if algorithm == Algorithm.HS256:
            self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY")
            if not self.secret_key:
                raise ValueError("SECRET_KEY is required for HS256 algorithm")
            self.signing_key = self.secret_key
            self.verification_key = self.secret_key
        
        elif algorithm == Algorithm.RS256:
            self.private_key = private_key or os.getenv("JWT_PRIVATE_KEY")
            self.public_key = public_key or os.getenv("JWT_PUBLIC_KEY")
            
            if not self.private_key:
                raise ValueError("PRIVATE_KEY is required for RS256 algorithm")
            if not self.public_key:
                raise ValueError("PUBLIC_KEY is required for RS256 algorithm")
                
            self.signing_key = self.private_key
            self.verification_key = self.public_key
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def encode_token(
        self,
        payload: Dict[str, Any],
        expiry_minutes: Optional[int] = None,
        include_standard_claims: bool = True
    ) -> str:
        """
        Encode and sign a JWT token with the given payload.
        
        Args:
            payload: Custom claims to include in the token
            expiry_minutes: Token expiry in minutes (overrides default)
            include_standard_claims: Whether to include iss, aud, iat, exp claims
            
        Returns:
            Signed JWT token string
            
        Raises:
            TokenError: If encoding fails
        """
        try:
            # Create a copy to avoid modifying the original payload
            token_payload = payload.copy()
            
            if include_standard_claims:
                now = datetime.now(timezone.utc)
                expiry = expiry_minutes or self.default_expiry_minutes
                
                # Add standard claims
                token_payload.update({
                    "iat": now,  # Issued at
                    "exp": now + timedelta(minutes=expiry),  # Expiry
                })
                
                if self.issuer:
                    token_payload["iss"] = self.issuer
                    
                if self.audience:
                    token_payload["aud"] = self.audience
            
            # Encode the token
            token = jwt.encode(
                token_payload,
                self.signing_key,
                algorithm=self.algorithm.value
            )
            
            logger.info(f"Token encoded successfully for payload keys: {list(payload.keys())}")
            return token
            
        except Exception as e:
            error_msg = f"Failed to encode token: {str(e)}"
            logger.error(error_msg)
            raise TokenError(error_msg) from e

    def decode_token(
        self,
        token: str,
        verify_expiry: bool = True,
        verify_signature: bool = True,
        verify_audience: bool = True
    ) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string to decode
            verify_expiry: Whether to verify token expiry
            verify_signature: Whether to verify token signature
            verify_audience: Whether to verify audience claim
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is malformed
            ExpiredTokenError: If token has expired
            InvalidSignatureError: If signature verification fails
        """
        try:
            # Prepare verification options
            options = {
                "verify_signature": verify_signature,
                "verify_exp": verify_expiry,
                "verify_aud": verify_audience and bool(self.audience),
            }
            
            # Decode the token
            payload = jwt.decode(
                token,
                self.verification_key,
                algorithms=[self.algorithm.value],
                audience=self.audience if verify_audience else None,
                options=options
            )
            
            logger.debug("Token decoded successfully")
            return payload
            
        except jwt.ExpiredSignatureError as e:
            error_msg = "Token has expired"
            logger.warning(error_msg)
            raise ExpiredTokenError(error_msg) from e
            
        except jwt.InvalidSignatureError as e:
            error_msg = "Invalid token signature"
            logger.warning(error_msg)
            raise InvalidSignatureError(error_msg) from e
            
        except jwt.InvalidTokenError as e:
            error_msg = f"Invalid token: {str(e)}"
            logger.warning(error_msg)
            raise InvalidTokenError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to decode token: {str(e)}"
            logger.error(error_msg)
            raise TokenError(error_msg) from e

    def extract_claim(
        self,
        token: str,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Extract a specific claim from a JWT token.
        
        Args:
            token: JWT token string
            key: Claim key to extract
            default: Default value if claim is not found
            required: Whether the claim is required
            
        Returns:
            Claim value or default
            
        Raises:
            MissingClaimError: If required claim is missing
            TokenError: If token decoding fails
        """
        try:
            payload = self.decode_token(token)
            
            if key not in payload:
                if required:
                    error_msg = f"Required claim '{key}' is missing from token"
                    logger.warning(error_msg)
                    raise MissingClaimError(error_msg)
                return default
                
            return payload[key]
            
        except (InvalidTokenError, ExpiredTokenError, InvalidSignatureError):
            # Re-raise token validation errors
            raise
        except Exception as e:
            error_msg = f"Failed to extract claim '{key}': {str(e)}"
            logger.error(error_msg)
            raise TokenError(error_msg) from e

    def verify_token(self, token: str) -> bool:
        """
        Verify if a token is valid without decoding the full payload.
        
        Args:
            token: JWT token string to verify
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            self.decode_token(token)
            return True
        except TokenError:
            return False

    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Get the expiry datetime of a token.
        
        Args:
            token: JWT token string
            
        Returns:
            Token expiry datetime or None if not available
        """
        try:
            exp_timestamp = self.extract_claim(token, "exp")
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            return None
        except TokenError:
            return None

    def is_token_expired(self, token: str) -> bool:
        """
        Check if a token is expired.
        
        Args:
            token: JWT token string
            
        Returns:
            True if token is expired, False otherwise
        """
        try:
            expiry = self.get_token_expiry(token)
            if expiry:
                return datetime.now(timezone.utc) > expiry
            return False
        except TokenError:
            return True

    def refresh_token(
        self,
        token: str,
        extend_minutes: Optional[int] = None,
        preserve_claims: bool = True
    ) -> str:
        """
        Create a new token with extended expiry from an existing token.
        
        Args:
            token: Existing JWT token
            extend_minutes: Minutes to extend from now (overrides default)
            preserve_claims: Whether to preserve all existing claims
            
        Returns:
            New JWT token with extended expiry
            
        Raises:
            TokenError: If token refresh fails
        """
        try:
            # Decode the existing token (ignoring expiry for refresh)
            payload = self.decode_token(token, verify_expiry=False)
            
            if preserve_claims:
                # Remove standard time-based claims that will be regenerated
                refresh_payload = {
                    k: v for k, v in payload.items() 
                    if k not in ["iat", "exp", "nbf"]
                }
            else:
                # Only preserve essential claims
                essential_claims = ["sub", "user_id", "role", "scope", "permissions"]
                refresh_payload = {
                    k: v for k, v in payload.items() 
                    if k in essential_claims
                }
            
            # Create new token with extended expiry
            return self.encode_token(
                refresh_payload,
                expiry_minutes=extend_minutes
            )
            
        except Exception as e:
            error_msg = f"Failed to refresh token: {str(e)}"
            logger.error(error_msg)
            raise TokenError(error_msg) from e


# Convenience functions for backward compatibility and ease of use
_default_handler: Optional[JWTHandler] = None


def _get_default_handler() -> JWTHandler:
    """Get or create the default JWT handler instance."""
    global _default_handler
    if _default_handler is None:
        _default_handler = JWTHandler()
    return _default_handler


def encode_token(payload: Dict[str, Any], **kwargs) -> str:
    """Convenience function to encode a token using the default handler."""
    return _get_default_handler().encode_token(payload, **kwargs)


def decode_token(token: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to decode a token using the default handler."""
    return _get_default_handler().decode_token(token, **kwargs)


def extract_claim(token: str, key: str, **kwargs) -> Any:
    """Convenience function to extract a claim using the default handler."""
    return _get_default_handler().extract_claim(token, key, **kwargs)


def verify_token(token: str) -> bool:
    """Convenience function to verify a token using the default handler."""
    return _get_default_handler().verify_token(token)




# In[4]:


# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Example: Basic usage with HMAC
    os.environ["JWT_SECRET_KEY"] = "your-super-secret-key-here"
    
    handler = JWTHandler(
        algorithm=Algorithm.HS256,
        default_expiry_minutes=45,
        issuer="your-service",
        audience="your-api"
    )
    
    # Create a token
    user_payload = {
        "user_id": "12345",
        "username": "john_doe",
        "role": "admin",
        "permissions": ["read", "write", "delete"],
        "scope": "api:full"
    }
    
    try:
        # Encode token
        token = handler.encode_token(user_payload)
        print(f"Generated token: {token[:50]}...")
        
        # Decode token
        decoded = handler.decode_token(token)
        print(f"Decoded payload: {decoded}")
        
        # Extract specific claims
        user_id = handler.extract_claim(token, "user_id")
        role = handler.extract_claim(token, "role")
        permissions = handler.extract_claim(token, "permissions")
        
        print(f"User ID: {user_id}")
        print(f"Role: {role}")
        print(f"Permissions: {permissions}")
        
        # Verify token
        is_valid = handler.verify_token(token)
        print(f"Token valid: {is_valid}")
        
        # Check expiry
        expiry = handler.get_token_expiry(token)
        print(f"Token expires at: {expiry}")
        
    except TokenError as e:
        print(f"Token error: {e}")


# In[ ]:




