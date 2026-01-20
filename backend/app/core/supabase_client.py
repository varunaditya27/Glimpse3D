"""
Supabase client initialization and configuration.

This module provides a singleton Supabase client instance
for database and storage operations throughout the application.
"""

import os
from typing import Optional
from supabase import create_client, Client
from app.core.logger import logger


class SupabaseClient:
    """Singleton wrapper for Supabase client."""
    
    _instance: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Client:
        """
        Get or create the Supabase client instance.
        
        Returns:
            Supabase client instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        if cls._instance is None:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError(
                    "Missing Supabase credentials. Please set SUPABASE_URL and "
                    "SUPABASE_ANON_KEY environment variables."
                )
            
            cls._instance = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
            
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the client instance (useful for testing)."""
        cls._instance = None


# Convenience function for getting the client
def get_supabase() -> Client:
    """Get the Supabase client instance."""
    return SupabaseClient.get_client()
