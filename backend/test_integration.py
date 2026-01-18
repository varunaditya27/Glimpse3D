"""
Test script to verify Supabase integration is working correctly.

Run this after completing the setup steps in SUPABASE_SETUP.md

Usage:
    python test_integration.py
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_environment():
    """Test that environment variables are set."""
    print("=" * 60)
    print("TEST 1: Environment Variables")
    print("=" * 60)
    
    required_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask the key for security
            display_value = value[:20] + "..." if len(value) > 20 else value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing.append(var)
    
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("Please create a .env file with your Supabase credentials.")
        return False
    
    print("\n✅ All environment variables set!\n")
    return True


def test_supabase_connection():
    """Test connection to Supabase."""
    print("=" * 60)
    print("TEST 2: Supabase Connection")
    print("=" * 60)
    
    try:
        from app.core.supabase_client import get_supabase
        client = get_supabase()
        print("✅ Supabase client initialized")
        
        # Test a simple query
        response = client.table("projects").select("id").limit(1).execute()
        print(f"✅ Database query successful (found {len(response.data)} projects)")
        
        print("\n✅ Supabase connection working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run supabase_schema.sql in the SQL Editor")
        print("2. Check that your SUPABASE_URL and SUPABASE_ANON_KEY are correct")
        print("3. Verify your Supabase project is not paused")
        return False


def test_database_tables():
    """Test that all required tables exist."""
    print("=" * 60)
    print("TEST 3: Database Tables")
    print("=" * 60)
    
    try:
        from app.core.supabase_client import get_supabase
        client = get_supabase()
        
        required_tables = [
            "projects",
            "multiview_generation",
            "depth_maps",
            "gaussian_splat_models",
            "enhancement_iterations",
            "enhanced_views",
            "refinement_metrics",
            "export_history"
        ]
        
        missing_tables = []
        
        for table_name in required_tables:
            try:
                client.table(table_name).select("*").limit(1).execute()
                print(f"✅ Table '{table_name}' exists")
            except Exception as e:
                print(f"❌ Table '{table_name}' not found")
                missing_tables.append(table_name)
        
        if missing_tables:
            print(f"\n❌ Missing tables: {', '.join(missing_tables)}")
            print("Please run supabase_schema.sql in the Supabase SQL Editor")
            return False
        
        print("\n✅ All database tables exist!\n")
        return True
        
    except Exception as e:
        print(f"❌ Database table check failed: {str(e)}")
        return False


def test_storage_buckets():
    """Test that all required storage buckets exist."""
    print("=" * 60)
    print("TEST 4: Storage Buckets")
    print("=" * 60)
    
    try:
        from app.core.supabase_client import get_supabase
        import warnings
        
        # Suppress the trailing slash warning
        warnings.filterwarnings('ignore', message='.*trailing slash.*')
        
        client = get_supabase()
        
        required_buckets = [
            "project-uploads",
            "processed-images",
            "multiview-images",
            "depth-maps",
            "enhanced-views",
            "3d-models"
        ]
        
        # Get list of buckets - try multiple methods
        try:
            buckets_response = client.storage.list_buckets()
            existing_buckets = [bucket['name'] for bucket in buckets_response]
        except Exception as e:
            # Fallback: try to access each bucket individually
            print("Note: Using fallback bucket detection method")
            existing_buckets = []
            for bucket_name in required_buckets:
                try:
                    # Try to list files in bucket (will fail if bucket doesn't exist)
                    client.storage.from_(bucket_name).list()
                    existing_buckets.append(bucket_name)
                except:
                    pass
        
        missing_buckets = []
        
        for bucket_name in required_buckets:
            if bucket_name in existing_buckets:
                print(f"✅ Bucket '{bucket_name}' exists")
            else:
                print(f"❌ Bucket '{bucket_name}' not found")
                missing_buckets.append(bucket_name)
        
        if missing_buckets:
            print(f"\n❌ Missing buckets: {', '.join(missing_buckets)}")
            print("Please create these buckets in Supabase Storage (make them public)")
            print("\nNote: If buckets exist but still show as missing, this may be a")
            print("Supabase Python client issue. Verify buckets exist in dashboard.")
            return False
        
        print("\n✅ All storage buckets exist!\n")
        return True
        
    except Exception as e:
        print(f"❌ Storage bucket check failed: {str(e)}")
        print("\nIf you've created the buckets in Supabase dashboard, you can")
        print("safely ignore this error and proceed with testing uploads.")
        return False


def test_database_operations():
    """Test basic database operations."""
    print("=" * 60)
    print("TEST 5: Database Operations")
    print("=" * 60)
    
    try:
        from app.core.database import DatabaseManager
        
        # Test creating a project
        print("Testing: Create project...")
        project_id = DatabaseManager.create_project()
        print(f"✅ Created test project: {project_id}")
        
        # Test updating project status
        print("Testing: Update project status...")
        DatabaseManager.update_project_status(
            project_id, "preprocessing", "Test status update"
        )
        print("✅ Updated project status")
        
        # Test retrieving project
        print("Testing: Retrieve project...")
        project = DatabaseManager.get_project(project_id)
        if project and project["status"] == "preprocessing":
            print("✅ Retrieved project successfully")
        else:
            print("❌ Failed to retrieve project")
            return False
        
        # Clean up test project
        print("Cleaning up test project...")
        from app.core.supabase_client import get_supabase
        client = get_supabase()
        client.table("projects").delete().eq("id", project_id).execute()
        print("✅ Cleaned up test project")
        
        print("\n✅ Database operations working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Database operations failed: {str(e)}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("GLIMPSE3D SUPABASE INTEGRATION TEST")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Environment Variables", test_environment()))
    
    if results[-1][1]:  # Only continue if env vars are set
        results.append(("Supabase Connection", test_supabase_connection()))
        
        if results[-1][1]:  # Only continue if connection works
            results.append(("Database Tables", test_database_tables()))
            results.append(("Storage Buckets", test_storage_buckets()))
            results.append(("Database Operations", test_database_operations()))
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your Glimpse3D backend is ready to use!")
        print("\nNext steps:")
        print("1. Start the backend: uvicorn app.main:app --reload")
        print("2. Visit http://localhost:8000/docs to see the API")
        print("3. Test the upload endpoint with a sample image")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before continuing.")
        print("\nRefer to SUPABASE_SETUP.md for detailed setup instructions.")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded .env file\n")
    except ImportError:
        print("python-dotenv not installed. Make sure .env variables are set.\n")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
