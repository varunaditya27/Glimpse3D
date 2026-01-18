"""
Test script for the compare endpoints.

This script tests:
1. Backend server is running
2. Compare endpoints are accessible
3. Database helper functions work correctly
"""

import sys
import requests
from app.core.database import DatabaseManager
from app.core.supabase_client import get_supabase

def test_backend_health():
    """Test if backend server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ Backend server is running")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend server")
        print("  Make sure the backend is running: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Health check error: {str(e)}")
        return False


def test_compare_endpoints():
    """Test compare endpoints."""
    try:
        # Test /compare/projects endpoint
        print("\nTesting GET /api/v1/compare/projects...")
        response = requests.get("http://localhost:8000/api/v1/compare/projects", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Projects list endpoint works")
            print(f"  Total projects: {data.get('total', 0)}")
            
            if data.get('total', 0) >= 2:
                projects = data.get('projects', [])
                project_1 = projects[0]['id']
                project_2 = projects[1]['id']
                
                # Test dual comparison endpoint
                print(f"\nTesting GET /api/v1/compare/{project_1}/{project_2}...")
                comp_response = requests.get(
                    f"http://localhost:8000/api/v1/compare/{project_1}/{project_2}",
                    timeout=10
                )
                
                if comp_response.status_code == 200:
                    comp_data = comp_response.json()
                    print("✓ Dual comparison endpoint works")
                    print(f"  Project A: {comp_data['project_a']['id']}")
                    print(f"  Project B: {comp_data['project_b']['id']}")
                    print(f"  Comparison: {comp_data['comparison']}")
                    return True
                else:
                    print(f"✗ Dual comparison failed: {comp_response.status_code}")
                    return False
            else:
                print("  Note: Less than 2 projects available for comparison test")
                print("  This is expected if you haven't created projects yet")
                return True
        else:
            print(f"✗ Projects list failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Endpoint test error: {str(e)}")
        return False


def test_database_functions():
    """Test database helper functions."""
    try:
        print("\nTesting database helper functions...")
        
        # Test get_projects_for_comparison
        projects = DatabaseManager.get_projects_for_comparison(limit=5)
        print(f"✓ get_projects_for_comparison works")
        print(f"  Found {len(projects)} projects")
        
        if len(projects) > 0:
            # Test get_project_comparison_data
            project_id = projects[0]['id']
            comparison_data = DatabaseManager.get_project_comparison_data(project_id)
            
            if comparison_data:
                print(f"✓ get_project_comparison_data works")
                print(f"  Project: {comparison_data['id']}")
                print(f"  Status: {comparison_data['status']}")
                print(f"  Model URL: {comparison_data['model']['url']}")
                return True
            else:
                print(f"✗ get_project_comparison_data returned None")
                return False
        else:
            print("  Note: No projects available for testing")
            print("  This is expected if you haven't created projects yet")
            return True
            
    except Exception as e:
        print(f"✗ Database function test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Glimpse3D Compare Feature - Integration Test")
    print("=" * 60)
    
    # Test 1: Backend health
    if not test_backend_health():
        print("\n❌ Backend server is not running. Start it first:")
        print("   cd backend")
        print("   uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Test 2: Database functions (works without backend running)
    db_success = test_database_functions()
    
    # Test 3: API endpoints
    api_success = test_compare_endpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"Backend Health:      {'✓ PASS' if True else '✗ FAIL'}")
    print(f"Database Functions:  {'✓ PASS' if db_success else '✗ FAIL'}")
    print(f"API Endpoints:       {'✓ PASS' if api_success else '✗ FAIL'}")
    print("=" * 60)
    
    if db_success and api_success:
        print("\n🎉 All tests passed! Compare feature is ready to use.")
        print("\nNext steps:")
        print("1. Start the frontend: cd frontend && npm run dev")
        print("2. Navigate to /compare in your browser")
        print("3. Select two projects to compare side-by-side")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        sys.exit(1)
