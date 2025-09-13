"""
Run the complete test suite
"""
import subprocess
import sys
import os

def run_tests():
    """Run all tests with pytest"""
    test_dir = os.path.dirname(__file__)
    
    # Test command
    cmd = [
        sys.executable, '-m', 'pytest',
        os.path.join(test_dir, 'tests'),
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--cov=pipeline',  # Coverage for pipeline module
        '--cov=models',  # Coverage for models module
        '--cov=api',  # Coverage for api module
        '--cov=security',  # Coverage for security module
        '--cov=monitoring',  # Coverage for monitoring module
        '--cov-report=html',  # HTML coverage report
        '--cov-report=term-missing',  # Terminal coverage report
        '--junitxml=test_results.xml',  # JUnit XML report
        '--html=test_report.html',  # HTML test report
        '--self-contained-html'  # Self-contained HTML report
    ]
    
    try:
        print("Running test suite...")
        result = subprocess.run(cmd, cwd=test_dir)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            print("📊 Coverage report generated: htmlcov/index.html")
            print("📋 Test report generated: test_report.html")
        else:
            print("\n❌ Some tests failed!")
            print("Check the output above for details.")
        
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def run_specific_tests(test_pattern):
    """Run specific tests matching pattern"""
    test_dir = os.path.dirname(__file__)
    
    cmd = [
        sys.executable, '-m', 'pytest',
        os.path.join(test_dir, 'tests'),
        '-k', test_pattern,
        '-v'
    ]
    
    try:
        print(f"Running tests matching pattern: {test_pattern}")
        result = subprocess.run(cmd, cwd=test_dir)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def run_unit_tests():
    """Run only unit tests"""
    return run_specific_tests("not integration")

def run_integration_tests():
    """Run only integration tests"""
    return run_specific_tests("integration")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SLM Business Insights test suite")
    parser.add_argument('--pattern', '-k', help='Run tests matching pattern')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    
    args = parser.parse_args()
    
    if args.pattern:
        exit_code = run_specific_tests(args.pattern)
    elif args.unit:
        exit_code = run_unit_tests()
    elif args.integration:
        exit_code = run_integration_tests()
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
