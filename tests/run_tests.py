#!/usr/bin/env python3
# tests/run_tests.py - Test runner script
"""
Test runner script for Crop Disease Prediction System
Provides convenient commands to run different test suites
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return False


def run_unit_tests(args):
    """Run unit tests"""
    cmd = [sys.executable, '-m', 'pytest', 'tests/test_services.py', '-v']
    if args.coverage:
        cmd.extend(['--cov=backend', '--cov-report=html'])
    return run_command(cmd, "Unit Tests")


def run_api_tests(args):
    """Run API integration tests"""
    # Start Flask app in background for API tests
    import time
    import signal
    import threading

    # Import here to avoid circular imports
    from backend.app import create_app

    def start_test_server():
        app = create_app()
        app.config['TESTING'] = True
        app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

    # Start server in background thread
    server_thread = threading.Thread(target=start_test_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start

    try:
        cmd = [sys.executable, '-m', 'pytest', 'tests/test_api.py', '-v',
               '-k', 'not test_load' if not args.load else '']
        if args.coverage:
            cmd.extend(['--cov=backend/routes', '--cov-report=html'])
        success = run_command(cmd, "API Integration Tests")
    finally:
        # Cleanup - try to kill the server
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except:
            pass

    return success


def run_e2e_tests(args):
    """Run end-to-end tests"""
    # Set environment variables for e2e tests
    env = os.environ.copy()
    env['BASE_URL'] = args.base_url or 'http://localhost:5000'
    env['HEADLESS'] = 'true' if args.headless else 'false'
    env['BROWSER'] = args.browser or 'chrome'

    cmd = [sys.executable, '-m', 'pytest', 'tests/test_e2e.py', '-v', '-m', 'e2e']
    if args.coverage:
        cmd.extend(['--cov-report=html'])

    # Run with modified environment
    result = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=Path(__file__).parent.parent, env=env)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    return result.returncode == 0


def run_load_tests(args):
    """Run load/performance tests"""
    cmd = [sys.executable, '-m', 'pytest', 'tests/test_api.py::TestLoadHandling', '-v', '-s']
    return run_command(cmd, "Load Tests")


def run_all_tests(args):
    """Run all test suites"""
    print("Running complete test suite...")

    results = []

    # Unit tests
    print("\n1. Running Unit Tests...")
    results.append(run_unit_tests(args))

    # API tests
    print("\n2. Running API Integration Tests...")
    results.append(run_api_tests(args))

    # E2E tests (only if not in CI or explicitly requested)
    if not os.environ.get('CI') or args.include_e2e:
        print("\n3. Running End-to-End Tests...")
        results.append(run_e2e_tests(args))

    # Load tests
    if args.load:
        print("\n4. Running Load Tests...")
        results.append(run_load_tests(args))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print('='*60)
    print(f"Unit Tests: {'PASS' if results[0] else 'FAIL'}")
    print(f"API Tests: {'PASS' if results[1] else 'FAIL'}")
    if len(results) > 2:
        print(f"E2E Tests: {'PASS' if results[2] else 'FAIL'}")
    if len(results) > 3:
        print(f"Load Tests: {'PASS' if results[3] else 'FAIL'}")

    all_passed = all(results)
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


def setup_test_environment():
    """Setup test environment and dependencies"""
    print("Setting up test environment...")

    # Install test dependencies
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'tests/requirements.txt']
    if not run_command(cmd, "Installing test dependencies"):
        return False

    # Create test directories
    test_dirs = ['tests/test_data', 'tests/coverage', 'tests/reports']
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("Test environment setup complete.")
    return True


def generate_test_report():
    """Generate test report from coverage and results"""
    print("Generating test report...")

    # Check if coverage report exists
    coverage_dir = Path('tests/coverage')
    if coverage_dir.exists():
        print(f"Coverage report available at: {coverage_dir}/index.html")

    # Generate simple test summary
    print("Test execution complete. Check individual test outputs above.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Crop Disease Prediction System - Test Runner')
    parser.add_argument('command', choices=['unit', 'api', 'e2e', 'load', 'all', 'setup', 'report'],
                       help='Test command to run')
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage reports')
    parser.add_argument('--base-url', default='http://localhost:5000',
                       help='Base URL for e2e tests')
    parser.add_argument('--browser', choices=['chrome', 'firefox'], default='chrome',
                       help='Browser for e2e tests')
    parser.add_argument('--headless', action='store_true',
                       help='Run e2e tests in headless mode')
    parser.add_argument('--load', action='store_true',
                       help='Include load tests')
    parser.add_argument('--include-e2e', action='store_true',
                       help='Include e2e tests even in CI')

    args = parser.parse_args()

    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)

    if args.command == 'setup':
        success = setup_test_environment()
    elif args.command == 'unit':
        success = run_unit_tests(args)
    elif args.command == 'api':
        success = run_api_tests(args)
    elif args.command == 'e2e':
        success = run_e2e_tests(args)
    elif args.command == 'load':
        success = run_load_tests(args)
    elif args.command == 'all':
        success = run_all_tests(args)
    elif args.command == 'report':
        success = generate_test_report()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()