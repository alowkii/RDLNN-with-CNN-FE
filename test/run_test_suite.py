#!/usr/bin/env python3
"""
Run Test Suite for Image Forgery Detection Techniques

This script automates running the comparative test suite against different datasets
and with different configuration options.

Usage:
  python run_test_suite.py --config config.json

Requirements:
  - Python 3.6+
  - All dependencies for test_forgery_detection_comparative.py
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_suite.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_suite")

def run_test(config, test_name, output_base_dir):
    """Run a single test configuration"""
    logger.info(f"Running test: {test_name}")
    
    # Create output directory for this test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{test_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare command line arguments
    cmd = ["python", "test_forgery_detection_comparative.py"]
    
    # Add required arguments
    cmd.extend(["--test-dir", config["test_dir"]])
    cmd.extend(["--output-dir", output_dir])
    
    # Add model paths
    if "dwt_model" in config:
        cmd.extend(["--dwt-model", config["dwt_model"]])
    if "dywt_model" in config:
        cmd.extend(["--dywt-model", config["dywt_model"]])
    if "rdlnn_model" in config:
        cmd.extend(["--rdlnn-model", config["rdlnn_model"]])
    
    # Add optional parameters
    if "dywt_decomp_level" in config:
        cmd.extend(["--dywt-decomp-level", str(config["dywt_decomp_level"])])
    if config.get("optimize_thresholds", False):
        cmd.append("--optimize-thresholds")
    
    # Add specialized tests
    if config.get("min_forgery_test", False):
        cmd.append("--min-forgery-test")
    if config.get("complex_bg_test", False):
        cmd.append("--complex-bg-test")
    if config.get("compression_test", False):
        cmd.append("--compression-test")
    if config.get("multiple_forgery_test", False):
        cmd.append("--multiple-forgery-test")
    
    # Run the test
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Test completed successfully")
        
        # Save command output
        with open(os.path.join(output_dir, "test_output.log"), "w") as f:
            f.write(result.stdout)
        
        # Generate report
        generate_report(output_dir, config.get("report_format", "markdown"))
        
        return {
            "status": "success",
            "output_dir": output_dir,
            "report_file": os.path.join(output_dir, "test_report." + config.get("report_format", "md"))
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Test failed with error code {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        
        # Save error output
        with open(os.path.join(output_dir, "test_error.log"), "w") as f:
            f.write(e.stdout)
            f.write("\n\nERROR OUTPUT:\n\n")
            f.write(e.stderr)
        
        return {
            "status": "failed",
            "output_dir": output_dir,
            "error_code": e.returncode,
            "error_file": os.path.join(output_dir, "test_error.log")
        }

def generate_report(test_output_dir, format="markdown"):
    """Generate a report from test results"""
    ext = "md" if format == "markdown" else "tex"
    output_file = os.path.join(test_output_dir, f"test_report.{ext}")
    
    cmd = [
        "python", "generate_test_report.py",
        "--results-dir", test_output_dir,
        "--output-file", output_file,
        "--format", format
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Report generated: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation failed: {e}")
        return None

def run_test_suite(config_file):
    """Run a complete test suite from a configuration file"""
    logger.info(f"Loading test suite configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Create output directory
    output_base_dir = config.get("output_base_dir", "test_results")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get default configuration
    default_config = config.get("default", {})
    
    # Run each test
    results = {}
    for test_name, test_config in config.get("tests", {}).items():
        # Merge default config with test-specific config
        full_config = {**default_config, **test_config}
        
        # Run the test
        results[test_name] = run_test(full_config, test_name, output_base_dir)
    
    # Save summary of all tests
    summary_file = os.path.join(output_base_dir, "test_suite_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config_file": config_file,
            "results": results
        }, f, indent=2)
    
    # Print summary
    logger.info("\n== Test Suite Summary ==")
    logger.info(f"Total tests: {len(results)}")
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    logger.info(f"Successful tests: {success_count}")
    logger.info(f"Failed tests: {len(results) - success_count}")
    
    for test_name, result in results.items():
        status = "✅ SUCCESS" if result["status"] == "success" else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        logger.info(f"  Output directory: {result['output_dir']}")
    
    logger.info(f"Summary saved to {summary_file}")
    return 0

def main():
    parser = argparse.ArgumentParser(description='Run a test suite for image forgery detection')
    parser.add_argument('--config', required=True, help='Configuration file for the test suite')
    
    args = parser.parse_args()
    
    return run_test_suite(args.config)

if __name__ == '__main__':
    sys.exit(main())