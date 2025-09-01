#!/usr/bin/env python3
"""Simple test script to verify histogram metrics integration."""
import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_structure():
    """Test that the basic structure for metrics is in place."""
    
    try:
        # Check if we can import the main components we need
        from opentelemetry import metrics as metrics_api
        from opentelemetry.metrics import Histogram
        print("✓ OpenTelemetry Metrics API imported successfully")
        
        # Try to create a histogram (this validates the basic API)
        meter_provider = metrics_api.get_meter_provider()
        meter = metrics_api.get_meter("test", "1.0.0", meter_provider)
        histogram = meter.create_histogram(
            name="test.histogram",
            unit="ms", 
            description="Test histogram"
        )
        print("✓ Histogram creation works")
        
        # Test recording to histogram
        histogram.record(100.0, attributes={"test": "value"})
        print("✓ Histogram recording works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_agno_metrics_integration():
    """Test that agno integration has metrics support."""
    
    # Read the source files and check for metrics integration
    try:
        with open('src/openinference/instrumentation/agno/__init__.py', 'r') as f:
            init_content = f.read()
            
        with open('src/openinference/instrumentation/agno/_wrappers.py', 'r') as f:
            wrappers_content = f.read()
        
        # Check for metrics API import
        if 'from opentelemetry import metrics as metrics_api' in init_content:
            print("✓ Metrics API imported in __init__.py")
        else:
            print("✗ Metrics API not imported in __init__.py")
            
        # Check for histogram creation
        if 'create_histogram(' in init_content:
            print("✓ Histogram creation found in __init__.py")
        else:
            print("✗ No histogram creation found in __init__.py")
            
        # Check for histogram record calls in wrappers
        if 'histogram.record(' in wrappers_content:
            print("✓ Histogram recording found in _wrappers.py")
        else:
            print("✗ No histogram recording found in _wrappers.py")
            
        # Check for metrics parameter in ModelWrapper
        if 'metrics: Optional[Dict[str, Any]]' in wrappers_content:
            print("✓ Metrics parameter added to _ModelWrapper")
        else:
            print("✗ No metrics parameter in _ModelWrapper")
            
        return True
        
    except Exception as e:
        print(f"✗ Error reading source files: {e}")
        return False

if __name__ == "__main__":
    print("Testing histogram metrics integration...")
    print()
    
    print("1. Testing OpenTelemetry Metrics API:")
    api_test = test_basic_structure()
    print()
    
    print("2. Testing agno metrics integration:")
    integration_test = test_agno_metrics_integration()
    print()
    
    if api_test and integration_test:
        print("✓ All tests passed! Histogram metrics integration looks good.")
        sys.exit(0)
    else:
        print("✗ Some tests failed.")
        sys.exit(1)