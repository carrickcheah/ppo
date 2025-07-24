"""
Run all fixed tests and save results
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fixed test modules
from test_data_pipeline_fixed import run_data_pipeline_tests
from test_environment_fixed import run_environment_tests  
from test_ppo_components_fixed import run_ppo_component_tests


def main():
    """Main test runner."""
    print("\n" + "="*80)
    print("PPO SCHEDULING SYSTEM - FIXED TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    output_lines = []
    
    # Capture all output
    output_lines.append("="*80)
    output_lines.append("PPO SCHEDULING SYSTEM - FIXED TEST SUITE")
    output_lines.append("="*80)
    output_lines.append(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test categories
    test_suites = [
        ("Data Pipeline", run_data_pipeline_tests),
        ("Environment", run_environment_tests),
        ("PPO Components", run_ppo_component_tests)
    ]
    
    total_passed = 0
    total_failed = 0
    
    for category, test_func in test_suites:
        print(f"\nRunning {category} tests...")
        output_lines.append(f"\nRunning {category} tests...")
        
        start_time = time.time()
        
        try:
            # Capture test output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                test_func()
            test_output = f.getvalue()
            
            # Print and save output
            print(test_output)
            output_lines.append(test_output)
            
            # Test passed
            total_passed += 1
            results.append(f"✅ {category}: PASSED")
            
        except Exception as e:
            # Test failed
            total_failed += 1
            error_msg = f"❌ {category}: FAILED - {str(e)}"
            print(error_msg)
            output_lines.append(error_msg)
            output_lines.append(traceback.format_exc())
            results.append(error_msg)
            
        duration = time.time() - start_time
        output_lines.append(f"Duration: {duration:.2f}s\n")
        
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    output_lines.append("\n" + "="*80)
    output_lines.append("TEST SUMMARY")
    output_lines.append("="*80)
    
    for result in results:
        print(result)
        output_lines.append(result)
        
    print(f"\nTotal: {total_passed + total_failed} test suites")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {total_passed/(total_passed + total_failed)*100:.1f}%")
    
    output_lines.append(f"\nTotal: {total_passed + total_failed} test suites")
    output_lines.append(f"Passed: {total_passed}")
    output_lines.append(f"Failed: {total_failed}")
    output_lines.append(f"Success Rate: {total_passed/(total_passed + total_failed)*100:.1f}%")
    
    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED! The system is ready for training.")
        output_lines.append("\n✅ ALL TESTS PASSED! The system is ready for training.")
        output_lines.append("\n🎉 NEXT STEP: Start training the PPO model!")
        output_lines.append("\nTo start training, run:")
        output_lines.append("  cd /Users/carrickcheah/Project/ppo/app_2")
        output_lines.append("  uv run python phase2/train.py --config configs/training.yaml")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        output_lines.append("\n❌ Some tests failed. Please review the errors above.")
        
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results to file
    result_file = "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result_fixed.txt"
    with open(result_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\n📄 Test results saved to: {result_file}")


if __name__ == "__main__":
    # Import torch here to avoid issues
    import torch
    main()