#!/usr/bin/env python3
"""
Quick test of the profiling functionality.
"""

import json
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dqsim import StatevectorSimulator
import qasmpi
from bosonic_model.qasm import Translator


def test_basic_profiling():
    """Test basic profiling with a real circuit"""
    
    print("\n=== Testing Profiling Functionality ===\n")
    
    # Load a real test circuit
    circuit = Translator().from_qasm(qasmpi.get_circuit("deutsch_n2"))
    
    # Test 1: Without profiling (backward compatibility)
    print("Test 1: simulate_shots WITHOUT profiling...")
    sim = StatevectorSimulator(seed=42)
    result = sim.simulate_shots(circuit, shots=100)
    print(f"  Result type: {type(result)}")
    if isinstance(result, dict):
        print(f"  Keys: {result.keys()}")
        print(f"  Sample counts: {dict(list(result.items())[:3])}")
    else:
        print(f"  Result counts (first 3): {list(result.items())[:3]}")
    
    # Test 2: With profiling
    print("\nTest 2: simulate_shots WITH profiling (collect_profile=True)...")
    sim = StatevectorSimulator(seed=42)
    result = sim.simulate_shots(circuit, shots=100, collect_profile=True)
    print(f"  Result type: {type(result)}")
    
    if isinstance(result, dict) and "profile" in result:
        print(f"  Keys: {result.keys()}")
        print(f"\n  Profile data:")
        profile = result["profile"]
        print(json.dumps(profile, indent=2))
        
        print("\n  OK Profiling data successfully collected!")
        
        # Verify profile has expected fields
        expected_fields = ["num_shots", "num_qubits", "num_instructions", 
                          "preprocessing_ms", "gate_fusion_ms", "parallel_execution_ms", 
                          "total_time_ms"]
        missing_fields = []
        for field in expected_fields:
            if field in profile:
                print(f"    OK {field}: {profile[field]:.4f}" if isinstance(profile[field], float) else f"    OK {field}: {profile[field]}")
            else:
                print(f"    FAIL Missing field: {field}")
                missing_fields.append(field)
        
        # Verify counts are also present
        if "counts" in result:
            print(f"\n  OK Counts data present with {len(result['counts'])} unique states")
        else:
            print(f"\n  FAIL Counts data missing!")
        
        if not missing_fields:
            print("\n  OK All tests passed!")
            return True
        else:
            print(f"\n  ❌ {len(missing_fields)} fields missing")
            return False
    else:
        print(f"  Result: {result}")
        print("  FAIL Profiling data not found in result!")
        return False

if __name__ == "__main__":
    success = test_basic_profiling()
    sys.exit(0 if success else 1)

