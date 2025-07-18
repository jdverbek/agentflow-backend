"""
Comprehensive Testing Suite for AgentFlow
Tests real AI execution, API endpoints, and system functionality
NO MOCKS - Only real execution validation
"""
import requests
import time
import json
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
import traceback

class AgentFlowTester:
    """
    Comprehensive testing system that validates real AgentFlow functionality
    """
    
    def __init__(self, base_url: str = "https://agentflow-backend-99xa.onrender.com"):
        self.base_url = base_url
        self.test_results: List[Dict] = []
        self.failed_tests: List[Dict] = []
        self.passed_tests: List[Dict] = []
        
    def log_test(self, test_name: str, success: bool, duration: float, 
                details: Dict = None, error: str = None):
        """Log test results with comprehensive details"""
        result = {
            "test_name": test_name,
            "success": success,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "error": error
        }
        
        self.test_results.append(result)
        
        if success:
            self.passed_tests.append(result)
            print(f"✅ PASS | {test_name} | {duration:.3f}s")
        else:
            self.failed_tests.append(result)
            print(f"❌ FAIL | {test_name} | {duration:.3f}s | {error}")
            
        if details:
            print(f"   Details: {json.dumps(details, indent=2)}")
    
    def test_backend_health(self) -> bool:
        """Test if backend is accessible and healthy"""
        test_name = "Backend Health Check"
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test(test_name, True, duration, {
                    "status_code": response.status_code,
                    "response": data
                })
                return True
            else:
                self.log_test(test_name, False, duration, 
                            {"status_code": response.status_code},
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(test_name, False, duration, error=str(e))
            return False
    
    def test_orchestration_endpoints(self) -> bool:
        """Test if orchestration endpoints are accessible"""
        endpoints = [
            "/api/orchestration/health",
            "/api/orchestration/capabilities",
            "/api/orchestration/examples"
        ]
        
        all_passed = True
        
        for endpoint in endpoints:
            test_name = f"Orchestration Endpoint: {endpoint}"
            start_time = time.time()
            
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(test_name, True, duration, {
                        "status_code": response.status_code,
                        "response_keys": list(data.keys()) if isinstance(data, dict) else "non-dict"
                    })
                else:
                    self.log_test(test_name, False, duration,
                                {"status_code": response.status_code},
                                f"HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(test_name, False, duration, error=str(e))
                all_passed = False
        
        return all_passed
    
    def test_real_ai_execution(self) -> bool:
        """Test REAL AI execution with actual task processing"""
        test_name = "Real AI Execution Test"
        start_time = time.time()
        
        # Simple but real task that should complete quickly
        test_task = "Analyze the current state of AI in healthcare and provide 3 key insights"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/orchestration/execute",
                json={
                    "task": test_task,
                    "max_iterations": 1
                },
                timeout=120  # Allow up to 2 minutes for real AI processing
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate that this is REAL execution, not mock
                is_real_execution = self._validate_real_execution(data, duration)
                
                if is_real_execution:
                    self.log_test(test_name, True, duration, {
                        "task": test_task,
                        "response_structure": list(data.keys()) if isinstance(data, dict) else "non-dict",
                        "execution_time": duration,
                        "validation": "REAL_EXECUTION_CONFIRMED"
                    })
                    return True
                else:
                    self.log_test(test_name, False, duration,
                                {"validation": "MOCK_EXECUTION_DETECTED"},
                                "Detected mock/fake execution instead of real AI processing")
                    return False
            else:
                self.log_test(test_name, False, duration,
                            {"status_code": response.status_code, "response": response.text},
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(test_name, False, duration, error=str(e))
            return False
    
    def _validate_real_execution(self, response_data: Dict, duration: float) -> bool:
        """Validate that execution was real, not mock"""
        
        # Check 1: Duration should be reasonable for real AI processing (>5 seconds)
        if duration < 5.0:
            print(f"   ⚠️  Suspicious: Execution too fast ({duration:.3f}s) for real AI processing")
            return False
        
        # Check 2: Response should contain actual AI-generated content
        if isinstance(response_data, dict):
            # Look for signs of real content vs mock data
            content_fields = ['result', 'response', 'content', 'analysis', 'insights']
            
            for field in content_fields:
                if field in response_data:
                    content = str(response_data[field])
                    
                    # Check for mock indicators
                    mock_indicators = [
                        "mock", "fake", "placeholder", "example", "demo",
                        "lorem ipsum", "test data", "sample"
                    ]
                    
                    if any(indicator in content.lower() for indicator in mock_indicators):
                        print(f"   ⚠️  Suspicious: Found mock indicators in {field}")
                        return False
                    
                    # Check for reasonable content length (real AI generates substantial content)
                    if len(content) < 100:
                        print(f"   ⚠️  Suspicious: Content too short ({len(content)} chars) for real AI")
                        return False
        
        # Check 3: Look for execution metadata that indicates real processing
        if 'execution_time' in response_data:
            exec_time = response_data.get('execution_time', 0)
            if exec_time < 5.0:
                print(f"   ⚠️  Suspicious: Reported execution time too fast ({exec_time}s)")
                return False
        
        print(f"   ✅ Validation passed: Real execution confirmed ({duration:.3f}s)")
        return True
    
    def test_powerpoint_task(self) -> bool:
        """Test the specific PowerPoint task that the user requested"""
        test_name = "PowerPoint Task Test"
        start_time = time.time()
        
        powerpoint_task = '''Create a powerpoint presentation in English about "Grote Taalmodellen en agentic AI in cardiologie". Search for the latest novelties. Fyi: we're based in Belgium. Provide narrative text for each slide. the presentation itself should have a duration of about 30 min.'''
        
        try:
            response = requests.post(
                f"{self.base_url}/api/orchestration/execute",
                json={
                    "task": powerpoint_task,
                    "max_iterations": 2
                },
                timeout=180  # Allow up to 3 minutes for complex task
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate PowerPoint-specific requirements
                is_valid_ppt = self._validate_powerpoint_response(data, duration)
                
                if is_valid_ppt:
                    self.log_test(test_name, True, duration, {
                        "task": "PowerPoint about LLMs in Cardiology",
                        "duration": duration,
                        "validation": "POWERPOINT_REQUIREMENTS_MET"
                    })
                    return True
                else:
                    self.log_test(test_name, False, duration,
                                {"validation": "POWERPOINT_REQUIREMENTS_NOT_MET"},
                                "PowerPoint task requirements not satisfied")
                    return False
            else:
                self.log_test(test_name, False, duration,
                            {"status_code": response.status_code},
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(test_name, False, duration, error=str(e))
            return False
    
    def _validate_powerpoint_response(self, response_data: Dict, duration: float) -> bool:
        """Validate PowerPoint-specific response requirements"""
        
        # Must be real execution (not too fast)
        if duration < 10.0:
            print(f"   ⚠️  PowerPoint task too fast ({duration:.3f}s) for real research and creation")
            return False
        
        # Look for PowerPoint-related content
        content_str = json.dumps(response_data).lower()
        
        required_elements = [
            "slide", "presentation", "cardiology", "belgium"
        ]
        
        for element in required_elements:
            if element not in content_str:
                print(f"   ⚠️  Missing required element: {element}")
                return False
        
        # Check for substantial content (real PowerPoint should have significant content)
        if len(content_str) < 1000:
            print(f"   ⚠️  PowerPoint response too short ({len(content_str)} chars)")
            return False
        
        print(f"   ✅ PowerPoint validation passed: All requirements met")
        return True
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid requests"""
        test_cases = [
            {
                "name": "Empty Task",
                "payload": {"task": "", "max_iterations": 1},
                "expected_status": 400
            },
            {
                "name": "Missing Task",
                "payload": {"max_iterations": 1},
                "expected_status": 400
            },
            {
                "name": "Invalid JSON",
                "payload": "invalid json",
                "expected_status": 400
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            test_name = f"Error Handling: {test_case['name']}"
            start_time = time.time()
            
            try:
                if isinstance(test_case['payload'], str):
                    # Test invalid JSON
                    response = requests.post(
                        f"{self.base_url}/api/orchestration/execute",
                        data=test_case['payload'],
                        headers={'Content-Type': 'application/json'},
                        timeout=10
                    )
                else:
                    response = requests.post(
                        f"{self.base_url}/api/orchestration/execute",
                        json=test_case['payload'],
                        timeout=10
                    )
                
                duration = time.time() - start_time
                
                if response.status_code == test_case['expected_status']:
                    self.log_test(test_name, True, duration, {
                        "expected_status": test_case['expected_status'],
                        "actual_status": response.status_code
                    })
                else:
                    self.log_test(test_name, False, duration,
                                {"expected": test_case['expected_status'], 
                                 "actual": response.status_code},
                                f"Expected {test_case['expected_status']}, got {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(test_name, False, duration, error=str(e))
                all_passed = False
        
        return all_passed
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🧪 Starting Comprehensive AgentFlow Testing Suite")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Run all test categories
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Orchestration Endpoints", self.test_orchestration_endpoints),
            ("Real AI Execution", self.test_real_ai_execution),
            ("PowerPoint Task", self.test_powerpoint_task),
            ("Error Handling", self.test_error_handling)
        ]
        
        category_results = {}
        
        for category_name, test_func in tests:
            print(f"\n🔍 Testing: {category_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                category_results[category_name] = result
            except Exception as e:
                print(f"❌ Category {category_name} failed with exception: {e}")
                category_results[category_name] = False
        
        overall_duration = time.time() - overall_start
        
        # Generate comprehensive report
        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_count,
                "failed": failed_count,
                "success_rate": round(success_rate, 2),
                "overall_duration": round(overall_duration, 3),
                "timestamp": datetime.now().isoformat()
            },
            "category_results": category_results,
            "detailed_results": self.test_results,
            "failed_tests": self.failed_tests,
            "system_status": "HEALTHY" if success_rate >= 80 else "DEGRADED" if success_rate >= 60 else "CRITICAL"
        }
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_count} ✅")
        print(f"Failed: {failed_count} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Overall Duration: {overall_duration:.3f}s")
        print(f"System Status: {report['system_status']}")
        
        if failed_count > 0:
            print(f"\n❌ FAILED TESTS:")
            for failed_test in self.failed_tests:
                print(f"   • {failed_test['test_name']}: {failed_test['error']}")
        
        return report

def run_agentflow_tests():
    """Main function to run AgentFlow tests"""
    tester = AgentFlowTester()
    return tester.run_comprehensive_tests()

if __name__ == "__main__":
    # Run tests when script is executed directly
    results = run_agentflow_tests()
    
    # Save results to file
    with open("/tmp/agentflow_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Full results saved to: /tmp/agentflow_test_results.json")

