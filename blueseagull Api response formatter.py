#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install fastapi uvicorn pydantic')


# In[4]:


"""
Standalone API Response Formatter
No external dependencies required - works with standard Python library only.
"""
import uuid
import json
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONResponse:
    """Simple JSON response class for non-FastAPI environments."""
    
    def __init__(self, status_code: int, content: Dict[str, Any]):
        self.status_code = status_code
        self.content = content
        self.body = json.dumps(content, default=str).encode()
    
    def dict(self) -> Dict[str, Any]:
        return {"status_code": self.status_code, "content": self.content}
    
    def __str__(self) -> str:
        return f"JSONResponse(status_code={self.status_code}, content={json.dumps(self.content, indent=2)})"


class ResponseFormatter:
    """Centralized response formatter with tracing and logging capabilities."""
    
    def __init__(self, enable_tracing: bool = True, log_responses: bool = False):
        self.enable_tracing = enable_tracing
        self.log_responses = log_responses
        self._current_trace_id: Optional[str] = None
    
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID for request tracking."""
        return f"req_{uuid.uuid4().hex[:12]}"
    
    def _get_trace_id(self) -> Optional[str]:
        """Get current trace ID or generate a new one if tracing is enabled."""
        if not self.enable_tracing:
            return None
        
        if self._current_trace_id is None:
            self._current_trace_id = self._generate_trace_id()
        
        return self._current_trace_id
    
    @contextmanager
    def trace_context(self, trace_id: Optional[str] = None):
        """Context manager for setting trace ID for a request."""
        old_trace_id = self._current_trace_id
        self._current_trace_id = trace_id or self._generate_trace_id()
        try:
            yield self._current_trace_id
        finally:
            self._current_trace_id = old_trace_id
    
    def success_response(
        self,
        data: Any,
        message: str = "OK",
        status_code: int = 200,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized success response.
        
        Args:
            data: The response payload
            message: Human-readable success message
            status_code: HTTP status code (default: 200)
            trace_id: Optional trace ID for request tracking
            
        Returns:
            Standardized success response dictionary
        """
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if trace_id or self.enable_tracing:
            response["trace_id"] = trace_id or self._get_trace_id()
        
        if self.log_responses:
            logger.info(f"Success response: {message}", extra={
                "trace_id": response.get("trace_id"),
                "status_code": status_code
            })
        
        return response
    
    def error_response(
        self,
        message: str,
        status_code: int = 400,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        field_errors: Optional[Dict[str, List[str]]] = None,
        trace_id: Optional[str] = None
    ) -> Union[JSONResponse, Dict[str, Any]]:
        """
        Create a standardized error response.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code (default: 400)
            error_code: Machine-readable error code
            details: Additional error details
            field_errors: Field-specific validation errors
            trace_id: Optional trace ID for request tracking
            
        Returns:
            JSONResponse with standardized error format
        """
        response_data = {
            "success": False,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if error_code:
            response_data["error_code"] = error_code
        
        if details:
            response_data["details"] = details
        
        if field_errors:
            response_data["field_errors"] = field_errors
        
        if trace_id or self.enable_tracing:
            response_data["trace_id"] = trace_id or self._get_trace_id()
        
        if self.log_responses:
            logger.error(f"Error response: {message}", extra={
                "trace_id": response_data.get("trace_id"),
                "status_code": status_code,
                "error_code": error_code
            })
        
        return JSONResponse(status_code=status_code, content=response_data)
    
    def paginated_response(
        self,
        data: List[Any],
        page: int,
        per_page: int,
        total: int,
        message: str = "OK",
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized paginated response.
        
        Args:
            data: List of items for current page
            page: Current page number (1-based)
            per_page: Items per page
            total: Total number of items
            message: Human-readable success message
            trace_id: Optional trace ID for request tracking
            
        Returns:
            Standardized paginated response dictionary
        """
        total_pages = (total + per_page - 1) // per_page  # Ceiling division
        
        pagination = {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
        
        response = self.success_response(data, message, trace_id=trace_id)
        response["pagination"] = pagination
        
        return response


# Global formatter instance
formatter = ResponseFormatter(enable_tracing=True, log_responses=True)


# Convenience functions for ease of use
def success_response(data: Any, message: str = "OK", status_code: int = 200) -> Dict[str, Any]:
    """Create a standardized success response."""
    return formatter.success_response(data, message, status_code)


def error_response(
    message: str,
    status_code: int = 400,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    field_errors: Optional[Dict[str, List[str]]] = None
) -> JSONResponse:
    """Create a standardized error response."""
    return formatter.error_response(message, status_code, error_code, details, field_errors)


def paginated_response(
    data: List[Any],
    page: int,
    per_page: int,
    total: int,
    message: str = "OK"
) -> Dict[str, Any]:
    """Create a standardized paginated response."""
    return formatter.paginated_response(data, page, per_page, total, message)


# Response validation helpers
class ResponseValidator:
    """Helper class to validate response structures."""
    
    @staticmethod
    def is_valid_success_response(response: Dict[str, Any]) -> bool:
        """Check if response follows success format."""
        required_fields = ["success", "message", "data", "timestamp"]
        return all(field in response for field in required_fields) and response["success"] is True
    
    @staticmethod
    def is_valid_error_response(response: Union[Dict[str, Any], JSONResponse]) -> bool:
        """Check if response follows error format."""
        if isinstance(response, JSONResponse):
            data = response.content
        else:
            data = response
        
        required_fields = ["success", "message", "timestamp"]
        return all(field in data for field in required_fields) and data["success"] is False
    
    @staticmethod
    def extract_data(response: Union[Dict[str, Any], JSONResponse]) -> Dict[str, Any]:
        """Extract response data regardless of format."""
        if isinstance(response, JSONResponse):
            return response.content
        return response


# Demo and testing functions
def demo_response_formatter():
    """Demonstrate the response formatter capabilities."""
    
    print("API Response Formatter Demo")
    print("=" * 50)
    
    # 1. Basic success response
    print("\n1. Basic Success Response:")
    user_data = {"id": 123, "name": "John Doe", "email": "john@example.com"}
    success_resp = success_response(user_data, "User retrieved successfully")
    print(json.dumps(success_resp, indent=2))
    
    # 2. Error response with details
    print("\n2. Error Response with Details:")
    error_resp = error_response(
        message="User not found",
        status_code=404,
        error_code="USER_NOT_FOUND",
        details={"attempted_id": 999, "search_method": "database_lookup"}
    )
    print(f"Status Code: {error_resp.status_code}")
    print(json.dumps(error_resp.content, indent=2))
    
    # 3. Validation error response
    print("\n3. Validation Error Response:")
    validation_error = error_response(
        message="Validation failed",
        status_code=422,
        error_code="VALIDATION_ERROR",
        field_errors={
            "email": ["Invalid email format", "Email already exists"],
            "password": ["Password too short"]
        }
    )
    print(f"Status Code: {validation_error.status_code}")
    print(json.dumps(validation_error.content, indent=2))
    
    # 4. Paginated response
    print("\n4. Paginated Response:")
    users_list = [
        {"id": 1, "name": "Alice", "role": "admin"},
        {"id": 2, "name": "Bob", "role": "user"},
        {"id": 3, "name": "Charlie", "role": "user"}
    ]
    paginated_resp = paginated_response(
        data=users_list,
        page=1,
        per_page=3,
        total=25,
        message="Users retrieved successfully"
    )
    print(json.dumps(paginated_resp, indent=2))
    
    # 5. Trace context demonstration
    print("\n5. Trace Context Demo:")
    with formatter.trace_context("custom_trace_abc123") as trace_id:
        print(f"Current Trace ID: {trace_id}")
        
        # Multiple operations within same trace
        operation1 = success_response({"step": 1, "status": "processing"}, "Step 1 completed")
        operation2 = success_response({"step": 2, "status": "completed"}, "Step 2 completed")
        
        print("Operation 1:")
        print(json.dumps(operation1, indent=2))
        print("\nOperation 2:")
        print(json.dumps(operation2, indent=2))
    
    # 6. Response validation
    print("\n6. Response Validation:")
    validator = ResponseValidator()
    
    test_success = success_response({"test": True})
    test_error = error_response("Test error")
    
    print(f"Success response valid: {validator.is_valid_success_response(test_success)}")
    print(f"Error response valid: {validator.is_valid_error_response(test_error)}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


def run_unit_tests():
    """Simple unit tests for the response formatter."""
    
    print("Running Unit Tests")
    print("=" * 30)
    
    def assert_equal(actual, expected, test_name):
        if actual == expected:
            print(f"✓ {test_name}")
        else:
            print(f"✗ {test_name}: Expected {expected}, got {actual}")
    
    def assert_true(condition, test_name):
        if condition:
            print(f"✓ {test_name}")
        else:
            print(f"✗ {test_name}: Condition was False")
    
    # Test 1: Success response structure
    resp = success_response({"test": "data"}, "Test message")
    assert_true(resp["success"], "Success response has success=True")
    assert_equal(resp["message"], "Test message", "Success response message")
    assert_equal(resp["data"]["test"], "data", "Success response data")
    assert_true("timestamp" in resp, "Success response has timestamp")
    
    # Test 2: Error response structure
    error_resp = error_response("Test error", 400, "TEST_ERROR")
    assert_equal(error_resp.status_code, 400, "Error response status code")
    assert_true(error_resp.content["success"] is False, "Error response has success=False")
    assert_equal(error_resp.content["error_code"], "TEST_ERROR", "Error response error_code")
    
    # Test 3: Pagination calculation
    paginated = paginated_response([1, 2, 3], page=2, per_page=5, total=23)
    assert_equal(paginated["pagination"]["total_pages"], 5, "Pagination total_pages calculation")
    assert_true(paginated["pagination"]["has_prev"], "Pagination has_prev for page 2")
    assert_true(paginated["pagination"]["has_next"], "Pagination has_next when more pages exist")
    
    # Test 4: Trace context
    test_formatter = ResponseFormatter(enable_tracing=True)
    with test_formatter.trace_context("test_trace_123") as trace_id:
        assert_equal(trace_id, "test_trace_123", "Trace context sets custom trace ID")
        traced_resp = test_formatter.success_response({"traced": True})
        assert_equal(traced_resp["trace_id"], "test_trace_123", "Response includes trace ID")
    
    print("=" * 30)
    print("Unit tests completed!")


# Example Flask integration (bonus)
def flask_example():
    """Example of how to use with Flask framework."""
    
    try:
        from flask import Flask, jsonify, request
        
        app = Flask(__name__)
        
        @app.route('/api/users/<int:user_id>')
        def get_user(user_id):
            if user_id <= 0:
                error_resp = error_response("Invalid user ID", 400, "INVALID_ID")
                return jsonify(error_resp.content), error_resp.status_code
            
            user_data = {"id": user_id, "name": f"User {user_id}"}
            return jsonify(success_response(user_data, "User found"))
        
        @app.route('/api/users')
        def list_users():
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
            
            # Mock data
            total = 100
            users = [{"id": i, "name": f"User {i}"} for i in range(1, 11)]
            
            return jsonify(paginated_response(users, page, per_page, total))
        
        print("Flask integration example created successfully!")
        print("Routes: GET /api/users/<id>, GET /api/users?page=1&per_page=10")
        
    except ImportError:
        print("Flask not installed - skipping Flask example")


if __name__ == "__main__":
    # Run the demo
    demo_response_formatter()
    
    print("\n" + "="*70 + "\n")
    
    # Run unit tests
    run_unit_tests()
    
    print("\n" + "="*70 + "\n")
    
    # Flask example
    flask_example()


# In[ ]:




