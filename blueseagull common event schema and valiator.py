#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install email-validator')


# In[10]:


# Complete Event Schema & Validator System - Single File Version
# Perfect for Jupyter notebooks and standalone execution

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field, EmailStr, ValidationError


# =============================================================================
# SHARED TYPES AND ENUMS
# =============================================================================

class EventStatus(str, Enum):
    """Common event status values."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceName(str, Enum):
    """Service identifiers."""
    BODEGA = "bodega"
    DESPACHO = "despacho"
    IVORY_GULL = "ivory_gull"


# Event type literals for type safety
UserEventTypes = Literal[
    "user.created.v1",
    "user.updated.v1", 
    "user.deleted.v1"
]

SessionEventTypes = Literal[
    "session.created.v1",
    "session.expired.v1",
    "session.destroyed.v1"
]

OrderEventTypes = Literal[
    "order.placed.v1",
    "order.confirmed.v1",
    "order.cancelled.v1"
]

# Union of all event types
AllEventTypes = Union[UserEventTypes, SessionEventTypes, OrderEventTypes]


# =============================================================================
# BASE EVENT MODEL
# =============================================================================

class BaseEvent(BaseModel):
    """
    Base event model that defines common fields for all events.
    
    All specific event schemas must inherit from this class.
    """
    type: str = Field(..., description="Event type identifier (e.g., 'user.created.v1')")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="ISO 8601 timestamp")
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique event identifier")
    source_service: Optional[str] = Field(None, description="Service that generated the event")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    payload: Dict[str, Any] = Field(..., description="Event-specific data")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        extra = "forbid"  # Prevent additional fields


# =============================================================================
# SPECIFIC EVENT SCHEMAS
# =============================================================================

# User Created Event
class UserCreatedPayload(BaseModel):
    """Payload schema for user created events."""
    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., description="Username")
    first_name: Optional[str] = Field(None, description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    is_active: bool = Field(True, description="Whether user account is active")
    created_at: datetime = Field(..., description="Account creation timestamp")
    metadata: Optional[dict] = Field(None, description="Additional user metadata")


class UserCreatedEvent(BaseEvent):
    """User created event schema."""
    type: str = Field(default="user.created.v1", const=True)
    payload: UserCreatedPayload


# Session Expired Event
class SessionExpiredPayload(BaseModel):
    """Payload schema for session expired events."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User ID associated with session")
    expired_at: datetime = Field(..., description="Session expiration timestamp")
    reason: str = Field(..., description="Reason for expiration (timeout, logout, etc.)")
    duration_minutes: Optional[int] = Field(None, description="Session duration in minutes")
    ip_address: Optional[str] = Field(None, description="Last known IP address")


class SessionExpiredEvent(BaseEvent):
    """Session expired event schema."""
    type: str = Field(default="session.expired.v1", const=True)
    payload: SessionExpiredPayload


# Order Placed Event
class OrderItem(BaseModel):
    """Individual order item."""
    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(..., description="Product name")
    quantity: int = Field(..., gt=0, description="Quantity ordered")
    unit_price: Decimal = Field(..., gt=0, description="Price per unit")
    total_price: Decimal = Field(..., gt=0, description="Total price for this item")


class OrderPlacedPayload(BaseModel):
    """Payload schema for order placed events."""
    order_id: str = Field(..., description="Unique order identifier")
    user_id: str = Field(..., description="User who placed the order")
    status: EventStatus = Field(default=EventStatus.PENDING, description="Order status")
    items: List[OrderItem] = Field(..., description="List of ordered items")
    total_amount: Decimal = Field(..., gt=0, description="Total order amount")
    currency: str = Field(default="USD", description="Currency code")
    shipping_address: dict = Field(..., description="Shipping address details")
    payment_method: str = Field(..., description="Payment method used")
    placed_at: datetime = Field(..., description="Order placement timestamp")


class OrderPlacedEvent(BaseEvent):
    """Order placed event schema."""
    type: str = Field(default="order.placed.v1", const=True)
    payload: OrderPlacedPayload


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

class SchemaRegistry:
    """Registry for managing event schemas."""
    
    def __init__(self):
        self._schemas: Dict[str, Type[BaseEvent]] = {}
        self._register_default_schemas()
    
    def _register_default_schemas(self):
        """Register all default event schemas."""
        schemas = [
            UserCreatedEvent,
            SessionExpiredEvent,
            OrderPlacedEvent,
        ]
        
        for schema_class in schemas:
            self.register(schema_class)
    
    def register(self, schema_class: Type[BaseEvent]) -> None:
        """
        Register a new event schema.
        
        Args:
            schema_class: Event schema class that inherits from BaseEvent
        """
        if not issubclass(schema_class, BaseEvent):
            raise ValueError(f"Schema {schema_class} must inherit from BaseEvent")
        
        # Get the event type from the schema's default type field
        event_type = schema_class.__fields__['type'].default
        if not event_type:
            raise ValueError(f"Schema {schema_class} must have a default type field")
        
        self._schemas[event_type] = schema_class
    
    def get_schema(self, event_type: str) -> Type[BaseEvent]:
        """
        Get schema class for an event type.
        
        Args:
            event_type: Event type identifier
            
        Returns:
            Schema class
            
        Raises:
            KeyError: If event type is not registered
        """
        if event_type not in self._schemas:
            raise KeyError(f"Unknown event type: {event_type}")
        return self._schemas[event_type]
    
    def list_event_types(self) -> List[str]:
        """List all registered event types."""
        return list(self._schemas.keys())
    
    def is_registered(self, event_type: str) -> bool:
        """Check if an event type is registered."""
        return event_type in self._schemas
    
    def get_schema_info(self, event_type: str) -> dict:
        """Get detailed information about a schema."""
        schema_class = self.get_schema(event_type)
        return {
            "event_type": event_type,
            "class_name": schema_class.__name__,
            "description": schema_class.__doc__ or "No description",
            "fields": list(schema_class.__fields__.keys()),
            "payload_fields": list(schema_class.__fields__['payload'].type_.__fields__.keys()) if hasattr(schema_class.__fields__['payload'].type_, '__fields__') else []
        }


# Global registry instance
registry = SchemaRegistry()


# =============================================================================
# VALIDATION LOGIC
# =============================================================================

class EventValidationError(Exception):
    """Custom exception for event validation errors."""
    
    def __init__(self, event_type: str, errors: list, original_data: dict):
        self.event_type = event_type
        self.errors = errors
        self.original_data = original_data
        super().__init__(f"Validation failed for event type '{event_type}': {errors}")


def validate_event(event: Dict[str, Any], event_type: Optional[str] = None) -> BaseEvent:
    """
    Validate an event against its registered schema.
    
    Args:
        event: Raw event data as dictionary
        event_type: Optional event type override. If not provided, uses event['type']
        
    Returns:
        Validated event instance
        
    Raises:
        EventValidationError: If validation fails
        KeyError: If event type is not registered
    """
    # Determine event type
    if event_type is None:
        if 'type' not in event:
            raise EventValidationError(
                "unknown", 
                ["Event must contain 'type' field"], 
                event
            )
        event_type = event['type']
    
    # Get schema class
    try:
        schema_class = registry.get_schema(event_type)
    except KeyError as e:
        raise EventValidationError(
            event_type,
            [f"Unknown event type: {event_type}"],
            event
        ) from e
    
    # Validate event
    try:
        return schema_class(**event)
    except ValidationError as e:
        raise EventValidationError(
            event_type,
            e.errors(),
            event
        ) from e


def validate_event_safe(event: Dict[str, Any], event_type: Optional[str] = None) -> tuple[bool, Optional[BaseEvent], Optional[str]]:
    """
    Safely validate an event without raising exceptions.
    
    Args:
        event: Raw event data as dictionary
        event_type: Optional event type override
        
    Returns:
        Tuple of (is_valid, validated_event_or_none, error_message_or_none)
    """
    try:
        validated_event = validate_event(event, event_type)
        return True, validated_event, None
    except EventValidationError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Unexpected validation error: {str(e)}"


def validate_batch_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a batch of events and return summary statistics.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        "total_events": len(events),
        "valid_events": 0,
        "invalid_events": 0,
        "validation_errors": [],
        "event_type_counts": {},
        "validated_events": []
    }
    
    for i, event in enumerate(events):
        is_valid, validated_event, error = validate_event_safe(event)
        
        if is_valid:
            results["valid_events"] += 1
            results["validated_events"].append(validated_event)
            
            # Count event types
            event_type = validated_event.type
            results["event_type_counts"][event_type] = results["event_type_counts"].get(event_type, 0) + 1
        else:
            results["invalid_events"] += 1
            results["validation_errors"].append({
                "index": i,
                "event": event,
                "error": error
            })
    
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_user_event(user_id: str, email: str, username: str, **kwargs) -> UserCreatedEvent:
    """Helper function to create a valid user created event."""
    payload_data = {
        "user_id": user_id,
        "email": email,
        "username": username,
        "created_at": datetime.utcnow(),
        **kwargs
    }
    
    event_data = {
        "type": "user.created.v1",
        "payload": payload_data
    }
    
    return UserCreatedEvent(**event_data)


def create_session_event(session_id: str, user_id: str, reason: str, **kwargs) -> SessionExpiredEvent:
    """Helper function to create a valid session expired event."""
    payload_data = {
        "session_id": session_id,
        "user_id": user_id,
        "expired_at": datetime.utcnow(),
        "reason": reason,
        **kwargs
    }
    
    event_data = {
        "type": "session.expired.v1",
        "payload": payload_data
    }
    
    return SessionExpiredEvent(**event_data)


def create_order_event(order_id: str, user_id: str, items: List[dict], 
                       shipping_address: dict, payment_method: str, **kwargs) -> OrderPlacedEvent:
    """Helper function to create a valid order placed event."""
    # Convert items to OrderItem objects and calculate total
    order_items = []
    total_amount = Decimal('0')
    
    for item_data in items:
        order_item = OrderItem(**item_data)
        order_items.append(order_item)
        total_amount += order_item.total_price
    
    payload_data = {
        "order_id": order_id,
        "user_id": user_id,
        "items": [item.dict() for item in order_items],
        "total_amount": total_amount,
        "shipping_address": shipping_address,
        "payment_method": payment_method,
        "placed_at": datetime.utcnow(),
        **kwargs
    }
    
    event_data = {
        "type": "order.placed.v1",
        "payload": payload_data
    }
    
    return OrderPlacedEvent(**event_data)


def export_schemas_to_json() -> Dict[str, Any]:
    """Export all registered schemas to JSON format for documentation."""
    schemas_info = {}
    
    for event_type in registry.list_event_types():
        try:
            schema_info = registry.get_schema_info(event_type)
            schema_class = registry.get_schema(event_type)
            
            # Get the JSON schema
            json_schema = schema_class.schema()
            
            schemas_info[event_type] = {
                "info": schema_info,
                "json_schema": json_schema
            }
        except Exception as e:
            schemas_info[event_type] = {"error": str(e)}
    
    return schemas_info




# In[9]:


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def run_examples():
    """Run comprehensive examples demonstrating the event system."""
    print("🚀 Event Schema & Validator System Demo")
    print("=" * 50)
    
    # Example 1: Create and validate user event using helper
    print("\n1️⃣ Creating User Event with Helper Function")
    try:
        user_event = create_user_event(
            user_id="user_123",
            email="john.doe@example.com", 
            username="johndoe",
            first_name="John",
            last_name="Doe",
            metadata={"signup_source": "web"}
        )
        print("✅ User event created successfully!")
        print(f"   Event ID: {user_event.event_id[:8]}...")
        print(f"   User: {user_event.payload.username} ({user_event.payload.email})")
    except Exception as e:
        print(f"❌ User event creation failed: {e}")
    
    # Example 2: Validate raw event data
    print("\n2️⃣ Validating Raw Event Data")
    raw_order_data = {
        "type": "order.placed.v1",
        "payload": {
            "order_id": "order_789",
            "user_id": "user_123", 
            "status": "pending",
            "items": [
                {
                    "product_id": "prod_001",
                    "product_name": "Premium Coffee",
                    "quantity": 2,
                    "unit_price": "15.99",
                    "total_price": "31.98"
                }
            ],
            "total_amount": "31.98",
            "currency": "USD",
            "shipping_address": {
                "street": "123 Main St",
                "city": "Anytown", 
                "state": "CA",
                "zip": "12345"
            },
            "payment_method": "credit_card",
            "placed_at": datetime.utcnow().isoformat()
        },
        "source_service": "bodega"
    }
    
    try:
        validated_order = validate_event(raw_order_data)
        print("✅ Order event validation successful!")
        print(f"   Order ID: {validated_order.payload.order_id}")
        print(f"   Total: ${validated_order.payload.total_amount}")
        print(f"   Items: {len(validated_order.payload.items)}")
    except EventValidationError as e:
        print(f"❌ Order validation failed: {e}")
    
    # Example 3: Handle invalid events
    print("\n3️⃣ Testing Invalid Event Handling")
    invalid_events = [
        {
            "type": "user.created.v1",
            "payload": {
                "user_id": "invalid_user",
                # Missing required email field
                "username": "baduser"
            }
        },
        {
            "type": "unknown.event.v1",
            "payload": {"data": "test"}
        }
    ]
    
    for i, invalid_event in enumerate(invalid_events):
        is_valid, validated, error = validate_event_safe(invalid_event)
        if not is_valid:
            print(f"✅ Invalid event {i+1} correctly rejected: {error.split(':')[0]}")
        else:
            print(f"❌ Invalid event {i+1} was incorrectly accepted")
    
    # Example 4: Batch validation
    print("\n4️⃣ Batch Event Validation")
    batch_events = [
        raw_order_data,
        user_event.dict(),
        invalid_events[0]  # This one should fail
    ]
    
    batch_results = validate_batch_events(batch_events)
    print(f"📊 Batch Results:")
    print(f"   Total: {batch_results['total_events']}")
    print(f"   Valid: {batch_results['valid_events']}")
    print(f"   Invalid: {batch_results['invalid_events']}")
    print(f"   Event Types: {list(batch_results['event_type_counts'].keys())}")
    
    # Example 5: Registry inspection
    print("\n5️⃣ Schema Registry Information")
    print(f"📋 Registered Event Types ({len(registry.list_event_types())}):")
    for event_type in registry.list_event_types():
        info = registry.get_schema_info(event_type)
        print(f"   • {event_type}")
        print(f"     Fields: {len(info['payload_fields'])} payload fields")
    
    # Example 6: Export schemas
    print("\n6️⃣ Schema Export")
    exported = export_schemas_to_json()
    print(f"📄 Exported {len(exported)} schema definitions to JSON")
    
    print("\n🎉 Demo completed successfully!")


# Run examples if executed directly
if __name__ == "__main__":
    run_examples()


# In[ ]:




