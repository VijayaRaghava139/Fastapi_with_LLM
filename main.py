"""
FastAPI Backend for Calorie Estimation System
==============================================

This backend receives RGB images from mobile apps, uses GPT-4o to detect food
items with bounding boxes, and returns structured data for mobile processing.

File: main.py
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import base64
import io
from PIL import Image
import logging
import os
from datetime import datetime

# LangChain imports for LLM integration
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# ==================== CONFIGURATION ====================

load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FASTAPI APP INITIALIZATION ====================

# Create FastAPI application
app = FastAPI(
    title="Calorie Estimation API",
    description="Backend API for food detection with depth sensor support",
    version="2.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Enable CORS (Cross-Origin Resource Sharing) for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PYDANTIC MODELS (DATA STRUCTURES) ====================

class BoundingBox(BaseModel):
    """
    Bounding box coordinates for food item location
    Origin (0,0) is top-left of image
    """
    x: int = Field(..., description="Top-left x coordinate in pixels", ge=0)
    y: int = Field(..., description="Top-left y coordinate in pixels", ge=0)
    width: int = Field(..., description="Width in pixels", gt=0)
    height: int = Field(..., description="Height in pixels", gt=0)
    
    class Config:
        schema_extra = {
            "example": {
                "x": 100,
                "y": 150,
                "width": 200,
                "height": 200
            }
        }

class DetectedFoodItem(BaseModel):
    """
    Single detected food item with all necessary information
    """
    name: str = Field(..., description="Name of the food item (e.g., 'apple', 'grilled chicken')")
    count: int = Field(..., description="Number of this item", ge=1)
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    calories_per_100g: float = Field(..., description="Calories per 100g", gt=0)
    estimated_density_g_cm3: float = Field(
        0.75, 
        description="Estimated density in g/cm³ (used for weight calculation)",
        gt=0
    )
    confidence: float = Field(..., description="Detection confidence (0.0 to 1.0)", ge=0.0, le=1.0)
    image_width: int = Field(..., description="Image width (same as request)")
    image_height: int = Field(..., description="Image height (same as request)")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "apple",
                "count": 2,
                "bbox": {"x": 100, "y": 150, "width": 200, "height": 200},
                "calories_per_100g": 52.0,
                "estimated_density_g_cm3": 0.64,
                "confidence": 0.92,
                "image_width": 1920,
                "image_height": 1080
            }
        }

class FoodDetectionRequest(BaseModel):
    """
    Request from mobile app for food detection
    """
    image_base64: str = Field(..., description="Base64 encoded JPEG image")
    image_width: int = Field(..., description="Image width in pixels", gt=0)
    image_height: int = Field(..., description="Image height in pixels", gt=0)
    user_text: Optional[str] = Field(None, description="Optional user context (e.g., 'my lunch')")
    
    @validator('image_base64')
    def validate_base64(cls, v):
        """Validate that the string is valid base64"""
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 string")
    
    class Config:
        schema_extra = {
            "example": {
                "image_base64": "/9j/4AAQSkZJRg...(truncated)",
                "image_width": 1920,
                "image_height": 1080,
                "user_text": "Calculate calories for my lunch"
            }
        }

class FoodDetectionResponse(BaseModel):
    """
    Response sent back to mobile app
    """
    success: bool = Field(..., description="Whether detection succeeded")
    items: List[DetectedFoodItem] = Field(
        default_factory=list, 
        description="List of detected food items"
    )
    message: Optional[str] = Field(None, description="Error or info message")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "items": [
                    {
                        "name": "apple",
                        "count": 2,
                        "bbox": {"x": 100, "y": 150, "width": 200, "height": 200},
                        "calories_per_100g": 52.0,
                        "estimated_density_g_cm3": 0.64,
                        "confidence": 0.92,
                        "image_width": 1920,
                        "image_height": 1080
                    }
                ],
                "message": None,
                "processing_time_ms": 8534
            }
        }

# ==================== FOOD DATABASE ====================

class FoodDatabase:
    """
    Simple in-memory database for food calories and densities
    In production, use a real database (PostgreSQL, MongoDB, etc.)
    """
    
    # Calories per 100g
    CALORIES_DB = {
        "apple": 52, "banana": 89, "orange": 47, "watermelon": 30,
        "strawberry": 32, "grape": 69, "mango": 60, "pineapple": 50,
        "tomato": 18, "cucumber": 15, "carrot": 41, "lettuce": 15,
        "broccoli": 34, "potato": 77, "sweet potato": 86,
        "bread": 265, "white rice": 130, "brown rice": 112, "pasta": 131,
        "chicken breast": 165, "chicken": 165, "beef": 250, "pork": 242,
        "salmon": 208, "tuna": 184, "shrimp": 99, "fish": 180,
        "egg": 155, "milk": 42, "cheese": 402, "yogurt": 59,
        "pizza": 266, "hamburger": 295, "burger": 295, "sandwich": 230,
        "salad": 50, "soup": 60, "coffee": 2, "tea": 1,
        "soda": 41, "juice": 45, "beer": 43, "wine": 83
    }
    
    # Density in g/cm³
    DENSITY_DB = {
        "apple": 0.64, "banana": 0.94, "orange": 0.84, "watermelon": 0.92,
        "strawberry": 0.60, "grape": 0.70, "mango": 0.85, "pineapple": 0.90,
        "tomato": 0.95, "cucumber": 0.96, "carrot": 1.03, "lettuce": 0.45,
        "broccoli": 0.65, "potato": 1.08, "sweet potato": 1.01,
        "bread": 0.27, "rice": 0.92, "pasta": 0.60,
        "chicken": 1.05, "beef": 1.04, "pork": 1.03,
        "salmon": 1.08, "tuna": 1.09, "shrimp": 1.05, "fish": 1.05,
        "egg": 1.03, "milk": 1.03, "cheese": 1.15, "yogurt": 1.04,
        "pizza": 0.70, "burger": 0.65, "sandwich": 0.55,

        "salad": 0.50, "soup": 1.00
    }
    
    @classmethod
    def get_calories(cls, food_name: str) -> float:
        """Get calories per 100g for a food item"""
        food_lower = food_name.lower()
        
        # Exact match
        if food_lower in cls.CALORIES_DB:
            return cls.CALORIES_DB[food_lower]
        
        # Partial match
        for key, value in cls.CALORIES_DB.items():
            if key in food_lower or food_lower in key:
                return value
        
        # Default
        return 150.0  # Average food calories
    
    @classmethod
    def get_density(cls, food_name: str) -> float:
        """Get density in g/cm³ for a food item"""
        food_lower = food_name.lower()
        
        # Exact match
        if food_lower in cls.DENSITY_DB:
            return cls.DENSITY_DB[food_lower]
        
        # Partial match
        for key, value in cls.DENSITY_DB.items():
            if key in food_lower or food_lower in key:
                return value
        
        # Default
        return 0.75  # Average food density

# ==================== LLM SERVICE ====================

class FoodDetectionService:
    """
    Service that uses GPT-4o Vision to detect food items
    """
    
    def __init__(self, api_key: str):
        """Initialize the LLM"""
        logger.info("Initializing GPT-4o Vision model...")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.2,
            max_tokens=2000
        )
        self.parser = PydanticOutputParser(pydantic_object=FoodDetectionResponse)
        logger.info("✅ LLM initialized successfully")
    
    def detect_food(
        self,
        image_base64: str,
        image_width: int,
        image_height: int,
        user_text: Optional[str] = None
    ) -> FoodDetectionResponse:
        """
        Main detection function
        
        Args:
            image_base64: Base64 encoded image
            image_width: Image width in pixels
            image_height: Image height in pixels
            user_text: Optional context from user
            
        Returns:
            FoodDetectionResponse with detected items
        """
        
        start_time = datetime.now()
        
        try:
            # Create prompt for GPT-4o
            format_instructions = self.parser.get_format_instructions()
            
            prompt = f"""You are a food detection AI. Analyze this image and detect ALL food items.

FOR EACH FOOD ITEM:
1. **Name**: Identify the specific food (e.g., "apple", "grilled chicken breast", "white rice")
2. **Count**: How many of this item (if multiple identical items, count them)
3. **Bounding Box**: Provide TIGHT coordinates (x, y, width, height) in pixels
   - x, y: top-left corner (0,0 is top-left of image)
   - width, height: box dimensions
   - Make box as TIGHT as possible around just the food item
4. **Calories**: Provide accurate calories per 100g from nutritional knowledge
5. **Density**: Estimate density in g/cm³ (examples: water=1.0, bread=0.27, meat=1.05, fruits=0.6-0.9)
6. **Confidence**: Your confidence in this detection (0.0 to 1.0)

IMAGE INFO:
- Dimensions: {image_width} × {image_height} pixels
- User context: {user_text or "None provided"}

IMPORTANT RULES:
- BE PRECISE with bounding boxes - they will be used for depth-based volume calculation
- If food is partially visible, still detect it with visible bbox
- For food on a plate, detect individual items separately (not "plate of food")
- Confidence should reflect detection certainty AND bbox accuracy
- If NO food detected, set success=false with explanatory message

{format_instructions}
"""
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"  # Request high detail analysis
                        }
                    }
                ]
            )
            
            # Call GPT-4o
            logger.info("🤖 Calling GPT-4o Vision API...")
            response = self.llm.invoke([message])
            
            # Parse response
            logger.info("📝 Parsing response...")
            result = self.parser.parse(response.content)
            
            # Enrich with database values
            for item in result.items:
                # Update calories if we have better data
                db_calories = FoodDatabase.get_calories(item.name)
                if db_calories != 150.0:  # If not default
                    item.calories_per_100g = db_calories
                
                # Update density if we have better data
                db_density = FoodDatabase.get_density(item.name)
                if db_density != 0.75:  # If not default
                    item.estimated_density_g_cm3 = db_density
            
            # Set metadata
            result.items.image_width = image_width
            result.items.image_height = image_height
            result.success = len(result.items) > 0
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = int(processing_time)
            
            logger.info(f"✅ Detection complete: {len(result.items)} items found in {processing_time:.0f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {str(e)}", exc_info=True)
            
            # Return error response
            return FoodDetectionResponse(
                success=False,
                items=[],
                image_width=image_width,
                image_height=image_height,
                message=f"Detection failed: {str(e)}",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )

# ==================== INITIALIZE SERVICE ====================

# Create global service instance (singleton pattern)
logger.info("="*60)
logger.info("Initializing Calorie Estimation API...")
logger.info("="*60)

food_detection_service = FoodDetectionService(OPENAI_API_KEY)

logger.info("="*60)
logger.info("✅ API Ready to serve requests")
logger.info("="*60)

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "service": "Calorie Estimation API",
        "version": "2.0.0",
        "status": "healthy",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "detect_food": "/detect_food (POST)",
            "database_search": "/database/search?q=apple (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint - used by load balancers and monitoring
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "llm_status": "connected",
        "database_items": {
            "calories": len(FoodDatabase.CALORIES_DB),
            "densities": len(FoodDatabase.DENSITY_DB)
        }
    }

@app.post("/detect_food", response_model=FoodDetectionResponse)
async def detect_food(request: FoodDetectionRequest):
    """
    Main endpoint: Detect food items with bounding boxes
    
    This is the PRIMARY endpoint that mobile apps call.
    
    Flow:
    1. Mobile app captures RGB image
    2. Mobile app converts image to base64
    3. Mobile app sends POST request to this endpoint
    4. Backend uses GPT-4o to detect food items
    5. Backend returns list of items with bounding boxes
    6. Mobile app uses bounding boxes to extract depth ROI
    7. Mobile app calculates volume and calories on-device
    
    Args:
        request: FoodDetectionRequest with image and metadata
        
    Returns:
        FoodDetectionResponse with detected food items
    """
    
    logger.info(f"📸 Received detection request for {request.image_width}x{request.image_height} image")
    
    # Validate image can be decoded
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"✅ Image decoded: {image.format} {image.size}")
    except Exception as e:
        logger.error(f"❌ Invalid image data: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}"
        )
    
    # Call detection service
    result = food_detection_service.detect_food(
        image_base64=request.image_base64,
        image_width=request.image_width,
        image_height=request.image_height,
        user_text=request.user_text
    )
    
    return result

@app.get("/database/search")
async def search_database(q: str):
    """
    Search food database
    
    Args:
        q: Search query (food name)
        
    Returns:
        Matching food items with calories and density
    """
    
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    
    query_lower = q.lower()
    results = {}
    
    for food_name in FoodDatabase.CALORIES_DB.keys():
        if query_lower in food_name:
            results[food_name] = {
                "calories_per_100g": FoodDatabase.get_calories(food_name),
                "density_g_cm3": FoodDatabase.get_density(food_name)
            }
    
    return {
        "success": True,
        "query": q,
        "results": results,
        "count": len(results)
    }

@app.get("/database/stats")
async def database_stats():
    """
    Get database statistics
    """
    return {
        "food_items": len(FoodDatabase.CALORIES_DB),
        "density_items": len(FoodDatabase.DENSITY_DB),
        "version": "2.0.0"
    }

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return {
        "success": False,
        "message": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return {
        "success": False,
        "message": "Internal server error",
        "status_code": 500
    }

# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("🚀 Application startup complete")
    logger.info(f"📊 Database loaded: {len(FoodDatabase.CALORIES_DB)} food items")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("👋 Application shutting down")

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    
    # Run with: python main.py
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
