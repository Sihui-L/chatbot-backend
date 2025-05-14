from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json
import time
import asyncio
import os
import uuid
from typing import List, Dict, Union, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from nltk.sentiment.vader import SentimentIntensityAnalyzer

router = APIRouter()

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: Missing OPENAI_API_KEY environment variable")

# Session cleanup configuration
MAX_SESSIONS = 1000  # Maximum number of sessions to store in memory
SESSION_TIMEOUT = 3600  # Session timeout in seconds (1 hour)

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=openai_api_key)

# Initialize VADER sentiment analyzer
sentiment_analyzer = None
try:
    # Try to initialize VADER
    sentiment_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Warning: Could not initialize VADER sentiment analyzer: {e}")
    # Try to download NLTK data if needed
    try:
        import nltk
        nltk.download('vader_lexicon')
        sentiment_analyzer = SentimentIntensityAnalyzer()
        print("Successfully downloaded VADER lexicon and initialized sentiment analyzer")
    except Exception as e:
        print(f"Failed to initialize VADER even after download attempt: {e}")
        # Fallback to simple sentiment analysis will be used

# Models
class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict]]
    timestamp: float = Field(default_factory=time.time)

class ChatSession(BaseModel):
    id: str
    messages: List[Message] = []
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)

# In-memory storage for chat sessions
chat_sessions: Dict[str, ChatSession] = {}

async def cleanup_old_sessions():
    """Periodically clean up old chat sessions to prevent memory issues"""
    while True:
        try:
            current_time = time.time()
            # Find sessions that have been inactive for too long
            inactive_sessions = [
                session_id for session_id, session in chat_sessions.items()
                if current_time - session.last_activity > SESSION_TIMEOUT
            ]
            
            # Remove inactive sessions
            for session_id in inactive_sessions:
                del chat_sessions[session_id]
                print(f"Cleaned up inactive session: {session_id}")
            
            # If we still have too many sessions, remove the oldest ones
            if len(chat_sessions) > MAX_SESSIONS:
                # Sort sessions by last activity time
                sorted_sessions = sorted(
                    chat_sessions.items(),
                    key=lambda x: x[1].last_activity
                )
                # Remove the oldest sessions to get under the limit
                sessions_to_remove = len(chat_sessions) - MAX_SESSIONS
                for i in range(sessions_to_remove):
                    session_id = sorted_sessions[i][0]
                    del chat_sessions[session_id]
                    print(f"Removed old session due to capacity limit: {session_id}")
        except Exception as e:
            print(f"Error in cleanup task: {e}")
        
        # Run cleanup every 5 minutes
        await asyncio.sleep(300)

# Start cleanup task
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_sessions())

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    # Initialize chat session if it doesn't exist
    if client_id not in chat_sessions:
        chat_sessions[client_id] = ChatSession(id=client_id)

    # Update session activity time
    chat_sessions[client_id].last_activity = time.time()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_message(json.dumps({
                    "type": "error",
                    "content": "Invalid JSON format received"
                }), client_id)
                continue

            # Update session activity time
            chat_sessions[client_id].last_activity = time.time()

            # Handle different message types
            if message_data.get("type") == "message":
                user_message = message_data.get("content", "")
                image_data = message_data.get("image")

                # Create message content based on whether an image is included
                if image_data:
                    # Validate image data format
                    if not isinstance(image_data, str) or not image_data.startswith("data:image/"):
                        await manager.send_message(json.dumps({
                            "type": "error",
                            "content": "Invalid image format. Expected data URL."
                        }), client_id)
                        continue
                        
                    # For messages with images, we create a list of content parts
                    message_content = [
                        {"type": "text", "text": user_message}
                    ]
                    
                    # Add the image as a content part
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_data,
                            "detail": "high"
                        }
                    })
                    
                    # Add user message with image to session history
                    chat_sessions[client_id].messages.append(
                        Message(role="user", content=message_content)
                    )
                else:
                    # Regular text message
                    chat_sessions[client_id].messages.append(
                        Message(role="user", content=user_message)
                    )

                # Generate a unique ID for this message exchange
                message_id = str(uuid.uuid4())

                # Start response time measurement
                start_time = time.time()

                try:
                    # For non-streaming response
                    if not message_data.get("stream", False):
                        # Call OpenAI API
                        response = await get_ai_response(chat_sessions[client_id].messages, image_data is not None)

                        # Calculate response time
                        response_time = time.time() - start_time

                        # Add AI response to session history
                        chat_sessions[client_id].messages.append(
                            Message(role="assistant", content=response)
                        )

                        # Send response with metadata
                        response_data = {
                            "type": "message",
                            "message_id": message_id,
                            "content": response,
                            "metadata": {
                                "response_time": round(response_time, 2),
                                "length": len(response),
                                "sentiment": analyze_sentiment(response),
                                "contains_image_analysis": image_data is not None,
                                "timestamp": time.time()
                            },
                        }
                        await manager.send_message(json.dumps(response_data), client_id)

                    # For streaming response
                    else:
                        # Create a variable to accumulate the complete response
                        complete_response = ""
                        
                        async for chunk in stream_ai_response(chat_sessions[client_id].messages, image_data is not None):
                            chunk_data = {
                                "type": "stream",
                                "message_id": message_id,
                                "content": chunk,
                            }
                            await manager.send_message(json.dumps(chunk_data), client_id)
                            complete_response += chunk
                        
                        # Add the complete response to chat history
                        chat_sessions[client_id].messages.append(
                            Message(role="assistant", content=complete_response)
                        )

                        # Send complete message signal with metadata
                        response_time = time.time() - start_time
                        complete_data = {
                            "type": "stream_complete",
                            "message_id": message_id,
                            "metadata": {
                                "response_time": round(response_time, 2),
                                "length": len(complete_response),
                                "sentiment": analyze_sentiment(complete_response),
                                "contains_image_analysis": image_data is not None,
                                "timestamp": time.time()
                            },
                        }
                        await manager.send_message(json.dumps(complete_data), client_id)

                except Exception as e:
                    # Handle API errors
                    error_message = str(e)
                    error_data = {
                        "type": "error",
                        "message_id": message_id,
                        "content": f"Error: {error_message}",
                    }
                    await manager.send_message(json.dumps(error_data), client_id)
                    print(f"API error for client {client_id}: {error_message}")

            # Handle feedback messages
            elif message_data.get("type") == "feedback":
                message_id = message_data.get("message_id")
                if not message_id:
                    await manager.send_message(json.dumps({
                        "type": "error",
                        "content": "Missing message_id in feedback"
                    }), client_id)
                    continue
                    
                # Store feedback (in a real app, save this to a database)
                feedback_data = {
                    "session_id": client_id,
                    "message_id": message_id,
                    "rating": message_data.get("rating"),
                    "timestamp": time.time()
                }
                # In a real implementation, save this feedback to a database
                print(f"Received feedback: {feedback_data}")

                # Acknowledge feedback receipt
                await manager.send_message(json.dumps({
                    "type": "feedback_received",
                    "message_id": message_id
                }), client_id)
            
            # Handle ping messages to keep connection alive
            elif message_data.get("type") == "ping":
                await manager.send_message(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }), client_id)
            
            # Handle unknown message types
            else:
                await manager.send_message(json.dumps({
                    "type": "error",
                    "content": f"Unknown message type: {message_data.get('type')}"
                }), client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error in websocket connection for client {client_id}: {str(e)}")
        try:
            error_data = {
                "type": "error",
                "content": "Connection error occurred"
            }
            await manager.send_message(json.dumps(error_data), client_id)
        except:
            pass
        manager.disconnect(client_id)


async def get_ai_response(messages: List[Message], has_image: bool = False):
    """Get a response from OpenAI API (non-streaming)"""
    try:
        # Convert messages to the format OpenAI expects
        openai_messages = []
        
        for m in messages:
            if isinstance(m.content, str):
                # Simple text message
                openai_messages.append({"role": m.role, "content": m.content})
            elif isinstance(m.content, list):
                # For messages with content parts (like images)
                openai_messages.append({"role": m.role, "content": m.content})

        # For vision capabilities, use GPT-4o
        model = "gpt-4o"

        response = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            max_tokens=1000,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling AI service: {str(e)}")

async def stream_ai_response(messages: List[Message], has_image: bool = False):
    """Stream a response from OpenAI API"""
    try:
        # Convert our messages to the format OpenAI expects
        openai_messages = []
        
        for m in messages:
            if isinstance(m.content, str):
                # Simple text message
                openai_messages.append({"role": m.role, "content": m.content})
            elif isinstance(m.content, list):
                # For messages with content parts (like images)
                openai_messages.append({"role": m.role, "content": m.content})

        # For vision capabilities, use GPT-4o
        model = "gpt-4o"
        
        # If using vision model and it doesn't support streaming with images, fall back to simulated streaming
        if has_image:
            # Vision model with images may not support streaming
            response = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=1000,
            )
            
            full_response = response.choices[0].message.content
            # Simulate streaming by yielding chunks of the response
            chunk_size = 10  # Characters per chunk
            for i in range(0, len(full_response), chunk_size):
                yield full_response[i:i+chunk_size]
                await asyncio.sleep(0.03)  # Small delay to simulate streaming
        else:
            # Regular streaming for text-only conversations
            stream = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                stream=True,
                max_tokens=1000,
            )

            # Process the streaming response
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"Error streaming from OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error streaming from AI service: {str(e)}")

def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment using VADER sentiment analysis tool.
    Returns "positive", "negative", or "neutral" based on the compound score.
    
    If VADER is not available, falls back to the simple keyword-based method.
    """
    # Check if VADER sentiment analyzer is available
    if sentiment_analyzer is not None:
        try:
            # Get sentiment scores
            scores = sentiment_analyzer.polarity_scores(text)
            
            # VADER compound score: Ranges from -1 (negative) to 1 (positive)
            compound = scores['compound']
            
            # Classify sentiment based on compound score
            if compound >= 0.05:
                return "positive"
            elif compound <= -0.05:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            print(f"Error in VADER sentiment analysis: {e}")
            # Fall back to simple method if VADER fails
            return simple_sentiment_analysis(text)
    else:
        # Fallback to simple keyword-based method
        return simple_sentiment_analysis(text)

def simple_sentiment_analysis(text: str) -> str:
    """
    Simple estimation of sentiment based on keywords.
    """
    positive_words = [
        "happy", "good", "great", "excellent", "positive", "wonderful", 
        "amazing", "love", "helpful", "thanks", "pleased", "appreciate",
        "enjoy", "like", "useful", "delighted", "glad", "fantastic"
    ]
    negative_words = [
        "sad", "bad", "terrible", "poor", "negative", "awful", "hate",
        "disappointed", "angry", "annoyed", "frustrating", "useless",
        "dislike", "awful", "horrible", "unhappy", "unfortunately"
    ]

    text = text.lower()
    words = text.split()

    positive_count = sum(1 for word in words if word.strip(".,!?") in positive_words)
    negative_count = sum(1 for word in words if word.strip(".,!?") in negative_words)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


@router.get("/sessions/{client_id}")
async def get_chat_history(client_id: str):
    """Get chat history for a specific client"""
    if client_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Update session activity time
    chat_sessions[client_id].last_activity = time.time()
    
    return chat_sessions[client_id]


@router.delete("/sessions/{client_id}")
async def delete_chat_session(client_id: str):
    """Delete a chat session"""
    if client_id in chat_sessions:
        del chat_sessions[client_id]
        return JSONResponse(content={"status": "success", "message": "Chat session deleted"})
    else:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Chat session not found"}
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "sessions": len(chat_sessions),
        "connections": len(manager.active_connections),
        "timestamp": time.time()
    }