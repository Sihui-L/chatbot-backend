from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json
import time
import asyncio
import os
import base64
from typing import List, Dict, Union
from openai import AsyncOpenAI
from pydantic import BaseModel

router = APIRouter()

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

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
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=openai_api_key)

# Models
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict]]  # Can be a string or a list of content parts

class ChatSession(BaseModel):
    id: str
    messages: List[Message] = []

# In-memory storage for chat sessions
chat_sessions: Dict[str, ChatSession] = {}

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    # Initialize chat session if it doesn't exist
    if client_id not in chat_sessions:
        chat_sessions[client_id] = ChatSession(id=client_id, messages=[])

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle different message types
            if message_data.get("type") == "message":
                user_message = message_data.get("content", "")
                image_data = message_data.get("image")

                # Create message content based on whether an image is included
                if image_data:
                    # For messages with images, we create a list of content parts
                    message_content = [
                        {"type": "text", "text": user_message}
                    ]
                    
                    # Add the image as a content part
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_data,
                            "detail": "high"  # Use high detail for better analysis
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
                            "content": response,
                            "metadata": {
                                "response_time": response_time,
                                "length": len(response),
                                "sentiment": estimate_sentiment(response),
                                "contains_image_analysis": image_data is not None,
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
                            "metadata": {
                                "response_time": response_time,
                                "length": len(complete_response),
                                "sentiment": estimate_sentiment(complete_response),
                                "contains_image_analysis": image_data is not None,
                            },
                        }
                        await manager.send_message(json.dumps(complete_data), client_id)

                except Exception as e:
                    # Handle API errors
                    error_data = {
                        "type": "error",
                        "content": f"Error: {str(e)}",
                    }
                    await manager.send_message(json.dumps(error_data), client_id)

            # Handle feedback messages
            elif message_data.get("type") == "feedback":
                # Store feedback (in a real app, you might want to save this to a database)
                feedback_data = {
                    "session_id": client_id,
                    "message_id": message_data.get("message_id"),
                    "rating": message_data.get("rating"),
                }
                # In a real implementation, you would save this feedback
                print(f"Received feedback: {feedback_data}")

                # Acknowledge feedback receipt
                await manager.send_message(json.dumps({"type": "feedback_received"}), client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error in websocket connection: {str(e)}")
        manager.disconnect(client_id)


async def get_ai_response(messages: List[Message], has_image: bool = False):
    """Get a response from OpenAI API (non-streaming)"""
    try:
        # Convert our messages to the format OpenAI expects
        openai_messages = []
        
        for m in messages:
            # For text messages
            if isinstance(m.content, str):
                openai_messages.append({"role": m.role, "content": m.content})
            # For messages with image content
            elif isinstance(m.content, list):
                openai_messages.append({"role": m.role, "content": m.content})

        # For image analysis, use GPT-4o which supports vision capabilities
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
            # For text messages
            if isinstance(m.content, str):
                openai_messages.append({"role": m.role, "content": m.content})
            # For messages with image content
            elif isinstance(m.content, list):
                openai_messages.append({"role": m.role, "content": m.content})

        # For image analysis, use GPT-4o which supports vision capabilities
        model = "gpt-4o"
        
        # If using vision model and it doesn't support streaming, fall back to non-streaming
        if has_image:
            # Vision model may not support streaming, so we'll simulate it
            response = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=1000,
            )
            
            full_response = response.choices[0].message.content
            # Simulate streaming by yielding chunks of the response
            chunk_size = 10
            for i in range(0, len(full_response), chunk_size):
                yield full_response[i:i+chunk_size]
                await asyncio.sleep(0.05)  # Add a small delay to simulate streaming
        else:
            stream = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                stream=True,
                max_tokens=1000,
            )

            # 'stream' is an async generator
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"Error streaming from OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error streaming from AI service: {str(e)}")

def estimate_sentiment(text: str) -> str:
    """
    Simple estimation of sentiment based on keywords.
    """
    positive_words = [
        "happy",
        "good",
        "great",
        "excellent",
        "positive",
        "wonderful",
        "amazing",
        "love",
    ]
    negative_words = [
        "sad",
        "bad",
        "terrible",
        "poor",
        "negative",
        "awful",
        "hate",
    ]

    text = text.lower()

    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

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

    return chat_sessions[client_id]


@router.delete("/sessions/{client_id}")
async def delete_chat_session(client_id: str):
    """Delete a chat session"""
    if client_id in chat_sessions:
        del chat_sessions[client_id]

    return JSONResponse(content={"status": "success", "message": "Chat session deleted"})