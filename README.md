# ChatApp Backend

## Overview

This backend implementation powers a real-time chat application with AI capabilities, featuring WebSocket communication, OpenAI integration, and image analysis functionality. Users can engage in conversations with an AI assistant, upload images for analysis, and receive visualized metrics on AI responses.

## Features

- WebSocket-based real-time communication
- OpenAI GPT integration with streaming responses
- Image upload and analysis using GPT-4o vision capabilities
- Multiple concurrent chat sessions
- Response metrics (time, length, sentiment)
- User feedback collection (thumbs up/down)

## Architecture

The application follows a clean, modular architecture built with FastAPI:

- **Connection Manager**: Handles WebSocket connections, messaging, and disconnections
- **Data Models**: Structured representation of messages and chat sessions
- **OpenAI Integration**: Processes user inputs and generates AI responses
- **HTTP Endpoints**: Provides additional session management functionality

## Technical Implementation

### WebSocket Communication

WebSockets enable persistent, real-time communication between the frontend and backend. The implementation includes:

- Client identification with unique IDs
- Structured JSON message protocol
- Error handling and reconnection management
- Support for both text messages and binary data

### OpenAI Integration

The application integrates with OpenAI's Chat Completions API to:

- Generate contextual responses based on conversation history
- Stream responses for a more natural experience
- Process and analyze images using GPT-4o

### Session Management

Chat sessions are maintained with:

- In-memory storage of conversation history
- Support for multiple concurrent sessions
- Simple retrieval and deletion via HTTP endpoints

## Design Considerations & Challenges

### Message Protocol Design

Implemented a type-based message protocol for clear separation of concerns:

- `message`: User/AI text or image messages
- `stream`: Incremental response chunks
- `feedback`: User ratings on AI responses

### Streaming Implementation

Vision models don't natively support streaming, so I implemented a chunking mechanism that simulates streaming for image-based conversations, providing a consistent experience regardless of message type.

### Image Processing Challenges

Ensuring proper formatting of image data for OpenAI's vision models required careful implementation of:

- Correct content structure for multimodal messages
- Proper image URL formatting
- Detailed error handling for image processing failures

## Future Improvements

While this implementation meets all requirements, potential enhancements include:

1. Persistent storage with a database
2. Authentication and user management
3. Enhanced sentiment analysis
4. Performance optimizations for large conversations
5. Additional visualization metrics

## Getting Started

1. Set up environment variables (see `.env.example`)
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `make app`
4. Run the frontend: `npm run dev`
