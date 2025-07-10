# Open-Source Community Support Chatbot

A chatbot system designed to provide technical support for open-source software communities, particularly Ubuntu.

## Features
- Intent classification and entity extraction using fine-tuned LLMs
- Retrieval-Augmented Generation (RAG) over the Ubuntu Dialogue Corpus
- Scalable microservices architecture
- Real-time chat widget for end-users
- Admin console for monitoring and training

## Getting Started
1. Clone this repository
2. Run `docker-compose up` to start all services
3. Visit http://localhost:3000 for the chat widget
4. Visit http://localhost:3001 for the admin console

## Architecture
This project uses a microservices architecture with the following components:
- Intent Classification Service (Python/FastAPI)
- Dialog Manager Service (Node.js/Express)
- RAG Service (Python/FastAPI)
- Frontend (React/Tailwind CSS)
