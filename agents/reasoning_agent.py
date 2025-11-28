"""
Reasoning Agent with Tool Calling
Uses Ollama with tool-calling capable models and maintains conversation memory
"""

import ollama
import json
from typing import List, Dict, Any, Optional
from tools.github_tool import GitHubTool


class ReasoningAgent:
    """Agent with reasoning capabilities, tool calling, and conversation memory."""
    
    def __init__(self, model: str = "llama3.1:8b"):
        """Initialize reasoning agent with Ollama model."""
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.github_tool = GitHubTool()
        
        # Define available tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_github_repo",
                    "description": "Analyze a GitHub repository by fetching its README and metadata. Use when user asks about a GitHub repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_url": {
                                "type": "string",
                                "description": "The GitHub repository URL (e.g., https://github.com/owner/repo)"
                            }
                        },
                        "required": ["repo_url"]
                    }
                }
            }
        ]
    
    def add_to_memory(self, role: str, content: str):
        """Add message to conversation memory."""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep last 10 messages to prevent context overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if tool_name == "analyze_github_repo":
            repo_url = arguments.get("repo_url", "")
            repo_info = self.github_tool.analyze_repository(repo_url)
            return self.github_tool.format_repo_info(repo_info)
        
        return f"Unknown tool: {tool_name}"
    
    def reason(self, query: str, context: Optional[str] = None) -> str:
        """
        Reason about a query with optional RAG context.
        Supports tool calling and maintains conversation memory.
        """
        # Build system prompt
        system_prompt = """You are a helpful AI assistant with reasoning capabilities. 
You can analyze questions, use tools when needed, and provide thoughtful answers.
When given context from documents, use it to inform your answers."""
        
        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current query with context if available
        if context:
            user_message = f"""Context from knowledge base:
{context}

Question: {query}"""
        else:
            user_message = query
        
        messages.append({"role": "user", "content": user_message})
        
        # Call Ollama with tools
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=self.tools
            )
            
            # Check if model wants to use tools
            if response.get("message", {}).get("tool_calls"):
                # Execute tool calls
                tool_results = []
                for tool_call in response["message"]["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    
                    result = self.execute_tool(tool_name, tool_args)
                    tool_results.append(result)
                
                # Add tool results to messages
                messages.append(response["message"])
                messages.append({
                    "role": "tool",
                    "content": "\n\n".join(tool_results)
                })
                
                # Get final response with tool results
                final_response = ollama.chat(
                    model=self.model,
                    messages=messages
                )
                
                answer = final_response["message"]["content"]
            else:
                answer = response["message"]["content"]
            
            # Update conversation memory
            self.add_to_memory("user", query)
            self.add_to_memory("assistant", answer)
            
            return answer
            
        except Exception as e:
            return f"Error in reasoning: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_memory_summary(self) -> str:
        """Get a summary of conversation history."""
        if not self.conversation_history:
            return "No conversation history"
        
        summary_parts = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            role = msg["role"].upper()
            content = msg["content"][:100]  # Truncate
            summary_parts.append(f"{role}: {content}...")
        
        return "\n".join(summary_parts)
