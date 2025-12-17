import json
import os
from typing import Any

from openrouter import OpenRouter

from src.infrastructure.prompts import clarify_intentions_system
from main import list_tables, list_columns, describe_column


class SchemaAgent:
    """Agent that uses tools to explore database schema and answer queries."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.conversation_history = []

    def _define_tools(self):
        """Define the tools available to the agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_tables",
                    "description": "Get a list of all available database tables with their metadata including granularity, purpose, passage, and relationships",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_columns",
                    "description": "Get a list of columns for a specific table with their names and descriptions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "The name of the table to get columns for"
                            }
                        },
                        "required": ["table"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "describe_column",
                    "description": "Get detailed information about a specific column in a table, including all metadata",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "The name of the table"
                            },
                            "column": {
                                "type": "string",
                                "description": "The name of the column"
                            }
                        },
                        "required": ["table", "column"]
                    }
                }
            }
        ]

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments."""
        # Import tools from main

        if tool_name == "list_tables":
            return list_tables()
        elif tool_name == "list_columns":
            return list_columns(**arguments)
        elif tool_name == "describe_column":
            return describe_column(**arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Send a message to the agent and get a response.
        The agent can use tools to explore the schema.

        Args:
            user_message: The user's message
            max_iterations: Maximum number of tool-calling iterations to prevent infinite loops

        Returns:
            The agent's final response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": clarify_intentions_system("strict")}
        ] + self.conversation_history

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Call the model using context manager
            with OpenRouter(api_key=self.api_key) as client:
                response = client.chat.send(
                    model=self.model,
                    messages=messages,
                    tools=self._define_tools(),
                    tool_choice="auto"
                )

            message = response.choices[0].message
            print(f"HIDDEN: {message}")

            # Check if the model wants to call tools
            if message.tool_calls:
                # Add assistant's message with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name

                    # Parse arguments (they come as JSON string)
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Execute the tool
                    try:
                        print(f"HIDDEN: Executing {tool_name} with {arguments}")
                        result = self._execute_tool(tool_name, arguments)
                        print(f"HIDDEN: Result: {result}")
                        result_str = json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        result_str = f"Error executing {tool_name}: {str(e)}"
                        print(f"HIDDEN: {result_str}")

                    # Add a tool result to the history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result_str
                    })

                # Update messages for the next iteration
                messages = [
                    {"role": "system", "content": clarify_intentions_system("strict")}
                ] + self.conversation_history

            else:
                # No more tool calls, we have the final response
                final_response = message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                return final_response

        # If we hit max iterations, return the last message
        return "Maximum iterations reached. Please try a simpler query."

    def reset(self):
        """Reset the conversation history."""
        self.conversation_history = []


def main():
    """Example usage of the agent."""
    agent = SchemaAgent()

    print("Schema Agent initialized. Type 'exit' to quit, 'reset' to clear history.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            break

        if user_input.lower() == 'reset':
            agent.reset()
            print("Conversation history cleared.\n")
            continue

        if not user_input:
            continue

        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
