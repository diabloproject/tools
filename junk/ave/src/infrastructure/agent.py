"""Schema agent implementation using OpenRouter."""
import json
import logging
import os
from typing import List, Dict, Any

from openrouter import OpenRouter

logger = logging.getLogger(__name__)

from .prompts import clarify_intentions_system
from .tools import ListTablesTool, DescribeColumnTool
from ..application.schema_service import SchemaService


class SchemaAgent:
    """Agent for exploring database schema using OpenRouter."""

    def __init__(
        self,
        schema_service: SchemaService,
        model_id: str = "openai/gpt-oss-20b",
    ):
        """Initialize agent with schema service and OpenRouter."""
        logger.info(f"Initializing SchemaAgent with model: {model_id}")
        self.schema_service = schema_service
        self.model_id = model_id
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = OpenRouter(api_key=self.api_key)
        logger.debug("OpenRouter client initialized")

        # Initialize tools
        self.tools = [
            ListTablesTool(schema_service),
            DescribeColumnTool(schema_service),
        ]
        logger.info(f"Initialized {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

        # Create tool lookup
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Conversation history
        self.conversation_history = [
            {
                "role": "system",
                "content": clarify_intentions_system("strict"),
            },
        ]
        logger.debug("Conversation history initialized with system prompt")

    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for function calling."""
        return [tool.get_schema() for tool in self.tools]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name with arguments."""
        logger.debug(f"Executing tool: {tool_name} with arguments: {arguments}")
        tool = self.tool_map.get(tool_name)
        if tool is None:
            logger.error(f"Tool '{tool_name}' not found in tool map")
            return f"Error: Tool '{tool_name}' not found"

        try:
            result = tool.execute(**arguments)
            logger.debug(f"Tool {tool_name} executed successfully, result length: {len(result)}")
            return result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            raise

    def run(self, query: str, max_iterations: int = 150) -> str:
        """Run a query through the agent with tool calling."""
        logger.info(f"Starting agent run with query: {query[:100]}...")
        # Add the user message to the history
        self.conversation_history.append(
            {
                "role": "user",
                "content": query,
            },
        )
        logger.debug(f"Added user message to history, total messages: {len(self.conversation_history)}")

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"Starting iteration {iteration}/{max_iterations}")

            # Call OpenRouter API
            logger.debug(f"Calling OpenRouter API with model: {self.model_id}")
            response = self.client.chat.send(
                model=self.model_id,
                messages=self.conversation_history,
                tools=self._get_tool_schemas(),
            )
            logger.debug("Received response from OpenRouter API")

            choice = response.choices[0]
            message = choice.message

            print("Thought: ", message.reasoning)

            # Check if model wants to call tools
            if message.tool_calls:
                logger.info(f"Model requested {len(message.tool_calls)} tool call(s)")
                # Add assistant message with tool calls
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls,
                    },
                )

                # Execute each tool call
                for tool_call in message.tool_calls:
                    logger.info(
                        f"Processing tool call (#{tool_call.id}: {tool_call.function.name} with "
                        f"arguments: {tool_call.function.arguments})",
                    )
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    result = self._execute_tool(tool_name, arguments)
                    logger.debug(f"Result (#{tool_call.id}): {result}")

                    self.conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                        },
                    )
                    logger.debug(f"Added tool result to history for {tool_name}")
            else:
                logger.info("No tool calls requested, returning final response")
                final_response = message.content
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": final_response,
                    },
                )
                logger.debug(f"Final response length: {len(final_response)}")
                return final_response

        logger.warning(f"Maximum iterations ({max_iterations}) reached")
        return "Maximum iterations reached. Please try a simpler query."

    def reset(self):
        """Reset the conversation history."""
        logger.info("Resetting conversation history")
        self.conversation_history = []
        logger.debug("Conversation history cleared")
