import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, TypedDict, Annotated
import json
import re
from pprint import pprint

# Add the LLM directory to Python path
llm_path = Path(__file__).parent / 'LLM'
sys.path.append(str(llm_path))

# Import required LangChain/LangGraph components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
# from langchain_community.llms import VLLM
# Import the existing tools and system prompt

from main import create_database_tools
from prompt import get_system_prompt

# State definition for the graph
class AgentState(TypedDict):
    messages: Annotated[List[Any], "The messages in the conversation"]
    tool_calls_count: Dict[str, int]
    iteration_count: int

class StreamingQwenAgent:
    def __init__(self):
        """Initialize the streaming agent with Qwen3:4b model."""
        print("🚀 Initializing Streaming Qwen Agent...")
        
        # self.llm = ChatOllama(
        #     model="qwen3:4b",
        #     temperature=0.1,
        #     streaming=True,
        #     callbacks=[]
        # )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv('OPENAI_API_KEY'),
        )


        # Available tools
        llm_db_url = os.getenv('CRYPTO_DATABASE_URL')
        self.tools = create_database_tools(llm_db_url)
        
        # Try to bind tools to the model (may not work with all Ollama models)
        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            self.use_llm_tools = True
            print("✅ Tools bound to LLM successfully")
        except Exception as e:
            print(f"⚠️  Tool binding failed: {e}")
            print("📝 Will use LLM without tool binding")
            self.llm_with_tools = self.llm
            self.use_llm_tools = False
        
        # Create the tool node
        self.tool_node = ToolNode(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
        
        print("✅ Agent initialized successfully!")
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": "__end__"
            }
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent node that processes messages and decides on actions."""
        messages = state["messages"]
        iteration_count = state.get("iteration_count", 0) + 1
        
        print(f"🔄 Agent iteration {iteration_count}")
        
        prompt_messages = [SystemMessage(content=get_system_prompt())]
        prompt_messages.extend(messages)
        
        # Print the prompt for debugging
        print("\n===== LLM PROMPT (all messages) =====")
        pprint([msg.dict() if hasattr(msg, 'dict') else str(msg) for msg in prompt_messages])
        print("===== END PROMPT =====\n")
        
        try:
            response = self.llm_with_tools.invoke(prompt_messages)
            
            # Check if response has tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"🔧 Detected {len(response.tool_calls)} tool call(s)")
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"  Tool {i+1}: {tool_call['name']} with args {tool_call['args']}")
                return {"messages": [response], "iteration_count": iteration_count}
            
            # If no tool calls but we have tool results, generate final response
            elif any(isinstance(msg, ToolMessage) for msg in messages):
                print("💬 Generating final response after tool execution...")
                # Use streaming for final response
                return {"messages": [self._generate_streaming_response(prompt_messages)], "iteration_count": iteration_count}
            
            # If no tool calls and no previous tool results, generate direct response
            else:
                print("💬 No tool calls needed, generating direct response...")
                return {"messages": [self._generate_streaming_response(prompt_messages)], "iteration_count": iteration_count}
                
        except Exception as e:
            print(f"❌ Error with LLM: {e}")
            return {"messages": [AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")], "iteration_count": iteration_count} 

    # def _needs_tools(self, query: str) -> bool:
    #     """Check if the query likely needs tool calls."""
    #     query_lower = query.lower()
    #     tool_keywords = [
    #         'find table', 'search table', 'table', 'database',
    #         'dq results', 'data quality', 'statistics', 'stats',
    #         'update description', 'generate description', 'description',
    #         'monitoring project', 'sample data', 'ci sample'
    #     ]
    #     return any(keyword in query_lower for keyword in tool_keywords)

    def _generate_streaming_response(self, messages: List[Any]) -> AIMessage:
        """Generate a streaming response."""
        response_text = ""
        print("🔄 Streaming response: ", end="", flush=True)
        try:
            for chunk in self.llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                print(token, end="", flush=True)
                response_text += token
        except Exception as e:
            print(f"\n❌ Error during streaming: {e}")
            response_text = "I apologize, but I encountered an error while generating the response."
        print("\n")  # New line after streaming
        return AIMessage(content=response_text)

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        iteration_count = state.get("iteration_count", 0)
        
        # Safety check: prevent infinite loops
        MAX_ITERATIONS = 10
        if iteration_count >= MAX_ITERATIONS:
            print(f"⚠️ Maximum iterations ({MAX_ITERATIONS}) reached, ending conversation")
            return "end"
        
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # If the last message is an AIMessage with tool calls, go to tools
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print("🔧 Tool calls detected, continuing to tool execution...")
            return "continue"
        
        # If we just got tool results and need to generate a response
        if isinstance(last_message, ToolMessage):
            print("🔄 Tool result received, returning to agent for final answer...")
            return "continue"
        
        # Otherwise, end
        print("🏁 Ending conversation flow")
        return "end"

    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Tool execution node."""
        print("🔧 Executing tools...")
        messages = state["messages"]
        if not messages:
            return state
            
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                print(f"🔧 Executing {tool_name} with args: {tool_args}")
                
                try:
                    # Find the tool function
                    tool_func = None
                    for tool in self.tools:
                        if tool.name == tool_name:
                            tool_func = tool
                            break
                    
                    if tool_func:
                        result = tool_func.invoke(tool_args)
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id
                        )
                        tool_messages.append(tool_message)
                        print(f"✅ Tool {tool_name} executed successfully")
                        print(f"📊 Tool result (first 200 chars): {str(result)[:200]}...")
                    else:
                        error_msg = f"Tool {tool_name} not found"
                        tool_message = ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_id
                        )
                        tool_messages.append(tool_message)
                        print(f"❌ {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    tool_message = ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_id
                    )
                    tool_messages.append(tool_message)
                    print(f"❌ {error_msg}")
                    
            return {"messages": state["messages"] + tool_messages}
        
        return state

    async def chat(self, message: str) -> str:
        """Main chat interface."""
        print(f"\n💬 User: {message}")
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "tool_calls_count": get_tool_call_counts(),
            "iteration_count": 0
        }
        config = {"configurable": {"thread_id": "main_thread"}}
        try:
            result = await self.graph.ainvoke(initial_state, config)
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I apologize, but I couldn't generate a response."
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            return f"An error occurred: {str(e)}"

    def sync_chat(self, message: str) -> str:
        """Synchronous chat interface that properly handles the full workflow."""
        print(f"\n💬 User: {message}")
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "tool_calls_count": get_tool_call_counts(),
            "iteration_count": 0
        }
        config = {"configurable": {"thread_id": "main_thread"}}
        try:
            result = self.graph.invoke(initial_state, config)
            messages = result.get("messages", [])
            
            # Get the final AI response
            final_response = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                    final_response = msg.content
                    break
                elif isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and not msg.tool_calls:
                    final_response = msg.content
                    break
            
            if not final_response:
                final_response = "I apologize, but I couldn't generate a response."
                
            print(f"\n🤖 Final Response: {final_response}")
            return final_response
            
        except Exception as e:
            print(f"❌ Error in sync_chat: {e}")
            return f"An error occurred: {str(e)}"

    def stream_chat(self, messages, session_id=None, user=None):
        """
        Generator that yields the LLM's response in real time, including tool calls and final answer.
        Accepts a list of LangChain message objects as chat memory.
        Now yields tool responses as ('tool', content) tuples for memory saving.
        Optionally saves conversation to database if session_id and user provided.
        """
        initial_state = {
            "messages": messages,
            "iteration_count": 0
        }
        print("📝 Input messages for streaming:")
        for i, msg in enumerate(messages):
            print(f"  {i}: {type(msg).__name__} - {str(msg)[:100]}...")
        
        config = {"configurable": {"thread_id": session_id or "main_thread"}}
        state = initial_state
        
        # Track tool calls and responses for database saving
        current_tool_calls = []
        current_tool_responses = []
        ai_response_content = ""
        
        while True:
            # Agent node
            prompt_messages = [SystemMessage(content=get_system_prompt())]
            prompt_messages.extend(state["messages"])
            
            response = self.llm_with_tools.invoke(prompt_messages)
            
            
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Stream the LLM's reasoning text before tool execution
                if response.content:
                    for char in response.content:
                        yield char
                
                print(f"🔧 Processing {len(response.tool_calls)} tool call(s)")
                tool_messages = []
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    # Track tool call for database
                    current_tool_calls.append({
                        "name": tool_name,
                        "args": tool_args,
                        "id": tool_id
                    })
                    
                    tool_func = next((t for t in self.tools if t.name == tool_name), None)
                    if tool_func:
                        yield ("tool_name", tool_name)
                        result = tool_func.invoke(tool_args)
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id
                        )
                        tool_messages.append(tool_message)
                        
                        # Track tool response for database
                        current_tool_responses.append({
                            "tool_name": tool_name,
                            "result": str(result)
                        })
                        
                        # Yield tool response for memory saving
                        yield ("tool", str(result), tool_name)
                    else:
                        error_msg = f"[ERROR] Tool {tool_name} not found"
                        tool_message = ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_id
                        )
                        tool_messages.append(tool_message)
                        
                        # Track error response for database
                        current_tool_responses.append({
                            "tool_name": tool_name,
                            "result": error_msg
                        })
                        
                        yield ("tool", error_msg, tool_name)
                
                state["messages"].append(response)
                state["messages"].extend(tool_messages)
                
            else:
                # Stream the final answer token by token
                print("🔄 Streaming final response...")
                for chunk in self.llm.stream(prompt_messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    ai_response_content += token
                    yield token
                
                # Save AI response to database if session provided
                if session_id and user:
                    try:
                        self._save_ai_message_to_db(session_id, ai_response_content, current_tool_calls, current_tool_responses)
                    except Exception as e:
                        print(f"❌ Error saving AI message to database: {e}")
                break

    def _save_ai_message_to_db(self, session_id, content, tool_calls, tool_responses):
        """Save AI message to database"""
        from django.apps import apps
        ConversationMessage = apps.get_model('core', 'ConversationMessage')
        ConversationSession = apps.get_model('core', 'ConversationSession')
        
        try:
            session = ConversationSession.objects.get(id=session_id)
            ConversationMessage.objects.create(
                session=session,
                content=content,
                is_user=False,
                tool_calls=tool_calls if tool_calls else None,
                tool_responses=tool_responses if tool_responses else None
            )
            print(f"✅ AI message saved to session {session_id}")
        except Exception as e:
            print(f"❌ Error saving AI message: {e}")
            raise

    def get_conversation_history(self, session_id, limit=50):
        """Get conversation history from database and convert to LangChain messages"""
        from django.apps import apps
        ConversationMessage = apps.get_model('core', 'ConversationMessage')
        
        try:
            messages = ConversationMessage.objects.filter(
                session_id=session_id
            ).order_by('created_at')[:limit]
            
            langchain_messages = []
            for msg in messages:
                if msg.is_user:
                    langchain_messages.append(HumanMessage(content=msg.content))
                else:
                    langchain_messages.append(AIMessage(content=msg.content))
            
            return langchain_messages
        except Exception as e:
            print(f"❌ Error retrieving conversation history: {e}")
            return []

def get_tool_call_counts():
    """Return tool call counts - placeholder implementation"""
    return {
        'btc_ohlcv_data': 0,
        'funding_rates_data': 0,
        'news_data': 0,
        'open_interest_data': 0,
        'database_analysis': 0
    }

def main():
    """Main function to run the streaming agent."""
    print("🎯 Starting LangGraph Streaming App with Qwen3:4b")
    print("=" * 50)
    
    # Initialize the agent
    try:
        agent = StreamingQwenAgent()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    print("\n🎉 Agent ready! Type 'quit' to exit.")
    print("💡 Try asking about data quality analysis, statistics, or general questions.")
    print("📝 Example: 'find table called lab_test_table from database one_demo, generate brief description for it in 1 paragraph, and update the description'")
    print("-" * 50)
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get response
            response = agent.sync_chat(user_input)
            print(f"\n🤖 Assistant: {response}")
            
            # Print tool usage stats
            tool_stats = get_tool_call_counts()
            if any(count > 0 for count in tool_stats.values()):
                print(f"\n📊 Tool usage: {tool_stats}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()