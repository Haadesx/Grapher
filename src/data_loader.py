import json
import pandas as pd
from typing import List, Dict

def load_conversations(file_path: str) -> List[Dict]:
    """
    Load conversations from the JSON export.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {file_path}")
        return []

def extract_conversation_details(conversation: Dict) -> Dict:
    """
    Extracts text, message count, and a representative snippet.
    Handles both 'mapping' (graph) and 'messages' (linear) formats.
    """
    text_parts = []
    message_count = 0
    first_user_message = ""
    
    # Determine source of messages
    messages_list = []
    
    # Case 1: Linear 'messages' list (e.g. single export)
    if 'messages' in conversation and isinstance(conversation['messages'], list):
        messages_list = conversation['messages']
        
    # Case 2: Graph 'mapping' dict (e.g. bulk export)
    elif 'mapping' in conversation and isinstance(conversation.get('mapping'), dict):
        mapping = conversation['mapping']
        nodes = []
        for node_data in mapping.values():
            if isinstance(node_data, dict) and node_data.get('message'):
                nodes.append(node_data)
        # Sort by create_time
        nodes.sort(key=lambda x: (x.get('message') or {}).get('create_time') or 0)
        # Extract message objects
        messages_list = [n.get('message') for n in nodes if n.get('message')]
        
    for message in messages_list:
        if not isinstance(message, dict): continue
        
        # Extract Content
        content = message.get('content')
        role = None
        text = ""
        
        # Handle Author/Role
        if 'author' in message and isinstance(message['author'], dict):
            role = message['author'].get('role')
        elif 'role' in message:
            role = message['role']
            
        # Handle Content extraction
        if isinstance(content, dict):
            # Standard format: content.parts
            if content.get('content_type') == 'text' and 'parts' in content:
                parts = content['parts']
                text = " ".join([str(p) for p in parts if p])
        elif isinstance(content, list):
            # List of parts directly
            text = " ".join([str(p) for p in content if p])
        elif isinstance(content, str):
            # Direct string
            text = content
            
        if text and role:
            message_count += 1
            if role == 'user':
                text_parts.append(f"User: {text}")
                if not first_user_message:
                    first_user_message = text
            elif role == 'assistant':
                text_parts.append(f"Assistant: {text}")
                
    full_text = "\n".join(text_parts)
    
    return {
        'text': full_text,
        'message_count': message_count,
        'snippet': first_user_message[:300] + "..." if len(first_user_message) > 300 else first_user_message
    }

def process_conversations(conversations: List[Dict]) -> pd.DataFrame:
    """
    Process raw conversations into a DataFrame with ID, Title, and Full Text.
    """
    processed_data = []
    
    # Handle case where input is a dict
    if isinstance(conversations, dict):
        if 'conversations' in conversations:
            conversations = conversations['conversations']
        elif 'messages' in conversations and ('conversation_id' in conversations or 'id' in conversations):
            # It's a single conversation object
            conversations = [conversations]
        else:
            # If it's a dict of conversations (unlikely but possible), use values
            conversations = list(conversations.values())
            
    if not isinstance(conversations, list):
        print("Error: Input data is not a list or recognized dictionary format.")
        return pd.DataFrame()
    
    for conv in conversations:
        # Safety check: ensure conv is a dictionary
        if not isinstance(conv, dict):
            continue
            
        conv_id = conv.get('conversation_id') or conv.get('id')
        title = conv.get('title', 'Untitled')
        
        try:
            details = extract_conversation_details(conv)
            
            if details['text'].strip(): # Only include if there's text
                processed_data.append({
                    'id': conv_id,
                    'title': title,
                    'text': details['text'],
                    'snippet': details['snippet'],
                    'message_count': details['message_count'],
                    'create_time': conv.get('create_time')
                })
        except Exception as e:
            print(f"Skipping conversation {conv_id} due to error: {e}")
            continue
            
    return pd.DataFrame(processed_data)
