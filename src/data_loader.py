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
    """
    text_parts = []
    message_count = 0
    first_user_message = ""
    
    mapping = conversation.get('mapping', {})
    
    # Sort by creation time to get chronological order
    # We need to collect nodes first
    nodes = []
    for node_id, node_data in mapping.items():
        if node_data.get('message'):
            nodes.append(node_data)
            
    # Sort by create_time if available
    nodes.sort(key=lambda x: x.get('message', {}).get('create_time') or 0)
    
    for node_data in nodes:
        message = node_data.get('message')
        if message and message.get('content'):
            content = message['content']
            role = message.get('author', {}).get('role')
            
            if content.get('content_type') == 'text':
                parts = content.get('parts', [])
                text = " ".join([str(p) for p in parts if p])
                
                if text:
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
    
    for conv in conversations:
        conv_id = conv.get('conversation_id') or conv.get('id')
        title = conv.get('title', 'Untitled')
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
            
    return pd.DataFrame(processed_data)
