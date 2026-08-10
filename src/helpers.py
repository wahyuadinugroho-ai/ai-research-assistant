def build_chat_history(messages, max_messages=6):
    """Build recent conversation history."""

    recent_messages = messages[-max_messages:]

    history = []

    for message in recent_messages:
        role = message["role"].capitalize()

        history.append(f"{role}: {message['content']}")

    return "\n".join(history)


def extract_text(content):
    """Extract plain text from an LLM response content."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    text_blocks.append(text)
            elif isinstance(block, list):
                text_blocks.append(extract_text(block))
            elif isinstance(block, str):
                text_blocks.append(block)

        return "".join(text_blocks)

    return str(content)
