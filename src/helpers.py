def extract_text(content):
    """Extract plain text from an LLM response content safely."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if block is None:
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    text_blocks.append(str(text))
            elif isinstance(block, list):
                text_blocks.append(extract_text(block))
            elif isinstance(block, str):
                text_blocks.append(block)
            else:
                text_blocks.append(str(block))
        return "".join(text_blocks)

    if isinstance(content, dict):
        return content.get("text", "") or ""

    return str(content)
