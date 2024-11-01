import re

def remove_punctuations_symbols_emojis(text):
    # Emoji pattern (targeting emojis)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F100-\U0001F1FF"  # flags (iOS)
        u"\U0001F200-\U0001F2FF"  # additional emojis
        u"\U00002702-\U000027B0"  # Dingbats and other symbols
        u"\U0001F780-\U0001F7FF"
        u"\U0001F900-\U0001F9FF"
        "]+", flags=re.UNICODE
    )

    segments = []  # To store text segments
    emojis = []  # To store emojis and their positions
    
    last_end = 0  # Track the end of the last match to segment the text
    
    firstIndex = False

    # Iterate over each emoji match
    for match in emoji_pattern.finditer(text):
        start, end = match.span()

        if start == 0:
            firstIndex = True
        
        # Append text segment before the emoji
        if last_end != start:
            segments.append(text[last_end:start])
        
        # Store emoji info (position and emoji)
        emojis.append(text[start:end])
        
        # Update last_end to the end of the current emoji
        last_end = end
    
    # Append any remaining text after the last emoji
    if last_end < len(text):
        segments.append(text[last_end:])
    
    return segments, emojis, firstIndex

# Example usage
if __name__ == "__main__":
    text = "😊😊😊😊😊😊😊😊 😊😊😊😊"
    segments, emojis, firstIndex = remove_punctuations_symbols_emojis(text)
    
    print("Text Segments:", segments)
    print("Emojis Info:", emojis)
    print("Is first: ", firstIndex)
