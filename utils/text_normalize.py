import re
import string

def remove_punctuations_symbols_emojis(text):
    # Emoji pattern (targeting emojis)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # Dingbats and other symbols
        "]+", flags=re.UNICODE
    )

    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    
    return text

if __name__ == "__main__":
    text = "Hello, coding bro!😜"
    normalized_text = remove_punctuations_symbols_emojis(text)
    print(normalized_text) # Output: "Hello, coding bro!😜"
