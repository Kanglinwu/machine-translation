from flask import Flask, render_template, request
import requests  # Import the requests library

app = Flask(__name__)

# Custom filter for nl2br
# 自定義過濾器
@app.template_filter('nl2br')
def nl2br_filter(s):
    return s.replace('\n', '<br>\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    input_text: str = "" # Input text
    output_text: str = "" # Translated text 
    target_lang: str = "en"  # Target Language. Set default as English
    model_used: str = "None" # Model used
    source_lang:str = "Not detected" # Source Language
    
    if request.method == 'POST':
        input_text = request.form['input_text']
        target_lang = request.form['language']  # Changed from target_lang to language to match form field
        
        # Send the input_text to the external server
        try:
            # Send a POST request to the external server at 10.10.10.48:2486
            response = requests.post('http://10.10.10.48:2486/translate', json={'msg': input_text, 'target_lang': target_lang})
            
            # Check if the response is successful (status code 200)
            if response.status_code == 200:
                output_text = response.json().get("target_msg", "No output received")
                source_lang = response.json().get("source_lang", "None")
                is_trans = response.json().get("is_trans", False)
                model_used = response.json().get("model", "None")

            else:
                output_text = f"Error: Received status code {response.status_code} from the server."
        
        except requests.exceptions.RequestException as e:
            output_text = f"An error occurred: {e}"

    return render_template('index.html', input_text=input_text, output_text=output_text, target_lang=target_lang, model_used=model_used, is_trans=str(is_trans), source_lang=source_lang)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
