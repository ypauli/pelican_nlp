from itertools import product
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch

class PipelineSetup: 
    def __init__(self, config):
        self.device = self.set_device()
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map=self.device, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.excluded_tokens = self.setup_tokenizer(self.tokenizer)
        self.covariances = self.instantiate_covariances(config)

    @staticmethod
    def set_device():
        if torch.cuda.is_available() == False:
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
        return torch.device("cuda")
    
    @staticmethod
    def instantiate_covariances(config):  
        covariance_matrix = np.outer(config.std_devs, config.std_devs) * config.correlation_matrix
        return covariance_matrix

    @staticmethod
    def setup_tokenizer(tokenizer):
        return tokenizer([
            # Newlines and Whitespace Variations
            "\n", "\n\n", "\n\n\n", "\t", "\r", "\r\n"," \n", " \n\n", ".\n", ".\n\n", 
            "#", "##", "###", "####", "'<0x0A>'", "<0x0A>", "ASSISTANT",
            # URL Patterns
            "http", "https", "ftp", "www.", "ftp://", "https://", "http://",  
            ".com", ".org", ".net", ".gov", ".edu", ".co", ".io", ".uk",       
            # HTML Tags and Elements
            "<html>", "</html>", "<br>", "<p>", "<div>", "</div>", "<a>", "</a>", "<img>", "</img>",
            "<body>", "</body>", "<head>", "</head>", "<title>", "</title>", "<script>", "</script>",
            "<style>", "</style>", "<meta>", "<iframe>", "</iframe>", "<form>", "</form>", "export", "default",
            # Social Media Handles and Patterns
            "@username", "#hashtag", "RT", "DM", "PM", "like", "share", "follow", "subscribe", "retweet", 
            # Common Internet Slang, Abbreviations, and Acronyms
            "lol", "omg", "brb", "idk", "lmk", "tbh", "fyi", "smh", "btw", "ftw", "irl", "afaik", "imo", "fomo",
            "np", "gg", "omw", "rofl", "nvm", "yolo", "bff", "thx", "ttyl", "wyd", "wfh", "lfg",
            # Common Emoticons and Emojis
            ":)", ":(", ":D", ";)", ":-)", ":-(", ":-D", ";-)", ":/", ":|", ":'(", "XD", ":-P", ":P", 
            "😂", "❤️", "👍", "😭", "🙏", "🙌", "🔥", "💯", "🎉", "✨", "💕", "🎶", "📷", "🌟", "💔", "🎂",
            # Placeholder and System Text
            "user", "admin", "guest", "member", "page not found", "loading...", "click here", "submit", "login", 
            "logout", "password", "username", "sign in", "sign up", "terms of service", "privacy policy", 
            "powered by", "404", "403", "500", "401", "503", "site map", "cookie policy", "GDPR",
            # Legal and Compliance Terms
            "cookie policy", "privacy policy", "terms of service", "GDPR", "data protection", "personal data", 
            "opt-in", "opt-out", "consent", "disclaimer", "rights reserved", "trademark", "copyright",
            # Javascript and System Artifacts
            "javascript:void(0);", "NaN", "undefined", "null", "false", "true", "document.write", 
            "window.location", "alert", "console.log", "function()", "event.preventDefault()", "onclick",
            # Filler Text
            "lorem ipsum", "placeholder", "sample text", "dummy text", "example text", "text here", "caption", 
            "insert here", "replace with text",
            # Time and Date Formats
            "00:00", "12:00 AM", "12:00 PM", "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", 
            "yesterday", "today", "tomorrow", "AM", "PM",
            # Common Brand Names and Platforms
            "Google", "Facebook", "Twitter", "YouTube", "Instagram", "Snapchat", "TikTok", "Pinterest", 
            "LinkedIn", "Reddit", "Tumblr", "WhatsApp", "Spotify", "Netflix", "Amazon", "Apple",
            # File Extensions and Types
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", 
            ".csv", ".txt", ".html", ".xml", ".json", ".js", ".css", ".zip", ".rar", ".mp3", ".wav", ".mp4", ".mov",
            # Email Patterns
            "mailto:", "@example.com", "@gmail.com", "@yahoo.com", "@outlook.com", "@hotmail.com", "@icloud.com",
            # Common Placeholder Data and Nonsense Strings
            "asdf", "qwerty", "12345", "test", "test123", "hello world", "foo", "bar", "baz"
            # Programming Syntax and Code Artifacts
            "var", "let", "const", "function", "return", "if", "else", "for", "while", "switch", "case", "class",
            "this", "new", "null", "undefined", "true", "false", "try", "catch", "await", "async", "import",
            "$", "->", "=>", "::", "=>", "==", "===", "!==", "||", "&&", "(", ")", "{", "}", "[", "]", ":", ";", ",", 
            "\"", "'", "`", "@param", "export", "import", "return", "PERF_LOG", "INFO", "vg_c", "request_id", "data()", "push_back", ".push_back", "--no-create",
            # LaTeX Symbols and Math Expressions
            r"\\", r"$$", r"$", r"\cdot", r"\times", r"\pi", r"\frac", r"\sum", r"\prod", r"\int", r"\lim", r"\sin", r"\cos", r"\tan",
            "=", ">", "<", r"\\begin{equation}", r"\\end{equation}", r"\\bB$", r"\\I_C_p", r"\\end{", r"\alpha", r"\beta", r"\gamma",
            r"\sigma", r"\mu", r"\lambda", r"\delta", r"\theta", r"\rho", r"\kappa", r"\omega", r"\phi", r"\epsilon", r"\Gamma", r"\Delta",
            r"\Omega", r"\\mathcal", r"\\mathbb", "p05", "16", "9.19", "2.972", "-136", "35160", "30px", "5%", "9:15", "o json",
            # Statistical and Numerical Artifacts
            "0.11", "0", "1", "3rd", "2nd", "16px", "5px", "15px", "9.19", "0.11", "16", "9:15", "199.4%", "2.972",
            "-136", "35160", "30px", "5%", "20x0600", "0.11", "0", "16px", "5px", "15px", "-136", "199.4%",
            # Units and Measures
            "m.s.s.", "px", "rem", "kg", "cm", "mm", "km", "mi", "lb", "oz", "ft", "in", "%", "°C", "°F",
            # System and Placeholder Text
            "404", "403", "500", "401", "503", "admin", "user", "guest", "error", "request", "response", 
            "login", "logout", "sign in", "sign up", "submit", "loading", "click here", "terms of service",
            "privacy policy", "GDPR", "cookie policy", "powered by", "debug", "trace", "end", "begin",
            # Random Nonsense or Placeholders
            "asdf", "qwerty", "test", "test123", "sample", "example", "foo", "bar", "baz", "BACKWARDS",
            # Special Characters and Symbols
            "@", "#", "$", "%", "&", "*", "^", "~", "`", "+", "-", "=", "_", "[", "]", "{", "}",
            "|", "\\", "\"", "'", "<", ">", "/", "»", "£", "•", "‰", "→", "→", "÷", 
            "∘", "§", "©", "®", "†", "‡", "€", "¥", "₹",
        ]).input_ids