from itertools import product
from transformers import AutoModelForCausalLM, AutoTokenizer # BitsAndBytesConfig optional

import torch

class Setup: 
    """
    A class to initialize and configure a model, tokenizer, and associated parameters for text generation.
    """

    def __init__(self, config):
        """
        Initializes the Setup class by configuring the model, tokenizer, and parameters.

        Args:
            config (object): Configuration object containing model, device, and parameter settings.
        """
        self.model, self.tokenizer = self.setup_model(config)
        self.parameters = self.setup_parameters(config)
        self.excluded_tokens = self.setup_tokenizer()

    def setup_model(self, config):
        """
        Sets up the language model and tokenizer based on the given configuration.

        Args:
            config (object): Configuration object with the model name and device map.

        Returns:
            tuple: A tuple containing:
                - model (transformers.PreTrainedModel): The initialized language model.
                - tokenizer (transformers.PreTrainedTokenizer): The initialized tokenizer.
        """
        print("Setting up the model")
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model, device_map=config.device, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        
        return model, tokenizer

    def setup_parameters(self, config):
        """
        Configures the generation parameters by creating a product of all parameter combinations.

        Args:
            config (object): Configuration object containing parameter options.

        Returns:
            itertools.product: An iterator over all combinations of the parameter values.
        """
        print("Setting up the parameters")
        
        for prompt in config.parameters["prompts"]:
            tokenized_prompt_length = len(self.tokenizer(prompt))
            for retroactive_span in config.parameters["retroactive_spans"]:
                assert (tokenized_prompt_length <= retroactive_span) or (retroactive_span == -1), \
                    f"Error: Tokenized prompt length ({tokenized_prompt_length}) exceeds the retroactive span ({retroactive_span})."
        
        parameters = [
            config.parameters["prompts"], config.parameters["temperatures"], config.parameters["num_beams"]
        ]
        
        context = [
            (retro, pro)
            for retro in config.parameters["retroactive_spans"]
            for pro in config.parameters["proactive_spans"]
            if retro != -1 or (retro == -1 and pro == config.parameters["proactive_spans"][0])
        ]
        
        sampling_tuples = [
            (method, value)
            for method, values in config.parameters["sampling"].items()
            for value in values
        ]
        
        return product(*parameters, context, sampling_tuples)
    
    def setup_tokenizer(self):
        """
        Configures the tokenizer to exclude certain tokens during text generation.

        Returns:
            list of int: A list of token IDs to be excluded from generation.
        """
        return self.tokenizer([
            
            # Newlines and Whitespace Variations
            "\n", "\n\n", "\n\n\n", "\t", "\r", "\r\n",
            "#", "##", "###", "####",

            # URL Patterns
            "http", "https", "ftp", "www.", "ftp://", "https://", "http://",  
            ".com", ".org", ".net", ".gov", ".edu", ".co", ".io", ".uk",       

            # HTML Tags and Elements
            "<html>", "</html>", "<br>", "<p>", "<div>", "</div>", "<a>", "</a>", "<img>", "</img>",
            "<body>", "</body>", "<head>", "</head>", "<title>", "</title>", "<script>", "</script>",
            "<style>", "</style>", "<meta>", "<iframe>", "</iframe>", "<form>", "</form>", "export", "default",

            # Special Characters and Symbols
            "@", "#", "$", "%", "&", "*", "^", "!", "~", "`", "?", "+", "-", "=", "_", "[", "]", "{", "}",
            "|", "\\", ":", ";", "\"", "'", "<", ">", "/", ".", ",", "(", ")", "»", "£", "•", "‰", "→", "→", "÷", 
            "∘", "§", "©", "®", "†", "‡", "€", "¥", "₹",

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
        ]).input_ids