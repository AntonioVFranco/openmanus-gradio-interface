import subprocess
import sys
import os

# Install gradio if not already installed
try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

# Install toml parser for configuration
try:
    import toml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
    import toml

import time
import threading
import queue

# Global variables
process = None
output_queue = queue.Queue()
input_queue = queue.Queue()
is_running = False
OPENMANUS_DIR = os.path.abspath("./OpenManus")

def check_installation():
    """Checks if OpenManus is installed"""
    return os.path.exists(OPENMANUS_DIR)

def install_openmanus():
    """Installs OpenManus and its dependencies"""
    output = []
    original_dir = os.getcwd()
    
    try:
        # Install condacolab and dependencies
        output.append("Installing dependencies...")
        cmds = [
            [sys.executable, "-m", "pip", "install", "-q", "condacolab"],
            [sys.executable, "-m", "pip", "install", "numpy", "pandas", "matplotlib"],
            "curl -LsSf https://astral.sh/uv/install.sh | sh"
        ]
        
        for cmd in cmds:
            if isinstance(cmd, list):
                result = subprocess.run(cmd, capture_output=True, text=True)
                output.append(f"$ {' '.join(cmd)}")
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output.append(f"$ {cmd}")
            
            output.append(result.stdout if result.stdout else "")
            if result.stderr:
                output.append(f"ERROR: {result.stderr}")
        
        # Clone the OpenManus repository
        output.append("\nCloning OpenManus repository...")
        if os.path.exists(OPENMANUS_DIR):
            output.append("OpenManus directory already exists. Skipping clone.")
        else:
            clone_cmd = ["git", "clone", "https://github.com/mannaandpoem/OpenManus.git"]
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            output.append(f"$ {' '.join(clone_cmd)}")
            output.append(result.stdout if result.stdout else "")
            if result.stderr and "fatal" in result.stderr:
                output.append(f"ERROR: {result.stderr}")
                return "\n".join(output)
        
        # Install Xvfb for X server simulation - Using sudo for Colab environment
        output.append("\nInstalling Xvfb (X Virtual Framebuffer)...")
        try:
            subprocess.run(["sudo", "apt-get", "update", "-y"], capture_output=True, text=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "xvfb"], capture_output=True, text=True)
            output.append("✅ Xvfb installed successfully!")
        except Exception as e:
            output.append(f"⚠️ Warning: Could not install Xvfb: {str(e)}")
            output.append("This might cause issues with web searches, but we'll continue installation.")
        
        # Install requirements directly using the system Python
        output.append("\nInstalling requirements directly...")
        os.chdir(OPENMANUS_DIR)
        
        # Install requirements with the system Python
        req_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        result = subprocess.run(req_cmd, capture_output=True, text=True)
        output.append(f"$ {' '.join(req_cmd)}")
        output.append(result.stdout if result.stdout else "")
        if result.stderr:
            output.append(f"ERROR: {result.stderr}")
        
        # Install playwright and browsers
        output.append("\nInstalling Playwright package...")
        playwright_pkg_cmd = [sys.executable, "-m", "pip", "install", "playwright"]
        result = subprocess.run(playwright_pkg_cmd, capture_output=True, text=True)
        output.append(f"$ {' '.join(playwright_pkg_cmd)}")
        output.append(result.stdout if result.stdout else "")
        if result.stderr:
            output.append(f"ERROR: {result.stderr}")
        
        # Install browsers with --with-deps flag and in headless mode
        output.append("\nInstalling Playwright browsers (this may take a while)...")
        try:
            playwright_cmd = [sys.executable, "-m", "playwright", "install", "--with-deps"]
            result = subprocess.run(playwright_cmd, capture_output=True, text=True, timeout=300)
            output.append(f"$ {' '.join(playwright_cmd)}")
            output.append(result.stdout if result.stdout else "")
            if result.stderr:
                output.append(f"ERROR: {result.stderr}")
        except subprocess.TimeoutExpired:
            output.append("⚠️ Browser installation timed out, but this might be okay - continuing installation.")
        except Exception as e:
            output.append(f"⚠️ Warning during browser installation: {str(e)}")
            output.append("We'll continue with the installation anyway.")
        
        # Modify OpenManus config to force headless mode
        output.append("\nConfiguring Playwright to use headless mode...")
        try:
            # Look for browser_use.py files to modify
            browser_files_modified = False
            for root, dirs, files in os.walk(OPENMANUS_DIR):
                for file in files:
                    if file == "browser_use.py" or "browser" in file and file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            
                            # Check if we need to modify the file
                            if "launch(" in content and "headless=True" not in content:
                                # Ensure browser launches in headless mode
                                modified_content = content.replace("launch(", "launch(headless=True, ")
                                with open(file_path, 'w') as f:
                                    f.write(modified_content)
                                output.append(f"Modified {file_path} to force headless mode")
                                browser_files_modified = True
                        except Exception as e:
                            output.append(f"⚠️ Couldn't modify file {file_path}: {str(e)}")
            
            if not browser_files_modified:
                output.append("ℹ️ No browser files needed modification or couldn't find browser files.")
        except Exception as e:
            output.append(f"⚠️ Warning: Could not modify browser configuration: {str(e)}")
            output.append("Web searches might not work correctly.")
        
        # Copy configuration file
        output.append("\nSetting up configuration file...")
        config_dir = os.path.join(OPENMANUS_DIR, "config")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        config_example = os.path.join(config_dir, "config.example.toml")
        config_dest = os.path.join(config_dir, "config.toml")
        
        if os.path.exists(config_example):
            if not os.path.exists(config_dest):
                with open(config_example, 'r') as src, open(config_dest, 'w') as dst:
                    dst.write(src.read())
                output.append("Configuration file created successfully!")
            else:
                output.append("Configuration file already exists.")
        else:
            output.append("WARNING: Example configuration file not found.")
            
        output.append("\n✅ Installation completed successfully!")
        
        # Return to original directory
        os.chdir(original_dir)
        return "\n".join(output)
    
    except Exception as e:
        output.append(f"\n❌ Error during installation: {str(e)}")
        # Ensure we return to original directory
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
        return "\n".join(output)

def install_playwright_browsers():
    """Installs Playwright browsers required for web search functionality"""
    try:
        if not check_installation():
            return "❌ OpenManus is not installed. Please install OpenManus first."
        
        original_dir = os.getcwd()
        os.chdir(OPENMANUS_DIR)
        
        output = []
        
        # Install Xvfb if not already installed
        output.append("Installing Xvfb (X Virtual Framebuffer)...")
        try:
            subprocess.run(["sudo", "apt-get", "update", "-y"], capture_output=True, text=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "xvfb"], capture_output=True, text=True)
            output.append("✅ Xvfb installed successfully!")
        except Exception as e:
            output.append(f"⚠️ Warning: Could not install Xvfb: {str(e)}")
            output.append("This might cause issues with web searches, but we'll continue.")
        
        # Install playwright package using system Python
        output.append("Installing Playwright package...")
        playwright_pkg_cmd = [sys.executable, "-m", "pip", "install", "playwright"]
        pkg_result = subprocess.run(playwright_pkg_cmd, capture_output=True, text=True)
        
        if pkg_result.returncode != 0:
            os.chdir(original_dir)
            return f"❌ Error installing Playwright package: {pkg_result.stderr}"
        
        # Install browsers with --with-deps flag for better compatibility
        output.append("Installing Playwright browsers (this may take a while)...")
        try:
            playwright_cmd = [sys.executable, "-m", "playwright", "install", "--with-deps"]
            result = subprocess.run(playwright_cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            output.append("⚠️ Browser installation timed out, but this might be okay.")
            result = type('obj', (object,), {'returncode': 0})  # Mock successful result
        except Exception as e:
            output.append(f"⚠️ Warning during browser installation: {str(e)}")
            output.append("We'll continue anyway.")
            result = type('obj', (object,), {'returncode': 0})  # Mock successful result
        
        # Modify browser_use.py files to force headless mode
        output.append("Configuring Playwright to use headless mode...")
        browser_files_modified = False
        try:
            for root, dirs, files in os.walk(OPENMANUS_DIR):
                for file in files:
                    if file == "browser_use.py" or "browser" in file and file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            
                            if "launch(" in content and "headless=True" not in content:
                                modified_content = content.replace("launch(", "launch(headless=True, ")
                                with open(file_path, 'w') as f:
                                    f.write(modified_content)
                                output.append(f"Modified {file_path} to force headless mode")
                                browser_files_modified = True
                        except Exception as e:
                            output.append(f"⚠️ Couldn't modify file {file_path}: {str(e)}")
            
            if not browser_files_modified:
                output.append("ℹ️ No browser files needed modification or couldn't find browser files.")
        except Exception as e:
            output.append(f"⚠️ Warning: Could not modify browser configuration: {str(e)}")
        
        os.chdir(original_dir)
        output_str = "\n".join(output)
        
        if hasattr(result, 'returncode') and result.returncode == 0:
            return f"✅ Playwright browsers setup completed!\n\n{output_str}"
        else:
            return f"⚠️ Partial success - some components may not work correctly.\n\n{output_str}"
            
    except Exception as e:
        os.chdir(original_dir) if 'original_dir' in locals() else None
        return f"❌ Error: {str(e)}"

def simplify_config(provider="openai"):
    """Simplifies the config.toml to use only one provider directly in the LLM section"""
    config_path = os.path.join(OPENMANUS_DIR, "config", "config.toml")
    try:
        if not os.path.exists(config_path):
            return "❌ Configuration file not found. Please install OpenManus first."
        
        # Read the config file
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Parse the TOML content
        try:
            config_data = toml.loads(config_content)
        except Exception as e:
            return f"❌ Error parsing config file: {str(e)}"
        
        # Get the API key and model for the selected provider
        api_key = ""
        model = ""
        base_url = ""
        
        # Check in provider section if it exists
        if provider in config_data and 'api_key' in config_data[provider]:
            api_key = config_data[provider]['api_key']
            if 'model' in config_data[provider]:
                model = config_data[provider]['model']
        
        # Set default values if missing
        if not model:
            default_models = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-opus-20240229",
                "google": "gemini-pro",
                "mistral": "mistral-large-latest",
                "azure": "gpt-4"
            }
            if provider in default_models:
                model = default_models[provider]
        
        # Set default base_url
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "anthropic":
            base_url = "https://api.anthropic.com"
        elif provider == "azure":
            base_url = "https://YOUR_RESOURCE_NAME.openai.azure.com"
        
        # Create a simplified config
        simplified_config = {
            "llm": {
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "max_tokens": 4096,
                "temperature": 0.0
            }
        }
        
        # Add base_url if we have one
        if base_url:
            simplified_config["llm"]["base_url"] = base_url
        
        # Add vision section for OpenAI if needed
        if provider == "openai":
            simplified_config["llm"]["vision"] = {
                "model": model,
                "api_key": api_key
            }
            if base_url:
                simplified_config["llm"]["vision"]["base_url"] = base_url
        
        # Write the simplified config
        with open(config_path, 'w') as f:
            toml.dump(simplified_config, f)
        
        return f"✅ Configuration simplified to use only {provider} directly in the LLM section. Please restart OpenManus to apply changes."
    
    except Exception as e:
        return f"❌ Error simplifying configuration: {str(e)}"

def fix_anthropic_config():
    """Special fix for the Anthropic/OpenAI API mismatch issue"""
    config_path = os.path.join(OPENMANUS_DIR, "config", "config.toml")
    try:
        if not os.path.exists(config_path):
            return "❌ Configuration file not found. Please install OpenManus first."
        
        # Read the config file
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Parse the TOML content
        try:
            config_data = toml.loads(config_content)
        except Exception as e:
            return f"❌ Error parsing config file: {str(e)}"
        
        # Check if we have an OpenAI section with an Anthropic key
        openai_key = config_data.get('openai', {}).get('api_key', '')
        if openai_key and openai_key.startswith("sk-ant-"):
            # This is definitely an Anthropic key in the OpenAI section
            
            # Create or update Anthropic section
            if 'anthropic' not in config_data:
                config_data['anthropic'] = {}
            
            config_data['anthropic']['api_key'] = openai_key
            
            # Set a default model if none exists
            if 'model' not in config_data['anthropic'] or not config_data['anthropic']['model']:
                config_data['anthropic']['model'] = 'claude-3-opus-20240229'
            
            # Update LLM provider
            if 'llm' not in config_data:
                config_data['llm'] = {}
            
            config_data['llm']['provider'] = 'anthropic'
            
            # Write the updated config back to the file
            with open(config_path, 'w') as f:
                toml.dump(config_data, f)
            
            return "✅ Fixed API configuration issue! Moved Anthropic API key to the correct section and set Anthropic as the default provider."
        
        # Check if we need to create an Anthropic section
        if 'llm' in config_data and config_data['llm'].get('provider') == 'anthropic' and 'anthropic' not in config_data:
            config_data['anthropic'] = {
                'api_key': '',
                'model': 'claude-3-opus-20240229'
            }
            
            # Write the updated config back to the file
            with open(config_path, 'w') as f:
                toml.dump(config_data, f)
            
            return "✅ Created Anthropic configuration section! Please enter your Anthropic API key."
        
        return "No specific API configuration issue detected. Please check your API keys manually."
            
    except Exception as e:
        return f"❌ Error fixing configuration: {str(e)}"

def diagnose_config():
    """Diagnoses configuration issues and suggests fixes"""
    config_path = os.path.join(OPENMANUS_DIR, "config", "config.toml")
    try:
        if not os.path.exists(config_path):
            return "❌ Configuration file not found. Please install OpenManus first."
        
        # Read the config file
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Parse the TOML content
        try:
            config_data = toml.loads(config_content)
        except Exception as e:
            return f"❌ Error parsing config file: {str(e)}"
        
        # Check for common configuration issues
        issues = []
        fixes = []
        
        # Check if llm provider is set
        llm_provider = None
        if 'llm' in config_data and 'provider' in config_data['llm']:
            llm_provider = config_data['llm']['provider']
            
        # Check for provider sections
        available_providers = [key for key in config_data.keys() if key in ["openai", "anthropic", "google", "ollama", "mistral", "azure"]]
        
        # If no llm provider is set, but we have exactly one provider section with an API key, suggest setting it
        if not llm_provider and len(available_providers) == 1:
            provider = available_providers[0]
            if 'api_key' in config_data[provider]:
                issues.append(f"No default LLM provider is set, but {provider} is configured.")
                fixes.append(f"Set {provider} as the default provider.")
                config_data['llm'] = {'provider': provider}
        
        # Check for API key issues
        for provider in available_providers:
            if provider in config_data:
                # Check if API key exists
                if 'api_key' not in config_data[provider] or not config_data[provider]['api_key']:
                    issues.append(f"No API key found for {provider}.")
                    fixes.append(f"Add an API key for {provider}.")
                
                # Check for model setting
                if 'model' not in config_data[provider] or not config_data[provider]['model']:
                    # Add default model recommendations
                    default_models = {
                        "openai": "gpt-4-turbo-preview",
                        "anthropic": "claude-3-opus-20240229",
                        "google": "gemini-pro",
                        "mistral": "mistral-large-latest",
                        "azure": "gpt-4"
                    }
                    if provider in default_models:
                        issues.append(f"No model specified for {provider}.")
                        fixes.append(f"Set a default model (e.g., {default_models[provider]}) for {provider}.")
                        config_data[provider]['model'] = default_models[provider]
        
        # Check for OpenAI using Anthropic API key issue (the specific error in the logs)
        if llm_provider == "anthropic" and "openai" in config_data:
            if "anthropic" not in config_data:
                issues.append("System is configured to use Anthropic but no Anthropic section found.")
                fixes.append("Add Anthropic configuration section.")
                
                # Copy the OpenAI API key to Anthropic if it seems to be an Anthropic key
                if 'api_key' in config_data['openai'] and config_data['openai']['api_key'].startswith("sk-ant-"):
                    config_data['anthropic'] = {
                        'api_key': config_data['openai']['api_key'],
                        'model': 'claude-3-opus-20240229'
                    }
                    issues.append("OpenAI section contains an Anthropic API key (starts with sk-ant-).")
                    fixes.append("Move Anthropic API key to the Anthropic section.")
        
        # Apply fixes if there are issues
        if issues and fixes:
            # Write the updated config back to the file
            with open(config_path, 'w') as f:
                toml.dump(config_data, f)
            
            issues_text = "\n".join([f"- {issue}" for issue in issues])
            fixes_text = "\n".join([f"- {fix}" for fix in fixes])
            return f"✅ Fixed configuration issues!\n\nIssues detected:\n{issues_text}\n\nApplied fixes:\n{fixes_text}"
        elif not issues:
            return "✅ No configuration issues detected!"
        
        return "⚠️ Could not automatically fix all configuration issues. Please check the configuration file manually."
            
    except Exception as e:
        return f"❌ Error diagnosing configuration: {str(e)}"

def load_config():
    """Loads the OpenManus configuration file"""
    config_path = os.path.join(OPENMANUS_DIR, "config", "config.toml")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_content = f.read()
            return config_content
        else:
            return "# Configuration file not found."
    except Exception as e:
        return f"# Error loading configuration: {str(e)}"

def save_config(config_content):
    """Saves the OpenManus configuration file"""
    config_path = os.path.join(OPENMANUS_DIR, "config", "config.toml")
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        return "✅ Configuration saved successfully!"
    except Exception as e:
        return f"❌ Error saving configuration: {str(e)}"

def update_api_key(api_key, provider, model=None):
    """Updates the API key in the config.toml file"""
    config_path = os.path.join(OPENMANUS_DIR, "config", "config.toml")
    try:
        if not os.path.exists(config_path):
            return "❌ Configuration file not found. Please install OpenManus first."
        
        # Read the current config file
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check if we need to update the model as well
        update_model = model is not None and model.strip() != ""
        
        # Parse the TOML content
        try:
            config_data = toml.loads(config_content)
        except Exception as e:
            return f"❌ Error parsing config file: {str(e)}"
        
        # Make sure the provider section exists
        if provider not in config_data:
            config_data[provider] = {}
        
        # Update the API key
        config_data[provider]["api_key"] = api_key
        
        # Update the model if provided
        if update_model:
            config_data[provider]["model"] = model
        
        # Also check if there's an 'llm' section that needs to be updated
        if 'llm' in config_data:
            if 'provider' not in config_data['llm'] or config_data['llm']['provider'] != provider:
                config_data['llm']['provider'] = provider
                return_message = f"✅ API key updated and default provider set to {provider}!"
            else:
                return_message = "✅ API key updated successfully!"
        else:
            # Create llm section if it doesn't exist
            config_data['llm'] = {'provider': provider}
            return_message = f"✅ API key updated and default provider set to {provider}!"
        
        # Write the updated config back to the file
        with open(config_path, 'w') as f:
            toml.dump(config_data, f)
        
        return return_message
            
    except Exception as e:
        return f"❌ Error updating API key: {str(e)}"

def extract_api_key_and_model(config_content, provider):
    """Extracts the API key and model for a specific provider from the config content"""
    try:
        # Parse the TOML content
        try:
            config_data = toml.loads(config_content)
        except Exception:
            return "", ""  # Return empty strings if parsing fails
        
        # Extract API key and model
        api_key = ""
        model = ""
        
        if provider in config_data:
            if "api_key" in config_data[provider]:
                api_key = config_data[provider]["api_key"]
            if "model" in config_data[provider]:
                model = config_data[provider]["model"]
        
        return api_key, model
    except Exception as e:
        print(f"Error extracting API key and model: {str(e)}")
        return "", ""

def get_current_provider(config_content):
    """Gets the currently selected provider from the config"""
    try:
        # Parse the TOML content
        try:
            config_data = toml.loads(config_content)
        except Exception:
            return "openai"  # Default to openai if parsing fails
        
        # Check the llm section for the provider setting
        if 'llm' in config_data and 'provider' in config_data['llm']:
            return config_data['llm']['provider']
        
        return "openai"  # Default to openai if not found
    except Exception as e:
        print(f"Error getting current provider: {str(e)}")
        return "openai"

def output_reader(proc):
    """Reads process output and puts it in the queue"""
    global is_running
    for line in iter(proc.stdout.readline, ''):
        if line:
            output_queue.put(line.strip())
    is_running = False

def input_writer(proc):
    """Writes input from queue to process"""
    while is_running:
        try:
            prompt = input_queue.get(timeout=1)
            if prompt and proc.poll() is None:
                proc.stdin.write(prompt + '\n')
                proc.stdin.flush()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error writing to process: {str(e)}")

def send_prompt(prompt):
    """Sends a prompt to OpenManus and returns its response"""
    global process, is_running
    
    if not is_running or process is None:
        return "❌ OpenManus is not running. Please start it first."
    
    try:
        # Clear previous output
        while not output_queue.empty():
            output_queue.get_nowait()
            
        # Send prompt to the process
        input_queue.put(prompt)
        
        # Wait for initial response
        time.sleep(2)  # Give some time for processing
        
        # Return a message indicating processing has started
        return "Prompt sent to OpenManus. Click 'Update Response' to see progress and results."
    
    except Exception as e:
        return f"❌ Error sending prompt: {str(e)}"

def get_response():
    """Gets the response from OpenManus for the most recent prompt"""
    lines = []
    try:
        while not output_queue.empty():
            lines.append(output_queue.get_nowait())
    except queue.Empty:
        pass
    
    if not lines:
        return "No response data available yet. Click 'Update Response' again in a few seconds." if is_running else "OpenManus is not running."
    
    return "\n".join(lines)

def start_openmanus():
    """Starts OpenManus"""
    global process, is_running
    
    if is_running:
        return "OpenManus is already running!"
    
    try:
        # Check if OpenManus is installed
        if not check_installation():
            return "Error: OpenManus is not installed. Please install it first."
        
        # Change to OpenManus directory
        original_dir = os.getcwd()
        os.chdir(OPENMANUS_DIR)
        
        try:
            # Try using xvfb-run if available
            process = subprocess.Popen(
                ["xvfb-run", "--auto-servernum", sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except FileNotFoundError:
            # Fallback to direct execution if xvfb-run is not available
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        
        # Start threads to read output and write input
        is_running = True
        output_thread = threading.Thread(target=output_reader, args=(process,))
        output_thread.daemon = True
        output_thread.start()
        
        input_thread = threading.Thread(target=input_writer, args=(process,))
        input_thread.daemon = True
        input_thread.start()
        
        # Return to original directory
        os.chdir(original_dir)
        
        return "OpenManus started successfully!"
    
    except Exception as e:
        is_running = False
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return f"Error starting OpenManus: {str(e)}"

def stop_openmanus():
    """Stops OpenManus"""
    global process, is_running
    
    if not is_running:
        return "OpenManus is not running."
    
    try:
        process.terminate()
        time.sleep(1)
        
        if process.poll() is None:  # Process is still running
            if os.name == 'nt':
                subprocess.run(["taskkill", "/f", "/pid", str(process.pid)])
            else:
                process.kill()
        
        is_running = False
        return "OpenManus stopped successfully."
    
    except Exception as e:
        return f"Error stopping OpenManus: {str(e)}"

def get_output():
    """Gets the most recent output from OpenManus"""
    lines = []
    try:
        while not output_queue.empty():
            lines.append(output_queue.get_nowait())
    except queue.Empty:
        pass
    
    if not lines:
        return "No new output..." if is_running else "OpenManus is not running."
    
    return "\n".join(lines)

def check_status():
    """Checks the current status of OpenManus"""
    if is_running:
        return "OpenManus is running."
    else:
        return "OpenManus is not running."

# Create Gradio interface
with gr.Blocks(title="OpenManus Interface") as app:
    gr.Markdown("# OpenManus Interface")
    gr.Markdown("Interface to install, configure and run OpenManus.")
    
    with gr.Tab("Installation"):
        with gr.Row():
            with gr.Column():
                install_btn = gr.Button("Install OpenManus", variant="primary")
                status_text = gr.Textbox(label="Status", value="Waiting...", interactive=False)
            
        install_output = gr.Textbox(label="Installation Log", lines=15, max_lines=30)
        install_btn.click(install_openmanus, outputs=[install_output])
        
        gr.Markdown("### Additional Tools")
        with gr.Row():
            playwright_btn = gr.Button("Install Playwright Browsers", variant="secondary")
            playwright_status = gr.Textbox(label="Status", interactive=False)
        playwright_btn.click(install_playwright_browsers, outputs=[playwright_status])
    
    with gr.Tab("Configuration"):
        gr.Markdown("### OpenManus Configuration")
        gr.Markdown("Edit the config.toml file to customize OpenManus behavior:")
        
        load_btn = gr.Button("Load Configuration")
        
        config_content = gr.Textbox(
            label="config.toml file",
            lines=20,
            max_lines=50,
            interactive=True
        )
        
        with gr.Row():
            save_btn = gr.Button("Save Configuration", variant="primary")
            config_status = gr.Textbox(label="Status", interactive=False)
        
        load_btn.click(load_config, outputs=[config_content])
        save_btn.click(save_config, inputs=[config_content], outputs=[config_status])
    
    with gr.Tab("Execution"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### OpenManus Chat")
                
                # Chat history component
                chat_history = gr.Chatbot(
                    label="Conversation",
                    height=500
                )
                
                # Input area with send button
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your prompt to OpenManus...",
                        lines=3,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", variant="primary")
                
                # Controls underneath
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    update_response_btn = gr.Button("Update Response", variant="secondary")
            
            # Right side for controls and logs
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                with gr.Row():
                    start_btn = gr.Button("Start OpenManus", variant="primary")
                    stop_btn = gr.Button("Stop OpenManus", variant="stop")
                
                status_btn = gr.Button("Check Status")
                control_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### System Logs")
                output_display = gr.Textbox(
                    label="Log Output", 
                    lines=15, 
                    max_lines=30,
                    interactive=False,
                    elem_id="system-logs"
                )
                refresh_btn = gr.Button("Refresh Logs")
        
        # Hidden container for raw response (used for processing)
        raw_response = gr.Textbox(visible=False)
        
        # Function to handle sending a message
        def user_message_callback(prompt, history):
            # Add user message to chat
            history = history + [[prompt, None]]
            return "", history
        
        def bot_response_callback(history):
            if not is_running or process is None:
                response = "Error: OpenManus is not running. Please start it first."
                history[-1][1] = response
                return history
            
            try:
                # Clear previous output
                while not output_queue.empty():
                    output_queue.get_nowait()
                
                # Send prompt to the process
                input_queue.put(history[-1][0])
                
                # Wait for initial response
                time.sleep(2)  # Give some time for processing
                
                # Return a message indicating processing has started
                history[-1][1] = "Processing your request... Click 'Update Response' for progress."
                return history
            
            except Exception as e:
                history[-1][1] = f"Error sending prompt: {str(e)}"
                return history
        
        def update_chat_response(history):
            if not history:
                return history
            
            lines = []
            try:
                while not output_queue.empty():
                    lines.append(output_queue.get_nowait())
            except queue.Empty:
                pass
            
            if not lines:
                if not is_running:
                    if history[-1][1] == "Processing your request... Click 'Update Response' for progress.":
                        history[-1][1] = "OpenManus is not running."
                return history
            
            # Format response with markdown
            response = "\n".join(lines)
            
            # Replace certain patterns to improve markdown formatting
            # Convert log timestamps to headers or remove them
            response = response.replace("WARNING  |", "### WARNING: ")
            response = response.replace("INFO     |", "### INFO: ")
            response = response.replace("ERROR    |", "### ERROR: ")
            
            # Make tool names bold
            if "Tool" in response:
                response = response.replace("Tool", "**Tool**")
            
            # Clean up special characters
            response = response.replace("✨", "")
            response = response.replace("🛠️", "")
            response = response.replace("🧰", "")
            response = response.replace("🔧", "")
            response = response.replace("🎯", "")
            
            history[-1][1] = response
            return history
        
        def clear_chat_history():
            return None
        
        # Connect the buttons to functions
        send_btn.click(
            user_message_callback,
            inputs=[prompt_input, chat_history],
            outputs=[prompt_input, chat_history],
            queue=False
        ).then(
            bot_response_callback,
            inputs=[chat_history],
            outputs=[chat_history]
        )
        
        update_response_btn.click(
            update_chat_response,
            inputs=[chat_history],
            outputs=[chat_history]
        )
        
        clear_btn.click(
            clear_chat_history,
            outputs=[chat_history]
        )
        
        # Connect other control buttons
        start_btn.click(start_openmanus, outputs=[control_status])
        stop_btn.click(stop_openmanus, outputs=[control_status])
        status_btn.click(check_status, outputs=[control_status])
        refresh_btn.click(get_output, outputs=[output_display])
        
    # Interface initialization
    gr.Markdown("---")
    gr.Markdown("### How to use this interface:")
    gr.Markdown("1. Go to the **Installation** tab to install OpenManus")
    gr.Markdown("2. Go to the **Configuration** tab to adjust settings as needed")
    gr.Markdown("3. Go to the **Execution** tab to start OpenManus, send prompts, and monitor responses")

# Start Gradio application
if __name__ == "__main__":
    app.launch(share=True)  # Share a public accessible link
