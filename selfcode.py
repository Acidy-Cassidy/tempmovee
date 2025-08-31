import os
import sys
import subprocess
import tempfile
import importlib.util
import threading
import itertools
import time
import re
from openai import OpenAI

# Configure your OpenAI API key and OpenAI v1.x client

# Paths
tmp_dir     = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = sys.executable
ORIGIN_FILE = os.path.abspath(__file__)

# Registry for integrated tools: tool_name -> (callable, [arg_names])
tools = {}

class Spinner:
    def __init__(self, message="Working"):
        self.message = message
        self.done = False
        self.thread = None

    def spin(self):
        for c in itertools.cycle("|/-\\"):
            if self.done:
                break
            sys.stdout.write(f"\r{self.message}... {c}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r")

    def __enter__(self):
        self.done = False
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.done = True
        self.thread.join()
        sys.stdout.write("\r")


def generate_code(prompt: str, error: str = None) -> (str, str):
    system_msg = {
        "role": "system",
        "content": (
            "You are a helper that writes valid, runnable Python scripts. "
            "Use argparse for inputs and include a demo under `if __name__ == '__main__':` that sets "
            "`sys.argv` to sample arguments before calling the main function. Do NOT use input(). "
            "Respond strictly with code in a ```python``` block. Include a ```requirements.txt``` block if needed. "
            "Also include a top-level function called `test()` that exercises the script’s functionality "
            "with at least one assert (or returns True/False), so the runner can automatically validate "
            "correctness before integration. Make sure that the test function is custom to the script being generated."
        )
    }
    user_content = prompt
    if error:
        user_content += (
            f"\nThe previous run failed with this error:\n{error}\n"
            "Please correct the code accordingly and resend only the code. If this is happening more than 3 times in a row (with the same error) try a new approach"
        )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg, {"role": "user", "content": user_content}],
        temperature=0
    )
    content = resp.choices[0].message.content
    code_match = re.search(r"```python\n([\s\S]*?)```", content)
    code = code_match.group(1) if code_match else content
    req_match = re.search(r"```requirements\.txt\n([\s\S]*?)```", content)
    requirements = req_match.group(1).strip() if req_match else None
    return code.strip(), requirements


def generate_filename(description: str) -> str:
    """
    Ask GPT for a concise, descriptive snake_case filename
    (without extension) based on the command description.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a filename generator. Produce a short, "
                        "snake_case name (no extension) describing a Python script."},
            {"role": "user",
             "content": f"Generate a filename for a Python script that {description}."}
        ],
        temperature=0
    )
    # strip any backticks or quotes
    name = resp.choices[0].message.content.strip().strip('`"\'')
    # sanitize to [a-z0-9_]
    name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    return name.lower()


def save_to_file(code: str, filename: str = None) -> str:
    """
    Write code to disk. If filename is provided, use that (in tmp_dir);
    otherwise fall back to a random NamedTemporaryFile.
    Returns the full path.
    """
    if filename:
        path = os.path.join(tmp_dir, f"{filename}.py")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        return path

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir=tmp_dir)
    tmp.write(code.encode('utf-8'))
    tmp.flush()
    tmp.close()
    return tmp.name


def install_requirements(code: str, requirements: str = None):
    if requirements:
        req_path = os.path.join(tmp_dir, "requirements.txt")
        with open(req_path, "w") as f:
            f.write(requirements)
        subprocess.check_call([
            VENV_PYTHON, "-m", "pip", "install", "--upgrade", "-r", req_path
        ])
        return

    pkgs = set()
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import "):
            pkgs.add(line.split()[1].split(".")[0])
        elif line.startswith("from "):
            pkgs.add(line.split()[1].split(".")[0])

    to_install = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if to_install:
        subprocess.check_call([VENV_PYTHON, "-m", "pip", "install", "--upgrade"] + to_install)


def run_file(path: str, args: list[str] = None) -> (bool, str):
    cmd = [VENV_PYTHON, path] + (args or [])
    try:
        res = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=60
        )
        if res.returncode == 0:
            return True, res.stdout
        else:
            return False, res.stderr
    except Exception as e:
        return False, str(e)


def run_and_fix(path: str, extra_args: list[str] = None) -> (bool, str):
    ok, out = run_file(path, args=extra_args)
    if not ok:
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", out) or \
            re.search(r"Missing optional dependency ['\"]([^'\"]+)['\"]", out)
        if m:
            pkg = m.group(1).split('.')[0]
            print(f"Detected missing dependency '{pkg}', installing...")
            subprocess.check_call([
                VENV_PYTHON, "-m", "pip", "install", "--upgrade", pkg
            ])
            return run_file(path, args=extra_args)
    return ok, out


def integrate_module(path: str, func_name: str = None):
    spec = importlib.util.spec_from_file_location("integrated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tool_name = os.path.splitext(os.path.basename(path))[0]
    src = open(path, 'r', encoding='utf-8').read()
    arg_names = re.findall(r"parser\.add_argument\(\s*['\"]([^'\"]+)['\"]", src)

    func = None
    if hasattr(mod, 'calculate') and callable(mod.calculate):
        func = mod.calculate
    elif hasattr(mod, 'main') and callable(mod.main):
        func = mod.main
    elif all(hasattr(mod, op) for op in ('add','subtract','multiply','divide')):
        def dispatcher(op, x, y):
            if hasattr(mod, op):
                return getattr(mod, op)(x, y)
            raise ValueError(f"Unknown operation '{op}'")
        func = dispatcher
        arg_names = ('operation','x','y')

    if func:
        tools[tool_name] = (func, arg_names)

    with open(ORIGIN_FILE, 'a', encoding='utf-8') as out, \
         open(path,         'r', encoding='utf-8') as gen:
        out.write(f"\n# --- Begin integrated code: {tool_name} ---\n")
        out.write(gen.read())
        out.write(f"\n# --- End integrated code: {tool_name} ---\n")

    if func:
        usage = f"use {tool_name}"
        if arg_names:
            usage += " " + " ".join(f"<{a}>" for a in arg_names)
        print(f"New tool '{tool_name}' integrated. To run it: {usage}")
    else:
        print(f"New tool '{tool_name}' integrated, but no callable was found.")


def interactive_loop():
    print("Welcome to SelfCode Runner! Type 'help' for commands.")
    while True:
        cmd = input(">>> ").strip()
        if not cmd:
            continue
        parts = cmd.split()
        cmd0 = parts[0].lower()

        if cmd0 == 'help':
            print(
                "Commands:\n"
                "  help                Show this message\n"
                "  use <tool> <args>   Run an integrated tool\n"
                "  gen <description>   Generate and integrate new functionality\n"
                "  exit                Quit"
            )

        elif cmd0 == 'exit':
            break

        elif cmd0 == 'use':
            if len(parts) < 2:
                print("Usage: use <tool> <args>")
                continue
            tool, *args = parts[1:]
            if tool in tools:
                func, arg_names = tools[tool]
                if len(args) != len(arg_names):
                    usage = f"use {tool} " + " ".join(f"<{a}>" for a in arg_names)
                    print(f"Usage: {usage}")
                    continue
                cargs = []
                for v in args:
                    try:
                        cargs.append(float(v))
                    except:
                        cargs.append(v)
                try:
                    res = func(*cargs)
                    if res is not None:
                        print(res)
                except Exception as e:
                    print(f"Error running tool '{tool}': {e}")
            else:
                print(f"No integrated tool named '{tool}'")

        elif cmd0 == 'gen':
            description = ' '.join(parts[1:])
            error = None
            while True:
                with Spinner("Generating code"):
                    code, req = generate_code(description, error)
                print("Code generation complete.")

                # ask GPT for a human-friendly filename
                nice_name = generate_filename(description)
                path = save_to_file(code, filename=nice_name)

                with Spinner("Installing dependencies"):
                    install_requirements(code, req)
                print("Dependencies installed.")

                # discover positional args dynamically via '-h'
                help_ok, help_out = run_file(path, args=["-h"])
                sample_args = []
                if help_ok:
                    for line in help_out.splitlines():
                        if line.lower().startswith("usage"):
                            parts2 = line.split()
                            for token in parts2[2:]:
                                token = token.strip('[]')
                                if not token.startswith('-'):
                                    sample_args.append(token)
                            break

                with Spinner("Running generated code"):
                    ok, out = run_and_fix(path, extra_args=sample_args)

                if ok:
                    print("Output:\n", out)
                    spec = importlib.util.spec_from_file_location("testmod", path)
                    testmod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(testmod)
                    if hasattr(testmod, 'test') and callable(testmod.test):
                        try:
                            result = testmod.test()
                            if result is False:
                                raise AssertionError("test() returned False")
                            print("✅ test() passed.")
                        except AssertionError as e:
                            print(f"❌ test() failed: {e}")
                            error = out
                            continue
                    with Spinner("Integrating module"):
                        integrate_module(path)
                    print("Integration complete.")
                    break
                else:
                    print("Error:\n", out)
                    error = out

        else:
            print("Unknown command. Type 'help'.")


if __name__ == '__main__':
    interactive_loop()

# --- Begin integrated code: check_internet_connection ---
import argparse
import requests
import sys

def check_internet(url='http://www.google.com', timeout=5):
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        return False

def main():
    parser = argparse.ArgumentParser(description='Check for internet connectivity.')
    parser.add_argument('--url', type=str, default='http://www.google.com', help='URL to test connectivity (default: http://www.google.com)')
    parser.add_argument('--timeout', type=int, default=5, help='Timeout for the request in seconds (default: 5)')
    
    args = parser.parse_args()
    
    if check_internet(args.url, args.timeout):
        print("Internet is available.")
    else:
        print("No internet connection.")

def test():
    assert check_internet('http://www.google.com') == True, "Should be able to connect to Google"
    assert check_internet('http://nonexistent.url') == False, "Should not be able to connect to a nonexistent URL"
    return True

if __name__ == '__main__':
    # Demo with sample arguments
    sys.argv = ['script_name', '--url', 'http://www.google.com', '--timeout', '5']
    main()
# --- End integrated code: check_internet_connection ---

# --- Begin integrated code: check_internet_connection ---
import requests

def check_internet():
    try:
        # Attempt to connect to a reliable website
        requests.get("http://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def main():
    if check_internet():
        print("Internet is available.")
    else:
        print("No internet connection.")

def test():
    # This test will not be able to assert the actual internet connection,
    # but we can check if the function returns True or False.
    # We can mock the requests.get method to simulate both scenarios.
    import requests
    from unittest.mock import patch

    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        assert check_internet() == True

        mock_get.side_effect = requests.ConnectionError
        assert check_internet() == False

    return True

if __name__ == '__main__':
    import sys
    sys.argv = ['script_name']  # Simulate no arguments
    main()
    assert test()  # Run the test to validate functionality
# --- End integrated code: check_internet_connection ---

# --- Begin integrated code: hello_world_printer ---
import argparse

def main():
    parser = argparse.ArgumentParser(description='Print Hello World.')
    parser.add_argument('--greet', type=str, default='World', help='Name to greet')
    args = parser.parse_args()
    
    print(f'Hello, {args.greet}!')

def test():
    import io
    import sys

    # Redirect stdout to capture print statements
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the main function with a test argument
    parser = argparse.ArgumentParser(description='Print Hello World.')
    parser.add_argument('--greet', type=str, default='World', help='Name to greet')
    args = parser.parse_args(['--greet', 'Test'])
    print(f'Hello, {args.greet}!')

    # Reset redirect.
    sys.stdout = sys.__stdout__

    # Check if the output is as expected
    assert captured_output.getvalue().strip() == 'Hello, Test!', "Test failed!"
    return True

if __name__ == '__main__':
    import sys
    sys.argv = ['script_name', '--greet', 'World']  # Sample arguments
    main()
# --- End integrated code: hello_world_printer ---
