import os
import re, subprocess, tempfile, shutil

def remove_comments_from_sol(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Remove single-line comments (//)
    content = re.sub(r'//.*', '', content)
    # Remove multi-line comments (/* */)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    return content

def remove_labels_from_sol(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove comments that match: // <yes> <report> NAME_OF_VULNERABILITY
    content = re.sub(r'//\s*<yes>\s*<report>\s*[\w\-]+', '', content)
    # Remove: // @vulnerable_at_lines: NUMBER (with optional spaces)
    content = re.sub(r'^\s*\*?\s*@vulnerable_at_lines:.*$', '', content, flags=re.MULTILINE)
    return content


def format_solidity_code(sol_code: str, source_file: str = None) -> str:
    """
    Format Solidity code using Prettier with the solidity plugin.
    Assumes `prettier` and `prettier-plugin-solidity` are installed.

    Args:
        sol_code (str): The Solidity code as a string.
        source_file (str): Optional. Original source filename (for better error messages).

    Returns:
        str: The formatted Solidity code.
    """
    with tempfile.NamedTemporaryFile(suffix=".sol", delete=False, mode="w+", encoding="utf-8") as tmp_file:
        tmp_file.write(sol_code)
        tmp_file_path = tmp_file.name

    prettier_path = shutil.which("prettier")
    if prettier_path is None:
        raise FileNotFoundError("❌ 'prettier' non trovato nel PATH. Installa con `npm install -g prettier prettier-plugin-solidity`.")

    try:
        subprocess.run(
            [
                prettier_path,
                "--plugin", "prettier-plugin-solidity",
                "--write",
                tmp_file_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        with open(tmp_file_path, "r", encoding="utf-8") as f:
            formatted_code = f.read()

    except subprocess.CalledProcessError as e:
        msg = f"❌ Error while formatting"
        if source_file:
            msg += f" file: {source_file}"
        msg += f"\n{e.stderr.decode()}"
        print(msg)
        formatted_code = sol_code  # fallback

    finally:
        os.remove(tmp_file_path)

    return formatted_code


def process_files_in_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.sol'):
                file_path = os.path.join(root, file)
                cleaned_content = remove_labels_from_sol(file_path)
                formatted_content = format_solidity_code(cleaned_content, file_path)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(formatted_content)

if __name__ == "__main__":
    dataset_directory = "./smartbugs-curated/dataset_unlabelled"
    process_files_in_directory(dataset_directory)