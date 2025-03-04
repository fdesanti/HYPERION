#!/usr/bin/env python3
import os
import shutil

# The API directory where sphinx-apidoc generates the .rst files.
API_DIR = os.path.join(os.path.dirname(__file__), "api")

# Your package prefix to remove (ensure trailing dot)
PACKAGE_PREFIX = "hyperion."

def organize_api_files():
    """Move files from a flat structure into nested subfolders based on the full module path."""
    print(f"Organizing API files in: {API_DIR}")
    print(f"Using package prefix: '{PACKAGE_PREFIX}'")
    for filename in os.listdir(API_DIR):
        # Process only .rst files that start with the package prefix.
        if not filename.endswith(".rst") or not filename.startswith(PACKAGE_PREFIX):
            continue

        print(f"Processing file: {filename}")
        # Remove the package prefix and file extension.
        base_name = filename[len(PACKAGE_PREFIX):-4]  # e.g. "core.flow.transforms.permutation"
        parts = base_name.split(".")
        if not parts:
            print(f"Skipping file (unexpected format): {filename}")
            continue

        # Build the nested folder path from all parts except the last.
        if len(parts) > 1:
            target_folder = os.path.join(API_DIR, *parts[:-1])
        else:
            target_folder = API_DIR

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"Created folder: {target_folder}")

        # The new file name is the last part plus the extension.
        new_filename = parts[-1] + ".rst"
        src_path = os.path.join(API_DIR, filename)
        dst_path = os.path.join(target_folder, new_filename)
        print(f"Moving '{src_path}' to '{dst_path}'")
        shutil.move(src_path, dst_path)

def replace_package_prefix_in_files():
    """Replace package prefix in titles but skip lines with automodule directives."""
    for root, dirs, files in os.walk(API_DIR):
        for file in files:
            if file.endswith(".rst"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    # Skip replacing in automodule directives
                    if line.strip().startswith(".. automodule::"):
                        new_lines.append(line)
                    else:
                        new_lines.append(line.replace(PACKAGE_PREFIX, ""))
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

def generate_indexes_for_directory(dir_path):
    """
    Recursively generate an index.rst in dir_path that lists:
      - All subdirectories (referencing their own index.rst)
      - All .rst files in dir_path (except index.rst itself)
    """
    entries = []
    subdirectories = []
    for entry in sorted(os.listdir(dir_path)):
        full_entry = os.path.join(dir_path, entry)
        if os.path.isdir(full_entry):
            subdirectories.append(entry)
        elif entry.endswith(".rst") and entry != "index.rst":
            doc_name = os.path.splitext(entry)[0]
            entries.append(doc_name)

    # Create an index if there are any files or subdirectories
    if entries or subdirectories:
        index_path = os.path.join(dir_path, "index.rst")
        folder_title = os.path.basename(dir_path) if os.path.basename(dir_path) else "API Documentation"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"{folder_title}\n{'=' * len(folder_title)}\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 2\n\n")
            # List subdirectories first
            for sub in subdirectories:
                f.write(f"   {sub}/index\n")
            # Then list any individual .rst files
            for entry in entries:
                f.write(f"   {entry}\n")
        print(f"Created index in {dir_path}")

    # Recurse into subdirectories.
    for sub in subdirectories:
        generate_indexes_for_directory(os.path.join(dir_path, sub))

def generate_indexes():
    """Generate indexes for all directories under API_DIR recursively."""
    generate_indexes_for_directory(API_DIR)

def main():
    organize_api_files()
    replace_package_prefix_in_files()
    generate_indexes()

if __name__ == "__main__":
    main()