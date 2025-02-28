#!/usr/bin/env python3
import os
import shutil

# The API directory where sphinx-apidoc generates the .rst files.
API_DIR = os.path.join(os.path.dirname(__file__), "api")

# Your package prefix to remove (ensure trailing dot)
PACKAGE_PREFIX = "hyperion."

def organize_api_files():
    """Move files from a flat structure into subfolders based on the main module."""
    print(f"Organizing API files in: {API_DIR}")
    print(f"Using package prefix: '{PACKAGE_PREFIX}'")
    for filename in os.listdir(API_DIR):
        # Process only .rst files that start with the package prefix.
        if not filename.endswith(".rst") or not filename.startswith(PACKAGE_PREFIX):
            continue

        print(f"Processing file: {filename}")
        # Remove the package prefix.
        remainder = filename[len(PACKAGE_PREFIX):]  # e.g. "core.flow.transforms.permutation.rst"
        parts = remainder.split(".")
        if len(parts) < 2:
            print(f"Skipping file (unexpected format): {filename}")
            continue

        # Use the first part as the main module folder (e.g. "core")
        main_module = parts[0]
        target_folder = os.path.join(API_DIR, main_module)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"Created folder: {target_folder}")

        # Rebuild the new filename from the remaining parts.
        new_filename = ".".join(parts[1:])
        if not new_filename.endswith(".rst"):
            new_filename += ".rst"

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

def generate_subfolder_index(subfolder_path):
    """Generate an index.rst in a subfolder that lists all .rst files (except its own index)."""
    entries = []
    for fname in sorted(os.listdir(subfolder_path)):
        if fname.endswith(".rst") and fname != "index.rst":
            doc_name = os.path.splitext(fname)[0]
            entries.append(doc_name)
    if entries:
        index_path = os.path.join(subfolder_path, "index.rst")
        # Instead of appending " API", simply use the folder name capitalized.
        folder_title = os.path.basename(subfolder_path).capitalize()
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"{folder_title}\n{'=' * len(folder_title)}\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 2\n\n")
            for entry in entries:
                f.write(f"   {entry}\n")
        print(f"Created index in {subfolder_path}")

def generate_top_level_index():
    """Generate the top-level index.rst in API_DIR that includes each subfolder's index."""
    subfolder_indexes = []
    for entry in sorted(os.listdir(API_DIR)):
        full_path = os.path.join(API_DIR, entry)
        if os.path.isdir(full_path):
            index_file = os.path.join(full_path, "index.rst")
            if os.path.exists(index_file):
                # Reference the subfolder index relative to API_DIR.
                subfolder_indexes.append(f"{entry}/index")
    top_index_path = os.path.join(API_DIR, "index.rst")
    with open(top_index_path, "w", encoding="utf-8") as f:
        title = "API Documentation"
        f.write(f"{title}\n{'=' * len(title)}\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        for sub_index in subfolder_indexes:
            f.write(f"   {sub_index}\n")
    print(f"Created top-level API index at {top_index_path}")

def generate_indexes():
    """Generate indexes for all subfolders and the top-level API index."""
    for entry in os.listdir(API_DIR):
        subfolder_path = os.path.join(API_DIR, entry)
        if os.path.isdir(subfolder_path):
            generate_subfolder_index(subfolder_path)
    generate_top_level_index()

def main():
    organize_api_files()
    replace_package_prefix_in_files()
    generate_indexes()

if __name__ == "__main__":
    main()