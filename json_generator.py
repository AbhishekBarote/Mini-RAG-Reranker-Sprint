import os
import json

def generate_sources_json(documents_dir=r'E:\gen ai\minirag\minirag\data', output_file='sources.json'):
    """
    Generates a sources.json file based on the PDF files present in the documents directory.
    """
    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    
    sources = []
    for pdf_file in pdf_files:
        # Remove the .pdf extension for the title
        title = pdf_file[:-4]
        
        # Create a placeholder URL (you might want to update this with real URLs)
        url = f"https://example.com/{pdf_file.replace(' ', '%20')}"
        
        sources.append({
            "title": title,
            "url": url
        })
    
    # Write to sources.json
    with open(output_file, 'w') as f:
        json.dump(sources, f, indent=2)
    
    print(f"Generated {output_file} with {len(sources)} entries.")
    return sources

if __name__ == "__main__":
    generate_sources_json()