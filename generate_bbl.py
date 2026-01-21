
import re

def parse_bib_file(bib_path):
    entries = {}
    current_entry = {}
    current_key = None
    
    with open(bib_path, 'r') as f:
        content = f.read()

    # Regex to find entries like @article{key, ... }
    # This is a simple parser and might fail on complex nested braces
    entry_pattern = re.compile(r'@(\w+)\s*{\s*([^,]+),', re.MULTILINE)
    
    # Split content by @ to handle entries
    raw_entries = content.split('@')
    
    for raw in raw_entries:
        if not raw.strip(): continue
        
        # Extract type and key
        match = re.match(r'(\w+)\s*{\s*([^,]+),', raw)
        if not match: continue
        
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        
        # Extract fields
        fields = {}
        # Simple field extraction: field\s*=\s*{value} or "value"
        # Handling multiline values loosely
        field_iter = re.finditer(r'(\w+)\s*=\s*[{"\'](.*?)["\']\s*(?:,|\})', raw, re.DOTALL)
        
        # Better robust parsing for fields
        # Let's just grab the content inside the outer braces
        # This is tricky without a proper parser. 
        # Let's use a simpler field regex that assumes standard formatting
        
        cleaned_raw = raw.replace('\n', ' ')
        
        title_m = re.search(r'title\s*=\s*{(.*?)}', cleaned_raw, re.IGNORECASE)
        author_m = re.search(r'author\s*=\s*{(.*?)}', cleaned_raw, re.IGNORECASE)
        journal_m = re.search(r'journal\s*=\s*{(.*?)}', cleaned_raw, re.IGNORECASE)
        year_m = re.search(r'year\s*=\s*[{"]?(\d+)[}"]?', cleaned_raw, re.IGNORECASE)
        volume_m = re.search(r'volume\s*=\s*{(.*?)}', cleaned_raw, re.IGNORECASE)
        number_m = re.search(r'number\s*=\s*{(.*?)}', cleaned_raw, re.IGNORECASE)
        pages_m = re.search(r'pages\s*=\s*{(.*?)}', cleaned_raw, re.IGNORECASE)
        
        fields['title'] = title_m.group(1) if title_m else ""
        fields['author'] = author_m.group(1) if author_m else ""
        fields['journal'] = journal_m.group(1) if journal_m else ""
        fields['year'] = year_m.group(1) if year_m else ""
        fields['volume'] = volume_m.group(1) if volume_m else ""
        fields['number'] = number_m.group(1) if number_m else ""
        fields['pages'] = pages_m.group(1) if pages_m else ""
        
        fields['type'] = entry_type
        entries[key] = fields
        
    return entries

def format_author(author_str):
    # Simplistic formatting
    return author_str.replace(' and ', ', ')

def generate_bbl(entries, output_path):
    # Sort keys alphabetically by author
    sorted_keys = sorted(entries.keys(), key=lambda k: entries[k]['author'])
    
    with open(output_path, 'w') as f:
        f.write("\\begin{thebibliography}{1}\n\n")
        
        for key in sorted_keys:
            data = entries[key]
            f.write(f"\\bibitem{{{key}}}\n")
            
            # Author
            if data['author']:
                f.write(f"{data['author']}.\n")
            
            # Title
            if data['title']:
                f.write(f"\\newblock {data['title']}.\n")
            
            # Journal/Book
            if data['journal']:
                f.write(f"\\newblock \\emph{{{data['journal']}}}")
                if data['volume']:
                    f.write(f", {data['volume']}")
                if data['number']:
                    f.write(f"({data['number']})")
                if data['pages']:
                    f.write(f":{data['pages']}")
                f.write(f", {data['year']}.\n")
            elif data['year']:
                 f.write(f"\\newblock {data['year']}.\n")
            
            f.write("\n")
            
        f.write("\\end{thebibliography}\n")

if __name__ == "__main__":
    bib_path = 'paper/references.bib'
    bbl_path = 'paper/main.bbl'
    
    print(f"Reading {bib_path}...")
    entries = parse_bib_file(bib_path)
    print(f"Found {len(entries)} entries.")
    
    print(f"Generating {bbl_path}...")
    generate_bbl(entries, bbl_path)
    print("Done.")
