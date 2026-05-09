f = r'c:\Users\Arwa\OneDrive\Desktop\MP\greenAI\major-project\greenai-dashboard\src\pages\Playground.jsx'
with open(f, 'r', encoding='utf-8') as fh:
    c = fh.read()

# Fix all remaining garbled multi-byte sequences by replacing char-by-char
# These are UTF-8 bytes that got interpreted as Latin-1 code points
replacements = [
    ('\u00e2\u0094\u0080\u00e2\u0094\u0080 RESPONSE SECTION \u00e2\u0094\u0080\u00e2\u0094\u0080', 'RESPONSE SECTION'),
    ('\u00e2\u0094\u0080', '-'),
    ('\u00e2\u0080\u0094', '--'),  # em dash
    ('\u00e2\u0080\u0099', "'"),   # right single quote
    ('\u00e2\u0080\u009c', '"'),   # left double quote
    ('\u00e2\u0080\u009d', '"'),   # right double quote
    ('\u00c2\u00b7', '\u00b7'),    # middle dot double-encoding
    ('\u00c2', ''),                 # stray prefix byte
]
for bad, good in replacements:
    c = c.replace(bad, good)

with open(f, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(c)
print('Done')
