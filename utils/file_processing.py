
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages])
    elif file.type.endswith("wordprocessingml.document"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return file.getvalue().decode()

# # PII Redaction
#def anonymize(text):
#     return re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[REDACTED]", text)