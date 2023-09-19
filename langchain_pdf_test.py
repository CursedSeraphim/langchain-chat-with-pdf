from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('./pdfs/MÖRK BORG BARE BONES EDITION.pdf')
pages = loader.load_and_split()
print(pages[1])