import os
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_lg")

from ChebiLoader import ChebiLoader

class BC5CDRLoader:
    spans = []
    labels = []
    min_role_char_length = -1
    
    def __init__(self, bc5cdr_filepath, chebi, min_role_char_length=4):
        print("loading BC5CDR dataset from:", bc5cdr_filepath)
        print("loading Chebi to add roles which can be lexically found in the text snippets...")
        self.bc5cdr_filepath = bc5cdr_filepath        
        self.min_role_char_length = min_role_char_length
        print(f"using {self.min_role_char_length} as a minimum character length for roles to mark them in the text")
        self.chebi = chebi
        self.nlp = spacy.load("en_core_web_lg")
        print("adding special tokenizer rules for chemical roles in Chebi")
        self.__add_special_tokenizer_rules()
        print("collecting entities")
        self.__collect_entities()             

    def __collect_entities(self):
        filelist = []
        text_data = []
        for root, dirs, files in os.walk(os.path.join(self.bc5cdr_filepath)):
            for file in files:
                f = os.path.join(root, file)
                filelist.append(f)

        print(f"BC5CDR has {len(filelist)} files to parse")

        for f in tqdm(filelist):
            self.__extract_data(f)
        print(f"loaded {len(self.spans)} spans and the according labels.")
        

    # Function to extract text from an element and its children
    def __extract_text(self, element):
        text = element.text or ""
        for child in element:
            text += self.__extract_text(child)
        return text
    
    
    def __extract_data(self, file):        
        tree = ET.parse(file)
        root = tree.getroot()    
        # Iterate through passage elements and extract text, including annotations
        for passage in root.findall(".//passage"):
            text_element = passage.find("text")
            if text_element is not None:
                text = self.__extract_text(text_element)            
                text_offset = int(self.__extract_text(passage.find("offset")))            
                
                chemicals = []
                
                # Check for chemical annotations
                for annotation in passage.findall(".//annotation"):
                    infon = annotation.find("infon[@key='type']")
                    if infon is not None and infon.text == "Chemical":
                        annotation_text_element = annotation.find("text")   
                        chemical = ""
                        if annotation_text_element is not None:
                            chemical = annotation_text_element.text
                        annotation_offset_element = annotation.find("location")
                        if annotation_offset_element is not None:
                            start = int(annotation_offset_element.get("offset"))
                            length = int(annotation_offset_element.get("length"))
                            chemicals.append((start, length, chemical))
                
                self.__split_sentences(text, text_offset, chemicals)                

    def __split_sentences(self, text, text_offset, chemicals):        
        doc = nlp(text)
        for sent in doc.sents:            
            lower_boundary = text_offset + sent.start_char
            upper_boundary = lower_boundary + len(sent.text)
            filtered_chemicals = [chemical for chemical in chemicals if lower_boundary <= chemical[0] <= upper_boundary]            
            self.__label_spans(sent.text, lower_boundary, filtered_chemicals)       

    def __add_special_tokenizer_rules(self):        
        for key in tqdm(self.chebi.roles.keys()):
            if len(key) >= self.min_role_char_length: # ignore roles which have less than 3 characters --> Too random in text                       
                upper = key[0].upper() + key[1:]
                lower = key.lower()
                self.nlp.tokenizer.add_special_case(key, [{spacy.symbols.ORTH: key}]) # add normal role
                self.nlp.tokenizer.add_special_case(upper, [{spacy.symbols.ORTH: upper}]) # add role as if used in the beginning of a sentence
                self.nlp.tokenizer.add_special_case(lower, [{spacy.symbols.ORTH: lower}]) # add role all lowercase
    
    def __label_spans(self, text, text_offset, chemicals):        
        _spans = []
        _labels = []    
        
        pointer = 0
        chemicals = sorted(chemicals, key=lambda x: x[0])
           
        for c in chemicals:
            start = c[0] - text_offset                
            if pointer < start:
                _spans.append(text[pointer:start])
                _labels.append(0)
            
            _spans.append(c[-1])
            _labels.append(1)
            
            pointer = start + c[1]
            
        if pointer < len(text):
            _spans.append(text[pointer:])
            _labels.append(0)
        self.__add_roles_to_spans_and_labels(_spans, _labels)

    def __add_roles_to_spans_and_labels(self, spans, labels):
        _spans = []
        _labels = []
        for s, l in zip(spans, labels):
            doc = self.nlp(s)
            start_index = 0
            for token in doc:
                # if token is a role and token is not already labeled as part of a chemical, add label as 3
                if l==0 and len(token.text) >= self.min_role_char_length and self.chebi.get_role(token.text):
                    _spans.append(s[start_index:token.idx])
                    _labels.append(l)
                    _spans.append(token.text)
                    _labels.append(3)
                    start_index = token.idx + len(token)
            _spans.append(s[start_index:])
            _labels.append(l)
        self.spans.append(_spans)
        self.labels.append(_labels)
        
                    
    
        