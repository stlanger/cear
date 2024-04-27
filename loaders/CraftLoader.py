import sys
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import spacy

from ChebiLoader import ChebiLoader

class CraftEntity:   
    label = 0 # 0=nothing, 1=chemical, 3=role

    id2label = {
        0: "O",
        1: "B-chemical",
        2: "I-chemical",
        3: "B-role",
        4: "I-role"
    }
    
    label2id = {
        "O": 0,
        "B-chemical": 1,
        "I-chemical": 2,
        "B-role": 3,
        "I-role": 4
    }
    
    def __init__(self, mentionId, start, end, text, label=0):        
        self.start = int(start)
        self.end = int(end)
        self.text = text
        self.mentionId = mentionId 
        self.label = label

    def __str__(self):
        return f"({self.id2label[self.label]} | mentionId: {self.mentionId}) [{self.start}:{self.end}] {self.text}"

    def __repr__(self):
        return self.__str__()


class CraftLoader:
    text = {} # dict of <file_id, fulltext>
    entities = {} # dict of <file_id, entity>
    unknown_entity_mentions = set() # set of <entitytext>
    
    spans = []
    labels = []
    
    def __init__(self, craft_filepath, chebi): 
        print("loading CRAFT from:", craft_filepath)        
        self.chebi = chebi
        self.craft_filepath = craft_filepath
        print("collecting chemical entities...")
        self.__collectEntities() 
        print("collecting text...")
        self.__collectText()
        self.nlp = spacy.load("en_core_web_lg")
        
        print("cutting text into spans and labeling them.")
        for file_id in tqdm(self.text.keys()):
            self.__label_spans(self.text[file_id], self.entities[file_id])

    def __collectEntities(self):
        for file in os.listdir(os.path.join(self.craft_filepath, "CHEBI+extensions")):            
            filepath = os.path.join(self.craft_filepath, "CHEBI+extensions", file)
            if os.path.isfile(filepath):
                self.__loadEntities(filepath)        
   
    def __loadEntities(self, file):
        tree = ET.parse(file)
        root = tree.getroot()
        file_id = int(file[file.rfind("/") + 1: -18])

        chebiNameDict = {}
        for classMention in root.findall(".//classMention"):            
            id = classMention.get("id")
            name = classMention.find("mentionClass").text
            chebiNameDict[id] = name        

        self.entities[file_id] = []
        for annotation in root.findall(".//annotation"):            
            mentionId = annotation.find("mention").get("id")
            text = annotation.find("spannedText").text
            span = annotation.find("span")
            start = span.get("start")
            end = span.get("end")

            chebiName = chebiNameDict[mentionId]            
            if chebiName:
                if self.chebi.get_chem(chebiName) or self.chebi.get_chem(text):
                    self.entities[file_id].append(CraftEntity(mentionId, start, end, text, 1))
                elif self.chebi.get_role(chebiName) or self.chebi.get_role(text):       
                    self.entities[file_id].append(CraftEntity(mentionId, start, end, text, 3))
                else:
                    self.unknown_entity_mentions.add(text)

        self.entities[file_id] = sorted(self.entities[file_id], key=lambda x: x.start)

    def __collectText(self):
        for file in os.listdir(os.path.join(self.craft_filepath, "articles")):
            if file.endswith(".txt"):
                filepath = os.path.join(self.craft_filepath, "articles", file)
                if os.path.isfile(filepath):
                    self.__loadText(filepath)

    def __loadText(self, file):        
        file_id = int(file[file.rfind("/") + 1: -4])
        with open(file) as f:
            self.text[file_id] = f.read()
    
    def __label_spans(self, text, chemicals):
        doc = self.nlp(text)
        chemical_pointer = 0
        for sentence in doc.sents:
            chems = []        
            while chemical_pointer < len(chemicals) and chemicals[chemical_pointer].start < sentence.end_char:            
                chems.append(chemicals[chemical_pointer])
                chemical_pointer += 1
                
            _s, _l = self.__label_sentences(sentence.text, sentence.start_char, chems)
            self.spans.append(_s)
            self.labels.append(_l)

    def __label_sentences(self, text, text_offset, chemicals):
        _spans = []
        _labels = []    
    
        pointer = 0           
        for c in chemicals:
            start = c.start-text_offset
            if pointer < start:
                _spans.append(text[pointer:start])
                _labels.append(0)
            
            _spans.append(c.text)
            _labels.append(c.label)
            
            pointer = c.end - text_offset
            
        if pointer < len(text):
            _spans.append(text[pointer:])
            _labels.append(0)
        
        return _spans, _labels