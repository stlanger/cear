from owlready2 import get_ontology, default_world
import urllib
import gc
import psutil

class ChebiEntity:
    def __init__(self, chebi_id, label, smiles):
        self.chebi_id = chebi_id
        self.label = label
        self.smiles = smiles
        
    def __str__(self):
        return f"{self.label}: {self.chebi_id} ({self.smiles})"

    def __repr__(self):
        return self.__str__()
        

class ChebiLoader:
    chems = {}
    chems_lower = {}
    roles = {}
    roles_lower  ={}
    
    def __init__(self, chebi_filepath):
        print("loading chebi from:", chebi_filepath)
        self.onto = get_ontology(chebi_filepath).load()
        root_chem_id = "<http://purl.obolibrary.org/obo/CHEBI_24431>"
        root_role_id = "<http://purl.obolibrary.org/obo/CHEBI_50906>"

        print()
        print("loading chemicals and their synonyms")
        self.chems, self.chems_lower = self.__subclasses_from_chebi_resursively(root_chem_id)        
        print("loading roles and their synonyms")
        self.roles, self.roles_lower = self.__subclasses_from_chebi_resursively(root_role_id)

        print(f"\nfound {len(self.chems)} chemicals and {len(self.roles)} roles.");        
        print("Memory usage of ChebiLoader:", self.__get_memory_usage(), "MB")

    def __get_memory_usage(self):
        process = psutil.Process()
        mem_usage = process.memory_info().rss  # Get memory usage in bytes
        return mem_usage / (1024 ** 2)  # Convert bytes to megabytes and return

    
    def get_all_chems(self, lowercase=False):
        if lowercase:
            return self.chems_lower
        else:
            return self.chems

    def get_all_roles(self, lowercase=False):
        if lowercase:
            return self.roles_lower
        else:
            return self.roles

    def get_chem(self, chem_id):
        return self.chems_lower.get(chem_id.lower())

    def get_role(self, role_id):
        return self.roles_lower.get(role_id.lower())

    def __subclasses_from_chebi_resursively(self, chebi_id):
        results = list(default_world.sparql("""
        PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
        PREFIX chebi: <http://purl.obolibrary.org/obo/chebi/>
        SELECT DISTINCT ?id ?label ?smiles ?synonym WHERE
        {
            ?id rdfs:label ?label
            ?id chebi:smiles ?smiles
            ?id rdfs:subClassOf* %s .
            { ?id oboInOwl:hasExactSynonym ?synonym. }
            UNION
            { ?id oboInOwl:hasRelatedSynonym ?synonym. }
        }
        """ % (chebi_id)))
        
        entities = {}
        entities_lower = {}
        for result in results:
            label = result[1]
            smiles = result[2]
            synon = result[3]
            chebi_id = result[0]            

            entities[label] = ChebiEntity(chebi_id, label, smiles)
            entities[synon] = ChebiEntity(chebi_id, label, smiles)
            entities_lower[label.lower()] = ChebiEntity(chebi_id, label, smiles)
            entities_lower[synon.lower()] = ChebiEntity(chebi_id, label, smiles)
            
        return entities, entities_lower
        
        