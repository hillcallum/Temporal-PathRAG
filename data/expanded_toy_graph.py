import networkx as nx
from typing import Dict, Any

class ExpandedToyGraphBuilder:
    """
    Creates a larger KG for more compregensive PathRAG testing
    
    Graph structure:
    - ~50+ entities: Nobel Laureates, Inventors, Philosophers, Scientific Fields, Discoveries, Institutions, Countries, Historical Events
    - Up to 5-6 hops needed for complex questions.
    - Different relational types and there are multiple paths necessary for doing the path ranking evaluation
    - Includes temporal and location attributes
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.create_expanded_data()
    
    def create_expanded_data(self):
        # Nodes: id, type, name, description, and additional attributes like year and city
        # Nodes generated using Gemini 2.5 Flash
        nodes = [
            # Scientists/Inventors/Philosophers
            {'id': 'albert_einstein', 'entity_type': 'Person', 'name': 'Albert Einstein', 'description': 'Theoretical physicist known for relativity.', 'born_year': 1879, 'died_year': 1955},
            {'id': 'marie_curie', 'entity_type': 'Person', 'name': 'Marie Curie', 'description': 'Pioneer in radioactivity, first woman to win a Nobel Prize.', 'born_year': 1867, 'died_year': 1934},
            {'id': 'pierre_curie', 'entity_type': 'Person', 'name': 'Pierre Curie', 'description': 'French physicist, co-discoverer of radium and polonium.', 'born_year': 1859, 'died_year': 1906},
            {'id': 'niels_bohr', 'entity_type': 'Person', 'name': 'Niels Bohr', 'description': 'Danish physicist, foundational contributions to atomic structure and quantum theory.', 'born_year': 1885, 'died_year': 1962},
            {'id': 'max_planck', 'entity_type': 'Person', 'name': 'Max Planck', 'description': 'German theoretical physicist, originator of quantum theory.', 'born_year': 1858, 'died_year': 1947},
            {'id': 'isaac_newton', 'entity_type': 'Person', 'name': 'Isaac Newton', 'description': 'English mathematician and physicist, key figure in scientific revolution.', 'born_year': 1642, 'died_year': 1727},
            {'id': 'galileo_galilei', 'entity_type': 'Person', 'name': 'Galileo Galilei', 'description': 'Italian astronomer, physicist, engineer, father of observational astronomy.', 'born_year': 1564, 'died_year': 1642},
            {'id': 'ada_lovelace', 'entity_type': 'Person', 'name': 'Ada Lovelace', 'description': 'English mathematician, considered the first computer programmer.', 'born_year': 1815, 'died_year': 1852},
            {'id': 'alan_turing', 'entity_type': 'Person', 'name': 'Alan Turing', 'description': 'British mathematician and computer scientist, father of AI.', 'born_year': 1912, 'died_year': 1954},
            {'id': 'aristotle', 'entity_type': 'Person', 'name': 'Aristotle', 'description': 'Ancient Greek philosopher and polymath, student of Plato.', 'born_year': -384, 'died_year': -322}, # BCE dates
            {'id': 'socrates', 'entity_type': 'Person', 'name': 'Socrates', 'description': 'Classical Greek philosopher, one of the founders of Western philosophy.', 'born_year': -470, 'died_year': -399},
            {'id': 'marie_tharp', 'entity_type': 'Person', 'name': 'Marie Tharp', 'description': 'American geological cartographer who co-created the first map of the Atlantic Ocean floor.', 'born_year': 1920, 'died_year': 2006},
            {'id': 'charles_darwin', 'entity_type': 'Person', 'name': 'Charles Darwin', 'description': 'English naturalist, best known for his contributions to the science of evolution.', 'born_year': 1809, 'died_year': 1882},
            {'id': 'rosalind_franklin', 'entity_type': 'Person', 'name': 'Rosalind Franklin', 'description': 'British chemist and X-ray crystallographer whose work was central to the understanding of the molecular structures of DNA and RNA.', 'born_year': 1920, 'died_year': 1958},

            # Institutions
            {'id': 'princeton_university', 'entity_type': 'Institution', 'name': 'Princeton University', 'description': 'Ivy League research university in New Jersey, USA.'},
            {'id': 'university_of_paris', 'entity_type': 'Institution', 'name': 'University of Paris', 'description': 'Historic French university, key center of learning in Europe.'},
            {'id': 'university_of_copenhagen', 'entity_type': 'Institution', 'name': 'University of Copenhagen', 'description': 'Oldest university and research institution in Denmark.'},
            {'id': 'cambridge_university', 'entity_type': 'Institution', 'name': 'University of Cambridge', 'description': 'Collegiate public research university in Cambridge, UK.'},
            {'id': 'gottingen_university', 'entity_type': 'Institution', 'name': 'University of Göttingen', 'description': 'Public research university in Göttingen, Germany.'},
            {'id': 'royal_society', 'entity_type': 'Institution', 'name': 'Royal Society', 'description': 'A fellowship of many of the world\'s most eminent scientists, UK.', 'founded_year': 1660},
            {'id': 'bell_labs', 'entity_type': 'Institution', 'name': 'Bell Labs', 'description': 'American industrial research and scientific development company.', 'founded_year': 1925},

            # Countries/Cities
            {'id': 'germany', 'entity_type': 'Country', 'name': 'Germany', 'description': 'Country in Central Europe.'},
            {'id': 'france', 'entity_type': 'Country', 'name': 'France', 'description': 'Country in Western Europe.'},
            {'id': 'poland', 'entity_type': 'Country', 'name': 'Poland', 'description': 'Country in Central Europe.'},
            {'id': 'denmark', 'entity_type': 'Country', 'name': 'Denmark', 'description': 'Nordic country in Northern Europe.'},
            {'id': 'usa', 'entity_type': 'Country', 'name': 'United States', 'description': 'Country in North America.'},
            {'id': 'uk', 'entity_type': 'Country', 'name': 'United Kingdom', 'description': 'Island country in Europe.'},
            {'id': 'italy', 'entity_type': 'Country', 'name': 'Italy', 'description': 'Country in Southern Europe.'},
            {'id': 'greece', 'entity_type': 'Country', 'name': 'Greece', 'description': 'Country in Southeast Europe.'},
            {'id': 'london', 'entity_type': 'City', 'name': 'London', 'description': 'Capital and largest city of England and the United Kingdom.'},
            {'id': 'paris', 'entity_type': 'City', 'name': 'Paris', 'description': 'Capital and most populous city of France.'},
            {'id': 'princeton_city', 'entity_type': 'City', 'name': 'Princeton', 'description': 'Town in Mercer County, New Jersey, United States.'},
            {'id': 'athens', 'entity_type': 'City', 'name': 'Athens', 'description': 'Capital and largest city of Greece.'},
            {'id': 'ulm', 'entity_type': 'City', 'name': 'Ulm', 'description': 'City in the German state of Baden-Württemberg.'},

            # Awards
            {'id': 'nobel_prize_physics', 'entity_type': 'Award', 'name': 'Nobel Prize in Physics', 'description': 'Awarded annually for outstanding contributions in physics.'},
            {'id': 'nobel_prize_chemistry', 'entity_type': 'Award', 'name': 'Nobel Prize in Chemistry', 'description': 'Awarded annually for outstanding contributions in chemistry.'},
            {'id': 'nobel_prize_peace', 'entity_type': 'Award', 'name': 'Nobel Peace Prize', 'description': 'Awarded to the person who has done the most or best work for fraternity between nations.'},
            {'id': 'fields_medal', 'entity_type': 'Award', 'name': 'Fields Medal', 'description': 'Awarded to two, three, or four mathematicians under 40 years of age.'},
            {'id': 'copley_medal', 'entity_type': 'Award', 'name': 'Copley Medal', 'description': 'A scientific award given by the Royal Society, UK.'},

            # Scientific Concepts/Discoveries/Theories
            {'id': 'relativity_theory', 'entity_type': 'Concept', 'name': 'Theory of Relativity', 'description': 'Developed by Einstein, consisting of special and general relativity.'},
            {'id': 'quantum_mechanics', 'entity_type': 'Concept', 'name': 'Quantum Mechanics', 'description': 'Fundamental theory in physics describing nature at the smallest scales.'},
            {'id': 'radioactivity', 'entity_type': 'Discovery', 'name': 'Radioactivity', 'description': 'Process by which an unstable atomic nucleus loses energy by radiation.'},
            {'id': 'photoelectric_effect', 'entity_type': 'Concept', 'name': 'Photoelectric Effect', 'description': 'Emission of electrons when light shines on a material.'},
            {'id': 'atomic_model', 'entity_type': 'Concept', 'name': 'Bohr Model', 'description': 'Model of the atom proposed by Niels Bohr.'},
            {'id': 'gravity_law', 'entity_type': 'Concept', 'name': 'Law of Universal Gravitation', 'description': 'Newton\'s law describing the gravitational force between bodies.'},
            {'id': 'calculus', 'entity_type': 'Concept', 'name': 'Calculus', 'description': 'Mathematical study of continuous change.'},
            {'id': 'analytical_engine', 'entity_type': 'Invention', 'name': 'Analytical Engine', 'description': 'Proposed mechanical general-purpose computer by Charles Babbage.'},
            {'id': 'ai', 'entity_type': 'Field', 'name': 'Artificial Intelligence', 'description': 'Intelligence demonstrated by machines.'},
            {'id': 'evolution_theory', 'entity_type': 'Concept', 'name': 'Theory of Evolution by Natural Selection', 'description': 'Darwin\'s theory explaining biological evolution.'},
            {'id': 'dna_structure', 'entity_type': 'Discovery', 'name': 'DNA Structure', 'description': 'Double helix structure of deoxyribonucleic acid.'},
            {'id': 'plate_tectonics', 'entity_type': 'Theory', 'name': 'Plate Tectonics Theory', 'description': 'Scientific theory describing the large-scale motion of Earth\'s lithosphere.'},
            {'id': 'mid_atlantic_ridge', 'entity_type': 'Geographical Feature', 'name': 'Mid-Atlantic Ridge', 'description': 'Mid-ocean ridge, a divergent plate boundary.'},

            # Publications
            {'id': 'principia_mathematica', 'entity_type': 'Publication', 'name': 'Philosophiæ Naturalis Principia Mathematica', 'description': 'Newton\'s seminal work on classical mechanics.', 'year': 1687},
            {'id': 'origin_of_species', 'entity_type': 'Publication', 'name': 'On the Origin of Species', 'description': 'Darwin\'s foundational work in evolutionary biology.', 'year': 1859},
            {'id': 'unified_field_theory_paper', 'entity_type': 'Publication', 'name': 'Unified Field Theory Paper', 'description': 'Einstein\'s attempt to unify fundamental forces.', 'year': 1929},
            {'id': 'relativity_paper', 'entity_type': 'Publication', 'name': 'On the Electrodynamics of Moving Bodies', 'description': 'Einstein\'s 1905 paper introducing special relativity.', 'year': 1905},

            # Historical Events
            {'id': 'ww2', 'entity_type': 'Historical_Event', 'name': 'World War II', 'description': 'Global war that lasted from 1939 to 1945.', 'start_year': 1939, 'end_year': 1945},
            {'id': 'manhattan_project', 'entity_type': 'Historical_Event', 'name': 'Manhattan Project', 'description': 'Research and development undertaking during WWII that produced the first nuclear weapons.', 'start_year': 1942, 'end_year': 1946},
            {'id': 'cold_war', 'entity_type': 'Historical_Event', 'name': 'Cold War', 'description': 'Geopolitical rivalry between the US and Soviet Union and their respective allies, from the mid-1940s until 1991.', 'start_year': 1947, 'end_year': 1991},
        ]

        for node in nodes:
            self.graph.add_node(node['id'], **node)

        # Edges with varied weights and temporal context
        # Edges generated using Gemini 2.5 Flash
        edges = [
            # Birth/Origin
            {'source': 'albert_einstein', 'target': 'germany', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Einstein was born in Germany.', 'weight': 0.9},
            {'source': 'albert_einstein', 'target': 'ulm', 'relation_type': 'BORN_IN_CITY', 'description': 'Einstein was born in Ulm.', 'weight': 0.92},
            {'source': 'marie_curie', 'target': 'poland', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Marie Curie was born in Poland.', 'weight': 0.9},
            {'source': 'pierre_curie', 'target': 'france', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Pierre Curie was born in France.', 'weight': 0.9},
            {'source': 'niels_bohr', 'target': 'denmark', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Bohr was born in Denmark.', 'weight': 0.9},
            {'source': 'max_planck', 'target': 'germany', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Planck was born in Germany.', 'weight': 0.9},
            {'source': 'isaac_newton', 'target': 'uk', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Newton was born in the UK.', 'weight': 0.9},
            {'source': 'galileo_galilei', 'target': 'italy', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Galileo was born in Italy.', 'weight': 0.9},
            {'source': 'ada_lovelace', 'target': 'uk', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Lovelace was born in the UK.', 'weight': 0.9},
            {'source': 'alan_turing', 'target': 'uk', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Turing was born in the UK.', 'weight': 0.9},
            {'source': 'marie_tharp', 'target': 'usa', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Tharp was born in the USA.', 'weight': 0.9},
            {'source': 'charles_darwin', 'target': 'uk', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Darwin was born in the UK.', 'weight': 0.9},
            {'source': 'rosalind_franklin', 'target': 'uk', 'relation_type': 'BORN_IN_COUNTRY', 'description': 'Franklin was born in the UK.', 'weight': 0.9},

            # Institutional Affiliations/Work
            {'source': 'albert_einstein', 'target': 'princeton_university', 'relation_type': 'WORKED_AT', 'description': 'Einstein worked at Princeton University.', 'weight': 0.8, 'start_year': 1933, 'end_year': 1955},
            {'source': 'marie_curie', 'target': 'university_of_paris', 'relation_type': 'WORKED_AT', 'description': 'Marie Curie worked at the University of Paris.', 'weight': 0.8, 'start_year': 1906, 'end_year': 1934}, #Became professor in 1906
            {'source': 'pierre_curie', 'target': 'university_of_paris', 'relation_type': 'WORKED_AT', 'description': 'Pierre Curie worked at the University of Paris.', 'weight': 0.8, 'start_year': 1900, 'end_year': 1906},
            {'source': 'niels_bohr', 'target': 'university_of_copenhagen', 'relation_type': 'WORKED_AT', 'description': 'Bohr worked at the University of Copenhagen.', 'weight': 0.8, 'start_year': 1916, 'end_year': 1962},
            {'source': 'max_planck', 'target': 'gottingen_university', 'relation_type': 'STUDIED_AT', 'description': 'Planck studied at University of Göttingen.', 'weight': 0.75, 'year': 1874},
            {'source': 'isaac_newton', 'target': 'cambridge_university', 'relation_type': 'WORKED_AT', 'description': 'Newton was a professor at Cambridge University.', 'weight': 0.8, 'start_year': 1667, 'end_year': 1702},
            {'source': 'alan_turing', 'target': 'cambridge_university', 'relation_type': 'STUDIED_AT', 'description': 'Turing studied at Cambridge University.', 'weight': 0.75, 'start_year': 1931, 'end_year': 1934},
            {'source': 'alan_turing', 'target': 'bell_labs', 'relation_type': 'VISITED', 'description': 'Turing visited Bell Labs.', 'weight': 0.6, 'year': 1942},
            {'source': 'rosalind_franklin', 'target': 'cambridge_university', 'relation_type': 'STUDIED_AT', 'description': 'Franklin studied at Cambridge University.', 'weight': 0.75, 'start_year': 1938, 'end_year': 1941},
            {'source': 'rosalind_franklin', 'target': 'university_of_paris', 'relation_type': 'WORKED_AT', 'description': 'Franklin worked at University of Paris.', 'weight': 0.8, 'start_year': 1947, 'end_year': 1950}, # Lab des services chimiques de l'État

            # Location of Institutions
            {'source': 'princeton_university', 'target': 'usa', 'relation_type': 'LOCATED_IN_COUNTRY', 'description': 'Princeton University is in the USA.', 'weight': 1.0},
            {'source': 'princeton_university', 'target': 'princeton_city', 'relation_type': 'LOCATED_IN_CITY', 'description': 'Princeton University is in Princeton city.', 'weight': 1.0},
            {'source': 'university_of_paris', 'target': 'france', 'relation_type': 'LOCATED_IN_COUNTRY', 'description': 'University of Paris is in France.', 'weight': 1.0},
            {'source': 'university_of_paris', 'target': 'paris', 'relation_type': 'LOCATED_IN_CITY', 'description': 'University of Paris is in Paris.', 'weight': 1.0},
            {'source': 'university_of_copenhagen', 'target': 'denmark', 'relation_type': 'LOCATED_IN_COUNTRY', 'description': 'University of Copenhagen is in Denmark.', 'weight': 1.0},
            {'source': 'cambridge_university', 'target': 'uk', 'relation_type': 'LOCATED_IN_COUNTRY', 'description': 'Cambridge University is in the UK.', 'weight': 1.0},
            {'source': 'gottingen_university', 'target': 'germany', 'relation_type': 'LOCATED_IN_COUNTRY', 'description': 'University of Göttingen is in Germany.', 'weight': 1.0},
            {'source': 'royal_society', 'target': 'uk', 'relation_type': 'LOCATED_IN_COUNTRY', 'description': 'The Royal Society is in the UK.', 'weight': 1.0},
            {'source': 'royal_society', 'target': 'london', 'relation_type': 'LOCATED_IN_CITY', 'description': 'The Royal Society is in London.', 'weight': 1.0},
            {'source': 'bell_labs', 'target': 'usa', 'relation_type': 'LOCATED_IN_COUNTRY', 'description': 'Bell Labs is in the USA.', 'weight': 1.0},

            # Awards
            {'source': 'albert_einstein', 'target': 'nobel_prize_physics', 'relation_type': 'AWARDED', 'description': 'Einstein received Nobel Prize in Physics.', 'weight': 0.95, 'year': 1921},
            {'source': 'marie_curie', 'target': 'nobel_prize_physics', 'relation_type': 'AWARDED', 'description': 'Marie Curie received Nobel Prize in Physics.', 'weight': 0.95, 'year': 1903},
            {'source': 'marie_curie', 'target': 'nobel_prize_chemistry', 'relation_type': 'AWARDED', 'description': 'Marie Curie received Nobel Prize in Chemistry.', 'weight': 0.95, 'year': 1911},
            {'source': 'pierre_curie', 'target': 'nobel_prize_physics', 'relation_type': 'AWARDED', 'description': 'Pierre Curie received Nobel Prize in Physics.', 'weight': 0.95, 'year': 1903},
            {'source': 'niels_bohr', 'target': 'nobel_prize_physics', 'relation_type': 'AWARDED', 'description': 'Bohr received Nobel Prize in Physics.', 'weight': 0.95, 'year': 1922},
            {'source': 'max_planck', 'target': 'nobel_prize_physics', 'relation_type': 'AWARDED', 'description': 'Planck received Nobel Prize in Physics.', 'weight': 0.95, 'year': 1918},
            {'source': 'isaac_newton', 'target': 'royal_society', 'relation_type': 'CHAIRED', 'description': 'Newton chaired the Royal Society.', 'weight': 0.7, 'start_year': 1703, 'end_year': 1727},
            {'source': 'isaac_newton', 'target': 'copley_medal', 'relation_type': 'AWARDED', 'description': 'Newton received the Copley Medal.', 'weight': 0.9, 'year': 1707},
            {'source': 'charles_darwin', 'target': 'copley_medal', 'relation_type': 'AWARDED', 'description': 'Darwin received the Copley Medal.', 'weight': 0.9, 'year': 1864},

            # Scientific Contributions/Discoveries/Developments
            {'source': 'albert_einstein', 'target': 'relativity_theory', 'relation_type': 'DEVELOPED', 'description': 'Einstein developed the theory of relativity.', 'weight': 0.9},
            {'source': 'albert_einstein', 'target': 'photoelectric_effect', 'relation_type': 'EXPLAINED', 'description': 'Einstein explained the photoelectric effect.', 'weight': 0.9},
            {'source': 'marie_curie', 'target': 'radioactivity', 'relation_type': 'PIONEERED', 'description': 'Marie Curie pioneered research into radioactivity.', 'weight': 0.9},
            {'source': 'pierre_curie', 'target': 'radioactivity', 'relation_type': 'RESEARCHED', 'description': 'Pierre Curie researched radioactivity with Marie.', 'weight': 0.85},
            {'source': 'niels_bohr', 'target': 'atomic_model', 'relation_type': 'PROPOSED', 'description': 'Niels Bohr proposed the atomic model.', 'weight': 0.9},
            {'source': 'max_planck', 'target': 'quantum_mechanics', 'relation_type': 'FOUNDED', 'description': 'Max Planck founded quantum mechanics.', 'weight': 0.9},
            {'source': 'isaac_newton', 'target': 'gravity_law', 'relation_type': 'FORMULATED', 'description': 'Newton formulated the law of universal gravitation.', 'weight': 0.9},
            {'source': 'isaac_newton', 'target': 'calculus', 'relation_type': 'INVENTED', 'description': 'Newton invented calculus.', 'weight': 0.9},
            {'source': 'galileo_galilei', 'target': 'gravity_law', 'relation_type': 'CONTRIBUTED_TO', 'description': 'Galileo contributed to understanding gravity.', 'weight': 0.7},
            {'source': 'ada_lovelace', 'target': 'analytical_engine', 'relation_type': 'WROTE_PROGRAMS_FOR', 'description': 'Ada Lovelace wrote programs for the Analytical Engine.', 'weight': 0.9},
            {'source': 'alan_turing', 'target': 'ai', 'relation_type': 'CONSIDERED_FATHER_OF', 'description': 'Alan Turing is considered the father of AI.', 'weight': 0.9},
            {'source': 'marie_tharp', 'target': 'mid_atlantic_ridge', 'relation_type': 'MAPPED', 'description': 'Marie Tharp mapped the Mid-Atlantic Ridge.', 'weight': 0.9},
            {'source': 'marie_tharp', 'target': 'plate_tectonics', 'relation_type': 'SUPPORTED_THEORY', 'description': 'Marie Tharp\'s work supported the theory of plate tectonics.', 'weight': 0.85},
            {'source': 'charles_darwin', 'target': 'evolution_theory', 'relation_type': 'PROPOSED', 'description': 'Charles Darwin proposed the theory of evolution by natural selection.', 'weight': 0.9},
            {'source': 'rosalind_franklin', 'target': 'dna_structure', 'relation_type': 'CONTRIBUTED_TO_DISCOVERY', 'description': 'Rosalind Franklin\'s work was crucial to DNA structure discovery.', 'weight': 0.9},

            # Publications
            {'source': 'isaac_newton', 'target': 'principia_mathematica', 'relation_type': 'AUTHORED', 'description': 'Newton authored Principia Mathematica.', 'weight': 0.9},
            {'source': 'principia_mathematica', 'target': 'gravity_law', 'relation_type': 'DESCRIBES', 'description': 'Principia Mathematica describes the law of gravity.', 'weight': 0.85},
            {'source': 'albert_einstein', 'target': 'relativity_paper', 'relation_type': 'AUTHORED', 'description': 'Einstein authored the special relativity paper.', 'weight': 0.9},
            {'source': 'relativity_paper', 'target': 'relativity_theory', 'relation_type': 'INTRODUCED', 'description': 'The relativity paper introduced the theory of relativity.', 'weight': 0.85},
            {'source': 'albert_einstein', 'target': 'unified_field_theory_paper', 'relation_type': 'AUTHORED', 'description': 'Einstein authored unified field theory paper.', 'weight': 0.7}, # Less successful attempt
            {'source': 'charles_darwin', 'target': 'origin_of_species', 'relation_type': 'AUTHORED', 'description': 'Darwin authored On the Origin of Species.', 'weight': 0.9},
            {'source': 'origin_of_species', 'target': 'evolution_theory', 'relation_type': 'PRESENTS', 'description': 'On the Origin of Species presents the theory of evolution.', 'weight': 0.85},

            # Collaborations/Relationships/Influence
            {'source': 'marie_curie', 'target': 'pierre_curie', 'relation_type': 'COLLABORATED_WITH', 'description': 'Marie Curie collaborated with Pierre Curie.', 'weight': 0.95},
            {'source': 'marie_curie', 'target': 'pierre_curie', 'relation_type': 'MARRIED_TO', 'description': 'Marie Curie was married to Pierre Curie.', 'weight': 0.95}, # Different type of relationship
            {'source': 'albert_einstein', 'target': 'max_planck', 'relation_type': 'INFLUENCED_BY', 'description': 'Einstein was influenced by Planck\'s quantum theory.', 'weight': 0.7},
            {'source': 'max_planck', 'target': 'albert_einstein', 'relation_type': 'INFLUENCED', 'description': 'Planck influenced Einstein.', 'weight': 0.75}, # Bidirectional influence
            {'source': 'niels_bohr', 'target': 'albert_einstein', 'relation_type': 'DEBATED_WITH', 'description': 'Bohr debated with Einstein on quantum mechanics.', 'weight': 0.6},
            {'source': 'socrates', 'target': 'aristotle', 'relation_type': 'INFLUENCED', 'description': 'Socrates influenced Aristotle through Plato.', 'weight': 0.7}, # Indirect influence
            {'source': 'aristotle', 'target': 'philosophy', 'relation_type': 'CONTRIBUTED_TO_FIELD', 'description': 'Aristotle contributed significantly to philosophy.', 'weight': 0.8},
            {'source': 'albert_einstein', 'target': 'niels_bohr', 'relation_type': 'COLLEAGUE', 'description': 'Einstein and Bohr were colleagues.', 'weight': 0.65},

            # Concepts Relationships
            {'source': 'photoelectric_effect', 'target': 'quantum_mechanics', 'relation_type': 'EVIDENCE_FOR', 'description': 'Photoelectric effect provided evidence for quantum mechanics.', 'weight': 0.8},
            {'source': 'atomic_model', 'target': 'quantum_mechanics', 'relation_type': 'PART_OF_DEVELOPMENT_OF', 'description': 'Bohr model was part of the development of quantum mechanics.', 'weight': 0.75},
            {'source': 'relativity_theory', 'target': 'quantum_mechanics', 'relation_type': 'CONTRASTS_WITH', 'description': 'Relativity theory contrasts with quantum mechanics at certain scales.', 'weight': 0.6},
            {'source': 'dna_structure', 'target': 'radioactivity', 'relation_type': 'STUDIED_USING', 'description': 'DNA structure was studied using X-ray crystallography, which involves radiation.', 'weight': 0.6}, # A bit of a stretch, for multi-hop
            {'source': 'ai', 'target': 'alan_turing', 'relation_type': 'ROOTED_IN_WORK_OF', 'description': 'AI is rooted in the work of Alan Turing.', 'weight': 0.9},

            # Historical Event connections
            {'source': 'albert_einstein', 'target': 'manhattan_project', 'relation_type': 'ADVISED_ON', 'description': 'Einstein advised on the Manhattan Project (letter to FDR).', 'weight': 0.7, 'year': 1939},
            {'source': 'manhattan_project', 'target': 'ww2', 'relation_type': 'OCCURRED_DURING', 'description': 'Manhattan Project occurred during WWII.', 'weight': 0.95},
            {'source': 'alan_turing', 'target': 'ww2', 'relation_type': 'CONTRIBUTED_TO_EFFORT_IN', 'description': 'Alan Turing contributed to Allied efforts in WWII (codebreaking).', 'weight': 0.9, 'year': 1940},
            {'source': 'germany', 'target': 'ww2', 'relation_type': 'PARTICIPANT_IN', 'description': 'Germany was a major participant in WWII.', 'weight': 0.9},
            {'source': 'usa', 'target': 'ww2', 'relation_type': 'PARTICIPANT_IN', 'description': 'USA was a major participant in WWII.', 'weight': 0.9},
            {'source': 'uk', 'target': 'ww2', 'relation_type': 'PARTICIPANT_IN', 'description': 'UK was a major participant in WWII.', 'weight': 0.9},
            {'source': 'radioactivity', 'target': 'manhattan_project', 'relation_type': 'APPLIED_IN', 'description': 'Radioactivity principles were applied in the Manhattan Project.', 'weight': 0.85},
            {'source': 'cold_war', 'target': 'manhattan_project', 'relation_type': 'PRECEDED_BY', 'description': 'The Manhattan Project preceded the Cold War.', 'weight': 0.8}, # Temporal relationship
        ]
        
        # Add edges to graph
        for edge in edges:
            self.graph.add_edge(
                edge['source'], 
                edge['target'],
                relation_type=edge['relation_type'],
                description=edge['description'],
                weight=edge['weight']
                **{k: v for k, v in edge.items() if k not in ['source', 'target', 'relation_type', 'description', 'weight']} 
            )
    
    def get_graph(self) -> nx.DiGraph:
        """Return the expanded toy graph"""
        return self.graph
    
    def print_graph_info(self):
        """Print information about the expanded graph"""
        print("EXPANDED TOY GRAPH INFORMATION")
        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")
        print()
        
        # Group nodes by type
        node_types = {}
        for node_id, data in self.graph.nodes(data=True):
            entity_type = data['entity_type']
            if entity_type not in node_types:
                node_types[entity_type] = []
            node_types[entity_type].append((node_id, data['name']))
        
        print("NODES BY TYPE:")
        for entity_type, nodes in node_types.items():
            print(f"  {entity_type} ({len(nodes)}):")
            for node_id, name in nodes:
                print(f"    {node_id}: {name}")
        print()
        
        # Group edges by relation type
        relation_types = {}
        for source, target, data in self.graph.edges(data=True):
            rel_type = data['relation_type']
            if rel_type not in relation_types:
                relation_types[rel_type] = []
            relation_types[rel_type].append((source, target, data['weight']))
        
        print("EDGES BY RELATION TYPE:")
        for rel_type, edges in relation_types.items():
            print(f"  {rel_type} ({len(edges)}):")
            for source, target, weight in edges[:3]:  # Show first 3 examples
                print(f"    {source} --> {target} (weight: {weight})")
            if len(edges) > 3:
                print(f"    ... and {len(edges) - 3} more")
        print()
    
    def get_sample_queries(self) -> Dict[str, Dict[str, Any]]:
        """Return complex sample queries for testing PathRAG"""
        return {
            "query_1": {
                "description": "Which country was Marie Curie born in?",
                "source": "marie_curie",
                "target": "poland",
                "expected_hops": 1,
                "difficulty": "easy"
            },
            "query_2": {
                "description": "What scientific concept did Einstein develop?",
                "source": "albert_einstein",
                "target": "relativity_theory",
                "expected_hops": 1,
                "difficulty": "easy"
            },
            "query_3": {
                "description": "Which country is Princeton University located in?",
                "source": "princeton_university",
                "target": "usa",
                "expected_hops": 1,
                "difficulty": "easy"
            },
            "query_4": {
                "description": "What award did both Einstein and Marie Curie receive?",
                "source": "albert_einstein",
                "target": "marie_curie",
                "via": "nobel_prize_physics",
                "expected_hops": 2,
                "difficulty": "medium"
            },
            "query_5": {
                "description": "Which country did Einstein work in later in his career?",
                "source": "albert_einstein",
                "target": "usa",
                "via": "princeton_university",
                "expected_hops": 2,
                "difficulty": "medium"
            },
            "query_6": {
                "description": "What concept connects Heisenberg and Planck through their work?",
                "source": "werner_heisenberg",
                "target": "max_planck",
                "via": "quantum_mechanics",
                "expected_hops": 2,
                "difficulty": "medium"
            },
            "query_7": {
                "description": "Find a path from Marie Curie to quantum mechanics through collaborations",
                "source": "marie_curie",
                "target": "quantum_mechanics",
                "possible_via": ["pierre_curie", "university_of_paris"],
                "expected_hops": 3,
                "difficulty": "hard"
            },
            "query_8": {
                "description": "Connect Einstein to Bohr's atomic theory through their shared concepts",
                "source": "albert_einstein",
                "target": "atomic_structure",
                "possible_via": ["quantum_mechanics", "max_planck"],
                "expected_hops": 3,
                "difficulty": "hard"
            },
            "query_9": {
                "description": "Find connection between Marie Curie and Denmark through academic networks",
                "source": "marie_curie",
                "target": "denmark",
                "possible_via": ["nobel_prize_physics", "niels_bohr"],
                "expected_hops": 3,
                "difficulty": "hard"
            },
            "query_10": {
                "description": "Complex path: Connect Schrödinger to France through scientific collaborations",
                "source": "erwin_schrodinger",
                "target": "france",
                "possible_via": ["quantum_mechanics", "marie_curie", "university_of_paris"],
                "expected_hops": 4,
                "difficulty": "very_hard"
            }
        }

def create_expanded_toy_graph() -> nx.DiGraph:
    """Create and return the expanded toy graph"""
    builder = ExpandedToyGraphBuilder()
    return builder.get_graph()

if __name__ == "__main__":
    builder = ExpandedToyGraphBuilder()
    builder.print_graph_info()
    
    print("SAMPLE QUERIES:")
    queries = builder.get_sample_queries()
    for query_id, query_data in queries.items():
        print(f"{query_id}: {query_data['description']} (Difficulty: {query_data['difficulty']})")
    print()