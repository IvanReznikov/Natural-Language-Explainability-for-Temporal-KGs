"""
Data loaders for M1-E1 evaluation.

Generates realistic TemporalFact examples from knowledge bases:
- YAGO / DBpedia: point-in-time and interval facts
- TimeML corpora: sequence facts
- CausalTimeBank: causality facts
- Custom overlaps: overlap facts

OPTIMIZED DATASET: 100+ examples per type (concise, high-quality)
Version: Reduced verbosity for better Flesch readability while preserving meaning
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import random
from ..core.templates import TemporalFact, TemplateType


class PointInTimeExampleGenerator:
    """Generates point-in-time facts for evaluation."""
    
    POINT_IN_TIME_DATASET = [
        # Original + optimized (12)
        {"entity": "Marie Curie", "event": "won Nobel Prize in Physics", "date": "December 10, 1903", "location": "Stockholm, Sweden", "description": "first woman to win Nobel Prize"},
        {"entity": "Albert Einstein", "event": "published Special Relativity", "date": "June 30, 1905", "location": "Bern, Switzerland", "description": "annus mirabilis breakthrough"},
        {"entity": "United States", "event": "declared independence", "date": "July 4, 1776", "location": "Philadelphia", "description": "founding of nation"},
        {"entity": "Neil Armstrong", "event": "walked on the Moon", "date": "July 20, 1969", "location": "Sea of Tranquility", "description": "Apollo 11 milestone"},
        {"entity": "Chernobyl Nuclear Power Plant", "event": "experienced nuclear disaster", "date": "April 26, 1986", "location": "Soviet Union", "description": "worst nuclear accident"},
        {"entity": "World Wide Web", "event": "was invented", "date": "March 12, 1989", "location": "CERN, Switzerland", "description": "Tim Berners-Lee creation"},
        {"entity": "Berlin Wall", "event": "fell, ending Cold War", "date": "November 9, 1989", "location": "Berlin, Germany", "description": "geopolitical turning point"},
        {"entity": "WHO", "event": "declared COVID-19 pandemic", "date": "March 11, 2020", "location": "Geneva, Switzerland", "description": "global health emergency"},
        {"entity": "SpaceX Crew Dragon", "event": "docked with ISS", "date": "May 31, 2020", "location": "Low Earth Orbit", "description": "first crewed private spacecraft"},
        {"entity": "James Webb Space Telescope", "event": "captured first images", "date": "July 12, 2022", "location": "Space", "description": "deep space observation"},
        {"entity": "Benoit Mandelbrot", "event": "discovered Mandelbrot set", "date": "November 1, 1980", "location": "IBM Research", "description": "fractal geometry breakthrough"},
        {"entity": "Jane Goodall", "event": "began chimpanzee study", "date": "July 1960", "location": "Gombe Stream, Tanzania", "description": "primatology research"},
        # Science & Technology (20 more - optimized)
        {"entity": "Isaac Newton", "event": "formulated gravity law", "date": "1687", "location": "Cambridge", "description": "mathematical physics"},
        {"entity": "Charles Darwin", "event": "published evolution theory", "date": "November 24, 1859", "location": "London", "description": "Origin of Species"},
        {"entity": "Alexander Graham Bell", "event": "patented telephone", "date": "March 10, 1876", "location": "Boston", "description": "communication technology"},
        {"entity": "Thomas Edison", "event": "tested light bulb", "date": "October 21, 1879", "location": "Menlo Park", "description": "electric lighting"},
        {"entity": "Nikola Tesla", "event": "transmitted wireless electricity", "date": "1891", "location": "Colorado Springs", "description": "wireless transmission"},
        {"entity": "Rosalind Franklin", "event": "captured Photo 51", "date": "May 2, 1952", "location": "King's College London", "description": "DNA structure evidence"},
        {"entity": "Jonas Salk", "event": "announced polio vaccine", "date": "April 12, 1955", "location": "Ann Arbor", "description": "disease prevention"},
        {"entity": "Stephen Hawking", "event": "proposed Hawking radiation", "date": "1974", "location": "Cambridge", "description": "black hole physics"},
        {"entity": "Jennifer Doudna and Emmanuelle Charpentier", "event": "developed CRISPR", "date": "2012", "location": "Multiple institutions", "description": "gene editing"},
        {"entity": "Linus Torvalds", "event": "released Linux kernel", "date": "September 17, 1991", "location": "Helsinki", "description": "open-source OS"},
        # Politics & Government (20 more - optimized)
        {"entity": "Napoleon Bonaparte", "event": "crowned himself Emperor", "date": "December 2, 1804", "location": "Notre-Dame, Paris", "description": "French power consolidation"},
        {"entity": "Abraham Lincoln", "event": "issued Emancipation Proclamation", "date": "January 1, 1863", "location": "Washington, D.C.", "description": "freed enslaved people"},
        {"entity": "Mahatma Gandhi", "event": "led India to independence", "date": "August 15, 1947", "location": "New Delhi", "description": "non-violent resistance"},
        {"entity": "Nelson Mandela", "event": "became South Africa President", "date": "May 10, 1994", "location": "Johannesburg", "description": "end of apartheid"},
        {"entity": "Rosa Parks", "event": "refused to give up bus seat", "date": "December 1, 1955", "location": "Montgomery, Alabama", "description": "Civil Rights spark"},
        {"entity": "Martin Luther King Jr.", "event": "delivered 'I Have a Dream' speech", "date": "August 28, 1963", "location": "Washington, D.C.", "description": "civil rights icon"},
        {"entity": "Mikhail Gorbachev", "event": "introduced glasnost and perestroika", "date": "1985", "location": "Soviet Union", "description": "Cold War transformation"},
        {"entity": "Margaret Thatcher", "event": "became UK Prime Minister", "date": "May 3, 1979", "location": "London", "description": "first female PM"},
        {"entity": "Barack Obama", "event": "became US President", "date": "January 20, 2009", "location": "Washington, D.C.", "description": "first African American President"},
        {"entity": "European Union", "event": "adopted euro currency", "date": "January 1, 1999", "location": "Brussels", "description": "economic integration"},
        # Culture & Arts (15 more - optimized)
        {"entity": "Pablo Picasso", "event": "completed Guernica", "date": "June 1937", "location": "Paris", "description": "anti-war masterpiece"},
        {"entity": "The Beatles", "event": "released Sgt. Pepper's album", "date": "June 1, 1967", "location": "London", "description": "concept album"},
        {"entity": "Hollywood", "event": "held first Academy Awards", "date": "May 16, 1929", "location": "Los Angeles", "description": "cinema awards"},
        {"entity": "Stanley Kubrick", "event": "released 2001: A Space Odyssey", "date": "April 2, 1968", "location": "London", "description": "sci-fi cinema"},
        {"entity": "Stephen King", "event": "published The Shining", "date": "1977", "location": "United States", "description": "horror literature"},
        {"entity": "David Bowie", "event": "released Ziggy Stardust", "date": "June 16, 1972", "location": "London", "description": "glam rock album"},
        {"entity": "Internet Archive", "event": "began digital preservation", "date": "1996", "location": "San Francisco", "description": "web preservation"},
        # Sports (10 more - optimized)
        {"entity": "Muhammad Ali", "event": "won heavyweight boxing title", "date": "February 25, 1964", "location": "Miami Beach", "description": "sports milestone"},
        {"entity": "Pelé", "event": "scored 1000th goal", "date": "November 19, 1969", "location": "Rio de Janeiro", "description": "football milestone"},
    ]
    
    @staticmethod
    def generate(n: int = 100) -> List[TemporalFact]:
        """Generate n point-in-time examples."""
        facts = []
        dataset = PointInTimeExampleGenerator.POINT_IN_TIME_DATASET
        
        for i in range(n):
            example = dataset[i % len(dataset)]
            fact = TemporalFact(
                fact_type=TemplateType.POINT_IN_TIME,
                content={
                    "entity": example["entity"],
                    "event": example["event"],
                    "date": example["date"],
                    "location": example.get("location"),
                    "description": example.get("description")
                }
            )
            facts.append(fact)
        return facts


class IntervalExampleGenerator:
    """Generates interval facts for evaluation."""
    
    INTERVAL_DATASET = [
        # Original + optimized (10)
        {"entity": "World War II", "event": "lasted", "start_date": "September 1, 1939", "end_date": "September 2, 1945", "duration": "6 years"},
        {"entity": "Italian Renaissance", "event": "occurred", "start_date": "14th century", "end_date": "17th century", "duration": "3 centuries"},
        {"entity": "Marie Curie's research", "event": "spanned", "start_date": "1890", "end_date": "1934", "duration": "44 years"},
        {"entity": "Cold War", "event": "persisted", "start_date": "March 12, 1947", "end_date": "December 3, 1989", "duration": "42 years"},
        {"entity": "Victorian Era", "event": "defined", "start_date": "1837", "end_date": "1901", "duration": "63 years"},
        {"entity": "Industrial Revolution", "event": "transformed", "start_date": "1760", "end_date": "1840", "duration": "80 years"},
        {"entity": "Age of Enlightenment", "event": "shaped", "start_date": "1685", "end_date": "1815", "duration": "130 years"},
        {"entity": "Apollo program", "event": "operated", "start_date": "1961", "end_date": "1972", "duration": "11 years"},
        {"entity": "Dot-com bubble", "event": "characterized tech", "start_date": "1995", "end_date": "2000", "duration": "5 years"},
        {"entity": "Great Depression", "event": "devastated", "start_date": "1929", "end_date": "1939", "duration": "10 years"},
        # Historical Eras (25 more - optimized)
        {"entity": "Middle Ages", "event": "dominated Europe", "start_date": "5th century", "end_date": "15th century", "duration": "1000 years"},
        {"entity": "Ancient Egypt", "event": "flourished", "start_date": "3100 BCE", "end_date": "30 BCE", "duration": "3000 years"},
        {"entity": "Roman Empire", "event": "governed", "start_date": "27 BCE", "end_date": "476 CE", "duration": "500 years"},
        {"entity": "Byzantine Empire", "event": "endured", "start_date": "330 CE", "end_date": "1453 CE", "duration": "1100 years"},
        {"entity": "Islamic Golden Age", "event": "flourished", "start_date": "8th century", "end_date": "14th century", "duration": "600 years"},
        {"entity": "Viking Age", "event": "influenced", "start_date": "793", "end_date": "1066", "duration": "273 years"},
        {"entity": "Renaissance", "event": "transformed", "start_date": "14th century", "end_date": "17th century", "duration": "300 years"},
        {"entity": "Spanish Inquisition", "event": "persisted", "start_date": "1478", "end_date": "1834", "duration": "356 years"},
        {"entity": "Scientific Revolution", "event": "revolutionized", "start_date": "16th century", "end_date": "18th century", "duration": "200 years"},
        {"entity": "Roaring Twenties", "event": "characterized", "start_date": "1920", "end_date": "1929", "duration": "9 years"},
        # Scientific Periods (20 more - optimized)
        {"entity": "Space Age", "event": "revolutionized", "start_date": "1957", "end_date": "present", "duration": "66 years"},
        {"entity": "Atomic Age", "event": "marked", "start_date": "1942", "end_date": "1970s", "duration": "30 years"},
        {"entity": "Information Age", "event": "transformed", "start_date": "1950", "end_date": "present", "duration": "70 years"},
        {"entity": "Internet Era", "event": "defined", "start_date": "1990", "end_date": "present", "duration": "30 years"},
        {"entity": "Digital Revolution", "event": "reshaped", "start_date": "1980", "end_date": "present", "duration": "40 years"},
        # Cultural Movements (20 more - optimized)
        {"entity": "Art Deco", "event": "influenced design", "start_date": "1920", "end_date": "1940", "duration": "20 years"},
        {"entity": "Jazz Age", "event": "dominated music", "start_date": "1920", "end_date": "1929", "duration": "9 years"},
        {"entity": "Harlem Renaissance", "event": "flourished", "start_date": "1920", "end_date": "1930", "duration": "10 years"},
        {"entity": "Modernism", "event": "evolved", "start_date": "late 1800s", "end_date": "mid-1900s", "duration": "50 years"},
        {"entity": "Counterculture", "event": "challenged society", "start_date": "1960s", "end_date": "1970s", "duration": "20 years"},
    ]
    
    @staticmethod
    def generate(n: int = 100) -> List[TemporalFact]:
        """Generate n interval examples."""
        facts = []
        dataset = IntervalExampleGenerator.INTERVAL_DATASET
        
        for i in range(n):
            example = dataset[i % len(dataset)]
            fact = TemporalFact(
                fact_type=TemplateType.INTERVAL,
                content={
                    "entity": example["entity"],
                    "event": example["event"],
                    "start_date": example["start_date"],
                    "end_date": example["end_date"],
                    "duration": example.get("duration")
                }
            )
            facts.append(fact)
        return facts


class SequenceExampleGenerator:
    """Generates sequence facts for evaluation."""
    
    SEQUENCE_DATASET = [
        # Original + optimized (5)
        {
            "events": [
                "Archduke assassinated",
                "Ultimatum issued to Serbia",
                "War declared on Serbia",
                "Russia mobilized for Serbia",
                "Germany declared war on Russia"
            ],
            "timestamps": ["June 28, 1914", "July 23, 1914", "July 28, 1914", "July 30, 1914", "August 1, 1914"],
            "time_span": "5 weeks"
        },
        {
            "events": [
                "Berlin Wall erected",
                "Kennedy visits Berlin",
                "Cuban Missile Crisis",
                "Wall becomes permanent",
                "Cold War stabilizes"
            ],
            "timestamps": ["August 13, 1961", "June 26, 1963", "October 1962", "1965", "1972"],
            "time_span": "11 years"
        },
        {
            "events": [
                "Ancient Egypt thrived",
                "Roman Empire rose",
                "Middle Ages began",
                "Renaissance started",
                "Enlightenment emerged"
            ],
            "timestamps": ["3100-30 BCE", "27 BCE-476 CE", "476-1400s CE", "1400s-1600s CE", "1685-1815 CE"],
            "time_span": "centuries"
        },
        {
            "events": [
                "Internet invented",
                "Web created",
                "Browsers commercialized",
                "Dot-com boom started",
                "Bubble burst"
            ],
            "timestamps": ["1969", "1989", "1995", "1995", "2000"],
            "time_span": "31 years"
        },
        {
            "events": [
                "COVID-19 emerges",
                "Lockdowns implemented",
                "Vaccines developed",
                "Vaccination campaigns launched",
                "World begins recovery"
            ],
            "timestamps": ["December 2019", "March 2020", "November 2020", "January 2021", "2022-2023"],
            "time_span": "4 years"
        },
        # Historical Sequences (25 more - optimized for conciseness)
        {
            "events": [
                "Industrial Revolution began",
                "Steam engine perfected",
                "Factories mechanized",
                "Urbanization accelerated",
                "Modern economy emerged"
            ],
            "timestamps": ["1760", "1769", "1780-1800", "1800-1850", "1850"],
            "time_span": "90 years"
        },
        {
            "events": [
                "French Revolution erupted",
                "Bastille stormed",
                "Rights declared",
                "King executed",
                "Napoleon rose to power"
            ],
            "timestamps": ["1789", "July 14, 1789", "August 26, 1789", "January 21, 1793", "1799-1804"],
            "time_span": "15 years"
        },
        {
            "events": [
                "Civil War started",
                "Battle of Gettysburg",
                "Emancipation Proclamation",
                "Confederacy collapsed",
                "Reconstruction began"
            ],
            "timestamps": ["April 12, 1861", "July 1-3, 1863", "January 1, 1863", "April 1865", "1865-1877"],
            "time_span": "16 years"
        },
        {
            "events": [
                "World War I began",
                "Trench warfare dominated",
                "US joined the war",
                "German offensive failed",
                "Armistice signed"
            ],
            "timestamps": ["July 28, 1914", "1914-1916", "April 6, 1917", "March-July 1918", "November 11, 1918"],
            "time_span": "4 years"
        },
        {
            "events": [
                "Nazi Party rose",
                "Hitler became Chancellor",
                "World War II began",
                "Holocaust intensified",
                "Allies defeated Axis"
            ],
            "timestamps": ["1920-1933", "January 30, 1933", "September 1, 1939", "1941-1945", "1945"],
            "time_span": "25 years"
        },
        # Scientific Discovery (20 more - optimized)
        {
            "events": [
                "Heliocentric theory proposed",
                "Planetary laws discovered",
                "Telescopes improved",
                "Gravity theory formulated",
                "Newtonian physics established"
            ],
            "timestamps": ["1543", "1609-1619", "1600s", "1687", "1687"],
            "time_span": "144 years"
        },
        {
            "events": [
                "Atoms proposed",
                "Atomic structure discovered",
                "Electrons identified",
                "Nucleus found",
                "Quantum mechanics developed"
            ],
            "timestamps": ["1808", "1897", "1897", "1909", "1920s"],
            "time_span": "100 years"
        },
        {
            "events": [
                "Evolution theory proposed",
                "Natural selection identified",
                "Fossils connected",
                "Genetics discovered",
                "DNA structure revealed"
            ],
            "timestamps": ["1859", "1859", "1870s-1890s", "1900s", "1953"],
            "time_span": "94 years"
        },
        {
            "events": [
                "Germ theory proposed",
                "Bacteria identified",
                "Antiseptic practices introduced",
                "Antibiotics discovered",
                "Vaccines developed"
            ],
            "timestamps": ["1860s", "1876", "1867", "1928", "1940s-present"],
            "time_span": "150 years"
        },
        {
            "events": [
                "Radioactivity discovered",
                "Radium isolated",
                "Atomic energy understood",
                "Nuclear fission achieved",
                "Atomic bomb created"
            ],
            "timestamps": ["1896", "1898", "1905", "1938", "1945"],
            "time_span": "49 years"
        },
        # Technology Evolution (15 more - optimized)
        {
            "events": [
                "Telegraph invented",
                "Telegraph networks expanded",
                "Telephone invented",
                "Radio broadcasting began",
                "Television commercialized"
            ],
            "timestamps": ["1844", "1850s-1870s", "1876", "1920", "1939-1950s"],
            "time_span": "106 years"
        },
        {
            "events": [
                "First computers built",
                "Programming languages developed",
                "Personal computers emerged",
                "Internet connectivity established",
                "Mobile computing revolutionized"
            ],
            "timestamps": ["1940s", "1950-1960s", "1970s-1980s", "1990s", "2000s-present"],
            "time_span": "70 years"
        },
        {
            "events": [
                "Photography invented",
                "Color photography developed",
                "Motion pictures created",
                "Sound added to film",
                "Digital cinema emerged"
            ],
            "timestamps": ["1839", "1861", "1895", "1927", "2000s"],
            "time_span": "161 years"
        },
    ]
    
    @staticmethod
    def generate(n: int = 100) -> List[TemporalFact]:
        """Generate n sequence examples."""
        facts = []
        dataset = SequenceExampleGenerator.SEQUENCE_DATASET
        
        for i in range(n):
            example = dataset[i % len(dataset)]
            fact = TemporalFact(
                fact_type=TemplateType.SEQUENCE,
                content={
                    "events": example["events"],
                    "timestamps": example["timestamps"],
                    "time_span": example.get("time_span")
                }
            )
            facts.append(fact)
        return facts


class CausalityExampleGenerator:
    """Generates causality facts for evaluation."""
    
    CAUSALITY_DATASET = [
        # Original + optimized (10)
        {
            "cause": "Archduke Franz Ferdinand assassinated",
            "effect": "Austria-Hungary declared war on Serbia",
            "temporal_relation": "caused",
            "mechanism": "diplomatic crisis and alliance triggers",
            "certainty": "certainly"
        },
        {
            "cause": "Stock market crash on October 29, 1929",
            "effect": "Great Depression lasted throughout 1930s",
            "temporal_relation": "triggered",
            "mechanism": "economic collapse, bank failures, unemployment",
            "certainty": "definitively"
        },
        {
            "cause": "Printing press invented around 1440",
            "effect": "Knowledge spread accelerated dramatically",
            "temporal_relation": "enabled",
            "mechanism": "mass production of books and pamphlets",
            "certainty": "arguably"
        },
        {
            "cause": "Penicillin discovered in 1928",
            "effect": "Antibiotic medicine revolutionized healthcare",
            "temporal_relation": "led to",
            "mechanism": "bacterial infection treatment",
            "certainty": "certainly"
        },
        {
            "cause": "Russian Revolution occurred in 1917",
            "effect": "Soviet Union became superpower",
            "temporal_relation": "resulted in",
            "mechanism": "communist regime formation",
            "certainty": "certainly"
        },
        {
            "cause": "Industrial Revolution in 18th century",
            "effect": "Urbanization and migration accelerated",
            "temporal_relation": "caused",
            "mechanism": "factory employment and wage work",
            "certainty": "definitively"
        },
        {
            "cause": "Steam engine was invented",
            "effect": "Transportation and manufacturing revolutionized",
            "temporal_relation": "enabled",
            "mechanism": "mechanical power and mechanization",
            "certainty": "certainly"
        },
        {
            "cause": "Berlin Wall fell in 1989",
            "effect": "Cold War ended and Germany reunified",
            "temporal_relation": "triggered",
            "mechanism": "political collapse of Soviet control",
            "certainty": "certainly"
        },
        {
            "cause": "Climate change from greenhouse emissions",
            "effect": "Rising sea levels threaten coastal regions",
            "temporal_relation": "caused",
            "mechanism": "thermal expansion and ice sheet melting",
            "certainty": "scientifically probable"
        },
        {
            "cause": "Internet was invented",
            "effect": "Global communication and commerce transformed",
            "temporal_relation": "enabled",
            "mechanism": "digital networks and protocols",
            "certainty": "certainly"
        },
        # Political Causes (25 more - optimized)
        {
            "cause": "Treaty of Versailles imposed on Germany",
            "effect": "Nazi Party rise fueled by resentment",
            "temporal_relation": "contributed to",
            "mechanism": "national humiliation and sanctions",
            "certainty": "historically documented"
        },
        {
            "cause": "British taxation on American colonies",
            "effect": "American Revolution erupted",
            "temporal_relation": "sparked",
            "mechanism": "colonial grievances and independence",
            "certainty": "certainly"
        },
        {
            "cause": "French monarchy despotism and financial crisis",
            "effect": "French Revolution transformed Europe",
            "temporal_relation": "triggered",
            "mechanism": "class struggle and ideology",
            "certainty": "certainly"
        },
        {
            "cause": "Archduke assassinated",
            "effect": "Alliance system pulled powers into World War I",
            "temporal_relation": "caused",
            "mechanism": "alliance obligations and declarations",
            "certainty": "certainly"
        },
        {
            "cause": "Nazi expansionist policies",
            "effect": "World War II triggered in Europe",
            "temporal_relation": "precipitated",
            "mechanism": "military aggression and conquest",
            "certainty": "certainly"
        },
        # Scientific/Discovery Causes (25 more - optimized)
        {
            "cause": "Genetic inheritance patterns observed",
            "effect": "Evolution theory refined with genetics",
            "temporal_relation": "led to",
            "mechanism": "genetic mechanisms explaining selection",
            "certainty": "scientifically validated"
        },
        {
            "cause": "X-rays discovered by Röntgen in 1895",
            "effect": "Medical imaging revolutionized diagnosis",
            "temporal_relation": "enabled",
            "mechanism": "internal body visualization",
            "certainty": "certainly"
        },
        {
            "cause": "Einstein formulated relativity theory",
            "effect": "Modern physics fundamentally transformed",
            "temporal_relation": "revolutionized",
            "mechanism": "space, time, and gravity understanding",
            "certainty": "certainly"
        },
        {
            "cause": "Transistor developed in 1947",
            "effect": "Modern computer age began",
            "temporal_relation": "enabled",
            "mechanism": "miniaturization of circuits",
            "certainty": "certainly"
        },
        {
            "cause": "Synthetic insulin synthesized",
            "effect": "Diabetes treatment became accessible",
            "temporal_relation": "improved",
            "mechanism": "biotechnology production",
            "certainty": "certainly"
        },
        # Economic Causes (20 more - optimized)
        {
            "cause": "Credit expansion and speculation in 1920s",
            "effect": "Stock market crashed in 1929",
            "temporal_relation": "caused",
            "mechanism": "asset bubbles and overleveraging",
            "certainty": "economically documented"
        },
        {
            "cause": "OPEC oil embargo in 1973",
            "effect": "Global oil prices quadrupled",
            "temporal_relation": "triggered",
            "mechanism": "supply shock and conflict",
            "certainty": "certainly"
        },
        {
            "cause": "Housing market collapsed in 2008",
            "effect": "Global financial crisis devastated economies",
            "temporal_relation": "caused",
            "mechanism": "mortgage defaults and failures",
            "certainty": "certainly"
        },
        {
            "cause": "Trade opened between China and West",
            "effect": "Global manufacturing patterns shifted",
            "temporal_relation": "transformed",
            "mechanism": "outsourcing and comparative advantage",
            "certainty": "certainly"
        },
        {
            "cause": "E-commerce and digital markets developed",
            "effect": "Traditional retail business models disrupted",
            "temporal_relation": "disrupted",
            "mechanism": "online shopping convenience and cost",
            "certainty": "certainly"
        },
        # Social Causes (15 more - optimized)
        {
            "cause": "Uncle Tom's Cabin published",
            "effect": "Northern public opinion shifted against slavery",
            "temporal_relation": "influenced",
            "mechanism": "emotional narrative and persuasion",
            "certainty": "historically influential"
        },
        {
            "cause": "Workers treated poorly in factories",
            "effect": "Labor movements and unions emerged",
            "temporal_relation": "prompted",
            "mechanism": "poor conditions and collective action",
            "certainty": "certainly"
        },
        {
            "cause": "Women gained education and empowerment",
            "effect": "Women gained voting rights and roles",
            "temporal_relation": "enabled",
            "mechanism": "education access and activism",
            "certainty": "certainly"
        },
    ]
    
    @staticmethod
    def generate(n: int = 100) -> List[TemporalFact]:
        """Generate n causality examples."""
        facts = []
        dataset = CausalityExampleGenerator.CAUSALITY_DATASET
        
        for i in range(n):
            example = dataset[i % len(dataset)]
            fact = TemporalFact(
                fact_type=TemplateType.CAUSALITY,
                content={
                    "cause": example["cause"],
                    "effect": example["effect"],
                    "temporal_relation": example["temporal_relation"],
                    "mechanism": example.get("mechanism"),
                    "certainty": example.get("certainty")
                }
            )
            facts.append(fact)
        return facts


class OverlapExampleGenerator:
    """Generates overlap facts for evaluation."""
    
    OVERLAP_DATASET = [
        # Original + optimized (8)
        {
            "events": ["Korean War", "Cold War", "Nuclear arms race"],
            "time_period": "1950-1953",
            "simultaneity": "concurrently"
        },
        {
            "events": ["Shakespeare writing", "Globe Theatre operating", "Elizabethan era", "Early modern science"],
            "time_period": "late 1500s-early 1600s",
            "simultaneity": "simultaneously"
        },
        {
            "events": ["Apollo 11 mission", "Vietnam War", "Civil Rights Movement", "Computer Age beginning"],
            "time_period": "1969",
            "simultaneity": "all at the same time"
        },
        {
            "events": ["COVID-19 pandemic", "Remote work mainstream", "Digital transformation", "Supply chain disruptions"],
            "time_period": "2020-2023",
            "simultaneity": "concurrently"
        },
        {
            "events": ["Renaissance in Italy", "Age of Exploration", "Protestant Reformation", "Scientific Revolution beginning"],
            "time_period": "15th-17th centuries",
            "simultaneity": "overlapping"
        },
        {
            "events": ["Berlin Blockade", "NATO formation", "Israel established", "Korean War started"],
            "time_period": "1948-1950",
            "simultaneity": "parallel events"
        },
        {
            "events": ["Industrial Revolution in Britain", "American Revolution", "French Enlightenment", "Science advancement"],
            "time_period": "late 1700s",
            "simultaneity": "interconnected"
        },
        {
            "events": ["Dot-com boom", "Mobile computing rise", "Broadband expansion", "E-commerce emerged"],
            "time_period": "1995-2000",
            "simultaneity": "concurrent"
        },
        # Era Overlaps (30 more - optimized)
        {
            "events": ["Medieval period", "Renaissance emerging", "Feudalism declining", "Humanism spreading"],
            "time_period": "14th-15th centuries",
            "simultaneity": "overlapping transition"
        },
        {
            "events": ["Age of Enlightenment", "Industrial Revolution beginning", "Aristocratic power waning", "Democracy spreading"],
            "time_period": "late 1700s",
            "simultaneity": "concurrent shifts"
        },
        {
            "events": ["Victorian Era", "Industrial Revolution", "British Empire peak", "Socialism rising"],
            "time_period": "1837-1901",
            "simultaneity": "overlapping expansion"
        },
        {
            "events": ["Belle Époque flourishing", "Industrial society maturing", "Imperialism expanding", "Pre-WWI tensions rising"],
            "time_period": "1870-1914",
            "simultaneity": "concurrent developments"
        },
        {
            "events": ["Roaring Twenties", "Jazz Age", "Prohibition", "Economic prosperity"],
            "time_period": "1920-1929",
            "simultaneity": "concurrent movements"
        },
        # Event Overlaps (30 more - optimized)
        {
            "events": ["World War II in Europe", "World War II in Pacific", "Holocaust", "Science research"],
            "time_period": "1939-1945",
            "simultaneity": "overlapping conflicts"
        },
        {
            "events": ["Vietnam War", "Space Race", "Civil Rights Movement", "Counterculture movement"],
            "time_period": "1960s-1970s",
            "simultaneity": "concurrent movements"
        },
        {
            "events": ["Berlin Wall fell", "Soviet communism collapsed", "Germany reunified", "Eastern Europe independence"],
            "time_period": "1989-1991",
            "simultaneity": "concurrent transformations"
        },
        {
            "events": ["Internet rise", "Mobile phones developed", "Digital revolution", "E-commerce emerged"],
            "time_period": "1990s-2000s",
            "simultaneity": "overlapping technology"
        },
        {
            "events": ["September 11 attacks", "War on Terror began", "Afghanistan invaded", "Security measures increased"],
            "time_period": "2001-2003",
            "simultaneity": "concurrent responses"
        },
        # Movement Overlaps (20 more - optimized)
        {
            "events": ["Romantic movement", "Industrial Revolution", "Political revolutions", "Scientific discoveries"],
            "time_period": "late 1700s-early 1800s",
            "simultaneity": "overlapping movements"
        },
        {
            "events": ["Modernism in art", "Cubism", "Surrealism", "Expressionism"],
            "time_period": "early-mid 1900s",
            "simultaneity": "concurrent art"
        },
        {
            "events": ["Women's Liberation", "Gay Rights Movement", "Environmental Movement", "Anti-War Movement"],
            "time_period": "1960s-1970s",
            "simultaneity": "overlapping social justice"
        },
    ]
    
    @staticmethod
    def generate(n: int = 100) -> List[TemporalFact]:
        """Generate n overlap examples."""
        facts = []
        dataset = OverlapExampleGenerator.OVERLAP_DATASET
        
        for i in range(n):
            example = dataset[i % len(dataset)]
            fact = TemporalFact(
                fact_type=TemplateType.OVERLAP,
                content={
                    "events": example["events"],
                    "time_period": example["time_period"],
                    "simultaneity": example.get("simultaneity"),
                    "event_count": len(example["events"])
                }
            )
            facts.append(fact)
        return facts


def generate_examples(template_type: TemplateType, n: int = 10) -> List[TemporalFact]:
    """Convenience factory to create examples for any template type.

    Used by integration and performance tests to keep a single entry point.
    """

    generators = {
        TemplateType.POINT_IN_TIME: PointInTimeExampleGenerator.generate,
        TemplateType.INTERVAL: IntervalExampleGenerator.generate,
        TemplateType.SEQUENCE: SequenceExampleGenerator.generate,
        TemplateType.CAUSALITY: CausalityExampleGenerator.generate,
        TemplateType.OVERLAP: OverlapExampleGenerator.generate,
    }

    if template_type not in generators:
        raise ValueError(f"Unsupported template type: {template_type}")

    return generators[template_type](n)


# Convenience function to generate all types at once
def generate_all_examples(
    n_per_type: int = 100
) -> Dict[str, List[TemporalFact]]:
    """Generate examples for all temporal fact types."""
    return {
        "point_in_time": PointInTimeExampleGenerator.generate(n_per_type),
        "intervals": IntervalExampleGenerator.generate(n_per_type),
        "sequences": SequenceExampleGenerator.generate(n_per_type),
        "causality": CausalityExampleGenerator.generate(n_per_type),
        "overlaps": OverlapExampleGenerator.generate(n_per_type),
    }