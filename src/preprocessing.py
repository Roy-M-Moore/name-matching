import pandas as pd
import numpy as np
import regex as re
from unidecode import unidecode

# source = https://www.nordichq.com/guides/list-of-legal-entity-types-by-country-in-europe/
LEGAL_SUFFIXES = {
    "bv", "nv", "sa", "ag", "vof", "cv",
    "stichting", "maatschap",
    "vzw", "sl", "lp", "llp", "ohg", "ev",
    "sarl", "sca", "scs", "gmbh", "llc",
    "aps", "as", "ivs", "ks", "fmba",
    "ltd", "limited", "plc", "inc", "corp", 
    "kg", "kgaa", "oy",
    "srl", "spa", "pte", "ab", "as"
}

def basic_normalize(name: str) -> str:
    
    if not isinstance(name, str):
        return ""
    
    # lowercase
    name = name.lower()
    
    # unicode normalization (müller -> muller)
    name = unidecode(name)
    
    # replace punctuation with space
    name = re.sub(r"[^\w\s]", " ", name)
    
    # collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()
    
    return name

def remove_legal_suffixes(name: str) -> str:
    
    tokens = name.split()
    
    tokens = [
        t for t in tokens
        if t not in LEGAL_SUFFIXES
    ]
    
    return " ".join(tokens)

def remove_single_chars(name: str) -> str:
    
    tokens = name.split()
    
    tokens = [
        t for t in tokens
        if len(t) > 1
    ]
    
    return " ".join(tokens)

def preprocess_company_name(name: str) -> str:
    
    name = basic_normalize(name)
    name = remove_legal_suffixes(name)
    name = remove_single_chars(name)
    # name = sort_tokens(name)
    
    return name
