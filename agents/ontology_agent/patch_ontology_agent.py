"""
Patch utility for the ontology agent.
This script patches the OntologyAgent class with additional methods
for querying the ontology and handling messages.
"""

import os
import sys
import inspect
import asyncio

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import necessary modules
import logging
logger = logging.getLogger(__name__)

# Import the ontology agent
from agents.ontology_agent.ontology_agent import OntologyAgent
from agents.ontology_agent.ontology_query_interface import OntologyQueryInterface

# Import process_message from process_message.py
# This is the modular approach to add functionality
from agents.ontology_agent.process_message import process_message as external_process_message

# Import necessary types
from typing import Dict, Any, List

# Add process_message method to OntologyAgent
OntologyAgent.process_message = external_process_message

# Add all methods from OntologyQueryInterface to OntologyAgent
# First add non-private methods
for name, method in inspect.getmembers(OntologyQueryInterface, inspect.isfunction):
    if not name.startswith('_') or name == '_load_ontology':
        setattr(OntologyAgent, name, method)

# Then add private methods (except _load_ontology which we already added)
for name, method in inspect.getmembers(OntologyQueryInterface, inspect.isfunction):
    if name.startswith('_') and name != '_load_ontology':
        setattr(OntologyAgent, name, method)

print("Successfully patched OntologyAgent with query methods and process_message method")
