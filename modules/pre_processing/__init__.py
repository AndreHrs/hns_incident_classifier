from .contraction_handler import handle_contraction
from .numeric_handler import handle_numeric

def clean_dataset():

    # Example of cleaning handling
    handle_contraction()
    handle_numeric()

def pre_process_dataset():

    # Clean dataset
    clean_dataset()

    # Populate if more actions are needed
