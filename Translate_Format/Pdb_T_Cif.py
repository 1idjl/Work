from ase.io import read, write

def pdb_to_cif(input_pdb, output_cif):
    """
    Convert PDB file to CIF format
    
    Parameters:
    -----------
    input_pdb : str
        Path to input PDB file
    output_cif : str
        Path to output CIF file
    """
    try:
        # Read structure from PDB file
        structure = read(input_pdb)
        
        # Write structure to CIF format
        write(output_cif, structure)
        
        print(f"Conversion successful: {input_pdb} -> {output_cif}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

# Usage example
if __name__ == "__main__":
    input_file = "structure.pdb"
    output_file = "structure.cif"
    
    pdb_to_cif(input_file, output_file)
