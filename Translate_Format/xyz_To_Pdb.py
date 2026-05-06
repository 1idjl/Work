from ase.io import read, write

def xyz_to_pdb(input_xyz, output_pdb):
    """
    Convert XYZ file to PDB format
    
    Parameters:
    -----------
    input_xyz : str
        Path to input XYZ file
    output_pdb : str
        Path to output PDB file
    """
    try:
        # Read structure from XYZ file
        structure = read(input_xyz)
        
        # Write structure to PDB format
        write(output_pdb, structure)
        
        print(f"Conversion successful: {input_xyz} -> {output_pdb}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

# Usage example
if __name__ == "__main__":
    input_file = "structure.xyz"
    output_file = "structure.pdb"
    
    xyz_to_pdb(input_file, output_file)
