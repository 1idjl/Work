from ase.io import read, write

def pdb_to_xyz(input_pdb, output_xyz):
    """
    Convert PDB file to XYZ format
    
    Parameters:
    -----------
    input_pdb : str
        Path to input PDB file
    output_xyz : str
        Path to output XYZ file
    """
    try:
        # Read structure from PDB file
        structure = read(input_pdb)
        
        # Write structure to XYZ format
        write(output_xyz, structure)
        
        print(f"Conversion successful: {input_pdb} -> {output_xyz}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

# Usage example
if __name__ == "__main__":
    input_file = "Material_studio/SiO2_21A_3d.pdb"
    output_file = "Material_studio/SiO2_21A_3d.xyz"
    
    pdb_to_xyz(input_file, output_file)
