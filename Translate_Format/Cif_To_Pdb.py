from ase.io import read, write

def cif_to_pdb(input_cif, output_pdb):
    try:
        # Read structure from CIF file
        structure = read(input_cif)
        
        # Write structure to PDB format
        write(output_pdb, structure)
        
        print(f"Conversion successful: {input_cif} -> {output_pdb}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

# Usage example
if __name__ == "__main__":
    input_file = "Material_studio/SiO2_21A_3d.cif"
    output_file = "SiO2_21A_3d.pdb"
    
    cif_to_pdb(input_file, output_file)
