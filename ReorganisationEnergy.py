# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:27:52 2024

@author: kyebchoo
"""
#%%
# Importing modules

# NUMPY
import numpy as np
from numpy import random

# SCIPY
import scipy as sci
import scipy.constants as constants
import scipy.stats as stats

# PANDAS
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)

# MATPLOTLIB-PYPLOT
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# SYSTEM CONTROLS
import os
import importlib
from copy import deepcopy
import paramiko

# RDKIT(CHEMISTRY)
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# import kora.install.rdkit

# DATE AND TIME CONTROLS
from datetime import datetime, timedelta
import time

# OTHERS
from pymongo import MongoClient
import urllib.parse
import pprint
import ast
import cclib
from pathlib import Path
from ase import Atoms
import warnings
from openbabel import pybel
from tqdm import tqdm
#%%

def timestamp() -> str:
    """
    Returns the standardised formatted timestamp for use in later methods, in the form (YYYYMMDDHHMMSS)

    Returns
    -------
    str
        Formatted timestamp.

    """
    return str(datetime.now()).split('.')[0].replace('-', '').replace(' ', '').replace(':', '')
    
def tag(molecule: str) -> str:
    """
    Returns the standardised formmated tag for use in later methods, in the form of {truncated molecule name}_{YYYYMMDDHHMMSS}.
    The code would automatically remove symbols and truncate the name to 8 letters.

    Parameters
    ----------
    molecule : str
        Name of molecule (preferrably IUPAC).

    Returns
    -------
    str
        Standardised tag.

    """
    molecule = molecule.replace('[', '').replace(']', '').replace('-', '').replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace('_', '')[0:8]
    tag_name = '%s_%s' % (molecule, timestamp())
    return tag_name

def is_notebook() -> bool:
    """
    Method to test if the script is being ran in a Jupyter notebook environment.

    Returns
    -------
    bool
        TRUE if Jupyter notebook detected, and FALSE otherwise.

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def display_if_notebook(display_item, otherwise: bool = False):
    """
    Display or print out the display_item if script is being ran in a Jupyter notebook environment.

    Parameters
    ----------
    display_item : TYPE
        Any item, whether it be string or table, to be displayed if in a Jupyter notebook environment.
    otherwise : bool, optional
        Whether to display or print out the display_item if not in a Jupyter notebook environment. The default is False.

    Returns
    -------
    None.

    """
    if is_notebook():
        try:
            display(display_item) # ignore error if running in Spyder
        except:
            pass
    else:
        if otherwise:
            display(display_item) # ignore error if running in Spyder
            
def check_directory(directory: str):
    """
    Method to check if a directory exists, and to create the directory if it does not exist.

    Parameters
    ----------
    directory : str
        Path to directory.

    Returns
    -------
    None.

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        pass
        
def convert_atomic_number_to_symbol(atomic_number: int) -> str:
    """
    Convert atomic number to atomic symbols representing the elements.

    Parameters
    ----------
    atomic_number : int
        Atomic number to be converted.

    Returns
    -------
    str
        Symbol of the corresponding element.

    """
    return Chem.GetPeriodicTable().GetElementSymbol(atomic_number)

def format_basis_set(basis_set: str):
    """
    Method to format the basis set, removing invalid characters when being used as part of a filename.

    Parameters
    ----------
    basis_set : str
        Basis set input.

    Returns
    -------
    TYPE
        Formatted basis set string.

    """
    return basis_set.replace('+', 'p').replace('*', 's').replace('(', '').replace(')', '').replace(',', '')

#%%
class Molecule():
    
    def __init__(self,
                 molecule_name: str,
                 molecule_id: str,
                 SMILES: str,
                 parent_folder: str = os.getcwd(),
                 debug: bool = False):
        """
        Creates a 'Molecule' class for later operations. The molecule could be declared directly from 'Molecule()',
        or be loaded from a previous documentation with 'Molecule.load_molecule()'.
        
        The path structure of the system would be:
            {parent_folder}/Data/{molecule_id}/{molecule_documents}

        Parameters
        ----------
        molecule_name : str
            Name of the molecule, preferrably IUPAC name.
        molecule_id : str
            A unique ID for this specific molecule. File and document naming would be dependent on this ID and thus it must be unique.
        SMILES : str
            SMILES string for the molecule.
        parent_folder : str, optional
            The main folder which you would be working in. All subfolders and files would be generated under the parent folder.
            The default is os.getcwd().
        debug : bool, optional
            Toggle to print additional lines during operation for debugging. The default is False.

        Returns
        -------
        None.

        """
        self._molecule_name = molecule_name
        self._molecule_id = molecule_id
        self._SMILES = SMILES
        self._parent_folder = parent_folder
        self._debug = debug
        
        check_directory("%s/Data/%s/" % (self._parent_folder, self._molecule_id))
    
    @classmethod
    def load_molecule(cls, 
                      molecule_id: str,
                      parent_folder: str = os.getcwd(),
                      debug: bool = False):
        """
        Alternative initialisation method where the key information is extracted from a previously generated documentation.
        Only the molecule ID and the parent folder has to be declared, additional entries would be searched within the parent folder.

        Parameters
        ----------
        molecule_id : str
            Molecule ID specific to the molecule. This should be previously generated when the molecule was first initialised.
            If the molecule has not been generated before, the molecule has to be initialised the ordinary way.
        parent_folder : str, optional
            The parent folder where all further files and folders are generated and located. The default is os.getcwd().
        debug : bool, optional
            Toggle to print additional lines during operation for debugging. The default is False.

        Raises
        ------
        Exception
            Fail to read documentation. Check if molecule ID is correct, or that the respective files are located in the correct location.

        Returns
        -------
        list
            Required entries to load molecule.

        """
        if debug:
            print('\nLoading molecule from %s/%s/%s_documentation.txt' % (parent_folder, molecule_id, molecule_id))
        try:
            with open("%s/Data/%s/%s_documentation.txt" % (parent_folder, molecule_id, molecule_id), "r") as documentation:
                content = documentation.read().split('\n')
                molecule_name = content[0]
                molecule_id = content[1]
                SMILES = content[2]
                parent_folder = content[3]
            return cls(molecule_name, molecule_id, SMILES, parent_folder, debug)   
        except:
            raise Exception('Failed to read documentation from %s/%s/%s_documentation.txt' % (parent_folder, molecule_id, molecule_id))
    
    def print_if_debug(self, lines: str):
        """
        Prints out the lines of in debug mode.

        Parameters
        ----------
        lines : str
            Lines to be printed out.

        Returns
        -------
        None.

        """
        if self._debug:
            print(lines)
            
    def create_molecule_documentation(self, allow_aromatic_bond: bool = True) -> None:
        """
        Creates molecule documentation so that future initialisation can be done with just specifying the molecule ID.
        The following documents would be genrated under parent_folder/Data/molecule_ID:
            (1) Text (.txt) document containing the initialisation lines.
            (2) Excel (.xlsx) file containing the initial geometry and also connectivity of the atoms within the molecule. 
                The connectivity would include aromatic bonds of order 1.5. To disable this, set allow_aromatic_bond to False. 
                Single and double bonds would be assigned by RDKit Kekulisation.

        Parameters
        ----------
        allow_aromatic_bond : bool, optional
            Allows aromatic bonds of order 1.5 to be generated in the connectivity document. The default is True.

        Returns
        -------
        None.

        """
        self.print_if_debug('\nIn function Molecule.create_molecule_documentation():')
        molecule = Chem.AddHs(Chem.MolFromSmiles(self._SMILES))
        kekulized_molecule = Chem.AddHs(Chem.MolFromSmiles(self._SMILES))
        Chem.Kekulize(kekulized_molecule)
        if self._debug:
            display_if_notebook(molecule, otherwise = True)
        dataframe = pd.DataFrame(columns = ['Index', 'Atomic Number', 'Element', 'Hybridisation', 'Neighbors[Index, Atomic Number, Element, Bond, Bond Order]'])
        for atom in molecule.GetAtoms():
            index = atom.GetIdx()
            atomic_number = atom.GetAtomicNum()
            element = atom.GetSymbol()
            hybridisation = str(Chem.rdchem.HybridizationType.values[atom.GetHybridization()]).split('.')[-1]
            self.print_if_debug('Atom with index: %s\n%s\n%s\nNeighbours:' % (index, atomic_number, element))
            neighbors = []
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                neighbor_atomic_number = neighbor.GetAtomicNum()
                neighbor_symbol = neighbor.GetSymbol()
                neighbor_bond_type = str(molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType())
                if allow_aromatic_bond == True:
                    neighbor_bond_order = str(molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble())
                else:
                    neighbor_bond_order = str(kekulized_molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble())
                neighbors.append([neighbor_index, neighbor_atomic_number, neighbor_symbol, neighbor_bond_type, neighbor_bond_order])
                self.print_if_debug(' %s\n %s\n %s\n %s\n %s' % (str(neighbor_index), str(neighbor_atomic_number), neighbor_symbol, str(neighbor_bond_type), str(neighbor_bond_order)))
            dataframe.loc[len(dataframe)] = [index, atomic_number, element, hybridisation, neighbors]
        if self._debug:
            display_if_notebook(dataframe, otherwise = True)
        dataframe.to_excel(r'%s/Data/%s/%s_documentation.xlsx' % (self._parent_folder, self._molecule_id, self._molecule_id))
        
        with open("%s/Data/%s/%s_documentation.txt" % (self._parent_folder, self._molecule_id, self._molecule_id), "w") as documentation:
            documentation.write("%s\n%s\n%s\n%s\n\nCreated on %s" % (self._molecule_name, self._molecule_id, self._SMILES, self._parent_folder, str(datetime.now())))
        self.print_if_debug('Molecule documentation created sucessfully.')
        
    def get_SMILES(self) -> str:
        """
        Returns the SMILES string of the molecule.

        Returns
        -------
        str
            SMILES string of the molecule.

        """
        self.print_if_debug('\nIn function Molecule.get_SMILES():')
        return self._SMILES
    
    def get_molecule_name(self) -> str:
        """
        Returns the name of the molecule.

        Returns
        -------
        str
            Name of the molecule.

        """
        self.print_if_debug('\nIn function Molecule.get_molecule_name():')
        return self._molecule_name
    
    def get_molecule_id(self) -> str:
        """
        Returns the ID of the molecule.

        Returns
        -------
        str
            ID of the molecule.

        """
        self.print_if_debug('\nIn function Molecule.get_molecule_id():')
        return self._molecule_id
    
    def generate_xyz_from_rdkit_mol(self,
                                    random_low: float = 0.00,
                                    random_high: float = 0.00) -> list[str, int]:
        """
        Generates the cartesian x,y,z coordinates using built-in RDKit functions.
        The molecule is optimised using Merck molecular force field methods.
        
        The returned xyz coordinate block DOES NOT terminate with two blank lines.

        Returns
        -------
        list[str, int]
            List with the [0] first entry being the xyz coordinates, and 
            the [1] second entry being the total number of atoms in the molecule.

        """
        self.print_if_debug('\nIn function Molecule.generate_xyz_from_rdkit_mol():')
        Mol = Chem.AddHs(Chem.MolFromSmiles(self._SMILES))
        AllChem.EmbedMolecule(Mol)
        AllChem.MMFFOptimizeMolecule(Mol, maxIters = 1000)
        
        for c in Mol.GetConformers():
            # only single conformer is chosen currently, might alter in future
            positions = c.GetPositions()
        atoms_symbols = np.array([at.GetSymbol() for at in Mol.GetAtoms()])
        atoms = Atoms(atoms_symbols, positions=positions)
        num_atoms: int = len(atoms)
        types = atoms.get_chemical_symbols()
        all_atoms = zip(types, atoms.get_positions())
        a_str = ""
        for atom in all_atoms:
            if random_low == 0.00 and random_high == 0.00:
                a_str += atom[0] + " " + " ".join([str(x) for x in atom[1]]) + "\n"
            elif random_low != random_high and random_low < random_high:
                a_str += atom[0] + " " + " ".join([str(x + random.uniform(low = random_low, high = random_high)) for x in atom[1]]) + "\n"
            else:
                raise Exception('Invalid random_low and/or random_high input.')
        # a_str = a_str + "\n\n"
        self.print_if_debug(a_str)
        return a_str, num_atoms
    
    def generate_xyz_from_pymol_mol(self, pbmol) -> list[str, int]:
        """
        Generates the cartesian x,y,z coordinates from a pymol object.

        Parameters
        ----------
        pbmol : TYPE
            pymol object.

        Returns
        -------
        list[str, int]
            List with the [0] first entry being the xyz coordinates, and 
            the [1] second entry being the total number of atoms in the molecule.

        """
        self.print_if_debug('\nIn function Molecule.generate_xyz_from_pymol_mol():')
        warnings.warn('This method has not been tested. If the xyz block generated has a double-blank line termination, it should be truncated as it would affect future processes.')
        # read mol and outputs xyz with a small optimisation
        pbmol.write("xyz", r"%s/Data/%s/%s_pybel_buffer.xyz" % (self._parent_folder, self._molecule_id, self._molecule_id), overwrite = True)
        with open(r"%s/Data/%s/%s_pybel_buffer.xyz" % (self._parent_folder, self._molecule_id, self._molecule_id), "r") as file:
            split: list = file.read().split("\n\n")
            pybabelxyz: str = split[1]
            num_atoms: int = int(split[0])
        self.print_if_debug(pybabelxyz)
        return pybabelxyz, num_atoms
    
    def generate_xyz_from_xyz_file(self, filename: str) -> list[str, int]:
        """
        Generates the cartesian x,y,z coordinates from a .xyz file. The filename should contain the filename with the .xyz suffix.
        The file should also be located in the molecule file under the folder {parent_folder}/Data/{molecule_id}/.

        Parameters
        ----------
        filename : str, optional
            Filename of the xyz file with the suffix (.xyz).

        Returns
        -------
        list[str, int]
            List with the [0] first entry being the xyz coordinates, and 
            the [1] second entry being the total number of atoms in the molecule.

        """
        self.print_if_debug('\nIn function Molecule.generate_xyz_from_xyz_file():')
        warnings.warn('This method has not been tested. If the xyz block generated has a double-blank line termination, it should be truncated as it would affect future processes.')
        if filename == '':
            filename = r"%s/Data/%s/%s" % (self._parent_folder, self._molecule_id, self._molecule_id)
        with open(filename, "r") as file:
            lines: str = file.readlines()[2:]
            mol_xyz: str = "".join(lines)
            num_atoms: int = len(lines)
        # mol_xyz: str = mol_xyz + "\n\n"
        mol_xyz: str = mol_xyz
        self.print_if_debug(mol_xyz)
        return mol_xyz, num_atoms
    
    def generate_xyz_from_log_file(self, 
                                   filename: str,
                                   random_low: float = 0.0000,
                                   random_high: float = 0.0001) -> list[str, int]:
        """
        Generates the cartesian x,y,z coordinates from a Gaussian16 .log output file. The filename should contain the filename with the .log suffix.
        The file should also be located in the molecule file under the folder {parent_folder}/Data/{molecule_id}/.

        Parameters
        ----------
        filename : str
            Filename of the log file with the suffix (.log).

        Returns
        -------
        list[str, int]
            List with the [0] first entry being the xyz coordinates, and 
            the [1] second entry being the total number of atoms in the molecule.

        """
        self.print_if_debug('\nIn function Molecule.generate_xyz_from_log_file():')
        # generate XYZ string from calculation output
        # gaussian_data = cclib.io.ccread(r'%s/Data/%s/%s' % (self._parent_folder, self._molecule_id, filename))
        gaussian_data = cclib.io.ccread(r'%s' % (filename))
        # generate xyz file from gaussian log output
        num_atoms: int = gaussian_data.natom
        coords = gaussian_data.atomcoords
        coords = coords[-1, :, :]
        atomnos = gaussian_data.atomnos
        xyz_str_gauss = ""
        for i in range(0, len(coords)):
            xyz_str_gauss += (str(convert_atomic_number_to_symbol(int(atomnos[i])))
                              + " "
                              + " ".join([str(float(x) + random.uniform(low = random_low, high = random_high)) for x in coords[i]])
                              + "\n")
        self.print_if_debug("There are %i atoms and %i MOs" % (gaussian_data.natom, gaussian_data.nmo))
        self.print_if_debug("Obtained opt geometry from optimised log file for: \n%s" % (filename))
        self.print_if_debug(xyz_str_gauss)
        return xyz_str_gauss, num_atoms
    
    def generate_molecule_xyz(self, 
                              input_object = '', 
                              name_extension: str = '',
                              save: bool = True,
                              random_low: float = 0.0000,
                              random_high: float = 0.0001) -> str:
        """
        Generates molecule xyz by detecting the input_object. The input_object can either be:
            1) an rdkit mol class
            2) a pymol class  
            3) an xyz file (located within the molecule file) with the structure already in it
            4) a Gaussian16 output .log file with previous calculation of the molecule
    
        If in the form of a filename, the file must be located within the folder:
            {parent_folder}/Data/{molecule_id}/ 
        
        The {name_extension} is the suffix to append at the end of the xyz filename: {molecule_id}_{name_extension}.xyz.

        Parameters
        ----------
        input_object : TYPE, optional
            Input object of the form of either (1) an RDKit molecule, (2) pymol molecule, (3) xyz file, or (4) .log Gaussian16 output file. 
            The default is '', where the geometry from the SMILES-generated RDKit molecule would be used.
        name_extension : str, optional
            Optional extension to be added to the end of the .xyz file. The default is ''.
        save : bool, optional
            Toggle to save the xyz geometry as a .xyz file. The default is True.

        Returns
        -------
        str
            Molecule xyz block.

        """
        self.print_if_debug('\nIn function Molecule.generate_molecule_xyz():')
        # 1) an rdkit mol class
        if (type(input_object) is Chem.rdchem.Mol) or (input_object == ''):
            a_str_gauss, num_atoms = self.generate_xyz_from_rdkit_mol(random_low = random_low,
                                                                      random_high = random_high)
            self.print_if_debug('RDKit molecule geometry used.')
        # 2) a pymol class
        try:
            from openbabel import pybel as pb
            if type(input_object) is pb.Molecule:
                a_str_gauss, num_atoms = self.generate_xyz_from_pymol_mol(pbmol = input_object)
                self.print_if_debug('Pymol molecule geometry used.')
        except ImportError:
            print("openbabel not found")
            pass
        
        if type(input_object) is str:
            if Path(input_object).is_file():
        # 3) path to an xyz file with the structure already in it
                if Path(input_object).suffix == ".xyz":
                    a_str_gauss, num_atoms = self.generate_xyz_from_xyz_file(filename = input_object)
                    self.print_if_debug('Geometry extracted from %s.' % input_object)
        # 4) path to a file with previous calculation of the molecule
                elif Path(input_object).suffix == ".log":
                    a_str_gauss, num_atoms = self.generate_xyz_from_log_file(filename = input_object, 
                                                                             random_low = random_low, 
                                                                             random_high = random_high)
                    self.print_if_debug('Geometry extracted from %s.' % input_object)
                else:
                    raise Exception('Input in generate_molecule_xyz not recognised.\n%s' % (input_object))
        if save:
            if name_extension != '':
                name_extension = '_%s' % (name_extension)
            with open(r"%s/Data/%s/%s%s.xyz" % (self._parent_folder, self._molecule_id, self._molecule_id, name_extension), "wt") as textfile: 
                a_str = "%s\n" % (str(num_atoms)) + "%s\n" % (self._molecule_name) + a_str_gauss
                textfile.write(a_str)
                self.print_if_debug('xyz geometry saved as; %s_%s.xyz.' % (self._molecule_id, name_extension))
        return a_str_gauss
    
    def keyword_line_generator(self, 
                               functional_basisset: str,
                               calculation_type: str = 'OptFreq',
                               state_TD: int = 0,
                               restart: bool = False, 
                               higher_max_cycles: int = 0,
                               set_trust_radius_bohr: float = 0,
                               explicitly_specify_connectivity: bool = True,
                               integral: str = 'SuperFine',
                               overwrite_Freq: list[list, list] = [[], []],
                               overwrite_Opt: list[list, list] = [[], []],
                               overwrite_NoSymm: list[list, list] = [[], []],
                               overwrite_state_TD: list[list, list] = [[], []],
                               overwrite_Geom: list[list, list] = [[], []],
                               overwrite_Integral: list[list, list] = [[], []]
                               ) -> str:
        """
        Method to write the keyword line for the Gaussian16 script. Supports by default the methods:
            (1) OptFreq:
                Optimise molecule followed by a frequency calculation.
            (2) Freq:
                Frequenmcy calculation only.
            (3) Manual:
                Manually input and specify keywords. For example to perform an optimisation job...
                ...with increased MaxCycles of 200 and reduced MaxStep of 0.10Bohr:
                    overwrite_Opt = [[True, 'Opt'], ['MaxCycles=200', MaxStep=10]]
                ...as a restart job:
                    overwrite_Opt = [[True, 'Opt'], ['Restart']]
        Overwrite lines would overwrite any inputs generated beforehand.

        Parameters
        ----------
        functional_basisset : str
            Functional and basisset in the form of <functional>/<basis_set>, e.g. B3LYP/6-31G(d,p).
        calculation_type : str, optional
            Calculation choice of 'OptFreq', 'Opt', or 'Manual'. The default is 'OptFreq'.
        state_TD : int, optional
            Excitation state of the molecule. The default is 0.
        restart : bool, optional
            Whether to include the Restart keyword under the Opt() line. The default is False.
        higher_max_cycles : int, optional
            Manually input an integer number of max cycles. The default is 0, where the Gaussian16 default would be used.
        set_trust_radius_bohr : float, optional
            Manually set the maximum trust radius (step size) in units of Bohr for the optimisation.
            The default is 0, where the Gaussian16 default would be used.
        explicitly_specify_connectivity : bool, optional
            Explicitly specify bonds and connectivity. The default is True.
        overwrite_Freq : list[list, list], optional
            Overwrite line for Freq keyword. The default is [[], []] for no overwrite.
        overwrite_Opt : list[list, list], optional
            Overwrite line for Opt keyword. The default is [[], []] for no overwrite.
        overwrite_NoSymm : list[list, list], optional
            Overwrite line for NoSymm keyword. The default is [[], []] for no overwrite.
        overwrite_state_TD : list[list, list], optional
            Overwrite line for TD keyword. The default is [[], []] for no overwrite.
        overwrite_Geom : list[list, list], optional
            Overwrite line for Geom keyword. The default is [[], []] for no overwrite.

        Raises
        ------
        ValueError
            Calculation type not recognised, check 'calculation_type' input.

        Returns
        -------
        str
            Keyword line to be passed into the method/function generating the Gaussian runscripts..

        """
        self.print_if_debug('\nIn function Molecule.keyword_line_generator():')
        # keyword_line: list[bool, list[]] = [bool, arguments]
        opt_line = [[False, 'Opt'], []]
        freq_line = [[False, 'Freq'], []]
        nosymm_line = [[False, 'NoSymm'], []]
        TD_line = [[False, 'TD'], []]
        Geom_line = [[False, 'Geom'], []]
        integral_line = [[False, 'Integral'], []]
        
        keyword_line = '#P'
        
        if calculation_type.lower() == 'optfreq' or calculation_type.lower() == 'freqopt':
            opt_line[0][0] = True
            freq_line[0][0] = True
            nosymm_line[0][0] = True
        elif calculation_type.lower() == 'opt':
            opt_line[0][0] = True
            nosymm_line[0][0] = True
        elif 'marcus':
            nosymm_line[0][0] = True
            Geom_line = [[True, 'Geom'], ['check']]
            explicitly_specify_connectivity = False
            restart = False
            higher_max_cycles = 0
            set_trust_radius_bohr = 0
            # state_TD = 0
        elif calculation_type.lower() == 'manual':
            print('Using manual input to generate keyword line for Gaussian16.')
        else:
            raise ValueError('\nCalculation type not recognised: %s' % (calculation_type))
            
        if higher_max_cycles != 0:
            opt_line[1].append('MaxCycles=%s' % (higher_max_cycles))
        
        if set_trust_radius_bohr != 0:
            set_trust_radius_bohr = round(set_trust_radius_bohr, 2)
            max_step = int(set_trust_radius_bohr/0.01)
            opt_line[1].append('MaxStep=%s' % (max_step))
            
        if state_TD != 0:
            TD_line[0][0] = True
            TD_line[1] = ['NStates=10', 'Root=%s' % (int(state_TD))]
        
        if restart == True:
            opt_line[1].append('Restart')
            Geom_line[0][0] = True
            Geom_line[1].append('check')
        
        if explicitly_specify_connectivity == True:
            Geom_line = [[True, 'Geom'], ['Connectivity']]
        
        if integral.lower() == 'superfine':
            integral_line = [[True, 'Integral'], ['SuperFine']]
        elif integral.lower() == 'ultrafine':
            integral_line = [[True, 'Integral'], ['UltraFine']]
        else:
            integral_line = [[True, 'Integral'], ['UltraFine']]
            warnings.warn('Intergral not recognised, switching to default = UltraFine')
        
        if overwrite_Freq != [[], []]:
            freq_line = overwrite_Freq
            warnings.warn('Overwriting Freq line.')
        if overwrite_Opt != [[], []]:
            opt_line = overwrite_Opt
            warnings.warn('Overwriting Opt line.')
        if overwrite_NoSymm != [[], []]:
            nosymm_line = overwrite_NoSymm
            warnings.warn('Overwriting NoSymm line.')
        if overwrite_state_TD != [[], []]:
            TD_line = overwrite_state_TD
            warnings.warn('Overwriting TD line.')
        if overwrite_Geom != [[], []]:
            Geom_line = overwrite_Geom
            warnings.warn('Overwriting Geom line.')
        if overwrite_Integral != [[], []]:
            integral_line = overwrite_Integral
            warnings.warn('Overwriting Geom line.')
        
        self.print_if_debug('freq_line: %s\nopt_line: %s\nnosymm_line: %s\nTD_line: %s\nGeom_line: %s' % (freq_line, opt_line, nosymm_line, TD_line, Geom_line))
        
        def keyword_line_argument_writer(keyword: list[[bool, str], list]) -> str:
            """
            Method to format the keyword array blocks into proper lines.

            Parameters
            ----------
            keyword : list[[bool, str], list]
                Keyword array block.

            Raises
            ------
            Exception
                Keyword_line_argument_writer failed to write arguments, check input list.

            Returns
            -------
            str
                Keyword line.

            """
            self.print_if_debug('\nIn function keyword_line_argument_writer():')
            if keyword[0][0] == True:
                keyword_line_argument = '%s' % (keyword[0][1])
                
                arguments: str = ''
                for i in range(0, len(keyword[1])):
                    if arguments == '':
                        arguments = '%s' % (keyword[1][i])
                    else:
                        arguments = arguments + ',' + '%s' % (keyword[1][i])
                if arguments != '':
                    keyword_line_argument = keyword_line_argument + '(' + arguments + ')'
                return keyword_line_argument
            elif keyword[0][0] == False:
                return ''
            else:
                raise Exception('\nKeyword_line_argument_writer failed to write arguments for %s' % (keyword))
        
        for keyword in [opt_line, freq_line, nosymm_line, TD_line, Geom_line, integral_line]:
            if keyword_line_argument_writer(keyword = keyword) != '':
                keyword_line = keyword_line + ' ' + keyword_line_argument_writer(keyword = keyword)
        keyword_line = keyword_line + ' ' + functional_basisset
        self.print_if_debug('Keyword line: %s' % (keyword_line))
        return keyword_line
    
    def get_connectivity(self) -> str:
        """
        Returns the connectivity of the bonds in a molecule as a block to be passed into the Gaussian16 runscript.

        Returns
        -------
        write_text : str
            Connectivity text block.

        """
        self.print_if_debug('\nIn function Molecule.get_connectivity():')
        dataframe = pd.read_excel(r'%s/Data/%s/%s_documentation.xlsx' % (self._parent_folder, self._molecule_id, self._molecule_id))
        if self._debug:
            display_if_notebook(dataframe, otherwise = True)
        write_text: str = ''
        for index in range(0, len(dataframe)):
            parent_index = dataframe.at[index, 'Index'] + 1
            line = '%s' % (parent_index)
            neighbors = dataframe.at[index, 'Neighbors[Index, Atomic Number, Element, Bond, Bond Order]']
            neighbors = ast.literal_eval(neighbors)
            for i in range(0, len(neighbors)):
                neighbor_index = neighbors[i][0] + 1
                bond_type = neighbors[i][4]
                line = line + ' ' + str(neighbor_index) + ' ' + str(bond_type)
            # print(line)
            if write_text == '':
                write_text = line
            else:
                write_text = write_text + '\n' + line
        write_text = write_text
        self.print_if_debug(write_text)
        with open(r'%s/Data/%s/%s_connectivity.txt' % (self._parent_folder, self._molecule_id, self._molecule_id), 'wt') as textfile:
            textfile.write(write_text)
            textfile.close()
        return write_text
        
    def generate_gjf_file(self, 
                          input_object,
                          excited_state: int,
                          charge: int,
                          functional: str, 
                          basisset: str,
                          number_of_proc: int,
                          memory: int,
                          calculation_type: str = 'OptFreq',
                          connectivity: bool = True,
                          keyword_line_arguments_input: list[[str, str]] = []) -> str:
        """
        Method to write the Gaussian runscript.

        Parameters
        ----------
        input_object : TYPE
            Input object to be fed into Molecule.generate_molecule_xyz().
        excited_state : int
            Excitation state.
        charge : int
            Charge of molecule.
        functional : str
            Functional to be used in DFT calculation.
        basisset : str
            Basis set to be used in DFT calculation.
        number_of_proc : int
            Number of processors.
        memory : int
            Total amount of memory to be divided between the processors.
        calculation_type : str, optional
            Calculation type. The default is 'OptFreq'.
        connectivity : bool, optional
            Whether to use molecule connectivity. The default is True.
        keyword_line_arguments_input : list[[str, str]], optional
            Overwite lines. The default is [].

        Returns
        -------
        gjf_file : str
            Gaussian16 job script.

        """
        self.print_if_debug('\nIn function Molecule.generate_gjf_file():')
        functional_basisset = '%s/%s' % (functional, basisset)
        keyword_line_arguments: dict = {'functional_basisset': functional_basisset,
                                        'calculation_type': calculation_type,
                                        'state_TD': excited_state,
                                        'restart': False,
                                        'higher_max_cycles': int(1024),
                                        'set_trust_radius_bohr': float(0.30),
                                        'explicitly_specify_connectivity': connectivity,
                                        'overwrite_Freq': [[], []],
                                        'overwrite_Opt': [[], []],
                                        'overwrite_NoSymm': [[], []],
                                        'overwrite_state_TD': [[], []]}
        
        # keyword_line_arguements_input = [['calculation_type', 'Opt'], ['restart', True], ['higher_max_cycles', 1024]]
        for i in range(0, len(keyword_line_arguments_input)):
            keyword_line_arguments['%s' % (keyword_line_arguments_input[i][0])] = keyword_line_arguments_input[i][1]
        
        keyword_line = self.keyword_line_generator(functional_basisset = keyword_line_arguments['functional_basisset'],
                                                   calculation_type = keyword_line_arguments['calculation_type'],
                                                   state_TD = keyword_line_arguments['state_TD'],
                                                   restart = keyword_line_arguments['restart'], 
                                                   higher_max_cycles = keyword_line_arguments['higher_max_cycles'],
                                                   set_trust_radius_bohr = keyword_line_arguments['set_trust_radius_bohr'],
                                                   explicitly_specify_connectivity = keyword_line_arguments['explicitly_specify_connectivity'],
                                                   overwrite_Freq = keyword_line_arguments['overwrite_Freq'],
                                                   overwrite_Opt = keyword_line_arguments['overwrite_Opt'],
                                                   overwrite_NoSymm = keyword_line_arguments['overwrite_NoSymm'],
                                                   overwrite_state_TD = keyword_line_arguments['overwrite_state_TD'])
        
        name_extension = '%s_%s_C%sS%s' % (functional, format_basis_set(basisset), charge, excited_state)
        Mol_xyz = self.generate_molecule_xyz(input_object = input_object, 
                                             name_extension = name_extension,
                                             save = True)
        
        gjf_file = "%s_%s" % (self._molecule_id, name_extension)
        
        title_line = '%s %s' % (gjf_file, keyword_line_arguments['calculation_type'])
        
        a_str_gaussian = (f"%nprocshared={number_of_proc} \n" + 
                          "\n" +
                          f"%mem={memory}GB \n" + 
                          f"%chk={gjf_file}.chk \n" + 
                          keyword_line + "\n" + 
                          "\n" +
                          title_line + "\n"
                          "\n" + 
                          "%s %s\n" % (charge, abs(charge) + 1) + 
                          Mol_xyz + '\n')
        
        if connectivity == True:
            a_str_gaussian = (a_str_gaussian + 
                              self.get_connectivity() + '\n' + 
                              '\n')
        
        self.print_if_debug('\nGaussian file:\n==============\n%s\n==============' % (a_str_gaussian))
        with open('%s/Data/%s/%s.gjf' % (self._parent_folder, self._molecule_id, gjf_file), mode = "wt") as writefile:
            writefile.write(a_str_gaussian)
          
        return gjf_file
    
    def generate_gjf_restart_file(self, 
                                  excited_state: int,
                                  charge: int,
                                  functional: str,
                                  basisset: str,
                                  number_of_proc: int,
                                  memory: int,
                                  calculation_type: str = 'OptFreq',
                                  keyword_line_arguments_input: list[[str, str]] = []) -> str:
        self.print_if_debug('\nIn function Molecule.generate_gjf_file():')
        functional_basisset = '%s/%s' % (functional, basisset)
        keyword_line_arguments: dict = {'functional_basisset': functional_basisset,
                                        'calculation_type': calculation_type,
                                        'state_TD': excited_state,
                                        'restart': True,
                                        'higher_max_cycles': int(1024),
                                        'set_trust_radius_bohr': float(0.30),
                                        'explicitly_specify_connectivity': False,
                                        'overwrite_Freq': [[], []],
                                        'overwrite_Opt': [[], []],
                                        'overwrite_NoSymm': [[], []],
                                        'overwrite_state_TD': [[], []]}
        
        # keyword_line_arguements_input = [['calculation_type', 'Opt'], ['restart', True], ['higher_max_cycles', 1024]]
        for i in range(0, len(keyword_line_arguments_input)):
            keyword_line_arguments['%s' % (keyword_line_arguments_input[i][0])] = keyword_line_arguments_input[i][1]
        
        keyword_line = self.keyword_line_generator(functional_basisset = keyword_line_arguments['functional_basisset'],
                                                   calculation_type = keyword_line_arguments['calculation_type'],
                                                   state_TD = keyword_line_arguments['state_TD'],
                                                   restart = keyword_line_arguments['restart'], 
                                                   higher_max_cycles = keyword_line_arguments['higher_max_cycles'],
                                                   set_trust_radius_bohr = keyword_line_arguments['set_trust_radius_bohr'],
                                                   explicitly_specify_connectivity = keyword_line_arguments['explicitly_specify_connectivity'],
                                                   overwrite_Freq = keyword_line_arguments['overwrite_Freq'],
                                                   overwrite_Opt = keyword_line_arguments['overwrite_Opt'],
                                                   overwrite_NoSymm = keyword_line_arguments['overwrite_NoSymm'],
                                                   overwrite_state_TD = keyword_line_arguments['overwrite_state_TD'])
        
        name_extension = '%s_%s_C%sS%s' % (functional, format_basis_set(basisset), charge, excited_state)
        
        gjf_file = "%s_%s" % (self._molecule_id, name_extension)
        
        title_line = '%s %s' % (gjf_file, keyword_line_arguments['calculation_type'])
        
        a_str_gaussian = (f"%nprocshared={number_of_proc} \n" + 
                          "\n" +
                          f"%mem={memory}GB \n" + 
                          f"%chk={gjf_file}.chk \n" + 
                          keyword_line + "\n" + 
                          "\n" +
                          title_line + "\n"
                          "\n" + 
                          "%s %s\n" % (charge, abs(charge) + 1))
        
        self.print_if_debug('\nGaussian file:\n==============\n%s\n==============' % (a_str_gaussian))
        with open('%s/Data/%s/%s.gjf' % (self._parent_folder, self._molecule_id, gjf_file), mode = "wt") as writefile:
            writefile.write(a_str_gaussian)
          
        return gjf_file
    
    def generate_gjf_marcus_file(self, 
                                 charge_geom: int,
                                 charge_eval: int,
                                 state_geom: int,
                                 state_eval: int,
                                 functional: str, 
                                 basisset: str,
                                 number_of_proc: int,
                                 memory: int,
                                 keyword_line_arguments_input: list[[str, str]] = []) -> str:
        self.print_if_debug('\nIn function Molecule.generate_gjf_marcus_file():')
        functional_basisset = '%s/%s' % (functional, basisset)
        keyword_line_arguments: dict = {'functional_basisset': functional_basisset,
                                        'calculation_type': 'marcus',
                                        'state_TD': state_eval,
                                        'restart': False,
                                        'overwrite_Freq': [[], []],
                                        'overwrite_Opt': [[], []],
                                        'overwrite_NoSymm': [[], []],
                                        'overwrite_state_TD': [[], []]}
        
        # keyword_line_arguements_input = [['calculation_type', 'Opt'], ['restart', True], ['higher_max_cycles', 1024]]
        for i in range(0, len(keyword_line_arguments_input)):
            keyword_line_arguments['%s' % (keyword_line_arguments_input[i][0])] = keyword_line_arguments_input[i][1]
        
        keyword_line = self.keyword_line_generator(functional_basisset = keyword_line_arguments['functional_basisset'],
                                                   calculation_type = keyword_line_arguments['calculation_type'],
                                                   state_TD = keyword_line_arguments['state_TD'],
                                                   restart = keyword_line_arguments['restart'], 
                                                   overwrite_Freq = keyword_line_arguments['overwrite_Freq'],
                                                   overwrite_Opt = keyword_line_arguments['overwrite_Opt'],
                                                   overwrite_NoSymm = keyword_line_arguments['overwrite_NoSymm'],
                                                   overwrite_state_TD = keyword_line_arguments['overwrite_state_TD'])
        
        oldchk_name_extension = '%s_%s_C%sS%s' % (functional, format_basis_set(basisset), charge_geom, state_geom)
        newchk_name_extension = '%s_%s_C%sS%s_opt_geom_at_C%sS%s' % (functional, format_basis_set(basisset), charge_geom, state_geom, charge_eval, state_eval)
        
        oldchk_gjf_file = "%s_%s" % (self._molecule_id, oldchk_name_extension)
        newchk_gjf_file = "%s_%s" % (self._molecule_id, newchk_name_extension)
        
        title_line = '%s %s' % (newchk_gjf_file, keyword_line_arguments['calculation_type'])
        
        a_str_gaussian = (f"%nprocshared={number_of_proc} \n" + 
                          "\n" +
                          f"%mem={memory}GB \n" + 
                          f"%oldchk={oldchk_gjf_file}.chk" + '\n' +
                          f"%chk={newchk_gjf_file}.chk \n" + 
                          keyword_line + "\n" + 
                          "\n" +
                          title_line + "\n"
                          "\n" + 
                          "%s %s\n" % (charge_eval, abs(charge_eval) + 1) + 
                          '\n')
        
        self.print_if_debug('\nGaussian file:\n==============\n%s\n==============' % (a_str_gaussian))
        with open('%s/Data/%s/%s.gjf' % (self._parent_folder, self._molecule_id, newchk_gjf_file), mode = "wt") as writefile:
            writefile.write(a_str_gaussian)
          
        return newchk_gjf_file
    
    def generate_sh_file(self, 
                         timeout_hr: int,
                         number_of_proc: int,
                         memory_gjf_gb: int,
                         charge: int,
                         excited_state: int,
                         functional_basisset: str) -> str:
        """
        Method to write the job script for a Gaussian16 job in the HPC.

        Parameters
        ----------
        timeout_hr : int
            Timeout settings in hours.
        number_of_proc : int
            Number of processors.
        memory_gjf_gb : int
            Total memory allocated for the Gaussian job.
        charge : int
            Charge of the molecule.
        excited_state : int
            Excitation state of the molecule.
        functional_basisset : str
            Functional and basis set in the form {functional}_{basis_set}.

        Returns
        -------
        qsub_script : str
            Submission script.

        """
        self.print_if_debug('\nIn function Molecule.generate_sh_file():')
        timeout = '%sh' % (timeout_hr)
        walltime = '%s:10:00' % (timeout_hr)
        gjf_file = "%s_%s_C%sS%s" % (self._molecule_id, functional_basisset, charge, excited_state)
        
        qsub_script  =  "#!/bin/sh\n"
        qsub_script +=  "#PBS -l walltime=%s\n" % (walltime)
        qsub_script += ("#PBS -l select=1:ncpus=%s" % (int(number_of_proc)) +
                        ":mem=%sGB" % (int(memory_gjf_gb + 8)) +
                        ":avx=true\n\n")
        qsub_script +=  "module load gaussian/g16-c01-avx\n"
        qsub_script +=  "cp $PBS_O_WORKDIR/" + gjf_file + ".gjf ./\n"
        qsub_script +=  "cp $PBS_O_WORKDIR/" + gjf_file + ".chk ./\n"
        qsub_script +=  "timeout %s g16 %s.gjf \n" % (timeout, gjf_file)
        qsub_script +=  "formchk %s.chk %s.fchk\n" % (gjf_file, gjf_file)
        qsub_script +=  "cp *.log  $PBS_O_WORKDIR\ncp *.chk  $PBS_O_WORKDIR\ncp *.fchk  $PBS_O_WORKDIR"
        with open('%s/Data/%s/%s.sh' % (self._parent_folder, self._molecule_id, gjf_file), mode = "wt", newline="\n") as text_file:
            text_file.write(qsub_script)
            text_file.close()
        self.print_if_debug('\nqsub_script:\n==============\n%s\n==============' % (qsub_script))
        return qsub_script
    
    def generate_sh_marcus_file(self, 
                                timeout_hr: int,
                                number_of_proc: int,
                                memory_gjf_gb: int,
                                charge_geom: int,
                                charge_eval: int,
                                state_geom: int,
                                state_eval: int,
                                functional_basisset: str) -> str:
        self.print_if_debug('\nIn function Molecule.generate_sh_marcus_file():')
        timeout = '%sh' % (timeout_hr)
        walltime = '%s:10:00' % (timeout_hr)
        old_gjf_file = "%s_%s_C%sS%s" % (self._molecule_id, functional_basisset, charge_geom, state_geom)
        newchk_name_extension = '%s_C%sS%s_opt_geom_at_C%sS%s' % (functional_basisset, charge_geom, state_geom, charge_eval, state_eval)
        gjf_file = "%s_%s" % (self._molecule_id, newchk_name_extension)
        
        qsub_script  =  "#!/bin/sh\n"
        qsub_script +=  "#PBS -l walltime=%s\n" % (walltime)
        qsub_script += ("#PBS -l select=1:ncpus=%s" % (int(number_of_proc)) +
                        ":mem=%sGB" % (int(memory_gjf_gb + 4)) +
                        ":avx=true\n\n")
        qsub_script +=  "module load gaussian/g16-c01-avx\n"
        qsub_script +=  "cp $PBS_O_WORKDIR/" + old_gjf_file + ".chk ./\n"
        qsub_script +=  "cp $PBS_O_WORKDIR/" + gjf_file + ".gjf ./\n"
        qsub_script +=  "cp $PBS_O_WORKDIR/" + gjf_file + ".chk ./\n"
        qsub_script +=  "timeout %s g16 %s.gjf \n" % (timeout, gjf_file)
        qsub_script +=  "formchk %s.chk %s.fchk\n" % (gjf_file, gjf_file)
        qsub_script +=  "cp *.log  $PBS_O_WORKDIR\ncp *.chk  $PBS_O_WORKDIR\ncp *.fchk  $PBS_O_WORKDIR"
        with open('%s/Data/%s/%s.sh' % (self._parent_folder, self._molecule_id, gjf_file), mode = "wt", newline="\n") as text_file:
            text_file.write(qsub_script)
            text_file.close()
        self.print_if_debug('\nqsub_script:\n==============\n%s\n==============' % (qsub_script))
        return qsub_script
    
    def generate_gjf_sh_files(self, 
                              input_object,
                              excited_state: int,
                              charge: int,
                              functional: str, 
                              basisset: str,
                              number_of_proc: int,
                              memory: int,
                              timeout_hr: int,
                              calculation_type: str = 'OptFreq', 
                              connectivity: bool = True,
                              keyword_line_arguments_input: list[[str, str]] = []):
        """
        Generates both Gaussian16 script and the HPC job script.

        Parameters
        ----------
        input_object : TYPE
            Input object to be fed into Molecule.generate_molecule_xyz().
        excited_state : int
            Excitation state of the molecule.
        charge : int
            Charge of molecule.
        functional : str
            Functional to be used in DFT calculation.
        basisset : str
            Basis set to be used in DFT calculation.
        number_of_proc : int
            Number of processors.
        memory : int
            Total amount of memory to be divided between the processors.
        timeout_hr : int
            Timeout settings in hours.
        calculation_type : str, optional
            Calculation type. The default is 'OptFreq'.
        connectivity : bool, optional
            Whether to use molecule connectivity. The default is True.
        keyword_line_arguments_input : list[[str, str]], optional
            Overwite lines. The default is [].

        Returns
        -------
        None.

        """
        self.print_if_debug('\nIn function Molecule.generate_gjf_sh_files():')
        gjf = self.generate_gjf_file(input_object = input_object, 
                                     excited_state = excited_state, 
                                     charge = charge, 
                                     functional = functional, 
                                     basisset = basisset, 
                                     number_of_proc = number_of_proc, 
                                     memory = memory,
                                     calculation_type = calculation_type,
                                     connectivity = connectivity,
                                     keyword_line_arguments_input = keyword_line_arguments_input)
        sh = self.generate_sh_file(timeout_hr = timeout_hr, 
                                   number_of_proc = number_of_proc, 
                                   memory_gjf_gb = memory, 
                                   charge = charge, 
                                   excited_state = excited_state, 
                                   functional_basisset = '%s_%s' % (functional, format_basis_set(basisset)))
        self.print_if_debug(gjf)
        self.print_if_debug(sh)
        
    def generate_gjf_sh_marcus_files(self, 
                                     charge_geom: int,
                                     charge_eval: int,
                                     state_geom: int,
                                     state_eval: int,
                                     functional: str, 
                                     basisset: str,
                                     number_of_proc: int,
                                     memory: int,
                                     timeout_hr: int,
                                     calculation_type: str = 'marcus', 
                                     keyword_line_arguments_input: list[[str, str]] = []):
        self.print_if_debug('\nIn function Molecule.generate_gjf_sh_marcus_files():')
        gjf = self.generate_gjf_marcus_file(charge_geom = charge_geom,
                                            charge_eval = charge_eval,
                                            state_geom = state_geom,
                                            state_eval = state_eval,
                                            functional = functional,
                                            basisset = basisset,
                                            number_of_proc = number_of_proc,
                                            memory = memory,
                                            keyword_line_arguments_input = keyword_line_arguments_input) 
        sh = self.generate_sh_marcus_file(timeout_hr = timeout_hr,
                                          number_of_proc = number_of_proc,
                                          memory_gjf_gb = memory,
                                          charge_geom = charge_geom,
                                          charge_eval = charge_eval,
                                          state_geom = state_geom,
                                          state_eval = state_eval,
                                          functional_basisset = '%s_%s' % (functional, format_basis_set(basisset)))
        self.print_if_debug(gjf)
        self.print_if_debug(sh)
        
    def generate_gjf_sh_restart_files(self, 
                                      excited_state: int,
                                      charge: int,
                                      functional: str, 
                                      basisset: str,
                                      number_of_proc: int,
                                      memory: int,
                                      timeout_hr: int,
                                      calculation_type: str = 'OptFreq', 
                                      keyword_line_arguments_input: list[[str, str]] = []):
        self.print_if_debug('\nIn function Molecule.generate_gjf_sh_restart_files():')
        gjf = self.generate_gjf_restart_file(excited_state = excited_state, 
                                             charge = charge, 
                                             functional = functional, 
                                             basisset = basisset, 
                                             number_of_proc = number_of_proc, 
                                             memory = memory,
                                             calculation_type = calculation_type,
                                             keyword_line_arguments_input = keyword_line_arguments_input)
        sh = self.generate_sh_file(timeout_hr = timeout_hr, 
                                   number_of_proc = number_of_proc, 
                                   memory_gjf_gb = memory, 
                                   charge = charge, 
                                   excited_state = excited_state, 
                                   functional_basisset = '%s_%s' % (functional, format_basis_set(basisset)))
        self.print_if_debug(gjf)
        self.print_if_debug(sh)

if __name__ == '__main__': # Molecule
    # test_molecule = Molecule(molecule_name = 'testbenzene',
    #                           molecule_id = 'testbenzene_id',
    #                           SMILES = 'C1=CC=CC=C1  ',
    #                           parent_folder = r"C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version",
    #                           debug = True)
    # test_molecule.create_molecule_documentation(allow_aromatic_bond = True)
    # del test_molecule
    # molecule_2 = Molecule.load_molecule(molecule_id = 'benzene_20240331000232',
    #                                     parent_folder = r"C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version",
    #                                     debug = True)
    # # print(molecule_2.get_molecule_name())
    # # print(molecule_2.get_molecule_id())
    # # print(molecule_2.get_SMILES())
    # # print(molecule_2.generate_xyz_from_rdkit_mol()[0])
    # # print(molecule_2.generate_molecule_xyz())
    # # print(molecule_2.generate_molecule_xyz(r"C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version\Data\testbenzene_id\testbenzene_id.xyz"))
    # # print(molecule_2.keyword_line_generator(functional_basisset = 'B3LYP/6-311+G(d,p)',
    # #                                         calculation_type = 'OptFreq'))
    # # print(molecule_2.keyword_line_generator(functional_basisset = 'B3LYP/6-311+G(d,p)',
    # #                                         calculation_type = 'OptFreq',
    # #                                         state_TD = 1))
    # # print(molecule_2.keyword_line_generator(functional_basisset = 'B3LYP/6-311+G(d,p)',
    # #                                         calculation_type = 'OptFreq', 
    # #                                         higher_max_cycles = 1024, 
    # #                                         set_trust_radius_bohr = 0.1))
    # # print(molecule_2.keyword_line_generator(functional_basisset = 'B3LYP/6-311+G(d,p)',
    # #                                         calculation_type = 'OptFreq',
    # #                                         state_TD = 1, 
    # #                                         higher_max_cycles = 1024, 
    # #                                         set_trust_radius_bohr = 0.1))
    # # print(molecule_2.keyword_line_generator(functional_basisset = 'B3LYP/6-311+G(d,p)',
    # #                                         calculation_type = 'OptFreq', 
    # #                                         higher_max_cycles = 1024, 
    # #                                         set_trust_radius_bohr = 0.1, 
    # #                                         restart = True))
    # # print(molecule_2.keyword_line_generator(functional_basisset = 'B3LYP/6-311+G(d,p)',
    # #                                         calculation_type = 'OptFreq',
    # #                                         state_TD = 1, 
    # #                                         higher_max_cycles = 1024, 
    # #                                         set_trust_radius_bohr = 0.1, 
    # #                                         restart = True))
    # print(molecule_2.generate_gjf_file(input_object = '',
    #                                     excited_state = 0,
    #                                     charge = 0,
    #                                     functional = 'B3LYP',
    #                                     basisset = '6-311+G(d,p)',
    #                                     number_of_proc = 8,
    #                                     memory = 32,
    #                                     connectivity = True))
    # # # print(molecule_2.generate_gjf_file(input_object = '',
    # # #                                     excited_state = 0,
    # # #                                     charge = 0,
    # # #                                     functional = 'B3LYP',
    # # #                                     basisset = '6-311+G(d,p)',
    # # #                                     number_of_proc = 8,
    # # #                                     memory = 32, 
    # # #                                     keyword_line_arguments_input = [['restart', True]]))
    # # # print(molecule_2.get_connectivity())
    # # print(molecule_2.generate_sh_file(timeout_hr = 3,
    # #                                   number_of_proc = 8,
    # #                                   memory_gjf_gb = 32,
    # #                                   charge = 0,
    # #                                   excited_state = 0))
    # molecule_2.generate_gjf_sh_files(input_object = r"C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version\Data\testbenzene_id\testbenzene_id_C0S0_B3LYP_6-311plusGdp.log",
    #                                   excited_state = 0,
    #                                   charge = 0,
    #                                   functional = 'B3LYP', 
    #                                   basisset = '6-31G',
    #                                   number_of_proc = 8,
    #                                   memory = 32,
    #                                   timeout_hr = 1,
    #                                   connectivity = True,
    #                                   keyword_line_arguments_input = [])
    # molecule_2.generate_gjf_sh_marcus_files(charge_geom = 1,
    #                                         charge_eval = 0,
    #                                         state_geom = 0,
    #                                         state_eval = 0,
    #                                         functional = 'B3LYP', 
    #                                         basisset = '6-31G',
    #                                         number_of_proc = 8,
    #                                         memory = 12,
    #                                         timeout_hr = 1)
    pass
#%%

class MongoDB():
    
    def __init__(self,
                 username: str,
                 password: str,
                 collection: str,
                 database: str = 'New_Reorganisation_Energy_Calc',
                 connection_string: str = 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
                 debug: bool = False):
        """
        Establishing connection to a MongoDB database.

        Parameters
        ----------
        username : str
            Username used to access that database.
        password : str
            Password correcponding to username.
        collection : str
            Name of the collection to be accessed.
        database : str, optional
            Name of the database to be accessed.. The default is 'Reorganisation_Energy_Calculation'.
        connection_string : str, optional
            Connection string to access the database. The default is 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'.
        debug : bool, optional
            Toggle debug mode. The default is False.

        Returns
        -------
        None.

        """
        self._username = username
        self._password = password
        self._collection = collection
        self._database = database
        self._connection_string = connection_string
        self._debug = debug
        
    def __repr__(self):
        """
        Official representation

        Returns
        -------
        string
            Official representation of class.
        """
        return '\nMongoDB connection:\n Username = %s\n Password = %s\n Database = %s\n Collection = %s\n Connection string = %s' % (self._username, self._password, self._database, self._collection, self._connection_string)
    
    def __str__(self):
        """
        Simplified string representation.

        Returns
        -------
        string
            String representation of class.

        """
        return '\n%s\n%s\n%s\n%s\n%s' % (self._username, self._password, self._database, self._collection, self._connection_string)
        
    def connect_mongo(self):
        """
        Connects to MongoDB and retyurns the database.

        Returns
        -------
        database : MongoDB Database
            Full database.

        """
        if self._debug:
            print('\nIn function MongoDB.connect_mongo():')
            print('Username: %s\nPassword: %s\nDatabase: %s\nConnection string: %s' % (self._username, self._password, self._collection, self._connection_string))
        client = MongoClient(self._connection_string.replace('<user>', self._username).replace('<password>', urllib.parse.quote_plus(self._password)))
        database = client[self._database]
        return database
    
    def pull(self):
        """
        Method to pull the collection from the online MongoDB database.

        Returns
        -------
        df : pandas.core.frame.DataFrame
            Dataframe of all the entries in the collection.

        """
        if self._debug:
            print('\nIn function MongoDB.pull():')
        database = self.connect_mongo()['%s' % (self._collection)].find()
        df =  pd.DataFrame(list(database)).iloc[:, :]
        return df
    
    def create_post(self, keys: list, values: list) -> dict:
        """
        Takes in keys and values in list/array form and organise it into a post for pushing onto MongoDB database.

        Parameters
        ----------
        keys : list
            List of keys in entry.
        values : list
            List of values corresonding to the keys.

        Raises
        ------
        ValueError
            Length of keys and values does not match.

        Returns
        -------
        dict
            A MongoDB post that can be pushed onto the collection.

        """
        if self._debug:
            print('\nIn function MongoDB.create_post():')
            print('keys: %s\nvalues: %s' % (keys, values))
            print('length (keys): %s\nlength (values): %s' % (len(keys), len(values)))
        if len(keys) != len(values):
            raise ValueError('Length of input is not the same:\n keys = %s\n values = %s' % ((len(keys)), len(values)))
        post: dict = {}
        for i in range(0, len(keys)):
            post[keys[i]] = values[i]
        return post
    
    def push_one_new_entry(self, keys: list, values: list):
        """
        Method to create a new entry on the online collection.

        Parameters
        ----------
        keys : list
            List of keys.
        values : list
            List of values corresponding to above keys.

        Raises
        ------
        Exception
            Push unsucessfull.

        Returns
        -------
        None.

        """
        post: dict = self.create_post(keys = keys, values = values)
        if self._debug:
            print('\nIn function MongoDB.push_one_entry():')
            pprint.pprint(post)
        try:
            self.connect_mongo()['%s' % (self._collection)].insert_one(post)
        except:
            raise Exception('\nPush failed.')
            
    def replace_one_entry(self, keys: list, values: list):
        """
        Method to replace one entry of the online collection.

        Parameters
        ----------
        keys : list
            List of keys.
        values : list
            List of values corresponding to above keys.

        Raises
        ------
        Exception
            Push unsucessfull.

        Returns
        -------
        None.

        """
        post: dict = self.create_post(keys = keys, values = values)
        if self._debug:
            print('\nIn function MongoDB.replace_one_entry():')
            pprint.pprint(post)
        try:
            self.connect_mongo()['%s' % (self._collection)].replace_one({'_id':post['_id']}, post)
        except:
            raise Exception('\nReplace failed.')
            
    def update_one_entry(self, _id: str, keys: list, values: list):
        """
        Method to only edit one entry on the MongoDB database.

        Parameters
        ----------
        _id : str
            ID of the entry.
        keys : list
            Keys to be changed.
        values : list
            The corresponding values to the keys.

        Raises
        ------
        Exception
            Update failed.

        Returns
        -------
        None.

        """
        post: dict = self.create_post(keys = keys, values = values)
        if self._debug:
            print('\nIn function MongoDB.update_one_entry():')
        try:
            self.connect_mongo()['%s' % (self._collection)].update_one({'_id': _id}, {'$set': post})
        except:
            raise Exception('\nUpdate failed.')
            
    def export_to_excel(self, path: str = ''):
        """
        Method to export the entire online collection to an Excel file saved locally.

        Parameters
        ----------
        path : str, optional
            Path to write the Excel file. The default is '': the current working directory would be used.

        Raises
        ------
        Exception
            Failed operation.

        Returns
        -------
        None.

        """
        if path == '':
            path = '%s/%s.xlsx' % (os.getcwd(), self._collection)
        dataframe = self.pull()
        if self._debug:
            print('\nIn function MongoDB.export_to_excel():')
            print('Export to location:\n%s' % (path))
            pprint.pprint(dataframe)
        try:
            dataframe.to_excel(r'%s' % (path))
        except:
            raise Exception('\nExport to excel not sucessfull.')

if __name__ == '__main__': # MongoDB
    # test = MongoDB(username = 'kbc121', password = 'Imperi@l020711', collection = 'SMILES_Database', debug = True)
    # test = MongoDB(username = 'kbc121', password = 'Imperi@l020711', collection = 'Tracker2', debug = True)
    # test = MongoDB(username = 'kbc121', password = 'Imperi@l020711', debug = True)
    # test.push_one_new_entry(keys = ['_id', 'name', 'age'], values = ['0008', 'kye', '7'])
    # test.replace_one_entry(keys = ['_id', 'name', 'age'], values = ['0003', 'melanie', '6.5'])
    # test.export_to_excel()
    # print(test.pull())
    pass

#%%

class HPC():
    
    def __init__(self, 
                 username: str,
                 password: str,
                 hostname: str = 'login.hpc.ic.ac.uk',
                 debug: bool = False):
        """
        A class to connect to the HPC and to submit operations.

        Parameters
        ----------
        username : str
            Username for the connection.
        password : str
            Password correcponding to the username.
        hostname : str, optional
            Hostname of the HPC. The default is 'login.hpc.ic.ac.uk'.
        debug : bool, optional
            Toggle for debugging. The default is False.

        Returns
        -------
        None.

        """
        self._username = username
        self._password = password
        self._hostname = hostname
        self._debug = debug
        
    def print_if_debug(self, lines: str):
        """
        Print lines if in debug mode

        Parameters
        ----------
        lines : str
            Lines to be printed.

        Returns
        -------
        None.

        """
        if self._debug:
            print(lines)
    
    def qstat(self) -> str:
        """
        Query to return the queue status under the username.

        Returns
        -------
        q_status : str
            String containing the queue status.

        """
        self.print_if_debug('\nIn function HPC.qstat():')
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.WarningPolicy)
            client.connect(hostname = self._hostname, 
                           username = self._username, 
                           password = self._password)
            stdin, stdout, stderr = client.exec_command("/opt/pbs/bin/qstat ")
            
            q_status = stdout.read()
            
            self.print_if_debug('Standard output:\n%s' % (stdout.read()))
            self.print_if_debug('Standard error:\n%s' % (stderr.read()))

            if stdout.channel.recv_exit_status() != 0:
                print('\nError encountered:')
                print("Standard error:\n%s" % (stderr.read()))
                print("Exit status: {}".format(stdout.channel.recv_exit_status()))
            client.close()
        q_status = str(q_status)[2:-1].replace('\\n', '\n')
        return q_status
    
    def formatted_qstat(self) -> pd.core.frame.DataFrame:
        """
        Method to return the queue status as a pandas dataframe.

        Returns
        -------
        formatted_q_status : pd.core.frame.DataFrame
            Queue status.

        """
        self.print_if_debug('\nIn function HPC.formatted_qstat():')
        q_status = self.qstat()
        self.print_if_debug('%s' % (q_status))
        q_status = q_status.split('\n')
        q_status = q_status[2:-1]
        formatted_q_status = pd.DataFrame(columns = ['Job ID', 'Class', 'Job Name', 'Status', 'Comment'])
        for i in range(0, len(q_status)):
            job_id = q_status[i][0:14].replace(' ', '')
            job_class = q_status[i][15:30].replace(' ', '')
            job_name = q_status[i][31:51].replace(' ', '')
            job_status = q_status[i][52:59].replace(' ', '')
            job_comment = q_status[i][60:]
            formatted_q_status.loc[len(formatted_q_status)] = [job_id, job_class, job_name, job_status, job_comment]
        return formatted_q_status
    
    def transfer_from_HPC(self, 
                          local_folder: str, 
                          remote_folder: str, 
                          filename: str):
        """
        Method to transfer files from the HPC to local device.

        Parameters
        ----------
        local_folder : str
            Path to local folder where the file would be saved.
        remote_folder : str
            Path to remote folder where the file is located.
        filename : str
            Filename of the file to be copied.

        Returns
        -------
        None.

        """
        self.print_if_debug('\nIn function HPC.transfer_from_HPC():')
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.WarningPolicy)
            client.connect(hostname = self._hostname, 
                           username = self._username, 
                           password = self._password)
            sftp_client = client.open_sftp()
            sftp_client.get(r'%s/%s' % (remote_folder, filename), 
                            r'%s/%s' % (local_folder, filename))
            sftp_client.close()
            client.close()
        self.print_if_debug('\nTransfer sucessfull.')
            
    def submit_job_gaussian(self,
                            local_folder: str,
                            remote_folder: str,
                            filename: str):
        """
        Method to submit a Gaussian16 job in the HPC. 
        The required scripts (.gjf and .sh) should be located in the local folder.

        Parameters
        ----------
        local_folder : str
            Local folder where the .gjf and .sh job scripts are located.
        remote_folder : str
            Remote folder where scripts will be transferred to.
        filename : str
            Full filename of the scripts. Example {filename}.sh and {filename}.gjf.

        Returns
        -------
        None.

        """
        self.print_if_debug('\nIn function HPC.submit_job_gaussian():')
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.WarningPolicy)

            # Establish SSH connection
            client.connect(hostname = self._hostname, 
                           username = self._username, 
                           password = self._password)

            # Establish SFTP connection ftp_client=ssh.open_sftp()
            stdin, stdout, stderr = client.exec_command("mkdir -p " + remote_folder)
            with client.open_sftp() as sftp0:
                # Push job submission script to a particular path on the cluster
                sftp0.put(r'%s/%s.sh' % (local_folder, filename), r'%s/%s.sh' % (remote_folder, filename))
                sftp0.put(r'%s/%s.gjf' % (local_folder, filename), r'%s/%s.gjf' % (remote_folder, filename))
            # Submit our Grid Engine job by running a remote 'qsub' command over SSH
            stdin, stdout, stderr = client.exec_command(f"cd {remote_folder} \n " + 
                                                        f"chmod +rwx {filename}.sh\n" + 
                                                        f"/opt/pbs/bin/qsub {filename}.sh \n")
            # Show the standard output and error of our job
            stdout_copy = deepcopy(stdout.read())
            self.print_if_debug('\nStandard output:\n%s' % (stdout_copy))
            self.print_if_debug('\nStandard error:\n%s' % (stderr.read()))
            print(stdout.channel.recv_exit_status())
            if stdout.channel.recv_exit_status() != 0:
                print('\nError encountered:')
                print("\nStandard error:\n%s" % (stderr.read()))
                print("Exit status: {}".format(stdout.channel.recv_exit_status()))
            client.close()
        job_id = int(''.join(filter(str.isdigit, str(stdout_copy))))
        return job_id
    
    def restart_job_gaussian(self,
                             local_folder: str,
                             remote_folder: str,
                             filename: str):
        self.print_if_debug('\nIn function HPC.restart_job_gaussian():')
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.WarningPolicy)

            # Establish SSH connection
            client.connect(hostname = self._hostname, 
                           username = self._username, 
                           password = self._password)

            # Establish SFTP connection ftp_client=ssh.open_sftp()
            stdin, stdout, stderr = client.exec_command("mkdir -p " + remote_folder)
            with client.open_sftp() as sftp0:
                # Push job submission script to a particular path on the cluster
                sftp0.put(r'%s/%s.sh' % (local_folder, filename), r'%s/%s.sh' % (remote_folder, filename))
                sftp0.put(r'%s/%s.gjf' % (local_folder, filename), r'%s/%s.gjf' % (remote_folder, filename))
                sftp0.put(r'%s/%s.chk' % (local_folder, filename), r'%s/%s.chk' % (remote_folder, filename))
            # Submit our Grid Engine job by running a remote 'qsub' command over SSH
            stdin, stdout, stderr = client.exec_command(f"cd {remote_folder} \n " + 
                                                        f"chmod +rwx {filename}.sh\n" + 
                                                        f"/opt/pbs/bin/qsub {filename}.sh \n")
            # Show the standard output and error of our job
            stdout_copy = deepcopy(stdout.read())
            self.print_if_debug('\nStandard output:\n%s' % (stdout_copy))
            self.print_if_debug('\nStandard error:\n%s' % (stderr.read()))
            print(stdout.channel.recv_exit_status())
            if stdout.channel.recv_exit_status() != 0:
                print('\nError encountered:')
                print("\nStandard error:\n%s" % (stderr.read()))
                print("Exit status: {}".format(stdout.channel.recv_exit_status()))
            client.close()
        job_id = int(''.join(filter(str.isdigit, str(stdout_copy))))
        return job_id
            
    def transfer_chk_fchk_log_files_from_HPC(self,
                                             local_folder: str,
                                             remote_folder: str,
                                             filename: str):
        """
        Transfer the calculation files from the HPC to local folder.

        Parameters
        ----------
        local_folder : str
            Local folder where the files will be saved.
        remote_folder : str
            Remote folder where the files will be transferred from.
        filename : str
            Full filename of the files. Example {filename}.fchk, {filename}.chk, and {filename}.log.

        Returns
        -------
        None.

        """
        self.print_if_debug('\nIn function HPC.transfer_chk_fchk_log_files_from_HPC():')
        try:
            self.print_if_debug('Transferring log files from HPC:\nlocal folder = %s\nremote folder = %s\nfile = %s.log' % (local_folder, remote_folder, filename))
            self.transfer_from_HPC(local_folder = local_folder, 
                                   remote_folder = remote_folder, 
                                   filename = r'%s.log' % (filename))
        except:
            print('\nFailed to transfer log files.')
        try:
            self.print_if_debug('Transferring chk files from HPC:\nlocal folder = %s\nremote folder = %s\nfile = %s.chk' % (local_folder, remote_folder, filename))
            self.transfer_from_HPC(local_folder = local_folder, 
                                   remote_folder = remote_folder, 
                                   filename = r'%s.chk' % (filename))
        except:
            print('\nFailed to transfer chk files.')
        try:
            self.print_if_debug('Transferring fchk files from HPC:\nlocal folder = %s\nremote folder = %s\nfile = %s.fchk' % (local_folder, remote_folder, filename))
            self.transfer_from_HPC(local_folder = local_folder, 
                                   remote_folder = remote_folder, 
                                   filename = r'%s.fchk' % (filename))
        except:
            print('\nFailed to transfer fchk files.')
    
    def copy_fchk_log_files_for_dushin(self,
                                       remote_dushin_folder: str,
                                       remote_log_folder: str,
                                       filename_initial_state:str,
                                       filename_final_state: str):
        """
        Transfer .fchk files and .log files between two folders in the HPC.

        Parameters
        ----------
        remote_dushin_folder : str
            Folder where DUSHIN would be ran.
        remote_log_folder : str
            Folder where the fchk and log files are saved.
        filename_initial_state : str
            Filename of the initial state calculation, e.g. in the form {filename_initial_state}.log.
        filename_final_state : str
            Filename of the final state calculation, e.g. in the form {filename_final_state}.log.

        Returns
        -------
        None.

        """
        self.print_if_debug('\nIn function HPC.copy_fchk_log_files_for_dushin():')
        
        self.print_if_debug('Copying fchk files for:\n %s.fchk and\n %s.fchk' % (filename_initial_state, filename_final_state))
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.WarningPolicy)
            client.connect(hostname = self._hostname, 
                           username = self._username, 
                           password = self._password)
            stdin, stdout, stderr = client.exec_command(f"cd {remote_log_folder} \n " + 
                                                        f"cp {filename_initial_state}.fchk {remote_dushin_folder} \n" + 
                                                        f"cp {filename_final_state}.fchk {remote_dushin_folder} \n")
            # Show the standard output and error of our job
            self.print_if_debug('Standard output:\n%s' % (stdout.read()))
            self.print_if_debug('Standard error:\n%s' % (stderr.read()))
            if stdout.channel.recv_exit_status() != 0:
                print('\nError encountered:')
                print("Standard error:\n%s" % (stderr.read()))
                print("Exit status: {}".format(stdout.channel.recv_exit_status()))
            client.close()
        self.print_if_debug('fchk files copied sucessfully')
        
        self.print_if_debug('Copying log files for:\n %s.log and\n %s.log' % (filename_initial_state, filename_final_state))
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.WarningPolicy)
            client.connect(hostname = self._hostname, 
                           username = self._username, 
                           password = self._password)
            stdin, stdout, stderr = client.exec_command(f"cd {remote_log_folder} \n " + 
                                                        f"cp {filename_initial_state}.log {remote_dushin_folder} \n" + 
                                                        f"cp {filename_final_state}.log {remote_dushin_folder} \n")
            # Show the standard output and error of our job
            self.print_if_debug('Standard output:\n%s' % (stdout.read()))
            self.print_if_debug('Standard error:\n%s' % (stderr.read()))
            if stdout.channel.recv_exit_status() != 0:
                print('\nError encountered:')
                print("Standard error:\n%s" % (stderr.read()))
                print("Exit status: {}".format(stdout.channel.recv_exit_status()))
            client.close()
        self.print_if_debug('log files copied sucessfully')
        
    def submit_job_dushin(self, 
                          remote_dushin_folder: str,
                          remote_bin_path: str,
                          initial_state: str,
                          final_state: str,
                          mol_id_functional_basisset: str):
        self.print_if_debug('\nIn function HPC.submit_job_dushin():')
        filename_initial_state = '%s_%s' % (mol_id_functional_basisset, initial_state)
        filename_final_state = '%s_%s' % (mol_id_functional_basisset, final_state)
        self.print_if_debug('\nDushin started for %s %s %s.' % (mol_id_functional_basisset, initial_state, final_state))
        with paramiko.SSHClient() as client:
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.WarningPolicy)
            client.connect(hostname = self._hostname, 
                           username = self._username, 
                           password = self._password)

            stdin, stdout, stderr = client.exec_command(
                f"cd {remote_dushin_folder} \n " + 
                f"export PATH=$PATH:{remote_bin_path} \n" + 
                f"find -type f -name '*.sh.*' -delete \n" + 
                f"cd {remote_dushin_folder} \n" + 
                
                f"split_log.sh {filename_initial_state}.log \n" + 
                f"mv {filename_initial_state}.log org_{filename_initial_state}.log\n" + 
                f"mv freq_1_{filename_initial_state}.log {filename_initial_state}.log \n" + 
                
                f"split_log.sh {filename_final_state}.log \n" + 
                f"mv {filename_final_state}.log org_{filename_final_state}.log\n" + 
                f"mv freq_1_{filename_final_state}.log {filename_final_state}.log \n" + 
                
                f"find -type f -name '*freq_2*' -delete \n" + 
                f"rundushin.sh {mol_id_functional_basisset} {initial_state} {final_state}")
            
            if stdout.channel.recv_exit_status() != 0:
                print('\nError encountered:')
                print("\nStandard error:\n%s" % (stderr.read()))
                print("Exit status: {}".format(stdout.channel.recv_exit_status()))
            client.close()
            self.print_if_debug('DUSHIN submission sucessful')

if __name__ == '__main__': # HPC
    # HPC = HPC(username = 'kbc121', password = 'Imperi@l020711', hostname = 'login.hpc.ic.ac.uk', debug = True)
    # # # print(HPC.formatted_qstat())
    # job_id = HPC.submit_job_gaussian(local_folder = r"C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version\Data\benzene_20240320003534", 
    #                         remote_folder = r'/rds/general/ephemeral/user/kbc121/ephemeral/test',
    #                         filename = 'benzene_20240320003534_B3LYP_STO-3G_C0S0')
    # # print(HPC.formatted_qstat())
    # # HPC.transfer_from_HPC(local_folder = r'C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version\Data\testbenzene_id', 
    # #                       remote_folder = r'/rds/general/ephemeral/user/kbc121/ephemeral/test', 
    # #                       filename = 'testbenzene_id_C0S0.log')
    # HPC.transfer_chk_fchk_log_files_from_HPC(local_folder = r'C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version\Data\testbenzene_id',
    #                                          remote_folder = r'/rds/general/ephemeral/user/kbc121/ephemeral/test',
    #                                          filename = 'testbenzene_id_C0S0_B3LYP_6-311plusGdp')
    pass

#%%

class SMILES_database():
    
    def __init__(self,
                 username: str,
                 password: str,
                 collection: str = 'SMILES_Database',
                 database: str = 'New_Reorganisation_Energy_Calc',
                 connection_string: str = 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
                 debug: bool = False):
        """
        Database of SMILES stored through a MongoDB connection.

        Parameters
        ----------
        username : str
            Username to connect to MongoDB databsse.
        password : str
            Password to connect to MongoDB database.
        collection : str, optional
            Name of the collection on the database. The default is 'SMILES_Database'.
        database : str, optional
            Name of the database. The default is 'New_Reorganisation_Energy_Calc'.
        connection_string : str, optional
            Connection string to connect to MongoDB. The default is 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'.
        debug : bool, optional
            Toggle for debugging. The default is False.

        Returns
        -------
        None.

        """
        self._username = username
        self._password = password
        self._collection = collection
        self._database = database
        self._connection_string = connection_string
        self._debug = debug
        self._MongoDB = MongoDB(username = self._username,
                                password = self._password,
                                collection = self._collection,
                                database = self._database,
                                connection_string = self._connection_string,
                                debug = self._debug)
        
    def __repr__(self):
        """
        Official representation

        Returns
        -------
        string
            Official representation of class.
        """
        return '\nMongoDB connection:\n Username = %s\n Password = %s\n Database = %s\n Collection = %s\n Connection string = %s' % (self._username, self._password, self._database, self._collection, self._connection_string)
    
    def __str__(self):
        """
        Simplified string representation.

        Returns
        -------
        string
            String representation of class.

        """
        return '\n%s\n%s\n%s\n%s\n%s' % (self._username, self._password, self._database, self._collection, self._connection_string)
        
    def pull_SMILES_database(self):
        """
        Get the SMILES database as a pandas dataframe.

        Raises
        ------
        Exception
            Failed to fetch SMILES database.

        Returns
        -------
        dataframe : pandas.core.frame.DataFrame
            SMILES database as a pandas dataframe.

        """
        if self._debug:
            print('\nIn function SMILES_database.pull_SMILES_database():')
        try:
            dataframe = self._MongoDB.pull()
            return dataframe
        except:
            raise Exception('Failed to fetch SMILES database.')
    
    def export_to_excel(self, path: str = ''):
        """
        Downloads a copy of the SMILES database as an .xlsx file.

        Parameters
        ----------
        path : str, optional
            Full path of the folder where the file would be saved. The default is ''.

        Raises
        ------
        Exception
            Export to excel not sucessfull..

        Returns
        -------
        None.

        """
        if path == '':
            path = '%s/SMILES_database.xlsx' % (os.getcwd())
        dataframe = self.pull_SMILES_database()
        if self._debug:
            print('\nIn function SMILES_database.export_to_excel():')
            print('Export to location:\n%s' % (path))
            pprint.pprint(dataframe)
        try:
            dataframe.to_excel(r'%s' % (path))
        except:
            raise Exception('\nExport to excel not sucessfull.')
        pass
            
    def check_against_database(self, iupac_name: str, SMILES: str = '') -> list[str, str]:
        """
        Check the molecule name and SMILES against database. If both molecule name and SMILES are declared, checks for a match.
        If only the Name is declared, checks whether the molecule is in the database, and returns the corresponding SMILES.

        Parameters
        ----------
        iupac_name : str
            Molecule name.
        SMILES : str, optional
            SMILES string. The default is ''.

        Raises
        ------
        Exception
            SMILES string not declared and not found in database.
        Exception
            Input SMILES does not match database.

        Returns
        -------
        list[str, str]
            The molecule name and SMILES string if no error is encountered.

        """
        if self._debug:
            print('\nIn function SMILES_database.in_database():')
            print('SMILES: %s\niupac_name: %s' % (SMILES, iupac_name))
        dataframe = self.pull_SMILES_database()
        # if SMILES not declared, ...
        if SMILES == '':
            # ... try looking for SMILES in database, ...
            try:
                SMILES = dataframe.at[dataframe[dataframe['_id'] == iupac_name].index[0], 'SMILES']
                print('Molecule found in database.')
                return iupac_name, SMILES
            # ... if not found in database, raise exception
            except:
                raise Exception('SMILES string not declared and not found in database.')
        # if SMILES declared, ...
        else:
            try:
                # ... compare SMILES to database, ...
                SMILES_found_in_database = dataframe.at[dataframe[dataframe['_id'] == iupac_name].index[0], 'SMILES']
                print('Molecule found in database.')
            except:
                SMILES_found_in_database = 'not found'
                print('Molecule not found in database.')
            # ... if SMILES from input and database does not match, raise exception, ...
            if SMILES_found_in_database != SMILES and SMILES_found_in_database != 'not found':
                raise Exception('Input SMILES does not match database:\nInput: %s\nDatabase: %s' % (SMILES, SMILES_found_in_database))
            # otherwise if SMILES not found in database, append and write to database
            elif SMILES_found_in_database == 'not found':
                self._MongoDB.push_one_new_entry(keys = ['_id', 'SMILES'], values = [iupac_name, SMILES])
                print('Molecule added to database:\nMolecule Name: %s\nSMILES: %s' % (iupac_name, SMILES))
                return iupac_name, SMILES
            else:
                print('Molecule matched with database')
                return iupac_name, SMILES
        
if __name__ == '__main__': # SMILES_database
    # SMILES_DB = SMILES_database(username = 'kbc121', password = 'Imperi@l020711', debug = True)
    # print(SMILES_DB.check_against_database(iupac_name = 'benzene', SMILES = 'C1=CC=CC=C1'))
    # print(SMILES_DB.check_against_database(iupac_name = 'benzene', SMILES = ''))
    # SMILES_DB.export_to_excel(path = '')
    # print(SMILES_DB.pull_SMILES_database())
    pass
#%%

class Funnel():
    
    def __init__(self,
                 path: str = r'%s\Funnel.xlsx' % (os.getcwd()),
                 debug: bool = False):
        self._funnel = pd.DataFrame(columns = ['MMFFO', 'Functional', 'Basis Set 1', 'Basis Set 2', 'Basis Set 3'])
        self._path = path
        self._debug = debug
        
    def retrieve(self):
        return self._funnel
        
    def add_workflow(self, 
                     MMFFO: str = 'Yes',
                     functional: str = 'B3LYP',
                     basis_set_1: str = '6-31G(d)',
                     basis_set_2: str = '6-311+G(d,p)',
                     basis_set_3: str = 'No'):
        if self._debug:
            print('\nIn function Funnel.add_workflow():')
            print('MMFFO = %s\nFunctional = %s\nBasis Set 1 = %s\nBasis Set 2 = %s\nBasis Set 3 = %s' % (MMFFO, functional, basis_set_1, basis_set_2, basis_set_3))
        self._funnel.loc[len(self._funnel)] = [MMFFO, functional, basis_set_1, basis_set_2, basis_set_3]
    
    def workflow_indices(self):
        return list(self._funnel.index)
    
    def MMFFO(self, index):
        return self._funnel.at[index, 'MMFFO']
    
    def functional(self, index):
        return self._funnel.at[index, 'Functional']
    
    def basis_set_1(self, index):
        return self._funnel.at[index, 'Basis Set 1']
    
    def basis_set_2(self, index):
        return self._funnel.at[index, 'Basis Set 2']
    
    def basis_set_3(self, index):
        return self._funnel.at[index, 'Basis Set 3']
    
    def export_to_excel(self, path: str = ''):
        if path == '':
            path = self._path
        dataframe = self.retrieve()
        if self._debug:
            print('\nIn function Funnel.export_to_excel():')
            print('Export to location:\n%s' % (path))
            pprint.pprint(dataframe)
        try:
            dataframe.to_excel(r'%s' % (path))
        except:
            raise Exception('\nExport to excel not sucessfull.')
    
    def load_from_excel(self, path: str = ''):
        if path == '':
            path = self._path
        if self._debug:
            print('\nIn function Funnel.load_from_excel():')
            print('Load from location:\n%s' % (path))
        try:
            self._funnel = pd.read_excel(path)[['MMFFO', 'Functional', 'Basis Set 1', 'Basis Set 2', 'Basis Set 3']]
        except:
            raise Exception('Read unsucessfull, unable to load from folder %s' % (path))
        
        
if __name__ == '__main__': # Funnel
    # test = Funnel(debug = True)
    # test.add_workflow()
    # test.add_workflow(MMFFO = 'Yes', functional = 'CAM-B3LYP', basis_set_1 = 'cc-pVDZ', basis_set_2 = 'AUG-cc-pVTZ')
    # print(test.retrieve())
    # print(test.MMFFO(0))
    # print(test.functional(0))
    # print(test.functional(1))
    # print(test.basis_set_2(0))
    # print(test.basis_set_3(1))
    # print(test.workflow_indices())
    # test.export_to_excel()
    # test.load_from_excel()
    # print(test.retrieve())
    pass
#%%

class Tracker():
    
    def __init__(self,
                 username: str,
                 password: str,
                 collection: str = 'Tracker',
                 database: str = 'New_Reorganisation_Energy_Calc',
                 connection_string: str = 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
                 debug: bool = False):
        self._username = username
        self._password = password
        self._collection = collection
        self._database = database
        self._connection_string = connection_string
        self._debug = debug
        
        self._MongoDB = MongoDB(username = self._username,
                                password = self._password,
                                collection = self._collection,
                                database = self._database,
                                connection_string = self._connection_string,
                                debug = self._debug)
        
    def print_if_debug(self, line):
        if self._debug == True:
            print(line)
    
    def pull_tracker(self):
        self.print_if_debug('\nIn function Tracker.pull_tracker():')
        try:
            dataframe = self._MongoDB.pull()
            return dataframe
        except:
            raise Exception('Failed to fetch tracker database.')
    
    def export_to_excel(self, parent_folder: str = ''):
        if parent_folder == '':
            parent_folder = r'%s' % (os.getcwd())
        
        dataframe = self.pull_tracker()
        self.print_if_debug('\nIn function Tracker.export_to_excel():')
        self.print_if_debug('Export to location:\n%s' % (parent_folder))
        self.print_if_debug(dataframe)
        try:
            dataframe.to_excel(r'%s/Tracker.xlsx' % (parent_folder))
        except:
            raise Exception('\nExport to excel not sucessfull.')
        
    def add_entry(self,
                  iupac_name: str,
                  SMILES: str,
                  MMFFO: str,
                  functional: str,
                  basis_set_1: str,
                  basis_set_2: str,
                  basis_set_3: str):
        self.print_if_debug('\nIn function Tracker.add_entry():')
        dataframe = self.pull_tracker()
        duplicates = dataframe.loc[((dataframe['IUPAC Name'] == iupac_name) &
                                    (dataframe['SMILES'] == SMILES) &
                                    (dataframe['MMFFO'] == MMFFO) &
                                    (dataframe['Functional'] == functional) &
                                    (dataframe['Basis Set 1'] == basis_set_1) &
                                    (dataframe['Basis Set 2'] == basis_set_2) &
                                    (dataframe['Basis Set 3'] == basis_set_3))]
        molecule_id = tag(iupac_name)
        
        def create_empty_entry(molecule_id, 
                               iupac_name, 
                               SMILES, 
                               MMFFO,
                               functional, 
                               basis_set_1, 
                               basis_set_2, 
                               basis_set_3,
                               username):
            self._MongoDB.push_one_new_entry(keys = ['_id', 'IUPAC Name', 'SMILES', 'MMFFO',
                                                     'Functional', 'Basis Set 1', 'Basis Set 2', 'Basis Set 3',
                                                     'C0S0: BS1', 'C0S0: BS2', 'C0S0: BS3',
                                                     'C0S1: BS1', 'C0S1: BS2', 'C0S1: BS3', 
                                                     'C0S1: BS1 - C0S0 geometry', 'C0S1: BS2 - C0S0 geometry', 'C0S1: BS3 - C0S0 geometry', 'C0S0: BS1 - C0S1 geometry', 'C0S0: BS2 - C0S1 geometry', 'C0S0: BS3 - C0S1 geometry',
                                                     'C1S0: BS1', 'C1S0: BS2', 'C1S0: BS3', 
                                                     'C1S0: BS1 - C0S0 geometry', 'C1S0: BS2 - C0S0 geometry', 'C1S0: BS3 - C0S0 geometry', 'C0S0: BS1 - C1S0 geometry', 'C0S0: BS2 - C1S0 geometry', 'C0S0: BS3 - C1S0 geometry',
                                                     'C-1S0: BS1', 'C-1S0: BS2', 'C-1S0: BS3', 
                                                     'C-1S0: BS1 - C0S0 geometry', 'C-1S0: BS2 - C0S0 geometry', 'C-1S0: BS3 - C0S0 geometry', 'C0S0: BS1 - C-1S0 geometry', 'C0S0: BS2 - C-1S0 geometry', 'C0S0: BS3 - C-1S0 geometry',
                                                     'User', 'Comment', 'Activity Log', 'Last Updated'],
                                             values = [molecule_id, iupac_name, SMILES, MMFFO,
                                                       functional, basis_set_1, basis_set_2, basis_set_3,
                                                       'Not Started', 'Not Started', 'Not Started',
                                                       'Not Started', 'Not Started', 'Not Started', 
                                                       'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started',
                                                       'Not Started', 'Not Started', 'Not Started',
                                                       'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started',
                                                       'Not Started', 'Not Started', 'Not Started', 
                                                       'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started',
                                                       username, 'No comment', 'Entry created on %s' % (str(datetime.now()).split('.')[0]), timestamp()]
                                             )
            self.print_if_debug('\nEntry sucessfully created.')
            # self._MongoDB.push_one_new_entry(keys = ['_id', 'IUPAC Name', 'SMILES', 'MMFFO',
            #                                          'Functional', 'Basis Set 1', 'Basis Set 2', 'Basis Set 3',
            #                                          'C0S0: BS1', 'C0S0: BS2', 'C0S0: BS3',
            #                                          'C0S1: BS1', 'C0S1: BS2', 'C0S1: BS3', 'C0S1-BS1: DUSHIN', 'C0S1-BS2: DUSHIN', 'C0S1-BS3: DUSHIN',
            #                                          'C1S0: BS1', 'C1S0: BS2', 'C1S0: BS3', 'C1S0-BS1: DUSHIN', 'C1S0-BS2: DUSHIN', 'C1S0-BS3: DUSHIN',
            #                                          'C-1S0: BS1', 'C-1S0: BS2', 'C-1S0: BS3', 'C-1S0-BS1: DUSHIN', 'C-1S0-BS2: DUSHIN', 'C-1S0-BS3: DUSHIN',
            #                                          'User', 'Comment', 'Activity Log', 'Last Updated'],
            #                                  values = [molecule_id, iupac_name, SMILES, MMFFO,
            #                                            functional, basis_set_1, basis_set_2, basis_set_3,
            #                                            'Not Started', 'Not Started', 'Not Started',
            #                                            'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started',
            #                                            'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started',
            #                                            'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started', 'Not Started',
            #                                            self._username, 'No comment', 'Entry created on %s' % (str(datetime.now()).split('.')[0]), timestamp()]
            #                                  )
            # self.print_if_debug('\nEntry sucessfully created.')
            
        if len(duplicates) == 0:
            create_empty_entry(molecule_id = molecule_id, 
                               iupac_name = iupac_name, 
                               SMILES = SMILES, 
                               MMFFO = MMFFO,
                               functional = functional, 
                               basis_set_1 = basis_set_1, 
                               basis_set_2 = basis_set_2, 
                               basis_set_3 = basis_set_3,
                               username = self._username)
        else:
            print('\nDuplicate found:')
            print(duplicates)
            print('\nDo you want to proceed with adding entry? (Y/N)')
            reply = input()
            if reply == 'Y' or reply == 'y':
                create_empty_entry(molecule_id = molecule_id, 
                                   iupac_name = iupac_name, 
                                   SMILES = SMILES, 
                                   MMFFO = MMFFO,
                                   functional = functional, 
                                   basis_set_1 = basis_set_1, 
                                   basis_set_2 = basis_set_2, 
                                   basis_set_3 = basis_set_3,
                                   username = self._username)
            del reply
        # self.export_to_excel()
        return molecule_id
        
    def update_entry(self,
                     _id: str,
                     keys: list,
                     values: list):
        self.print_if_debug('\nIn function Tracker.update_entry():')
        # dataframe = self.pull_tracker()
        keys.append('Last Updated')
        values.append('%s' % (timestamp()))
        # if _id not in dataframe['_id'].values:
            # raise ValueError('_id not found in database: %s' % (_id))
        if len(keys) != len(values):
            raise ValueError('Length of keys and values does not match.')
        self._MongoDB.update_one_entry(_id = _id, 
                                       keys = keys, 
                                       values = values)
        # self.export_to_excel()
        
if __name__ == '__main__': # Tracker
    # tracker = Tracker(username = 'kbc121', password = 'Imperi@l020711', debug = True)
    # print(tracker.pull_tracker())
    # tracker.add_entry(iupac_name = 'benzene',
    #                   SMILES = 'C1=CC=CC=C1',
    #                   MMFFO = 'Yes',
    #                   functional = 'B3LYP',
    #                   basis_set_1 = '6-31G(d)',
    #                   basis_set_2 = '6-311+G(d,p)',
    #                   basis_set_3 = 'No')
    # tracker.update_entry(_id = 'test_20240316233053',
    #                      keys = ['C0S0'],
    #                      values = ['new value'])
    pass

#%%

class Results:
    
    def __init__(self,
                 username: str,
                 password: str,
                 collection: str = 'Results', 
                 database: str = 'New_Reorganisation_Energy_Calc',
                 connection_string: str = 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
                 debug: bool = False):
        self._username = username
        self._password = password
        self._collection = collection
        self._database = database
        self._connection_string = connection_string 
        self._debug = debug 
        
        self._MongoDB = MongoDB(username = self._username, 
                                password = self._password, 
                                collection = self._collection,
                                database = self._database,
                                connection_string = self._connection_string,
                                debug = self._debug)
    
    def print_if_debug(self, line):
        if self._debug == True:
            print(line)
            
    def pull_results(self):
        if self._debug:
            self.print_if_debug('\nIn function Results.pull_results():')
        try:
            dataframe = self._MongoDB.pull()
            return dataframe
        except:
            raise Exception('Failed to fetch results database.')
    
    def export_to_excel(self, parent_folder: str = ''):
        if parent_folder == '':
            parent_folder = r'%s' % (os.getcwd())
        
        dataframe = self.pull_results()
        if self._debug:
            self.print_if_debug('\nIn function Results.export_to_excel():')
            self.print_if_debug('Export to location:\n%s' % (parent_folder))
            pprint.pprint(dataframe)
        try:
            dataframe.to_excel(r'%s/Results.xlsx' % (parent_folder))
        except:
            raise Exception('\nExport to excel not sucessfull.')

    def create_empty_entry(self, 
                           molecule_id: str, 
                           iupac_name: str, 
                           SMILES: str,
                           MMFFO: str, 
                           functional: str, 
                           basis_set: str, 
                           initial_charge_and_state: str, 
                           final_charge_and_state: str):
        # try:
        self._MongoDB.push_one_new_entry(keys = ['_id', 'IUPAC Name', 'SMILES', 'MMFFO', 'Functional', 'Basis Set', 
                                                 'Inital Charge and State', 'Final Charge and State', 'Alpha Occ Orbitals List', 'Alpha Virt Orbitals List', 'HOMO-LUMO Gap', 
                                                 'Initial CS at Initial Optimised Geometry', 'Initial CS at Final Optimised Geometry', 'Final CS at Initial Optimised Geometry', 'Final CS at Final Optimised Geometry', 
                                                 'Vibrational Frequencies', 'Displacements', 'Reorganisation Energies', 'Parsed SCF Summary', 'Initial Atoms Summary', 'Final Atoms Summary', 
                                                 '4-Point Reorganisation Energy', 'DUSHIN Reorganisation Energy', 
                                                 'Difference (4-point & DUSHIN)', 'Difference Percentage (4-point & DUSHIN)',
                                                 'Comment', 'Last Updated'],
                                         values = ['%s_%s_%s_%s_%s' % (molecule_id, functional, format_basis_set(basis_set), initial_charge_and_state, final_charge_and_state), iupac_name, SMILES, MMFFO, functional, basis_set, 
                                                   initial_charge_and_state, final_charge_and_state, [], [], 0,
                                                   None, None, None, None, 
                                                   [], [], [], '', '', '', 
                                                   None, None, 
                                                   None, None,
                                                   None, None])
        self.print_if_debug('Entry sucessfully created.')
        # except:
        #     self.print_if_debug('Entry already exist.')
        
    def update_entry(self,
                     molecule_id: str, 
                     keys: str, 
                     values: str):
        self.print_if_debug('\nIn function Results.update_entry():')
        # dataframe = self.pull_results()
        keys.append('Last Updated')
        values.append('%s' % (timestamp()))
        # if _id not in dataframe['_id'].values:
            # raise ValueError('_id not found in database: %s' % (_id))
        if len(keys) != len(values):
            raise ValueError('Length of keys and values does not match.')
        self._MongoDB.update_one_entry(_id = molecule_id, 
                                       keys = keys, 
                                       values = values)
        # self.export_to_excel()

#%%

class ReorganisationEnergy():
    
    def __init__(self,
                 HPC_username: str,
                 HPC_password: str,
                 HPC_hostname: str,
                 MongoDB_username: str,
                 MongoDB_password: str,
                 MongoDB_database: str = 'New_Reorganisation_Energy_Calc',
                 MongoDB_SMILES_database_collection: str = 'SMILES_Database',
                 MongoDB_tracker_collection: str = 'Tracker',
                 MongoDB_results_collection: str = 'Results',
                 MongoDB_connection_string: str = 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
                 local_folder: str = r'%s' % (os.getcwd()), 
                 remote_folder: str = r'/rds/general/ephemeral/user/kbc121/ephemeral/test',
                 debug: str = False):
        
        self._debug = debug
        self._current_working_directory = os.getcwd()
        
        self._SMILES_database = SMILES_database(username = MongoDB_username,
                                                password = MongoDB_password,
                                                collection = MongoDB_SMILES_database_collection,
                                                database = MongoDB_database,
                                                connection_string = MongoDB_connection_string,
                                                debug = self._debug)
        
        self._HPC_connection = HPC(username = HPC_username,
                                   password = HPC_password,
                                   hostname = HPC_hostname,
                                   debug = self._debug)
        
        self._local_folder = local_folder
        self._remote_folder = remote_folder
        
        self._funnel = Funnel(path = r'%s/Funnel.xlsx' % (local_folder),
                              debug = self._debug)
        
        self._tracker = Tracker(username = MongoDB_username,
                                password = MongoDB_password,
                                collection = MongoDB_tracker_collection,
                                database = MongoDB_database,
                                connection_string = MongoDB_connection_string,
                                debug = self._debug)     
        
        self._results = Results(username = MongoDB_username, 
                                password = MongoDB_password, 
                                collection = MongoDB_results_collection,
                                database = MongoDB_database,
                                connection_string = MongoDB_connection_string,
                                debug = self._debug)
    
    def print_if_debug(self, lines:str):
        if self._debug:
            print(lines)
    
    def get_tracker(self):
        # self._tracker.export_to_excel(parent_folder = self._local_folder)
        return self._tracker.pull_tracker()
    
    def get_results(self):
        # self._tracker.export_to_excel(parent_folder = self._local_folder)
        return self._results.pull_results()
    
    def download_tracker_as_excel(self):
        self._tracker.export_to_excel(parent_folder = self._local_folder)
        
    def download_results_as_excel(self):
        self._results.export_to_excel(parent_folder = self._local_folder)
    
    def get_funnel(self):
        return self._funnel.retrieve()
    
    def get_SMILES_database(self):
        return self._SMILES_database.pull_SMILES_database()
    
    def get_HPC_qstat(self):
        return self._HPC_connection.formatted_qstat()
            
    def add_workflow(self, 
                     MMFFO: str = 'Yes',
                     functional: str = 'B3LYP',
                     basis_set_1: str = '6-31G(d)',
                     basis_set_2: str = '6-311+G(d,p)',
                     basis_set_3: str = 'No'):
        self.print_if_debug('\nIn function ReorganisationEnergy.add_workflow():')
        try:
            self._funnel.load_from_excel()
            print('\nPrevious Workflow found and extracted.')
        except:
            pass
        self._funnel.add_workflow(MMFFO = MMFFO,
                                  functional = functional,
                                  basis_set_1 = basis_set_1,
                                  basis_set_2 = basis_set_2,
                                  basis_set_3 = basis_set_3)
        self._funnel.export_to_excel()
        print('\nWorkflow added sucessfully. Current workflow:')
        print(self._funnel.retrieve())
    
    def add_molecule(self, iupac_name: str, SMILES: str = ''):
        self.print_if_debug('\nIn function ReorganisationEnergy.add_molecule():')
        try:
            self._funnel.load_from_excel()
            print('\nWorkflow:')
            print(self._funnel.retrieve())
        except:
            raise Exception('No workflow found.')
        iupac_name, SMILES = self._SMILES_database.check_against_database(iupac_name = iupac_name,
                                                                          SMILES = SMILES)
        for index in self._funnel.workflow_indices():
            molecule_id = self._tracker.add_entry(iupac_name = iupac_name,
                                                  SMILES = SMILES,
                                                  MMFFO = self._funnel.MMFFO(index = index),
                                                  functional = self._funnel.functional(index = index),
                                                  basis_set_1 = self._funnel.basis_set_1(index = index),
                                                  basis_set_2 = self._funnel.basis_set_2(index = index),
                                                  basis_set_3 = self._funnel.basis_set_3(index = index))
            for basis_set in [self._funnel.basis_set_1(index = index), self._funnel.basis_set_2(index = index), self._funnel.basis_set_3(index = index)]:
                self._results.create_empty_entry(molecule_id = molecule_id, 
                                                 iupac_name = iupac_name, 
                                                 SMILES = SMILES, 
                                                 MMFFO = self._funnel.MMFFO(index = index), 
                                                 functional = self._funnel.functional(index = index), 
                                                 basis_set = basis_set, 
                                                 initial_charge_and_state = 'C0S0', 
                                                 final_charge_and_state = 'C0S1')
                self._results.create_empty_entry(molecule_id = molecule_id, 
                                                 iupac_name = iupac_name, 
                                                 SMILES = SMILES, 
                                                 MMFFO = self._funnel.MMFFO(index = index), 
                                                 functional = self._funnel.functional(index = index), 
                                                 basis_set = basis_set, 
                                                 initial_charge_and_state = 'C0S0', 
                                                 final_charge_and_state = 'C1S0')
                self._results.create_empty_entry(molecule_id = molecule_id, 
                                                 iupac_name = iupac_name, 
                                                 SMILES = SMILES, 
                                                 MMFFO = self._funnel.MMFFO(index = index), 
                                                 functional = self._funnel.functional(index = index), 
                                                 basis_set = basis_set, 
                                                 initial_charge_and_state = 'C0S0', 
                                                 final_charge_and_state = 'C-1S0')
            molecule = Molecule(molecule_name = iupac_name, 
                                molecule_id = molecule_id, 
                                SMILES = SMILES, 
                                parent_folder = self._local_folder)
            molecule.create_molecule_documentation(allow_aromatic_bond = True)
            self.print_if_debug('Molecule workflow added.')
    
    def batch_add_molecule(self, filepath: str):
        self.print_if_debug('\nIn function ReorganisationEnergy.batch_add_molecule():')
        dataframe = pd.read_excel(filepath)
        for line in range(0, len(dataframe)):
            iupac_name = dataframe.at[line, 'iupac_name']
            SMILES = dataframe.at[line, 'SMILES']
            self.add_molecule(iupac_name = iupac_name, SMILES = SMILES)
            print('Batch write: %s added.' % (iupac_name))
            
    def get_entries_from_molecule_id(self, molecule_id: str):
        dataframe = self.get_tracker()
        return dataframe.loc[dataframe['_id'] == '%s' % (molecule_id)]
    
    def submit_molecule_for_calculation(self,
                                        input_object,
                                        molecule_id: str,
                                        excitation_state: int,
                                        charge: int,
                                        functional: str,
                                        basisset: str,
                                        number_of_proc: int,
                                        memory: int,
                                        timeout_hr: int,
                                        connectivity: bool = True,
                                        keyword_line_arguments_input: list[[str, str]] = []):
        molecule = Molecule.load_molecule(molecule_id = molecule_id,
                                          parent_folder = self._local_folder,
                                          debug = self._debug)
        molecule.generate_gjf_sh_files(input_object = input_object,
                                       excited_state = excitation_state,
                                       charge = charge,
                                       functional = functional, 
                                       basisset = basisset,
                                       number_of_proc = number_of_proc,
                                       memory = memory,
                                       timeout_hr = timeout_hr,
                                       connectivity = connectivity,
                                       keyword_line_arguments_input = keyword_line_arguments_input)
        job_id = self._HPC_connection.submit_job_gaussian(local_folder = r'%s/Data/%s' % (self._local_folder, molecule_id),
                                                          remote_folder = self._remote_folder,
                                                          filename = r'%s_%s_%s_C%sS%s' % (molecule_id, functional, format_basis_set(basisset), charge, excitation_state))
        return job_id
    
    def restart_molecule_calculation(self,
                                     molecule_id: str,
                                     excitation_state: int,
                                     charge: int,
                                     functional: str,
                                     basisset: str,
                                     number_of_proc: int,
                                     memory: int,
                                     timeout_hr: int,
                                     keyword_line_arguments_input: list[[str, str]] = []):
        molecule = Molecule.load_molecule(molecule_id = molecule_id,
                                          parent_folder = self._local_folder,
                                          debug = self._debug)
        molecule.generate_gjf_sh_restart_files(excited_state = excitation_state,
                                               charge = charge,
                                               functional = functional, 
                                               basisset = basisset,
                                               number_of_proc = number_of_proc,
                                               memory = memory,
                                               timeout_hr = timeout_hr,
                                               keyword_line_arguments_input = keyword_line_arguments_input)
        job_id = self._HPC_connection.restart_job_gaussian(local_folder = r'%s/Data/%s' % (self._local_folder, molecule_id),
                                                           remote_folder = self._remote_folder,
                                                           
                                                           filename = r'%s_%s_%s_C%sS%s' % (molecule_id, functional, format_basis_set(basisset), charge, excitation_state))
        return job_id
    
    def submit_molecule_for_marcus_calculation(self,
                                               molecule_id: str,
                                               charge_geom: int,
                                               charge_eval: int,
                                               state_geom: int,
                                               state_eval: int,
                                               functional: str,
                                               basisset: str,
                                               number_of_proc: int,
                                               memory: int,
                                               timeout_hr: int,
                                               keyword_line_arguments_input: list[[str, str]] = []):
        molecule = Molecule.load_molecule(molecule_id = molecule_id,
                                          parent_folder = self._local_folder,
                                          debug = self._debug)
        molecule.generate_gjf_sh_marcus_files(charge_geom = charge_geom,
                                              charge_eval = charge_eval,
                                              state_geom = state_geom,
                                              state_eval = state_eval,
                                              functional = functional, 
                                              basisset = basisset,
                                              number_of_proc = number_of_proc,
                                              memory = memory,
                                              timeout_hr = timeout_hr,
                                              keyword_line_arguments_input = keyword_line_arguments_input)
        job_id = self._HPC_connection.submit_job_gaussian(local_folder = r'%s/Data/%s' % (self._local_folder, molecule_id),
                                                          remote_folder = self._remote_folder,
                                                          filename = r'%s_%s_%s_C%sS%s_opt_geom_at_C%sS%s' % (molecule_id, functional, format_basis_set(basisset), charge_geom, state_geom, charge_eval, state_eval))
        return job_id
    
    def transfer_chk_fchk_log_files_from_HPC(self, 
                                             molecule_id: str,
                                             charge: int,
                                             excitation_state: int,
                                             functional: str,
                                             basisset: str):
        self._HPC_connection.transfer_chk_fchk_log_files_from_HPC(local_folder = r'%s/Data/%s' % (self._local_folder, molecule_id),
                                                                  remote_folder = r'%s' % (self._remote_folder),
                                                                  filename = r'%s_%s_%s_C%sS%s' % (molecule_id, functional, format_basis_set(basisset), charge, excitation_state))
    
    def transfer_chk_fchk_log_marcus_files_from_HPC(self, 
                                                    molecule_id: str,
                                                    charge_geom: int,
                                                    charge_eval: int,
                                                    state_geom: int,
                                                    state_eval: int,
                                                    functional: str,
                                                    basisset: str):
        self._HPC_connection.transfer_chk_fchk_log_files_from_HPC(local_folder = r'%s/Data/%s' % (self._local_folder, molecule_id),
                                                                  remote_folder = r'%s' % (self._remote_folder),
                                                                  filename = r'%s_%s_%s_C%sS%s_opt_geom_at_C%sS%s' % (molecule_id, functional, format_basis_set(basisset), charge_geom, state_geom, charge_eval, state_eval))
        
    def memory_assignment(self, molecule_id: str):
        SMILES = self.get_entries_from_molecule_id(molecule_id = molecule_id)['SMILES'].values[0]
        molecular_weight = Descriptors.ExactMolWt(Chem.MolFromSmiles(SMILES))
        out: dict = {'number_of_proc': 0,
                     'memory': 0,
                     'timeout_hr': 0}
        if molecular_weight <= 50:                                             # ethene
            out['number_of_proc'] = 8    # 8
            out['memory']         = 24   # 8 * 3GB
            out['timeout_hr']     = 3    # 3 hr
        elif molecular_weight > 50 and molecular_weight <= 100:                # benzene
            out['number_of_proc'] = 16   # 16
            out['memory']         = 48   # 16 * 3GB
            out['timeout_hr']     = 5    # 7 hr
        elif molecular_weight > 100 and molecular_weight <= 200:               # napthalene, anthracene
            out['number_of_proc'] = 32   # 32
            out['memory']         = 96   # 32 * 3GB
            out['timeout_hr']     = 7   # 10 hr
        elif molecular_weight > 200 and molecular_weight <= 300:               # tetracene, pentacene
            out['number_of_proc'] = 64   # 64
            out['memory']         = 256  # 64 * 4GB
            out['timeout_hr']     = 9   # 15 hr
        elif molecular_weight > 300 and molecular_weight <= 400:               # hexacene, heptacene
            out['number_of_proc'] = 80   # 80
            out['memory']         = 320  # 80 * 4GB
            out['timeout_hr']     = 11   # 23 hr
        elif molecular_weight > 400:                                           # Octacene and above
            out['number_of_proc'] = 90   # 90
            out['memory']         = 360  # 90 * 4GB
            out['timeout_hr']     = 15   # 48 hr
        self.print_if_debug(out)
        return out
    
    def add_activity_log(self, molecule_id: str, line: str):
        filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        entry = filtered_line.loc[:, 'Activity Log'].values[0]
        new_entry = entry + '\n' + line
        self._tracker.update_entry(_id = molecule_id, keys = ['Activity Log'], values = [new_entry])
        self.print_if_debug('Activity log updated for %s:' % (molecule_id))
        self.print_if_debug('-> %s' % (line))
        
    def add_comment(self, user: str, molecule_id: str, line: str):
        filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        entry = filtered_line.loc[:, 'Comment'].values[0]
        if entry == 'No comment':
            new_entry = '%s: %s' % (user, line)
        else:
            new_entry = '%s\n%s: %s' % (entry, user, line)
        self._tracker.update_entry(_id = molecule_id, keys = ['Comment'], values = [new_entry])
        self.print_if_debug('Comment updated for %s:' % (molecule_id))
        self.print_if_debug('-> %s' % (line))
        
    def comment_result(self,
                       _id: str,
                       user: str,
                       string: str):
        results = self.get_results()
        current_comment = results.loc[results['_id'] == _id, 'Comment'].values[0]
        if current_comment != None:
            comment = (current_comment + '\n' + 
                       '%s: ' % (user) + string)
        else:
            comment = '%s: ' % (user) + string
        self._results.update_entry(molecule_id = _id,
                                   keys = ['Comment'], 
                                   values = [comment])        
        
    def gaussian_normal_termination(self, 
                                    molecule_id: str, 
                                    functional: str,
                                    basis_set: str,
                                    charge: int, 
                                    excitation_state: int):
        # try:
        #     with open(r'%s/Data/%s/%s_%s_%s_C%sS%s.log' % (self._parent_folder, molecule_id, molecule_id, functional,
        #                                                    format_basis_set(basis_set), charge, excitation_state)) as file:
        #         word = file.readlines()[-1].split(' ')
        #         if 'Normal' in word:
        #             return True
        #         else:
        #             return False
        # except:
        #     raise Exception('\nUnable to open file:\n %s/Data/%s/%s_%s_%s_C%sS%s.log' % (self._parent_folder, molecule_id, molecule_id, functional, 
        #                                                                                  format_basis_set(basis_set), charge, excitation_state))
        try:
            with open(r'%s/Data/%s/%s_%s_%s_C%sS%s.log' % (self._local_folder, molecule_id, molecule_id, functional,
                                                           format_basis_set(basis_set), charge, excitation_state)) as file:
                whole_file = file.read()
                lines = whole_file.split('\n')
                normal_count = 0
                warning_count = 0
                comment_string = ''
                for line in lines:
                    if 'Normal' in line:
                        normal_count += 1
                        self.print_if_debug('\nNormal termination statement %s: %s' % (normal_count, line))
                        comment_string = comment_string = '\nAuto_Generated: %s C%sS%s Normal termination statement %s: %s' % (basis_set, charge, excitation_state, normal_count, line)
                        # self.add_comment(user = 'Auto-Generated', 
                        #                  molecule_id = molecule_id, 
                        #                  line = '%s C%sS%s Normal termination statement %s: %s' % (basis_set, charge, excitation_state, normal_count, line))
                    if '**** Warning!!' in line:
                        warning_count += 1
                        self.print_if_debug('\nWarning statement %s: %s' % (warning_count, line))
                        comment_string = comment_string + '\nAuto_Generated: %s C%sS%s Warning statement %s: %s' % (basis_set, charge, excitation_state, warning_count, line)
                        # self.add_comment(user = 'Auto-Generated', 
                        #                  molecule_id = molecule_id, 
                        #                  line = '%s C%sS%s Warning statement %s: %s' % (basis_set, charge, excitation_state, warning_count, line))
                if normal_count == 2:
                    self.print_if_debug('Normal termination count = 2, SUCESSFULL OPT and FREQ calculation.')
                    comment_string = comment_string + '\nAuto-Generated: %s C%sS%s Normal termination count = 2, SUCESSFULL OPT and FREQ calculation.' % (basis_set, charge, excitation_state)
                    comment_string = comment_string.replace('\nAuto-Generated: ', '', 1)
                    self.add_comment(user = 'Auto-Generated', 
                                     molecule_id = molecule_id, 
                                     line = comment_string)
                    return 'Completed'
                elif normal_count == 1:
                    self.print_if_debug('Normal termination count = 1, SUCESSFULL OPT and FAILED FREQ calculation.')
                    comment_string = comment_string + '\nAuto-Generated: %s C%sS%s Normal termination count = 1, SUCESSFULL OPT and FAILED FREQ calculation.' % (basis_set, charge, excitation_state)
                    comment_string = comment_string.replace('\nAuto-Generated: ', '', 1)
                    self.add_comment(user = 'Auto-Generated', 
                                     molecule_id = molecule_id, 
                                     line = comment_string)
                    return 'Failed'
                else:
                    self.print_if_debug('Normal termination count = 0, FAILED OPT and FREQ calculation.')
                    comment_string = comment_string + '\nAuto-Generated: %s C%sS%s Normal termination count = 0, FAILED OPT and FREQ calculation.' % (basis_set, charge, excitation_state)
                    comment_string = comment_string.replace('\nAuto-Generated: ', '', 1)
                    self.add_comment(user = 'Auto-Generated', 
                                     molecule_id = molecule_id, 
                                     line = comment_string)
                    return 'Failed'
        except:
            raise Exception('\nUnable to open file:\n %s/Data/%s/%s_%s_%s_C%sS%s.log' % (self._local_folder, molecule_id, molecule_id, functional, 
                                                                                         format_basis_set(basis_set), charge, excitation_state))
    def gaussian_marcus_normal_termination(self, 
                                           molecule_id: str, 
                                           functional: str,
                                           basis_set: str,
                                           charge_geom: int,
                                           charge_eval: int,
                                           state_geom: int,
                                           state_eval: int):
        try:
            with open(r'%s/Data/%s/%s_%s_%s_C%sS%s_opt_geom_at_C%sS%s.log' % (self._local_folder, molecule_id, molecule_id, functional, format_basis_set(basis_set), charge_geom, state_geom, charge_eval, state_eval)) as file:
                whole_file = file.read()
                lines = whole_file.split('\n')
                normal_count = 0
                warning_count = 0
                comment_string = ''
                for line in lines:
                    if 'Normal' in line:
                        normal_count += 1
                        self.print_if_debug('\nNormal termination statement %s: %s' % (normal_count, line))
                        comment_string = comment_string + '\nAuto-Generated: %s C%sS%s_opt_geom_at_C%sS%s Normal termination statement %s: %s' % (basis_set, charge_geom, state_geom, charge_eval, state_eval, normal_count, line)
                        # self.add_comment(user = 'Auto-Generated', 
                        #                  molecule_id = molecule_id, 
                        #                  line = '%s C%sS%s_opt_geom_at_C%sS%s Normal termination statement %s: %s' % (basis_set, charge_geom, state_geom, charge_eval, state_eval, normal_count, line))
                    if '**** Warning!!' in line:
                        warning_count += 1
                        self.print_if_debug('\nWarning statement %s: %s' % (warning_count, line))
                        comment_string = comment_string + '\nAuto-Generated: %s C%sS%s_opt_geom_at_C%sS%s Warning statement %s: %s' % (basis_set, charge_geom, state_geom, charge_eval, state_eval, warning_count, line)
                        # self.add_comment(user = 'Auto-Generated', 
                        #                  molecule_id = molecule_id, 
                        #                  line = '%s C%sS%s_opt_geom_at_C%sS%s Warning statement %s: %s' % (basis_set, charge_geom, state_geom, charge_eval, state_eval, warning_count, line))
                if normal_count == 1:
                    self.print_if_debug('Normal termination count = 1, SUCESSFULL calculation.')
                    comment_string = comment_string + '\nAuto-Generated: %s C%sS%s_opt_geom_at_C%sS%s Normal termination count = 1, SUCESSFULL calculation.' % (basis_set, charge_geom, state_geom, charge_eval, state_eval)
                    comment_string = comment_string.replace('\nAuto-Generated: ', '', 1)
                    self.add_comment(user = 'Auto-Generated', 
                                     molecule_id = molecule_id, 
                                     line = comment_string)
                    return 'Completed'
                else:
                    self.print_if_debug('Normal termination count = 0, FAILED OPT and FREQ calculation.')
                    comment_string = comment_string + '\nAuto-Generated: %s C%sS%s_opt_geom_at_C%sS%s Normal termination count = 0, FAILED OPT and FREQ calculation.' % (basis_set, charge_geom, state_geom, charge_eval, state_eval)
                    comment_string = comment_string.replace('\nAuto-Generated: ', '', 1)
                    self.add_comment(user = 'Auto-Generated', 
                                     molecule_id = molecule_id, 
                                     line = comment_string)
                    return 'Failed'
        except:
            raise Exception('\nUnable to open file:\n %s/Data/%s/%s_%s_%s_C%sS%s_opt_geom_at_C%sS%s.log' % (self._local_folder, molecule_id, molecule_id, functional, format_basis_set(basis_set), charge_geom, state_geom, charge_eval, state_eval))
    
    def query_stage_started(self, 
                            molecule_id: str,
                            charge: int, 
                            excitation_state: int, 
                            basis_set: int) -> bool:
        filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        entry = filtered_line.loc[:, 'C%sS%s: BS%s' % (charge, excitation_state, basis_set)].values[0]
        if entry == 'Not Started':
            return False
        else:
            return True
    
    def count_running_optimisations(self, alternative: str = '') -> int:
        tracker = self.get_tracker()[['C0S0: BS1', 'C0S0: BS2', 'C0S0: BS3',
                                      'C0S1: BS1 - C0S0 geometry', 'C0S1: BS2 - C0S0 geometry', 'C0S1: BS3 - C0S0 geometry',
                                      'C1S0: BS1 - C0S0 geometry', 'C1S0: BS2 - C0S0 geometry', 'C1S0: BS3 - C0S0 geometry',
                                      'C-1S0: BS1 - C0S0 geometry', 'C-1S0: BS2 - C0S0 geometry', 'C-1S0: BS3 - C0S0 geometry',
                                      'C0S1: BS1', 'C0S1: BS2', 'C0S1: BS3',
                                      'C0S0: BS1 - C0S1 geometry', 'C0S0: BS2 - C0S1 geometry', 'C0S0: BS3 - C0S1 geometry',
                                      'C1S0: BS1', 'C1S0: BS2', 'C1S0: BS3',
                                      'C0S0: BS1 - C1S0 geometry', 'C0S0: BS2 - C1S0 geometry', 'C0S0: BS3 - C1S0 geometry',
                                      'C-1S0: BS1', 'C-1S0: BS2', 'C-1S0: BS3',
                                      'C0S0: BS1 - C-1S0 geometry', 'C0S0: BS2 - C-1S0 geometry', 'C0S0: BS3 - C-1S0 geometry']]
        if alternative == '':
            raise Exception('\nMethod depreciated, use count_running_optimisations_alt() instead.')
            return pd.DataFrame(tracker == 'Started').sum().sum()
        else:
            return pd.DataFrame(tracker == alternative).sum().sum()
        
    def count_running_optimisations_alt(self, exclude: list[str] = ['Not Started', 'Completed', 'Failed']):
        tracker = self.get_tracker()
        count = len(tracker) * 30
        for i in range(0, len(exclude)):
            count -= self.count_running_optimisations(alternative = exclude[i])
        return count
    
    def get_tracker_not_started(self):
        tracker = self.get_tracker()
        out = pd.DataFrame([])
        for index in range(0, len(tracker)):
            line = tracker.iloc[[index]]
            filtered_line = line[['C0S0: BS1', 'C0S0: BS2', 'C0S0: BS3',
                                  'C0S1: BS1 - C0S0 geometry', 'C0S1: BS2 - C0S0 geometry', 'C0S1: BS3 - C0S0 geometry',
                                  'C1S0: BS1 - C0S0 geometry', 'C1S0: BS2 - C0S0 geometry', 'C1S0: BS3 - C0S0 geometry',
                                  'C-1S0: BS1 - C0S0 geometry', 'C-1S0: BS2 - C0S0 geometry', 'C-1S0: BS3 - C0S0 geometry',
                                  'C0S1: BS1', 'C0S1: BS2', 'C0S1: BS3',
                                  'C0S0: BS1 - C0S1 geometry', 'C0S0: BS2 - C0S1 geometry', 'C0S0: BS3 - C0S1 geometry',
                                  'C1S0: BS1', 'C1S0: BS2', 'C1S0: BS3',
                                  'C0S0: BS1 - C1S0 geometry', 'C0S0: BS2 - C1S0 geometry', 'C0S0: BS3 - C1S0 geometry',
                                  'C-1S0: BS1', 'C-1S0: BS2', 'C-1S0: BS3',
                                  'C0S0: BS1 - C-1S0 geometry', 'C0S0: BS2 - C-1S0 geometry', 'C0S0: BS3 - C-1S0 geometry']]
            if any(filtered_line.isin(['Not Started']).values[0]):
                if len(out) == 0:
                    out = pd.DataFrame(line)
                else:
                    out = pd.concat([out, line])
        return out
    
    def get_tracker_failed(self):
        tracker = self.get_tracker()
        out = pd.DataFrame([])
        for index in range(0, len(tracker)):
            line = tracker.iloc[[index]]
            filtered_line = line[['C0S0: BS1', 'C0S0: BS2', 'C0S0: BS3',
                                  'C0S1: BS1 - C0S0 geometry', 'C0S1: BS2 - C0S0 geometry', 'C0S1: BS3 - C0S0 geometry',
                                  'C1S0: BS1 - C0S0 geometry', 'C1S0: BS2 - C0S0 geometry', 'C1S0: BS3 - C0S0 geometry',
                                  'C-1S0: BS1 - C0S0 geometry', 'C-1S0: BS2 - C0S0 geometry', 'C-1S0: BS3 - C0S0 geometry',
                                  'C0S1: BS1', 'C0S1: BS2', 'C0S1: BS3',
                                  'C0S0: BS1 - C0S1 geometry', 'C0S0: BS2 - C0S1 geometry', 'C0S0: BS3 - C0S1 geometry',
                                  'C1S0: BS1', 'C1S0: BS2', 'C1S0: BS3',
                                  'C0S0: BS1 - C1S0 geometry', 'C0S0: BS2 - C1S0 geometry', 'C0S0: BS3 - C1S0 geometry',
                                  'C-1S0: BS1', 'C-1S0: BS2', 'C-1S0: BS3',
                                  'C0S0: BS1 - C-1S0 geometry', 'C0S0: BS2 - C-1S0 geometry', 'C0S0: BS3 - C-1S0 geometry']]
            if any(filtered_line.isin(['Failed']).values[0]):
                if len(out) == 0:
                    out = pd.DataFrame(line)
                else:
                    out = pd.concat([out, line])
        return out
    
    def get_tracker_running_only(self):
        tracker = self.get_tracker()
        out = pd.DataFrame()
        for index in range(0, len(tracker)):
            line = tracker.iloc[[index]]
            filtered_line = line[['C0S0: BS1', 'C0S0: BS2', 'C0S0: BS3',
                                  'C0S1: BS1 - C0S0 geometry', 'C0S1: BS2 - C0S0 geometry', 'C0S1: BS3 - C0S0 geometry',
                                  'C1S0: BS1 - C0S0 geometry', 'C1S0: BS2 - C0S0 geometry', 'C1S0: BS3 - C0S0 geometry',
                                  'C-1S0: BS1 - C0S0 geometry', 'C-1S0: BS2 - C0S0 geometry', 'C-1S0: BS3 - C0S0 geometry',
                                  'C0S1: BS1', 'C0S1: BS2', 'C0S1: BS3',
                                  'C0S0: BS1 - C0S1 geometry', 'C0S0: BS2 - C0S1 geometry', 'C0S0: BS3 - C0S1 geometry',
                                  'C1S0: BS1', 'C1S0: BS2', 'C1S0: BS3',
                                  'C0S0: BS1 - C1S0 geometry', 'C0S0: BS2 - C1S0 geometry', 'C0S0: BS3 - C1S0 geometry',
                                  'C-1S0: BS1', 'C-1S0: BS2', 'C-1S0: BS3',
                                  'C0S0: BS1 - C-1S0 geometry', 'C0S0: BS2 - C-1S0 geometry', 'C0S0: BS3 - C-1S0 geometry']]
            count = 0
            scan_for = ['Not Started', 'Completed', 'Failed']
            for item in scan_for:
                count += pd.DataFrame(filtered_line == item).sum().sum()
            if count != 12: # start count from 12 and minus, and if more than 0 add, or != 12
                if len(out) == 0:
                    out = pd.DataFrame(line)
                else:
                    out = pd.concat([out, line])
        return out
    
    def tracker_summarise_running(self):
        running_only = self.get_tracker_running_only()
        scan = ['C0S0: BS1', 'C0S0: BS2', 'C0S0: BS3',
                'C0S1: BS1 - C0S0 geometry', 'C0S1: BS2 - C0S0 geometry', 'C0S1: BS3 - C0S0 geometry',
                'C1S0: BS1 - C0S0 geometry', 'C1S0: BS2 - C0S0 geometry', 'C1S0: BS3 - C0S0 geometry',
                'C-1S0: BS1 - C0S0 geometry', 'C-1S0: BS2 - C0S0 geometry', 'C-1S0: BS3 - C0S0 geometry',
                'C0S1: BS1', 'C0S1: BS2', 'C0S1: BS3',
                'C0S0: BS1 - C0S1 geometry', 'C0S0: BS2 - C0S1 geometry', 'C0S0: BS3 - C0S1 geometry',
                'C1S0: BS1', 'C1S0: BS2', 'C1S0: BS3',
                'C0S0: BS1 - C1S0 geometry', 'C0S0: BS2 - C1S0 geometry', 'C0S0: BS3 - C1S0 geometry',
                'C-1S0: BS1', 'C-1S0: BS2', 'C-1S0: BS3',
                'C0S0: BS1 - C-1S0 geometry', 'C0S0: BS2 - C-1S0 geometry', 'C0S0: BS3 - C-1S0 geometry']
        out = pd.DataFrame(columns = ['molecule_id', 'run', 'run_id'])
        for index in range(0, len(running_only)):
            line = running_only.iloc[index, :]
            for item in scan:
                if line[item] != 'Not Started' and line[item] != 'Completed' and line[item] != 'Failed':
                    out.loc[len(out)] = [line['_id'], item, line[item]]
        return out
            
    
    def next_step_for_molecule_id_with_charge_state(self, 
                                                    molecule_id: str, 
                                                    charge: int, 
                                                    excitation_state: int):
        raise Exception('Depreciated. Use next_step_for_molecule_id_with_basis_set.') 
        # charge_state = 'C%sS%s' % (charge, excitation_state)
        # filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        # if filtered_line.loc[:, '%s: BS1' % (charge_state)].values[0] == 'Not Started':
        #     job_parameters = self.memory_assignment(molecule_id = molecule_id)
        #     job_id = self.submit_molecule_for_calculation(input_object = '',            # use RDKit geometry
        #                                                   molecule_id = molecule_id,
        #                                                   excitation_state = excitation_state,
        #                                                   charge = charge,
        #                                                   functional = filtered_line.loc[:, 'Functional'].values[0],
        #                                                   basisset = filtered_line.loc[:, 'Basis Set 1'].values[0],
        #                                                   number_of_proc = int(job_parameters['number_of_proc']),
        #                                                   memory = int(job_parameters['memory']),
        #                                                   timeout_hr = int(job_parameters['timeout_hr']),
        #                                                   connectivity = True,
        #                                                   keyword_line_arguments_input = [])
        #     self._tracker.update_entry(_id = molecule_id, 
        #                                keys = ['%s: BS1' % (charge_state)], 
        #                                values = [job_id])
        #     self.add_activity_log(molecule_id = molecule_id, 
        #                           line = '%s BS1 optimisation started: %s' % (charge_state, str(datetime.now()).split('.')[0]))
        #     self.print_if_debug(('%s %s: BS1 started.' % (molecule_id, charge_state)))
        #     return True
        # elif filtered_line.loc[:, '%s: BS1' % (charge_state)].values[0] == 'Completed' and filtered_line.loc[:, '%s: BS2' % (charge_state)].values[0] == 'Not Started':
        #     job_parameters = self.memory_assignment(molecule_id = molecule_id)
        #     job_id = self.submit_molecule_for_calculation(input_object = '%s/Data/%s/%s_%s_%s_%s.log' % (self._local_folder, molecule_id, molecule_id, 
        #                                                                                                  filtered_line.loc[:, 'Functional'].values[0], 
        #                                                                                                  format_basis_set(filtered_line.loc[:, 'Basis Set 1'].values[0]),
        #                                                                                                  charge_state),            # use BS1 geometry
        #                                                   molecule_id = molecule_id,
        #                                                   excitation_state = excitation_state,
        #                                                   charge = charge,
        #                                                   functional = filtered_line.loc[:, 'Functional'].values[0],
        #                                                   basisset = filtered_line.loc[:, 'Basis Set 2'].values[0],
        #                                                   number_of_proc = int(job_parameters['number_of_proc']),
        #                                                   memory = int(job_parameters['memory']),
        #                                                   timeout_hr = int(job_parameters['timeout_hr']),
        #                                                   connectivity = True,
        #                                                   keyword_line_arguments_input = [])
        #     self._tracker.update_entry(_id = molecule_id, 
        #                                keys = ['%s: BS2' % (charge_state)], 
        #                                values = [job_id])
        #     self.add_activity_log(molecule_id = molecule_id, 
        #                           line = '%s BS2 optimisation started: %s' % (charge_state, str(datetime.now()).split('.')[0]))
        #     self.print_if_debug(('%s %s: BS2 started.' % (molecule_id, charge_state)))
        #     return True
        # elif filtered_line.loc[:, '%s: BS2' % (charge_state)].values[0] == 'Completed' and filtered_line.loc[:, '%s: BS3' % (charge_state)].values[0] == 'Not Started':
        #     job_parameters = self.memory_assignment(molecule_id = molecule_id)
        #     job_id = self.submit_molecule_for_calculation(input_object = '%s/Data/%s/%s_%s_%s_%s.log' % (self._local_folder, molecule_id, molecule_id, 
        #                                                                                                  filtered_line.loc[:, 'Functional'].values[0], 
        #                                                                                                  format_basis_set(filtered_line.loc[:, 'Basis Set 2'].values[0]),
        #                                                                                                  charge_state),            # use BS2 geometry
        #                                                   molecule_id = molecule_id,
        #                                                   excitation_state = excitation_state,
        #                                                   charge = charge,
        #                                                   functional = filtered_line.loc[:, 'Functional'].values[0],
        #                                                   basisset = filtered_line.loc[:, 'Basis Set 2'].values[0],
        #                                                   number_of_proc = int(job_parameters['number_of_proc']),
        #                                                   memory = int(job_parameters['memory']),
        #                                                   timeout_hr = int(job_parameters['timeout_hr']),
        #                                                   connectivity = True,
        #                                                   keyword_line_arguments_input = [])
        #     self._tracker.update_entry(_id = molecule_id, 
        #                                keys = ['%s: BS3' % (charge_state)], 
        #                                values = [job_id])
        #     self.add_activity_log(molecule_id = molecule_id, 
        #                           line = '%s BS3 optimisation started: %s' % (charge_state, str(datetime.now()).split('.')[0]))
        #     self.print_if_debug(('%s %s: BS3 started.' % (molecule_id, charge_state)))
        #     return True
        # else:
        #     return False
        
    def restart_gaussian_for_molecule(self, 
                                      molecule_id: str,
                                      basis_set: int, 
                                      excitation_state: int,
                                      charge: int):
        filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        job_parameters = self.memory_assignment(molecule_id = molecule_id)
        job_id = self.restart_molecule_calculation(molecule_id = molecule_id,
                                                   excitation_state = excitation_state,
                                                   charge = charge,
                                                   functional = filtered_line.loc[:, 'Functional'].values[0],
                                                   basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                   number_of_proc = int(job_parameters['number_of_proc']),
                                                   memory = int(job_parameters['memory']),
                                                   timeout_hr = int(job_parameters['timeout_hr']),
                                                   keyword_line_arguments_input = [])
        self._tracker.update_entry(_id = molecule_id, 
                                   keys = ['C%sS%s: BS%s' % (charge, excitation_state, basis_set)], 
                                   values = [job_id])
        self.add_activity_log(molecule_id = molecule_id, 
                              line = 'C%sS%s BS%s optimisation restarted: %s' % (charge, excitation_state, basis_set, str(datetime.now()).split('.')[0]))
        self.print_if_debug(('%s C%sS%s: BS%s started.' % (charge, excitation_state, molecule_id, basis_set)))
        
    def restart_failed_cases(self, 
                             stop_time = '',
                             max_jobs_parallel: int = 45):
        if stop_time == '':
            stop_time = datetime.now() + timedelta(minutes = 15)
        if datetime.now() >= stop_time:
            return
        current_running = self.count_running_optimisations_alt()
        to_be_run = self.get_tracker_failed().reset_index()
        for index in range(0, len(to_be_run)):
            molecule_id = to_be_run.at[index, '_id']
            filtered_line = to_be_run.loc[to_be_run['_id'] == molecule_id]
            columns = filtered_line.columns[filtered_line.isin(['Failed']).any()].tolist()
            for header in columns:
                if current_running <= max_jobs_parallel:
                    header = header.split(' - ') # 'C0S1: BS3 - opt geom at C0S0'.split(' - ') -->  ['C0S1: BS3', 'opt geom at C0S0']
                    if len(header) == 1: # only 'C0S0: BS1'
                        charge_state, basis_set = header[0].split(': ') # ['C0S1', 'BS3']
                        basis_set = int(basis_set.replace('BS', ''))
                        # basis_set = filtered_line['Basis Set %s' % (basis_set)].values[0]
                        state = int(charge_state.split('S')[1])
                        charge = int(charge_state.split('S')[0].replace('C', ''))
                        self.restart_gaussian_for_molecule(molecule_id = molecule_id,
                                                           basis_set = basis_set, 
                                                           excitation_state = state,
                                                           charge = charge)
                        current_running += 1
                    else: # ['C0S1: BS3', 'opt geom at C0S0']
                        raise Exception('Not coded')
        self.download_tracker_as_excel()
                    
        
        
    def next_step_for_molecule_id_with_basis_set(self,
                                                 molecule_id: str,
                                                 basis_set: int):
        filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        job_parameters = self.memory_assignment(molecule_id = molecule_id)
        # start ground state ##################################################
        if filtered_line.loc[:, 'C0S0: BS%s' % (basis_set)].values[0] == 'Not Started': # start ground state
            charge = 0
            excitation_state = 0
            charge_state = 'C%sS%s' % (charge, excitation_state)
            if basis_set == 1: # basis set 1
                job_id = self.submit_molecule_for_calculation(input_object = '',            # use RDKit geometry
                                                              molecule_id = molecule_id,
                                                              excitation_state = excitation_state,
                                                              charge = charge,
                                                              functional = filtered_line.loc[:, 'Functional'].values[0],
                                                              basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                              number_of_proc = int(job_parameters['number_of_proc']),
                                                              memory = int(job_parameters['memory']),
                                                              timeout_hr = int(job_parameters['timeout_hr']),
                                                              connectivity = True,
                                                              keyword_line_arguments_input = [])
            else: # not basis set 1: use geometry of previous basis set optimisation
                previous_basis_set = int(basis_set - 1)
                file_path_string = '%s/Data/%s/%s_%s_%s_%s.log' % (self._local_folder, molecule_id, molecule_id,
                                                                   filtered_line.loc[:, 'Functional'].values[0],
                                                                   format_basis_set(filtered_line.loc[:, 'Basis Set %s' % (previous_basis_set)].values[0]),
                                                                   charge_state)
                job_id = self.submit_molecule_for_calculation(input_object = r'%s' % (file_path_string), # use previous basis set geometry
                                                              molecule_id = molecule_id,
                                                              excitation_state = excitation_state,
                                                              charge = charge,
                                                              functional = filtered_line.loc[:, 'Functional'].values[0],
                                                              basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                              number_of_proc = int(job_parameters['number_of_proc']),
                                                              memory = int(job_parameters['memory']),
                                                              timeout_hr = int(job_parameters['timeout_hr']),
                                                              connectivity = True,
                                                              keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C0S0: BS%s' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C0S0 BS%s optimisation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C0S0: BS%s started.' % (molecule_id, basis_set)))
            return True
        # C0S0 optimal geometry completed #####################################
        elif filtered_line.loc[:, 'C0S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C0S1: BS%s' % (basis_set)].values[0] == 'Not Started':
            # start C0S1 ######################################################
            charge = 0
            excitation_state = 1
            charge_state = 'C%sS%s' % (charge, excitation_state)
            file_path_string = '%s/Data/%s/%s_%s_%s_C0S0.log' % (self._local_folder, molecule_id, molecule_id,
                                                                 filtered_line.loc[:, 'Functional'].values[0],
                                                                 format_basis_set(filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0]))
            job_id = self.submit_molecule_for_calculation(input_object = r'%s' % (file_path_string), # use C0S0 geometry
                                                          molecule_id = molecule_id,
                                                          excitation_state = excitation_state,
                                                          charge = charge,
                                                          functional = filtered_line.loc[:, 'Functional'].values[0],
                                                          basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                          number_of_proc = int(job_parameters['number_of_proc']),
                                                          memory = int(job_parameters['memory']),
                                                          timeout_hr = int(job_parameters['timeout_hr']),
                                                          connectivity = True,
                                                          keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C0S1: BS%s' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C0S1 BS%s optimisation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C0S0: BS%s started.' % (molecule_id, basis_set)))
            return True
        elif filtered_line.loc[:, 'C0S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C1S0: BS%s' % (basis_set)].values[0] == 'Not Started':
            # start C1S0 ######################################################
            charge = 1
            excitation_state = 0
            charge_state = 'C%sS%s' % (charge, excitation_state)
            file_path_string = '%s/Data/%s/%s_%s_%s_C0S0.log' % (self._local_folder, molecule_id, molecule_id,
                                                                 filtered_line.loc[:, 'Functional'].values[0],
                                                                 format_basis_set(filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0]))
            job_id = self.submit_molecule_for_calculation(input_object = r'%s' % (file_path_string), # use C0S0 geometry
                                                          molecule_id = molecule_id,
                                                          excitation_state = excitation_state,
                                                          charge = charge,
                                                          functional = filtered_line.loc[:, 'Functional'].values[0],
                                                          basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                          number_of_proc = int(job_parameters['number_of_proc']),
                                                          memory = int(job_parameters['memory']),
                                                          timeout_hr = int(job_parameters['timeout_hr']),
                                                          connectivity = True,
                                                          keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C1S0: BS%s' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C1S0 BS%s optimisation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C1S0: BS%s started.' % (molecule_id, basis_set)))
            return True
        elif filtered_line.loc[:, 'C0S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C-1S0: BS%s' % (basis_set)].values[0] == 'Not Started':
            # start C-1S0 #####################################################
            charge = -1
            excitation_state = 0
            charge_state = 'C%sS%s' % (charge, excitation_state)
            file_path_string = '%s/Data/%s/%s_%s_%s_C0S0.log' % (self._local_folder, molecule_id, molecule_id,
                                                                 filtered_line.loc[:, 'Functional'].values[0],
                                                                 format_basis_set(filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0]))
            job_id = self.submit_molecule_for_calculation(input_object = r'%s' % (file_path_string), # use C0S0 geometry
                                                          molecule_id = molecule_id,
                                                          excitation_state = excitation_state,
                                                          charge = charge,
                                                          functional = filtered_line.loc[:, 'Functional'].values[0],
                                                          basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                          number_of_proc = int(job_parameters['number_of_proc']),
                                                          memory = int(job_parameters['memory']),
                                                          timeout_hr = int(job_parameters['timeout_hr']),
                                                          connectivity = True,
                                                          keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C-1S0: BS%s' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C-1S0 BS%s optimisation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C-1S0: BS%s started.' % (molecule_id, basis_set)))
            return True
        elif filtered_line.loc[:, 'C0S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C0S1: BS%s - C0S0 geometry' % (basis_set)].values[0] == 'Not Started':
            # start C0S0 optimal geometry at C0S1 energy ######################
            job_id = self.submit_molecule_for_marcus_calculation(molecule_id = molecule_id,
                                                                 charge_geom = 0,
                                                                 charge_eval = 0,
                                                                 state_geom = 0,
                                                                 state_eval = 1,
                                                                 functional = filtered_line.loc[:, 'Functional'].values[0],
                                                                 basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                                 number_of_proc = int(job_parameters['number_of_proc']),
                                                                 memory = int(job_parameters['memory']),
                                                                 timeout_hr = int(job_parameters['timeout_hr']),
                                                                 keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C0S1: BS%s - C0S0 geometry' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C0S1: BS%s - C0S0 geometry calculation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C0S1: BS%s - C0S0 geometry started.' % (molecule_id, basis_set)))
            return True
        elif filtered_line.loc[:, 'C0S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C1S0: BS%s - C0S0 geometry' % (basis_set)].values[0] == 'Not Started':
            # start C0S0 optimal geometry at C1S0 energy ######################
            job_id = self.submit_molecule_for_marcus_calculation(molecule_id = molecule_id,
                                                                  charge_geom = 0,
                                                                  charge_eval = 1,
                                                                  state_geom = 0,
                                                                  state_eval = 0,
                                                                  functional = filtered_line.loc[:, 'Functional'].values[0],
                                                                  basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                                  number_of_proc = int(job_parameters['number_of_proc']),
                                                                  memory = int(job_parameters['memory']),
                                                                  timeout_hr = int(job_parameters['timeout_hr']),
                                                                  keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                        keys = ['C1S0: BS%s - C0S0 geometry' % (basis_set)], 
                                        values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C1S0: BS%s - C0S0 geometry calculation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C1S0: BS%s - C0S0 geometry started.' % (molecule_id, basis_set)))
            return True
        elif filtered_line.loc[:, 'C0S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C-1S0: BS%s - C0S0 geometry' % (basis_set)].values[0] == 'Not Started':
            # start C0S0 optimal geometry at C-1S0 energy #####################
            job_id = self.submit_molecule_for_marcus_calculation(molecule_id = molecule_id,
                                                                 charge_geom = 0,
                                                                 charge_eval = -1,
                                                                 state_geom = 0,
                                                                 state_eval = 0,
                                                                 functional = filtered_line.loc[:, 'Functional'].values[0],
                                                                 basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                                 number_of_proc = int(job_parameters['number_of_proc']),
                                                                 memory = int(job_parameters['memory']),
                                                                 timeout_hr = int(job_parameters['timeout_hr']),
                                                                 keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C-1S0: BS%s - C0S0 geometry' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C-1S0: BS%s - C0S0 geometry calculation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C-1S0: BS%s - C0S0 geometry started.' % (molecule_id, basis_set)))
            return True
        # C0S1 optimal geometry completed #####################################
        elif filtered_line.loc[:, 'C0S1: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C0S0: BS%s - C0S1 geometry' % (basis_set)].values[0] == 'Not Started':
            # start C0S1 optimal geometry at C0S0 energy ######################
            job_id = self.submit_molecule_for_marcus_calculation(molecule_id = molecule_id,
                                                                 charge_geom = 0,
                                                                 charge_eval = 0,
                                                                 state_geom = 1,
                                                                 state_eval = 0,
                                                                 functional = filtered_line.loc[:, 'Functional'].values[0],
                                                                 basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                                 number_of_proc = int(job_parameters['number_of_proc']),
                                                                 memory = int(job_parameters['memory']),
                                                                 timeout_hr = int(job_parameters['timeout_hr']),
                                                                 keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C0S0: BS%s - C0S1 geometry' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C0S0: BS%s - C0S1 geometry calculation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C0S0: BS%s - C0S1 geometry started.' % (molecule_id, basis_set)))
            return True
        # C1S0 optimal geometry completed #####################################
        elif filtered_line.loc[:, 'C1S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C0S0: BS%s - C1S0 geometry' % (basis_set)].values[0] == 'Not Started':
            # start C1S0 optimal geometry at C0S0 energy ######################
            job_id = self.submit_molecule_for_marcus_calculation(molecule_id = molecule_id,
                                                                 charge_geom = 1,
                                                                 charge_eval = 0,
                                                                 state_geom = 0,
                                                                 state_eval = 0,
                                                                 functional = filtered_line.loc[:, 'Functional'].values[0],
                                                                 basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                                 number_of_proc = int(job_parameters['number_of_proc']),
                                                                 memory = int(job_parameters['memory']),
                                                                 timeout_hr = int(job_parameters['timeout_hr']),
                                                                 keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C0S0: BS%s - C1S0 geometry' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C0S0: BS%s - C1S0 geometry calculation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C0S0: BS%s - C1S0 geometry started.' % (molecule_id, basis_set)))
            return True
        # C-1S0 optimal geometry completed ####################################
        elif filtered_line.loc[:, 'C-1S0: BS%s' % (basis_set)].values[0] == 'Completed' and filtered_line.loc[:, 'C0S0: BS%s - C-1S0 geometry' % (basis_set)].values[0] == 'Not Started':
            # start C-1S0 optimal geometry at C0S0 energy #####################
            job_id = self.submit_molecule_for_marcus_calculation(molecule_id = molecule_id,
                                                                 charge_geom = -1,
                                                                 charge_eval = 0,
                                                                 state_geom = 0,
                                                                 state_eval = 0,
                                                                 functional = filtered_line.loc[:, 'Functional'].values[0],
                                                                 basisset = filtered_line.loc[:, 'Basis Set %s' % (basis_set)].values[0],
                                                                 number_of_proc = int(job_parameters['number_of_proc']),
                                                                 memory = int(job_parameters['memory']),
                                                                 timeout_hr = int(job_parameters['timeout_hr']),
                                                                 keyword_line_arguments_input = [])
            self._tracker.update_entry(_id = molecule_id, 
                                       keys = ['C0S0: BS%s - C-1S0 geometry' % (basis_set)], 
                                       values = [job_id])
            self.add_activity_log(molecule_id = molecule_id, 
                                  line = 'C0S0: BS%s - C-1S0 geometry calculation started: %s' % (basis_set, str(datetime.now()).split('.')[0]))
            self.print_if_debug(('%s C0S0: BS%s - C-1S0 geometry started.' % (molecule_id, basis_set)))
            return True
        else:
            return False
                
            
    def next_step_for_molecule_id(self, molecule_id: str):
        filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        if len(filtered_line) == 0:
            raise Exception('molecule_id not found in tracker.')
        job_submitted: bool = False
        # basis set 1
        if filtered_line.loc[:, 'C0S0: BS1 - C-1S0 geometry'].values[0] != 'Completed':
            job_submitted = self.next_step_for_molecule_id_with_basis_set(molecule_id = molecule_id, 
                                                                          basis_set = 1)
        # basis set 2
        if job_submitted == False and (filtered_line.loc[:, 'C0S0: BS2 - C-1S0 geometry'].values[0] != 'Completed' or filtered_line.loc[:, 'C0S0: BS2 - C1S0 geometry'].values[0] != 'Completed' or filtered_line.loc[:, 'C0S0: BS2 - C0S1 geometry'].values[0] != 'Completed') and filtered_line.loc[:, 'C0S0: BS1'].values[0] == 'Completed':
            job_submitted = self.next_step_for_molecule_id_with_basis_set(molecule_id = molecule_id, 
                                                                          basis_set = 2)
        # basis set 3
        if job_submitted == False and (filtered_line.loc[:, 'C0S0: BS3 - C-1S0 geometry'].values[0] != 'Completed' or filtered_line.loc[:, 'C0S0: BS3 - C1S0 geometry'].values[0] != 'Completed' or filtered_line.loc[:, 'C0S0: BS3 - C0S1 geometry'].values[0] != 'Completed') and filtered_line.loc[:, 'C0S0: BS2'].values[0] == 'Completed':
            job_submitted = self.next_step_for_molecule_id_with_basis_set(molecule_id = molecule_id, 
                                                                          basis_set = 3)     
        if job_submitted == True:
            return True
        else:
            return False
        
    def check_qstat_and_transfer_files(self,
                                       stop_time = ''):
        if stop_time == '':
            stop_time = datetime.now() + timedelta(minutes = 15)
        else:
            stop_time = datetime.now() + timedelta(minutes = stop_time)
        if datetime.now() >= stop_time:
            return
        qstat = self.get_HPC_qstat()
        running_only = self.tracker_summarise_running()
        for i in range(0, len(running_only)):
            if datetime.now() >= stop_time:
                return
            if str(running_only.at[i, 'run_id']) not in qstat['Job ID'].values:
                molecule_id = running_only.at[i, 'molecule_id']
                filtered_line = self.get_entries_from_molecule_id(molecule_id = molecule_id)
                charge_state, basis_set = running_only.at[i, 'run'].split(': ')
                charge = charge_state.split('S')[0].replace('C', '')
                excitation_state = charge_state.split('S')[1]
                if basis_set == 'BS1' or basis_set == 'BS2' or basis_set == 'BS3':
                    if basis_set == 'BS1':
                        basis_set = 'Basis Set 1'
                    elif basis_set == 'BS2':
                        basis_set = 'Basis Set 2'
                    elif basis_set == 'BS3':
                        basis_set = 'Basis Set 3'
                    print(charge_state, basis_set, charge, excitation_state)
                    self.transfer_chk_fchk_log_files_from_HPC(molecule_id = molecule_id,
                                                              charge = charge,
                                                              excitation_state = excitation_state, 
                                                              functional = filtered_line['Functional'].values[0],
                                                              basisset = filtered_line[basis_set].values[0])
                    convergence = self.gaussian_normal_termination(molecule_id = molecule_id, 
                                                                   functional = filtered_line['Functional'].values[0],
                                                                   basis_set = format_basis_set(filtered_line['%s' % (basis_set)].values[0]),
                                                                   charge = charge, 
                                                                   excitation_state = excitation_state)
                    self._tracker.update_entry(_id = molecule_id, 
                                               keys = ['%s' % (running_only.at[i, 'run'])], 
                                               values = [convergence])
                    self.add_activity_log(molecule_id = molecule_id, 
                                          line = '%s BS%s files transferred from HPC' % (charge_state, basis_set.split(' ')[-1]))
                else:
                    geom_charge_state = basis_set.split(' - ')[1].split(' ')[0]
                    basis_set = basis_set.split(' - ')[0]
                    if basis_set == 'BS1':
                        basis_set = 'Basis Set 1'
                    elif basis_set == 'BS2':
                        basis_set = 'Basis Set 2'
                    elif basis_set == 'BS3':
                        basis_set = 'Basis Set 3'
                    charge_eval = charge
                    state_eval = excitation_state
                    charge_geom = geom_charge_state.split('S')[0].replace('C', '')
                    state_geom = geom_charge_state.split('S')[1]
                    self.transfer_chk_fchk_log_marcus_files_from_HPC(molecule_id = molecule_id,
                                                                     charge_geom = charge_geom,
                                                                     charge_eval = charge_eval,
                                                                     state_geom = state_geom,
                                                                     state_eval = state_eval,
                                                                     functional = filtered_line['Functional'].values[0],
                                                                     basisset = filtered_line[basis_set].values[0])
                    convergence = self.gaussian_marcus_normal_termination(molecule_id = molecule_id, 
                                                                          functional = filtered_line['Functional'].values[0],
                                                                          basis_set = filtered_line[basis_set].values[0],
                                                                          charge_geom = charge_geom,
                                                                          charge_eval = charge_eval,
                                                                          state_geom = state_geom,
                                                                          state_eval = state_eval)
                    self._tracker.update_entry(_id = molecule_id, 
                                               keys = ['%s' % (running_only.at[i, 'run'])], 
                                               values = [convergence])
                    self.add_activity_log(molecule_id = molecule_id, 
                                          line = '%s BS%s at C%sS%s geometry files transferred from HPC' % (charge_state, charge_geom, state_geom, basis_set.split(' ')[-1]))
        # self.get_tracker()
        
    def automate_submissions(self,
                             stop_time = '',
                             max_jobs_parallel: int = 40):
        if stop_time == '':
            stop_time = datetime.now() + timedelta(minutes = 15)
        if datetime.now() >= stop_time:
            return
        current_running = self.count_running_optimisations_alt()
        to_be_run = self.get_tracker_not_started().reset_index()
        for index in range(0, len(to_be_run)):
            molecule_id = to_be_run.at[index, '_id']
            job_submitted = True
            while job_submitted == True and current_running <= max_jobs_parallel:
                job_submitted = self.next_step_for_molecule_id(molecule_id = molecule_id)
                if job_submitted == True:
                    current_running += 1
                if datetime.now() >= stop_time:
                    # self.download_tracker_as_excel()
                    # raise Exception('\nSubmission terminated automatically, stop_time acheived. Total running currently = %s.' % (self.count_running_optimisations_alt()))
                    return
        self.print_if_debug('\nTotal running currently = %s' % (current_running))
        self.download_tracker_as_excel()
        
    def automate_submission_for_period_of_time(self,
                                               total_time_minutes: int = 10,
                                               pause_between_runs_minutes: int = 2,
                                               max_jobs_parallel: int = 40, 
                                               jobs_reserved_for_restart: int = 3):
        time_end = datetime.now() + timedelta(minutes = total_time_minutes)
        run = True
        while datetime.now() <= time_end - timedelta(minutes = pause_between_runs_minutes) and run == True:
            with open(r'%s/Currently Running.txt' % (self._local_folder), 'wt') as file:
                string = 'Started: %s\nEnding: %s' % (datetime.now(), time_end)
                file.write(string)
                file.close()
            if len(self.get_tracker_not_started()) == 0:
                run = False
            self.check_qstat_and_transfer_files(stop_time = time_end)
            self.automate_submissions(stop_time = time_end,
                                      max_jobs_parallel = max_jobs_parallel - jobs_reserved_for_restart)
            #self.restart_failed_cases(stop_time = time_end,
            #                          max_jobs_parallel = max_jobs_parallel)
            self.update_results_all(stop_time = time_end)
            if datetime.now() <= time_end - timedelta(minutes = pause_between_runs_minutes):
                pd.set_option('display.expand_frame_repr', False)
                print(test_reorg_e.get_HPC_qstat())
                pd.set_option('display.expand_frame_repr', True)
                print('\nEntering sleep mode for %s minutes.' % (pause_between_runs_minutes))
                print('Safe to terminate kernal. Program ending at %s.' % (time_end))
                wait_time = pause_between_runs_minutes * 60
                sleep_start = datetime.now()
                for i in tqdm(range(0, int(wait_time/5)), position = 0, leave = True):
                    while datetime.now() < sleep_start + timedelta(seconds = i * 5):
                        time.sleep(5)
        self.download_tracker_as_excel()
        print('\nSubmission terminated automatically. Total running currently = %s.' % (self.count_running_optimisations_alt()))
    
    def get_orbitals(self, 
                     molecule_id: str, 
                     filename: str):
        with open(r'%s/Data/%s/%s.log' % (self._local_folder, molecule_id, filename.replace('.log', '').replace('fchk', '').replace('chk', '')), 'r') as file:
            whole_file = file.read()
            lines = whole_file.split('\n')
            number_of_lines_in_whole_file = len(lines)
            population_analysis_start_line = 0
            population_analysis_end_line = 0
            
            found_start = False
            current_line = number_of_lines_in_whole_file
            while found_start == False:
                current_line -= 1
                if 'Population analysis using the SCF Density.' in lines[current_line]:
                    found_start = True
                    population_analysis_start_line = current_line - 2
            
            found_end = False
            current_line = population_analysis_start_line + 5
            while found_end == False:
                current_line += 1
                if 'Alpha' in lines[current_line]:
                    population_analysis_end_line = current_line
                    # print(lines[current_line])
                else:
                    found_end = True
                    
            alpha_occ_eigenvalues = []
            alpha_virt_eigenvalues = []
            
            parsed_lines = lines[population_analysis_start_line:population_analysis_end_line]
            parsed_lines_txt = '\n'.join(parsed_lines)
            
            for line in parsed_lines:
                if 'Alpha  occ. eigenvalues -- ' in line:
                    line = line.replace(' Alpha  occ. eigenvalues --  ', '').split(' ')
                    for energy in line:
                        if energy != '':
                            alpha_occ_eigenvalues.append(float(energy))
                if 'Alpha virt. eigenvalues --   ' in line:
                    line = line.replace('Alpha virt. eigenvalues --  ', '').split(' ')
                    for energy in line:
                        if energy != '':
                            alpha_virt_eigenvalues.append(float(energy))
            alpha_occ_eigenvalues = list(np.flip(alpha_occ_eigenvalues))
            alpha_virt_eigenvalues = alpha_virt_eigenvalues
        
        HOMO_LUMO_gap = float(alpha_virt_eigenvalues[0]) - float(alpha_occ_eigenvalues[0])
        
        return alpha_occ_eigenvalues, alpha_virt_eigenvalues, HOMO_LUMO_gap, parsed_lines_txt
    
    def get_SCF_energy_one(self, 
                           molecule_id: str,
                           filename:str):
        with open(r"%s/Data/%s/%s.log" % (self._local_folder, molecule_id, filename.replace('.log', '').replace('fchk', '').replace('chk', '')), 'r') as file:
            whole_file = file.read()
            lines = whole_file.split('\n')
            lines_to_print = 0
            continue_printing_until_blank = False
            summary = ''
            for line in lines:
                if 'SCF Done' in line:
                    # lines_to_print = 1
                    continue_printing_until_blank = True
                    final_SCF_energy = float(line.split('=')[1].split('A.U.')[0].replace(' ', ''))
                if 'Cycle  ' in line:
                    continue_printing_until_blank = True
                if 'Item               Value     Threshold  Converged?' in line:
                    lines_to_print = 5
                if lines_to_print > 0:
                    summary += line + '\n'
                    lines_to_print -= 1
                    if lines_to_print == 0:
                        summary += '\n'
                elif continue_printing_until_blank:
                    if line == '':
                        continue_printing_until_blank = False
                        summary += '\n'
                    else:
                        summary += line + '\n'
            self.print_if_debug(summary)
            
            warning = ''
            if 'Item               Value     Threshold  Converged?' in summary:
                convergence = []
                parsed_summary = summary.split('Item               Value     Threshold  Converged?\n')[-1]
                for line in parsed_summary.split('\n'):
                    if line != '':
                        word = line.split('     ')[-1].replace(' ', '')
                        convergence.append(word)
                self.print_if_debug(convergence)
                if convergence == ['YES', 'YES', 'YES', 'YES']:
                    self.print_if_debug('convergence acheived')
                else:
                    # raise Exception('\nConvergence not acheived:\n%s\n%s' % ('Item; Value; Threshold; Converged?', parsed_summary))
                    warning = '\n%s\nConvergence not acheived:\n%s\n%s' % (filename.replace('.log', '').replace('fchk', '').replace('chk', ''), 'Item; Value; Threshold; Converged?', parsed_summary)
                    warnings.warn(warning)
                    summary = 'WARNING: CONVERGENCE NOT ACHEIVED\n' + summary
                
            self.print_if_debug('Final SCF energy = %s' % (final_SCF_energy))
            file.close()
        return final_SCF_energy, summary, warning
    
    def get_SCF_energy_excited_one(self, 
                                   molecule_id: str,
                                   filename:str):
        with open(r"%s/Data/%s/%s.log" % (self._local_folder, molecule_id, filename.replace('.log', '').replace('fchk', '').replace('chk', '')), 'r') as file:
            whole_file = file.read()
            lines = whole_file.split('\n')
            lines_to_print = 0
            continue_printing_until_blank = False
            summary = ''
            for line in lines:
                if 'Excited State   1' in line:
                    # lines_to_print = 1
                    continue_printing_until_blank = True
                if 'Total Energy, E(TD-HF/TD-DFT)' in line:
                    final_SCF_energy = float(line.split('=')[1].replace(' ', ''))
                if 'Cycle  ' in line:
                    continue_printing_until_blank = True
                if 'Item               Value     Threshold  Converged?' in line:
                    lines_to_print = 5
                if lines_to_print > 0:
                    summary += line + '\n'
                    lines_to_print -= 1
                    if lines_to_print == 0:
                        summary += '\n'
                elif continue_printing_until_blank:
                    if line == '':
                        continue_printing_until_blank = False
                        summary += '\n'
                    else:
                        summary += line + '\n'
            self.print_if_debug(summary)
            
            warning = ''
            if 'Item               Value     Threshold  Converged?' in summary:
                convergence = []
                parsed_summary = summary.split('Item               Value     Threshold  Converged?\n')[-1]
                for line in parsed_summary.split('\n'):
                    if line != '':
                        word = line.split('     ')[-1].replace(' ', '')
                        convergence.append(word)
                self.print_if_debug(convergence)
                if convergence == ['YES', 'YES', 'YES', 'YES']:
                    self.print_if_debug('convergence acheived')
                else:
                    # raise Exception('\nConvergence not acheived:\n%s\n%s' % ('Item; Value; Threshold; Converged?', parsed_summary))
                    warning = '\n%s\nConvergence not acheived:\n%s\n%s' % (filename.replace('.log', '').replace('fchk', '').replace('chk', ''), 'Item; Value; Threshold; Converged?', parsed_summary)
                    warnings.warn(warning)
                    summary = 'WARNING: CONVERGENCE NOT ACHEIVED\n' + summary
                
            self.print_if_debug('Final SCF energy = %s' % (final_SCF_energy))
            file.close()
        return final_SCF_energy, summary, warning
        
        
    def get_SCF_energies(self,
                         molecule_id:str, 
                         initial_charge_state: str,
                         final_charge_state: str,
                         formatted_functional_basisset: str):
        if final_charge_state != 'C0S1':
            initial_SCF_at_initial_geom, summary1, warning1 = self.get_SCF_energy_one(molecule_id = molecule_id,
                                                                                      filename = '%s_%s_%s' % (molecule_id, formatted_functional_basisset, initial_charge_state))
            final_SCF_at_final_geom, summary2, warning2 = self.get_SCF_energy_one(molecule_id = molecule_id,
                                                                                  filename = '%s_%s_%s' % (molecule_id, formatted_functional_basisset, final_charge_state))
            initial_SCF_at_final_geom, summary3, warning3 = self.get_SCF_energy_one(molecule_id = molecule_id,
                                                                                    filename = '%s_%s_%s_opt_geom_at_%s' % (molecule_id, formatted_functional_basisset, final_charge_state, initial_charge_state))
            final_SCF_at_initial_geom, summary4, warning4 = self.get_SCF_energy_one(molecule_id = molecule_id,
                                                                                    filename = '%s_%s_%s_opt_geom_at_%s' % (molecule_id, formatted_functional_basisset, initial_charge_state, final_charge_state))
        else:
            initial_SCF_at_initial_geom, summary1, warning1 = self.get_SCF_energy_one(molecule_id = molecule_id,
                                                                                      filename = '%s_%s_%s' % (molecule_id, formatted_functional_basisset, initial_charge_state))
            final_SCF_at_final_geom, summary2, warning2 = self.get_SCF_energy_excited_one(molecule_id = molecule_id,
                                                                                          filename = '%s_%s_%s' % (molecule_id, formatted_functional_basisset, final_charge_state))
            initial_SCF_at_final_geom, summary3, warning3 = self.get_SCF_energy_one(molecule_id = molecule_id,
                                                                                    filename = '%s_%s_%s_opt_geom_at_%s' % (molecule_id, formatted_functional_basisset, final_charge_state, initial_charge_state))
            final_SCF_at_initial_geom, summary4, warning4 = self.get_SCF_energy_excited_one(molecule_id = molecule_id,
                                                                                            filename = '%s_%s_%s_opt_geom_at_%s' % (molecule_id, formatted_functional_basisset, initial_charge_state, final_charge_state))
            pass
            
        summary = ('initial_SCF_at_initial_geom\n' + 
                   '\n' + 
                   summary1 + '\n' + 
                   'final_SCF_at_final_geom\n' + 
                   '\n' + 
                   summary2 + '\n' + 
                   'initial_SCF_at_final_geom\n' + 
                   '\n' + 
                   summary3 + '\n' + 
                   'final_SCF_at_initial_geom\n' + 
                   '\n' + 
                   summary4 + '\n')
        warning = ''
        for warn in [warning1, warning2, warning3, warning4]:
            if warn != '':
                warning += warn + '\n'
            
                  
        return initial_SCF_at_initial_geom, initial_SCF_at_final_geom, final_SCF_at_initial_geom, final_SCF_at_final_geom, summary, warning
    
    def do_DUSHIN_calculation(self,
                              molecule_id: str, 
                              filename_initial_state: str,
                              filename_final_state: str):
        self._HPC_connection.copy_fchk_log_files_for_dushin(remote_dushin_folder = r'%s/Logs_for_dushin/' % (self._remote_folder),
                                                            remote_log_folder = self._remote_folder,
                                                            filename_initial_state = filename_initial_state,
                                                            filename_final_state = filename_final_state)
        initial_state = filename_initial_state.split('_')[-1]
        final_state = filename_final_state.split('_')[-1]
        if filename_initial_state.replace('_%s' % (initial_state), '') == filename_final_state.replace('_%s' % (final_state), ''):
            mol_id_functional_basisset = filename_initial_state.replace('_%s' % (initial_state), '')
        else:
            raise Exception('\nIssue with mol_id_functional_basisset.')
        self._HPC_connection.submit_job_dushin(remote_dushin_folder = r'%s/Logs_for_dushin/' % (self._remote_folder),
                                               remote_bin_path = r'%s/Logs_for_dushin/' % (self._remote_folder),
                                               initial_state = initial_state,
                                               final_state = final_state,
                                               mol_id_functional_basisset = mol_id_functional_basisset)
        self._HPC_connection.transfer_from_HPC(local_folder = r'%s/Data/%s/%s_%s/' % (self._local_folder, molecule_id, initial_state, final_state), 
                                               remote_folder = r'%s/Logs_for_dushin/' % (self._remote_folder),
                                               filename = 'dushin%s_%s_%s.log' % (mol_id_functional_basisset, initial_state, final_state))
        with open(r'%s/Data/%s/%s_%s/dushin%s_%s_%s.log' % (self._local_folder, molecule_id, initial_state, final_state, mol_id_functional_basisset, initial_state, final_state), 'r') as file:
            dushin_output = file.readlines()
            split_index_top = dushin_output.index(" Displacement: in terms of nc of 1 THEN of nc of 2\n") + 2
            dushin_output = dushin_output[split_index_top::]
            split_index_bot = dushin_output.index("\n")
            dushin_output = dushin_output[0:split_index_bot]
            if (dushin_output[-1] != " "):
                dushin_output = dushin_output[:-1]
            dushin_results = pd.DataFrame(columns = ['Freq', 'Q/cm', 'lam/eV'])
            for line in dushin_output:
                # print(line)
                freq1 = float(line.split('freq=')[1].split('Q')[0].replace(' ', ''))
                freq2 = float(line.split('freq=')[2].split('Q')[0].replace(' ', ''))
                Q1 = float(line.split('Q=')[1].split('lam')[0].replace(' ', ''))
                Q2 = float(line.split('Q=')[2].split('lam')[0].replace(' ', ''))
                lam1 = float(line.split('lam=')[1].split('freq=')[0].replace(' ', '')) * constants.physical_constants['inverse meter-electron volt relationship'][0] * 100
                lam2 = float(line.split('lam=')[2].split('\n')[0].replace(' ', '')) * constants.physical_constants['inverse meter-electron volt relationship'][0] * 100
                if freq1 != 0 or Q1 != 0 or lam1 != 0:
                    dushin_results.loc[len(dushin_results)] = [freq1, Q1, lam1]
                if freq2 != 0 or Q2 != 0 or lam2 != 0:
                    dushin_results.loc[len(dushin_results)] = [freq2, Q2, lam2]
        dushin_summarised = dushin_results.sort_values('Freq', axis = 'index').reset_index(drop = True)
        dushin_summarised.to_excel(r'%s/Data/%s/%s_%s/dushin%s_%s_%s.xlsx' % (self._local_folder, molecule_id, initial_state, final_state, mol_id_functional_basisset, initial_state, final_state))
        self.print_if_debug(dushin_summarised)
        imaginary_frequencies = dushin_results[['Freq']].lt(0).any(axis = 'index').values[0]
        return dushin_summarised, imaginary_frequencies       
    
    def get_atoms_summary(self, 
                          molecule_id: str, 
                          filename: str):
        summary = pd.DataFrame(columns = ['Atomic Mass', 'Atomic Number', 'Coordinates', 'Charge', 'Number of Bonds', 'Hybridisation'])
        mol = next(pybel.readfile("log", r"%s/Data/%s/%s.log" % (self._local_folder, molecule_id, filename.replace('.log', '').replace('fchk', '').replace('chk', ''))))
        for entry in mol.atoms:
            self.print_if_debug([entry.atomicmass, entry.atomicnum, entry.coords, entry.formalcharge,  entry.degree, entry.hyb])
            summary.loc[len(summary)] = [entry.atomicmass, entry.atomicnum, entry.coords, entry.formalcharge,  entry.degree, entry.hyb]
        del mol
        return summary
        
    def organise_results_one(self,
                              molecule_id: str, 
                              # new_entry: bool = False, 
                              download_results: bool = False,
                              tracker = '', 
                              results = '',
                              basis_set_list: list = [1, 2, 3], 
                              transition_list: list = ['C0S0_C0S1', 'C0S0_C1S0', 'C0S0_C-1S0'], 
                              overwrite_: bool = False):
        if type(tracker) == str:
            tracker = self.get_entries_from_molecule_id(molecule_id = molecule_id)
        else:
            tracker = tracker.loc[tracker['_id'] == molecule_id]
        if type(results) == str:
            results = self.get_results()
        else:
            tracker = tracker.loc[tracker['_id'] == molecule_id]
        for basis_set in basis_set_list:
            for transitions in transition_list:
                keys = []
                values = []
                initial = transitions.split('_')[0]
                final = transitions.split('_')[1]
                sub_directory = r'%s/Data/%s/%s/' % (self._local_folder, molecule_id, transitions)
                check_directory(r'%s' % (sub_directory))
                
                if overwrite_ == False:
                    if results.loc[results['_id'] == '%s_%s_%s_%s' % (molecule_id, tracker.loc[tracker['_id'] == molecule_id, 'Functional'].values[0], format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (basis_set)].values[0]), transitions), 'Last Updated'].values[0] == None:
                        if (tracker['%s: BS%s' % (initial, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s' % (final, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (final, basis_set, initial)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (initial, basis_set, final)].values[0] == 'Completed'):
                            run = True
                        else:
                            run = False
                    elif tracker.loc[tracker['_id'] == molecule_id, 'Last Updated'].values[0] > results.loc[results['_id'] == '%s_%s_%s_%s' % (molecule_id, tracker.loc[tracker['_id'] == molecule_id, 'Functional'].values[0], format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (basis_set)].values[0]), transitions), 'Last Updated'].values[0]:
                        if (tracker['%s: BS%s' % (initial, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s' % (final, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (final, basis_set, initial)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (initial, basis_set, final)].values[0] == 'Completed'):
                            run = True
                        else:
                            run = False
                    else:
                        run = False
                else:
                    if (tracker['%s: BS%s' % (initial, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s' % (final, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (final, basis_set, initial)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (initial, basis_set, final)].values[0] == 'Completed'):
                        run = True
                    else:
                        run = False
                
                if run == True:
                    print(r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), final))
                    # Orbitals and HOMO-LUMO handling
                    alpha_occ_eigenvalues, alpha_virt_eigenvalues, HOMO_LUMO_gap, parsed_lines_txt = self.get_orbitals(molecule_id = molecule_id, filename = r'%s_%s_%s_C0S0' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0])))
                    with open(r'%s/GS_Orbitals.txt' % (sub_directory), 'wt') as file:
                        file.write(parsed_lines_txt)
                        file.close()
                    keys.append('Alpha Occ Orbitals List')
                    values.append(alpha_occ_eigenvalues)
                    keys.append('Alpha Virt Orbitals List')
                    values.append(alpha_virt_eigenvalues)
                    keys.append('HOMO-LUMO Gap')
                    values.append(HOMO_LUMO_gap * constants.physical_constants['Hartree energy in eV'][0])
                    
                    # SCF energies
                    initial_SCF_at_initial_geom, initial_SCF_at_final_geom, final_SCF_at_initial_geom, final_SCF_at_final_geom, summary, warning = self.get_SCF_energies(molecule_id = molecule_id, 
                                                                                                                                                                         initial_charge_state = initial,
                                                                                                                                                                         final_charge_state = final,
                                                                                                                                                                         formatted_functional_basisset = '%s_%s' % (tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0])))
                    with open(r'%s/%s_%s_%s_SCF_Summary.txt' % (sub_directory, molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0])), 'wt') as file:
                        file.write(summary)
                        file.close()
                    keys.append('Initial CS at Initial Optimised Geometry')
                    values.append(initial_SCF_at_initial_geom)
                    keys.append('Initial CS at Final Optimised Geometry')
                    values.append(initial_SCF_at_final_geom)
                    keys.append('Final CS at Initial Optimised Geometry')
                    values.append(final_SCF_at_initial_geom)
                    keys.append('Final CS at Final Optimised Geometry')
                    values.append(final_SCF_at_final_geom)
                    keys.append('Parsed SCF Summary')
                    values.append(summary)
                    four_point = initial_SCF_at_final_geom - initial_SCF_at_initial_geom + final_SCF_at_initial_geom - final_SCF_at_final_geom
                    keys.append('4-Point Reorganisation Energy')
                    values.append(four_point * constants.physical_constants['Hartree energy in eV'][0])
                    if warning != '':
                        self.comment_result(_id = '%s_%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial, final),
                                            user = 'Auto-Generated',
                                            string = warning)
                    
                    # DUSHIN
                    filename_initial_state = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial)
                    filename_final_state = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), final)
                    dushin_summarised, imaginary_frequencies = self.do_DUSHIN_calculation(molecule_id = molecule_id, 
                                                                                          filename_initial_state = filename_initial_state,
                                                                                          filename_final_state = filename_final_state)
                    keys.append('Vibrational Frequencies')
                    values.append(list(dushin_summarised['Freq'].values))
                    keys.append('Displacements')
                    values.append(list(dushin_summarised['Q/cm'].values))
                    keys.append('Reorganisation Energies')
                    values.append(list(dushin_summarised['lam/eV'].values))
                    keys.append('DUSHIN Reorganisation Energy')
                    values.append(dushin_summarised['lam/eV'].sum())
                    
                    # atoms summary
                    initial_atoms_summary = self.get_atoms_summary(molecule_id = molecule_id, 
                                                                   filename = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial))
                    final_atoms_summary = self.get_atoms_summary(molecule_id = molecule_id, 
                                                                 filename = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), final))
                    keys.append('Initial Atoms Summary')
                    values.append(initial_atoms_summary.to_string())
                    keys.append('Final Atoms Summary')
                    values.append(final_atoms_summary.to_string())
                    
                    # simple energy analysis
                    difference = (four_point * constants.physical_constants['Hartree energy in eV'][0]) - (dushin_summarised['lam/eV'].sum())
                    frac_diff = difference/(four_point * constants.physical_constants['Hartree energy in eV'][0])
                    keys.append('Difference (4-point & DUSHIN)')
                    values.append(difference)
                    keys.append('Difference Percentage (4-point & DUSHIN)')
                    values.append('%.3f' % (frac_diff * 100))
                    
                    self._results.update_entry(molecule_id = '%s_%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial, final),
                                               keys = keys, 
                                               values = values)
                
                
                # if (tracker['%s: BS%s' % (initial, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s' % (final, basis_set)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (final, basis_set, initial)].values[0] == 'Completed') and (tracker['%s: BS%s - %s geometry' % (initial, basis_set, final)].values[0] == 'Completed'):
                #     if tracker.loc[tracker['_id'] == molecule_id, 'Last Updated'].values[0] > results.loc[results['_id'] == '%s_%s_%s_%s' % (molecule_id, tracker.loc[tracker['_id'] == molecule_id, 'Functional'].values[0], format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (1)].values[0]), transitions), 'Last Updated'].values[0]:
                #         # if new_entry == True and '%s_%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (basis_set)].values[0]), initial, final) not in results['_id'].values:
                #         #     self._results.create_empty_entry(molecule_id = molecule_id, 
                #         #                                      iupac_name = tracker['IUPAC Name'].values[0], 
                #         #                                      SMILES = tracker['SMILES'].values[0], 
                #         #                                      MMFFO = tracker['MMFFO'].values[0], 
                #         #                                      functional = tracker['Functional'].values[0], 
                #         #                                      basis_set = tracker['Basis Set %s' % (basis_set)].values[0], 
                #         #                                      initial_charge_and_state = initial, 
                #         #                                      final_charge_and_state = final)
                    
                #         # Orbitals and HOMO-LUMO handling
                #         alpha_occ_eigenvalues, alpha_virt_eigenvalues, HOMO_LUMO_gap, parsed_lines_txt = self.get_orbitals(molecule_id = molecule_id, filename = r'%s_%s_%s_C0S0' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0])))
                #         with open(r'%s/GS_Orbitals.txt' % (sub_directory), 'wt') as file:
                #             file.write(parsed_lines_txt)
                #             file.close()
                #         keys.append('Alpha Occ Orbitals List')
                #         values.append(alpha_occ_eigenvalues)
                #         keys.append('Alpha Virt Orbitals List')
                #         values.append(alpha_virt_eigenvalues)
                #         keys.append('HOMO-LUMO Gap')
                #         values.append(HOMO_LUMO_gap * constants.physical_constants['Hartree energy in eV'][0])
                        
                #         # SCF energies
                #         initial_SCF_at_initial_geom, initial_SCF_at_final_geom, final_SCF_at_initial_geom, final_SCF_at_final_geom, summary, warning = self.get_SCF_energies(molecule_id = molecule_id, 
                #                                                                                                                                                              initial_charge_state = initial,
                #                                                                                                                                                              final_charge_state = final,
                #                                                                                                                                                              formatted_functional_basisset = '%s_%s' % (tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0])))
                #         with open(r'%s/%s_%s_%s_SCF_Summary.txt' % (sub_directory, molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0])), 'wt') as file:
                #             file.write(summary)
                #             file.close()
                #         keys.append('Initial CS at Initial Optimised Geometry')
                #         values.append(initial_SCF_at_initial_geom)
                #         keys.append('Initial CS at Final Optimised Geometry')
                #         values.append(initial_SCF_at_final_geom)
                #         keys.append('Final CS at Initial Optimised Geometry')
                #         values.append(final_SCF_at_initial_geom)
                #         keys.append('Final CS at Final Optimised Geometry')
                #         values.append(final_SCF_at_final_geom)
                #         keys.append('Parsed SCF Summary')
                #         values.append(summary)
                #         four_point = initial_SCF_at_final_geom - initial_SCF_at_initial_geom + final_SCF_at_initial_geom - final_SCF_at_final_geom
                #         keys.append('4-Point Reorganisation Energy')
                #         values.append(four_point * constants.physical_constants['Hartree energy in eV'][0])
                #         if warning != '':
                #             self.comment_result(_id = '%s_%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial, final),
                #                                 user = 'Auto-Generated',
                #                                 string = warning)
                        
                #         # DUSHIN
                #         filename_initial_state = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial)
                #         filename_final_state = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), final)
                #         dushin_summarised, imaginary_frequencies = self.do_DUSHIN_calculation(molecule_id = molecule_id, 
                #                                                                               filename_initial_state = filename_initial_state,
                #                                                                               filename_final_state = filename_final_state)
                #         keys.append('Vibrational Frequencies')
                #         values.append(list(dushin_summarised['Freq'].values))
                #         keys.append('Displacements')
                #         values.append(list(dushin_summarised['Q/cm'].values))
                #         keys.append('Reorganisation Energies')
                #         values.append(list(dushin_summarised['lam/eV'].values))
                #         keys.append('DUSHIN Reorganisation Energy')
                #         values.append(dushin_summarised['lam/eV'].sum())
                        
                #         # atoms summary
                #         initial_atoms_summary = self.get_atoms_summary(molecule_id = molecule_id, 
                #                                                        filename = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial))
                #         final_atoms_summary = self.get_atoms_summary(molecule_id = molecule_id, 
                #                                                      filename = r'%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), final))
                #         keys.append('Initial Atoms Summary')
                #         values.append(initial_atoms_summary.to_string())
                #         keys.append('Final Atoms Summary')
                #         values.append(final_atoms_summary.to_string())
                        
                #         # simple energy analysis
                #         difference = (four_point * constants.physical_constants['Hartree energy in eV'][0]) - (dushin_summarised['lam/eV'].sum())
                #         frac_diff = difference/(four_point * constants.physical_constants['Hartree energy in eV'][0])
                #         keys.append('Difference (4-point & DUSHIN)')
                #         values.append(difference)
                #         keys.append('Difference Percentage (4-point & DUSHIN)')
                #         values.append('%.3f' % (frac_diff * 100))
                        
                #         self._results.update_entry(molecule_id = '%s_%s_%s_%s_%s' % (molecule_id, tracker['Functional'].values[0], format_basis_set(tracker['Basis Set %s' % (basis_set)].values[0]), initial, final),
                #                                    keys = keys, 
                #                                    values = values)
                    
        
                
    def update_results_all(self, 
                           stop_time: str = ''):
        if stop_time == '':
            stop_time = datetime.now() + timedelta(minutes = 15)
        if datetime.now() >= stop_time:
            return
        tracker = self.get_tracker()
        results = self.get_results()
        for molecule_id in tracker['_id'].values:
            functional = tracker.loc[tracker['_id'] == molecule_id, 'Functional'].values[0]
            # basis_set = format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set 1'].values[0])
            # basis_set = []
            # if '%s_%s_%s_C0S0_C0S1' % (molecule_id, functional, format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set 1'].values[0])) not in results['_id'].values:
            #     basis_set.append(1)
            # if '%s_%s_%s_C0S0_C0S1' % (molecule_id, functional, format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set 2'].values[0])) not in results['_id'].values:
            #     basis_set.append(2)
            # if '%s_%s_%s_C0S0_C0S1' % (molecule_id, functional, format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set 3'].values[0])) not in results['_id'].values:
            #     basis_set.append(3)
            # if basis_set != []: # basis_set is entries NOT PRESENT in collection
            #     self.organise_results_one(molecule_id = molecule_id, 
            #                               # new_entry = True, 
            #                               download_results = False, 
            #                               tracker = tracker, 
            #                               results = results, 
            #                               basis_set_list = basis_set)
            #     basis_set = [x for x in [1, 2, 3] if x not in basis_set] # basis_set is entries PRESENT in collection
            #     if basis_set != []:
            #         for x in basis_set:
            #             if tracker.loc[tracker['_id'] == molecule_id, 'Last Updated'].values[0] > results.loc[results['_id'] == '%s_%s_%s_C0S0_C0S1' % (molecule_id, functional, format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (x)].values[0])), 'Last Updated'].values[0]:
            #                 self.organise_results_one(molecule_id = molecule_id, 
            #                                           # new_entry = False, 
            #                                           download_results = False, 
            #                                           tracker = tracker, 
            #                                           results = results, 
            #                                           basis_set_list = [x])
            # else: # basis_set == []; all basis sets in collection
            #     basis_set = [x for x in [1, 2, 3] if x not in basis_set]
            #     for x in basis_set:
            #         if tracker.loc[tracker['_id'] == molecule_id, 'Last Updated'].values[0] > results.loc[results['_id'] == '%s_%s_%s_C0S0_C0S1' % (molecule_id, functional, format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (x)].values[0])), 'Last Updated'].values[0]:
            #             self.organise_results_one(molecule_id = molecule_id, 
            #                                       # new_entry = False, 
            #                                       download_results = False, 
            #                                       tracker = tracker, 
            #                                       results = results, 
            #                                       basis_set_list = [x])
            if results.loc[results['_id'] == '%s_%s_%s_C0S0_C0S1' % (molecule_id, functional, format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (1)].values[0])), 'Last Updated'].values[0] == None:
                self.organise_results_one(molecule_id = molecule_id, 
                                          # new_entry = False, 
                                          download_results = False, 
                                          tracker = tracker, 
                                          results = results, 
                                          basis_set_list = [1, 2, 3])
            else:
                if tracker.loc[tracker['_id'] == molecule_id, 'Last Updated'].values[0] > results.loc[results['_id'] == '%s_%s_%s_C0S0_C0S1' % (molecule_id, functional, format_basis_set(tracker.loc[tracker['_id'] == molecule_id, 'Basis Set %s' % (1)].values[0])), 'Last Updated'].values[0]:
                    self.organise_results_one(molecule_id = molecule_id, 
                                              # new_entry = False, 
                                              download_results = False, 
                                              tracker = tracker, 
                                              results = results, 
                                              basis_set_list = [1, 2, 3])
        self.download_results_as_excel()
        
if __name__ == '__main__': # ReorganisationEnergy
    test_reorg_e = ReorganisationEnergy(HPC_username = 'kbc121',
                                        HPC_password = 'Imperi@l020711',
                                        HPC_hostname = 'login.hpc.ic.ac.uk',
                                        MongoDB_username = 'kbc121',
                                        MongoDB_password = 'Imperi@l020711',
                                        MongoDB_database = 'New_Reorganisation_Energy_Calc',
                                        MongoDB_SMILES_database_collection = 'SMILES_Database',
                                        MongoDB_tracker_collection = 'Tracker',
                                        MongoDB_connection_string = 'mongodb+srv://<user>:<password>@cluster0.wxpvcwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0',
                                        local_folder = r"C:/Users/kyebchoo/OneDrive - Imperial College London/Desktop/Physics/Year 3/PHYS60016 - BSc Project T2/reorg_calculation/new_version/", 
                                        remote_folder = r'/rds/general/ephemeral/user/kbc121/ephemeral/test',
                                        debug = True)
    # test_reorg_e.add_workflow(MMFFO = 'Yes',
    #                           functional = 'CAM-B3LYP',
    #                           basis_set_1 = 'STO-3G',
    #                           basis_set_2 = 'STO-4G',
    #                           basis_set_3 = 'STO-6G')
    # test_reorg_e.add_molecule(iupac_name = 'phenanthrene', SMILES = 'C1=CC=C2C(=C1)C=CC3=CC=CC=C32')
    # test_reorg_e.add_molecule(iupac_name = 'chrysene', SMILES = 'C1=CC=C2C(=C1)C=CC3=C2C=CC4=CC=CC=C43')
    # test_reorg_e.add_molecule(iupac_name = 'picene', SMILES = 'C1=CC=C2C(=C1)C=CC3=C2C=CC4=C3C=CC5=CC=CC=C54')
    # test_reorg_e.add_molecule(iupac_name = 'hexacyclo[12.12.0.02,11.05,10.015,24.018,23]hexacosa-1(14),2(11),3,5,7,9,12,15(24),16,18,20,22,25-tridecaene', SMILES = 'C1=CC=C2C(=C1)C=CC3=C2C=CC4=C3C=CC5=C4C=CC6=CC=CC=C65')
    # test_reorg_e.add_molecule(iupac_name = 'heptacyclo[16.12.0.02,15.05,14.06,11.019,28.022,27]triaconta-1(18),2(15),3,5(14),6,8,10,12,16,19(28),20,22,24,26,29-pentadecaene', SMILES = 'C1=CC=C2C(=C1)C=CC3=C2C=CC4=C3C=CC5=C4C=CC6=C5C=CC7=CC=CC=C76')
    # test_reorg_e.add_molecule(iupac_name = 'octacyclo[16.16.0.02,15.05,14.06,11.019,32.022,31.023,28]tetratriaconta-1(18),2(15),3,5(14),6,8,10,12,16,19(32),20,22(31),23,25,27,29,33-heptadecaene', SMILES = 'C1=CC=C2C(=C1)C=CC3=C2C=CC4=C3C=CC5=C4C=CC6=C5C=CC7=C6C=CC8=CC=CC=C87')
    # test_reorg_e.submit_molecule_for_calculation(input_object = '', 
    #                                              molecule_id = 'benzene_20240319151751', 
    #                                              excitation_state = 0, 
    #                                              charge = 0, 
    #                                              functional = 'B3LYP', 
    #                                              basisset = '6-31G(d)', 
    #                                              number_of_proc = 4, 
    #                                              memory = 16, 
    #                                              timeout_hr = 1)
    # test_reorg_e.transfer_chk_fchk_log_files_from_HPC(molecule_id = 'benzene_20240319151751',
    #                                                   charge = 0,
    #                                                   excitation_state = 0,
    #                                                   functional = 'B3LYP',
    #                                                   basisset = '6-31G(d)')
    # test_reorg_e.submit_molecule_for_calculation(input_object = r"C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version\Data\benzene_20240319151751\benzene_20240319151751_C0S0_B3LYP_6-31Gd.log", 
    #                                               molecule_id = 'benzene_20240319151751', 
    #                                               excitation_state = 0, 
    #                                               charge = 0, 
    #                                               functional = 'B3LYP', 
    #                                               basisset = '6-311+G(d,p)', 
    #                                               number_of_proc = 4, 
    #                                               memory = 16, 
    #                                               timeout_hr = 1)
    # test_reorg_e.transfer_chk_fchk_log_files_from_HPC(molecule_id = 'benzene_20240319151751',
    #                                                   charge = 0,
    #                                                   excitation_state = 0,
    #                                                   functional = 'B3LYP',
    #                                                   basisset = '6-311+G(d,p)')
    # test_reorg_e.submit_molecule_for_calculation(input_object = r"C:\Users\kyebchoo\OneDrive - Imperial College London\Desktop\Physics\Year 3\PHYS60016 - BSc Project T2\reorg_calculation\new_version\Data\benzene_20240319151751\benzene_20240319151751_C0S0_B3LYP_6-311plusGdp.log", 
    #                                               molecule_id = 'benzene_20240319151751', 
    #                                               excitation_state = 1, 
    #                                               charge = 0, 
    #                                               functional = 'B3LYP', 
    #                                               basisset = '6-31G(d)', 
    #                                               number_of_proc = 8, 
    #                                               memory = 32, 
    #                                               timeout_hr = 1)
    # print(test_reorg_e.count_running_optimisations_alt())
    # print(test_reorg_e.next_step_for_molecule_id(molecule_id = 'benzene_20240331000232'))
    # print(test_reorg_e.get_tracker_running_only())
    # print(test_reorg_e.tracker_summarise_running())
    # test_reorg_e.check_qstat_and_transfer_files(stop_time = 60)
    # print(test_reorg_e.get_tracker_not_started()
    # print(test_reorg_e.get_tracker_running_only())
    # test_reorg_e.automate_submissions(max_jobs_parallel = 48)
    # test_reorg_e.automate_submission_for_period_of_time(total_time_minutes = 20,
    #                                                     pause_between_runs_minutes = 5,
                                                        # max_jobs_parallel = 45)
    # print(test_reorg_e.get_orbitals(molecule_id = 'napthale_20240331000319', filename = 'napthale_20240331000319_CAM-B3LYP_6-31G_C0S0_opt_geom_at_C-1S0'))
    # print(test_reorg_e.get_SCF_energy_one(molecule_id = 'napthale_20240331000319', filename = 'napthale_20240331000319_CAM-B3LYP_6-31G_C0S0_opt_geom_at_C-1S0'))
    # test_reorg_e.organise_results_one(molecule_id = 'benzene_20240331000232', new_entry = True)
    # initial_SCF_at_initial_geom, initial_SCF_at_final_geom, final_SCF_at_initial_geom, final_SCF_at_final_geom, summary = test_reorg_e.get_SCF_energies(molecule_id = 'napthale_20240331000319', 
    #                                                                                                                                                     initial_charge_state = 'C0S0',
    #                                                                                                                                                     final_charge_state = 'C0S1',
    #                                                                                                                                                     formatted_functional_basisset = 'CAM-B3LYP_6-31G')
    # test_reorg_e.do_DUSHIN_calculation(molecule_id = 'napthale_20240331000319',
    #                                    filename_initial_state = 'napthale_20240331000319_CAM-B3LYP_6-31G_C0S0',
    #                                    filename_final_state = 'napthale_20240331000319_CAM-B3LYP_6-31G_C0S1')
    # print(test_reorg_e.get_atoms_summary(molecule_id = 'napthale_20240331000319',
    #                                      filename = 'napthale_20240331000319_CAM-B3LYP_6-31G_C1S0'))
    test_reorg_e.update_results_all()
    # test_reorg_e.automate_submission_for_period_of_time(total_time_minutes = 8 * 60,
    #                                                     pause_between_runs_minutes = 45,
    #                                                     max_jobs_parallel = 45,
    #                                                     jobs_reserved_for_restart = 0)
    # test_reorg_e.restart_gaussian_for_molecule(molecule_id = 'pentacen_20240402070336',
    #                                   basis_set = 3, 
    #                                   excitation_state = 1,
    #                                   charge = 0)
    # test_reorg_e.restart_gaussian_for_molecule(molecule_id = 'pentacen_20240402070336',
    #                                   basis_set = 3, 
    #                                   excitation_state = 0,
    #                                   charge = 1)
    # test_reorg_e.restart_gaussian_for_molecule(molecule_id = 'tetracen_20240402070138',
    #                                   basis_set = 3, 
    #                                   excitation_state = 1,
    #                                   charge = 0)
    # test_reorg_e.restart_gaussian_for_molecule(molecule_id = 'anthrace_20240402070034',
    #                                   basis_set = 3, 
    #                                   excitation_state = 1,
    #                                   charge = 0)
    test_reorg_e.download_tracker_as_excel()
    test_reorg_e.download_results_as_excel()
    pass








































