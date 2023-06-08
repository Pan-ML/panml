import sys
import importlib
from panml.constants import IMPORT_PACKAGE_VER

class ImportEnvironment:
    '''
    Perform validation and import of relevant packages required
    '''
    def __init__(self, packages: list) -> None:
        self.packages = packages

    # Perform validation and import of required packages for the library
    def validate(self) -> None:
        '''
        Args:
        package: list of Python package names relevant for import

        Returns:
        None. Only checks whether package can be imported or otherwise throw error with message for resolution.
        '''
        packages_not_found = []
        for package in self.packages:
            package_var = self.get_package_var_name(package)
            if package_var not in sys.modules:
                try:
                    if package_var not in vars():
                        _ = importlib.import_module(package_var) # assess if package can be imported
                except ImportError:
                    packages_not_found.append(package)
        
        if len(packages_not_found) > 0:
            package_msg = '\n'.join([f"{p}: `pip install {p}=={IMPORT_PACKAGE_VER[p]}`" for p in packages_not_found])
            raise ImportError(f"Could not import required packages. \n\nThe following Python package(s) are required:\n{package_msg}")

    # Get the import package variable name
    def get_package_var_name(self, package: str) -> str:
        '''
        Args:
        package: name of the package

        Return:
        Get the import variable name
        '''
        if package == 'faiss-cpu':
            return 'faiss'
        if package == 'sentence-transformers':
            return 'sentence_transformers'
        return package