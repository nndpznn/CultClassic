from setuptools import setup, find_packages

setup(
    name="CultClassic",              # Name of your package
    version="0.1.0",                 # Version number
    packages=find_packages(),        # Automatically find subpackages
    install_requires=[               # Optional: Dependencies
        "torch",                     # Example: PyTorch
        "numpy",
		"sklearn",

    ],
    python_requires=">=3.7",         # Python version requirement
    author="Nolan Nguyen & Allen Ho",              # Optional: Author name
    description="Neural-network based collaborative filtering recommender system for movies.",
    long_description=open("README.md").read(),  # Read description from README
    long_description_content_type="text/markdown",
)