from setuptools import find_packages, setup

setup(
    name='CCDP',
    version='0.0.1',
    author='harshal',
    author_email='harshal3558@gmail.com',
    packages=find_packages(),
    install_requires=[
        'Flask',
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'shap',
        'dill',
        'python-dotenv',
        'gunicorn',
    ],
)
