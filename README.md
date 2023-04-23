# bda-analytics-challenge-template  

Please use this template for submitting your case challenge. 
Please only submit code and no datasets or models because the storage capacities are limited.

## Group Members: 
- Forename: Louis
- Surname: Skowronek
- Matriculation Number: 2222726

- Forename: Another
- Surname: Dude
- Matriculation Number: XXXXXX

## Reproducibility
Please explain each step to reproduce your results and also include key information about your Python environment. 
It is your responsibility to test the reproducibility. Please keep in mind that we can only test your code on a 
Windows or Linux machine. 

You can use the following template to document the necessary steps for executing your program:

**Operating System**: Windows

**Python Version**: 3.8.4

**Environment Setup**: 
````
conda create –n bda python=3.8.4
conda activate bda
pip install –r requirements.txt
pip install .
````

**Main Entry Point**
````
python main.py --dataset dummy.csv
````

**Unittest & docstring coverage**:
````
pytest --cov-report term --cov=src tests/
docstr-coverage src -i -f
````  


## Project Organization
------------
```
    ├── README.md 							<-- this file. insert group members here
    ├── .gitignore 						    <-- prevents you from submitting several clutter files
    ├── data
    │   ├── modeling
    │   │   ├── dev 						<-- your development set goes here
    │   │   ├── test 						<-- your test set goes here
    │   │   └── train 						<-- your train set goes here goes here
    │   ├── preprocessed 					<-- your preprocessed data goes here
    │   └── raw								<-- the provided raw data for modeling goes here
    ├── docs								<-- provided explanation of raw input data goes here
    │
    ├── models								<-- dump models here
    ├── presentation                        <-- please submit your presentation in this folder
    ├── notebooks							<-- your playground for juptyer notebooks
    ├── requirements.txt 					<-- required packages to run your submission (use a virtualenv!)
    ├── src
           ├── additional_features.py 			<-- your creation of additional features/data goes here
           ├── predict.py 						<-- your prediction script goes here
           ├── preprocessing.py 				<-- your preprocessing script goes here
           ├── train.py 						<-- your training script goes here
           └── demo.py                       <-- demo script
    └── tests
           └── test_demo.py                  <-- demo script for unittest                
	
```
## Code evaluation

To evaluate your code, we will run the following commands:

````
pytest --cov-report term --cov=src tests/
docstr-coverage src -i -f
````