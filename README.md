# MathSymbolParser

## Configurations

- ### Split

    `python symbolsegmentor.py split [path to data] [output file name]`

    `ex. python symbolsegmentor.py split ..\inkml\ splitName`

	....
*creates two files with splitName_train_data.txt and splitName_test_data.txt.
Each file contains inkml expression file names.*
	....

- ### Evaluate

    `python symbolparser.py evaluate [parser_name] [files] [path to data] [path to output]`

    `ex. python symbolparser.py evaluate baseline/kmean splitName_train_data.txt ..\inkml\ ..\output\`

	....
*Evaluates the expression from the file and using the parser mentioned, creates .lg file with output in output path.*
	....

# LPGA Parser

## New Config options

- ### RANDOM_FOREST_BALANCE 

    `RANDOM_FOREST_BALANCE = 0/1`

	....
*When set to 1 will use class weights to blance the training of the Random forest*
	....
 
 - ### FEATURES_USE_SYMBOLS and FEATURES_SYMBOLS_DICT
 
    `FEATURES_USE_SYMBOLS = 0/1`
    `FEATURES_SYMBOLS_DICT = ../output/trainedclass/classes.txt`
 
 	....
*When set to 1 will use symbol labels as a feature and requires that a path to a pickled dictonary to convert from class to int*
	....

## Run

   `symbol_parse_script.py configs/Inkml_bonus/full_system_inkml.conf 0/1`
   
   ....
*When using 0 a new parser is trained when using 1 the existing parser is loaded in*
    ....