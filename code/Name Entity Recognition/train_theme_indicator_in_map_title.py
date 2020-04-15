from __future__ import unicode_literals, print_function

import random
import numpy as np
import os
from spacy.util import minibatch, compounding
import spacy
from pathlib import Path
import plac
import en_core_web_sm

# new entity label
LABEL = "THEME"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

# import pickle
# path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\code\\Name Entity Recognition\\labledTitles.pkl'
# TRAIN_DATA = pickle.load( open( path, "rb" ) )
nlpTest = en_core_web_sm.load()

TRAIN_DATA = [(
"In 1961 by county Intergovernmental revenue of governments in South Korea ",
{"entities": [(18, 58, LABEL)]},
),
(
"In 1995 Annual payroll by state in the UK ",
{"entities": [(8, 22, LABEL)]},
),
(
"Average hours per day by men spent on Work-related activities by census tract ",
{"entities": [(0, 61, LABEL)]},
),
(
"Number of fire points by census tract ",
{"entities": [(0, 21, LABEL)]},
),
(
"Estimated annual sales for All oth. gen. merch. Stores by county in 1983 in South Korea ",
{"entities": [(0, 54, LABEL)]},
),
(
"Intergovernmental expenditure of governments in 1988 by township ",
{"entities": [(0, 44, LABEL)]},
),
(
"Estimated annual sales for Motor vehicle & parts Dealers in 1951 in the United States ",
{"entities": [(0, 56, LABEL)]},
),
(
"Renter occupied in Canada by census tract in 1985 ",
{"entities": [(0, 15, LABEL)]},
),
(
"By state import and export statistics ",
{"entities": [(9, 37, LABEL)]},
),
(
"In 1976 by township in the UK Estimated annual sales for Building material & garden eq. & supplies dealers ",
{"entities": [(30, 106, LABEL)]},
),
(
"License taxes of governments by county in the United States ",
{"entities": [(0, 28, LABEL)]},
),
(
"In the UK sale amounts of beer by state ",
{"entities": [(10, 30, LABEL)]},
),
(
"In the UK Number of paid employees in 1976 by census tract ",
{"entities": [(10, 34, LABEL)]},
),
(
"Population density of separated in 1987 in China ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 1983 Average household size by township in Canada ",
{"entities": [(8, 30, LABEL)]},
),
(
"Estimated annual sales for Food & beverage stores in 1960 ",
{"entities": [(0, 49, LABEL)]},
),
(
"In 1982 social vulnerability index by county ",
{"entities": [(8, 34, LABEL)]},
),
(
"By county GDP (nominal or ppp) ",
{"entities": [(10, 30, LABEL)]},
),
(
"In France Number of employees of Audio and video equipment manufacturing by county ",
{"entities": [(10, 72, LABEL)]},
),
(
"Total capital expenditures of Clay product and refractory manufacturing by census tract in the United States in 1973 ",
{"entities": [(0, 71, LABEL)]},
),
(
"In 2000 NSF funding for Catalogue in France by township ",
{"entities": [(8, 35, LABEL)]},
),
(
"By state in 1990 Household income in South Korea ",
{"entities": [(17, 33, LABEL)]},
),
(
"In 1985 Gross profit of companies by census tract ",
{"entities": [(8, 33, LABEL)]},
),
(
"In Canada Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) ",
{"entities": [(10, 92, LABEL)]},
),
(
"In 1988 Direct expenditure of governments ",
{"entities": [(8, 41, LABEL)]},
),
(
"By county in 2013 Annual payroll ",
{"entities": [(18, 32, LABEL)]},
),
(
"In 1953 in the UK Population density of White ",
{"entities": [(18, 45, LABEL)]},
),
(
"In 1952 average age ",
{"entities": [(8, 19, LABEL)]},
),
(
"In France Households with householder living alone by county ",
{"entities": [(10, 50, LABEL)]},
),
(
"In 1979 in Canada Insurance benefits and repayments of governments ",
{"entities": [(18, 66, LABEL)]},
),
(
"License Taxes of governments in Canada in 2003 by state ",
{"entities": [(0, 28, LABEL)]},
),
(
"In France in 1976 Households with male householder, no wife present, family ",
{"entities": [(18, 75, LABEL)]},
),
(
"Estimated annual sales for Clothing & clothing accessories stores by census tract in 1968 ",
{"entities": [(0, 65, LABEL)]},
),
(
"In Canada annual average temperature in 2012 ",
{"entities": [(10, 36, LABEL)]},
),
(
"Percent of farms with female principal operator in France ",
{"entities": [(0, 47, LABEL)]},
),
(
"By census tract Average percent of time engaged in Playing with household children, not sports ",
{"entities": [(16, 94, LABEL)]},
),
(
"In 2014 annual average temperature ",
{"entities": [(8, 34, LABEL)]},
),
(
"In 2004 Annual payroll of Seafood product preparation and packaging in Canada ",
{"entities": [(8, 67, LABEL)]},
),
(
"By state in South Korea number of fire points ",
{"entities": [(24, 45, LABEL)]},
),
(
"In Canada in 1965 number of schools by census tract ",
{"entities": [(18, 35, LABEL)]},
),
(
"By county in 1977 Average hours per day spent on Health-related self care ",
{"entities": [(18, 73, LABEL)]},
),
(
"In the UK Production workers annual hours of Other wood product manufacturing in 1956 by state ",
{"entities": [(10, 77, LABEL)]},
),
(
"In 2014 Exports value of firms by county in the United States ",
{"entities": [(8, 30, LABEL)]},
),
(
"Average hours per day by women spent on Relaxing and thinking by township ",
{"entities": [(0, 61, LABEL)]},
),
(
"In 1977 by township annual average temperature ",
{"entities": [(20, 46, LABEL)]},
),
(
"License Taxes of governments by township in China ",
{"entities": [(0, 28, LABEL)]},
),
(
"In 1968 by county Household income ",
{"entities": [(18, 34, LABEL)]},
),
(
"By township Median household income in France in 1991 ",
{"entities": [(12, 35, LABEL)]},
),
(
"By township in South Korea number of people of White ",
{"entities": [(27, 52, LABEL)]},
),
(
"By census tract in the United States in 1982 well-being index ",
{"entities": [(45, 61, LABEL)]},
),
(
"Average percent of time engaged in Telephone calls (to or from) in Canada in 1991 ",
{"entities": [(0, 63, LABEL)]},
),
(
"Difference in race diversity in the United States ",
{"entities": [(0, 28, LABEL)]},
),
(
"In 1984 Average percent of time engaged in by menCaring for household adults ",
{"entities": [(8, 76, LABEL)]},
),
(
"In 1985 Average hours per day by women spent on Arts and entertainment (other than sports) ",
{"entities": [(8, 90, LABEL)]},
),
(
"In France by census tract Number of paid employees ",
{"entities": [(26, 50, LABEL)]},
),
(
"In 1971 Elementary-secondary revenue from school lunch charges ",
{"entities": [(8, 62, LABEL)]},
),
(
"In China by census tract Average number of bedrooms of houses ",
{"entities": [(25, 61, LABEL)]},
),
(
"NBA player origins (per capita) by county ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 2020 Total value of shipments and receipts for services of Machinery manufacturing ",
{"entities": [(8, 85, LABEL)]},
),
(
"In 1954 Average monthly housing cost ",
{"entities": [(8, 36, LABEL)]},
),
(
"By census tract Estimated annual sales for Furniture stores ",
{"entities": [(16, 59, LABEL)]},
),
(
"In 1971 Family households (families) by county ",
{"entities": [(8, 36, LABEL)]},
),
(
"By county people living in poverty areas ",
{"entities": [(10, 40, LABEL)]},
),
(
"In 2001 Number of paid employees ",
{"entities": [(8, 32, LABEL)]},
),
(
"In France Households with male householder, no wife present, family by state ",
{"entities": [(10, 67, LABEL)]},
),
(
"In 2014 Elementary-secondary revenue from parent government contributions ",
{"entities": [(8, 73, LABEL)]},
),
(
"In the UK in 1960 by census tract Average percent of time engaged in Travel related to personal care] ",
{"entities": [(34, 101, LABEL)]},
),
(
"In France number of fire points by census tract ",
{"entities": [(10, 31, LABEL)]},
),
(
"Sale amounts of beer by census tract ",
{"entities": [(0, 20, LABEL)]},
),
(
"Average year built by county in France ",
{"entities": [(0, 18, LABEL)]},
),
(
"Number of paid employees in China by county in 2006 ",
{"entities": [(0, 24, LABEL)]},
),
(
"By township Estimated annual sales for Mens clothing stores in 1995 in China ",
{"entities": [(12, 59, LABEL)]},
),
(
"By township Average hours per day by women spent on Other income-generating activities ",
{"entities": [(12, 86, LABEL)]},
),
(
"Number of patent in the United States by township ",
{"entities": [(0, 16, LABEL)]},
),
(
"In 1978 in the UK by state Estimated annual sales for Home furnishings stores ",
{"entities": [(27, 77, LABEL)]},
),
(
"By county in 1995 Households with female householder, no husband present, family ",
{"entities": [(18, 80, LABEL)]},
),
(
"In the UK by county number of earthquake ",
{"entities": [(20, 40, LABEL)]},
),
(
"In Canada in 1971 General sales of governments by county ",
{"entities": [(18, 46, LABEL)]},
),
(
"By census tract in 1950 Number of paid employees ",
{"entities": [(24, 48, LABEL)]},
),
(
"In Canada by county Total households ",
{"entities": [(20, 36, LABEL)]},
),
(
"In 1965 Production workers annual hours of Nonmetallic mineral product manufacturing in France by township ",
{"entities": [(8, 84, LABEL)]},
),
(
"In 1993 helicobacter pylori rate ",
{"entities": [(8, 32, LABEL)]},
),
(
"In the United States in 1967 by township Average square footage of houses ",
{"entities": [(41, 73, LABEL)]},
),
(
"General revenue of governments in China ",
{"entities": [(0, 30, LABEL)]},
),
(
"Elementary-secondary revenue from property taxes in 1997 ",
{"entities": [(0, 48, LABEL)]},
),
(
"Married-couple family in China in 1954 by county ",
{"entities": [(0, 21, LABEL)]},
),
(
"In 2004 by township Average hours per day spent on Attending household children events ",
{"entities": [(20, 86, LABEL)]},
),
(
"In 1951 by state General revenue of governments ",
{"entities": [(17, 47, LABEL)]},
),
(
"In 1998 Elementary-secondary revenue from property taxes by township ",
{"entities": [(8, 56, LABEL)]},
),
(
"In China flu incidence by census tract in 1991 ",
{"entities": [(9, 22, LABEL)]},
),
(
"GDP (nominal or ppp) per capita by county in China in 1998 ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 1974 in the United States number of people of Catholic by county ",
{"entities": [(29, 57, LABEL)]},
),
(
"In China in 2003 annual average temperature ",
{"entities": [(17, 43, LABEL)]},
),
(
"In South Korea Number of firms in 1996 by county ",
{"entities": [(15, 30, LABEL)]},
),
(
"In China Family households (families) in 1956 by township ",
{"entities": [(9, 37, LABEL)]},
),
(
"In 2017 Average square footage of houses in South Korea ",
{"entities": [(8, 40, LABEL)]},
),
(
"By township in the United States Percent change of retailers of personal computer ",
{"entities": [(33, 81, LABEL)]},
),
(
"In 1966 Average household size ",
{"entities": [(8, 30, LABEL)]},
),
(
"In China by census tract Average square footage of houses ",
{"entities": [(25, 57, LABEL)]},
),
(
"By census tract Average poverty level for household ",
{"entities": [(16, 51, LABEL)]},
)]

@plac.annotations(
    model=(nlpTest, "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="en_core_web_sm", new_model_name="animal", output_dir=None, n_iter=100):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [
        pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text_list = ["By state Average percent of time engaged in Vehicles in South Korea in 2004 ",
    "In 1964 Household income ","By census tract in France in 1972 Household income ",
    "By township in 1969 Estimated annual sales for Electronics & appliance stores ",
    "By state Average monthly housing cost as percentage of income ",
    "By township rate of male in Canada ","Number of firms in the United States ",
    "In the UK in 1955 number of cell phones per 100 person ",
    "By county in the UK Average hours per day by women spent on Consumer goods purchases ",
    "Adult obesity by state in 1973 "]
    for test_text in test_text_list:
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
