# MachineLearning-Project
## 1 Introduction

Nata Visionaries is an ancient, not-for-profit brotherhood formed around Portuguese Custard Tarts,
the Pastel de Nata. It is a charming organization, with a positive attitude towards life and society,
with a significant obsession with the Pastel de Nata.

Bound by pastry and purpose, the Brotherhood is devoted to the ancient art of the Pastel de Nata.

Through careful tasting, endless testing, and the occasional sugar-induced revelation, members seek
the perfect union of crisp and cream. Eggs over egos.

We are the alchemists seeking the pure gold of a perfect Pastel de Nata. Our mission is humble
yet noble: to rule the palate, not the planet.

May every bite bring us closer to enlightenment... or at least to another Pastel de Nata.

Nata Visionaries heard about Data Visionaries and decided to hire your team!

## 2 The project

### 2.1 Background

Nata Visionaries are very methodical and organized, and have collected data from traditional bakeries
in Porto and Lisbon to uncover what makes an exceptional Pastel de Nata. The information collected
includes the recipe with quantities, information about the preparation, and information about the
baking process. For each recipe, the resulting Pastel de Nata is classified as either ”OK”, meaning

that it deserves the seal of approval of the Nata Visionaries, or ”KO”, meaning that yes, you can still
eat it, but it’s not the real thing.

This classification is obtained by a process that is extremely accurate, but that is kept even more
secret than the recipe for Past´eis de Bel´em. The only thing we know is that it involves destroying the
Pastel de Nata without actually eating it. And the brotherhood has had enough of that nonsense!

### 2.2 Objectives

Your team was hired to build a model that can predict the quality based on existing Nata existing
data. Just with the information about a certain production, your model will classify the Pastel de
Nata as either ”OK” or ”KO”.

The model will save thousands of Past´eis de Nata every year from the fate of ”destruction without
consumption”. It will also provide information that can be used by different bakeries to improve on
their recipes and production processes.

Your team is free to develop the work in whichever way you feel is preferable. It is nevertheless
suggested that you use these steps:

• Business Understanding: In crust we trust.
As most business info is secret, the only information provided by the Nata Visionaries was that
of the target variable and the metric that you should use;
• Data understanding: The path to wisdom is dusted with cinnamon.
Explore data to understand the main characteristics and limitations of the dataset;
• Data Preparation: May our crusts be crisp and our spirits flaky.
Building on the data exploration and insights, prepare data for modelling.
This may include the need for dropping existing features or for creating new ones.
• Modelling: Our philosophy is simple: bake it until you make it.
Experiment with different models that can predict the class of a Pastel de Nata; assess those
models, using one or more configurations, to identify the top-performing models;
• Evaluation: We don’t sugar-coat the truth — we caramelize it.
Evaluate the selected model on different metrics. Attempt further optimization.
• Deployment: From Portugal with crust.
Use this final model to predict the results for the test data for which you have no labels. Submit
it to Kaggle.

## 3 Nata Data provided

### 3.1 Files provided

Three data files are provided, all in CSV format:

• learn.csv has the measures performed and the ground truth (quality class);
• predict.csv has a structure similar to learn.csv, except that it does not include the ground truth.
You will need to make a prediction for each record in this file, and submit it for evaluation (more
on that later);
• sampred.csv is provided as an example of the structure of a predictions file. You could and
can, submit this file. Just don’t expect anyone to be impressed with the result.
