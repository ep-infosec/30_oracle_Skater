.. raw:: html

    <div align="center">
    <a href="https://www.datascience.com">
    <img src ="https://cdn2.hubspot.net/hubfs/532045/Logos/DS_Skater%2BDataScience_Colored.svg" height="300" width="400"/>
    </a>
    </div>
    <br>
    <br>



********
Overview
********


What is Model Interpretation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The concept of model interpretability in the field of machine learning is still new, largely subjective, and, at times,
controversial. Model interpretation is the ability to explain and validate the decisions of a predictive model to
enable fairness, accountability, and transparency in the algorithmic decision-making
(for a more detailed explanation on the definition of transparency in machine learning,
see “Challenges of Transparency” by Adrian Weller).Or, to state it formally, model interpretation can be defined as
the ability to better understand the decision policies of a machine-learned response function to explain the
relationship between independent (input) and dependent (target) variables, preferably in a human interpretable way.
Ideally, you should be able to query the model to understand the "what, why, and how" of it's algorithmic decisions.

Skater
~~~~~~
Skater is a unified framework to enable Model Interpretation for all forms of model to help one build an Interpretable
machine learning system often needed for real world use-cases. It is an open source python library designed to
demystify the learned structures of a black box model both globally(inference on the basis of a complete data set)
and locally(inference about an individual prediction). 

Note: The library has embraced object-oriented and functional programming paradigms as deemed necessary to provide
scalability and concurrency while keeping code brevity in mind. The project is still in beta phase and is
under active development.

Algorithms supported by Skater
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model Interpretation could be enabled in multiple different ways, but at a highlevel it could be broadly categorized as,

1. post hoc interpretation: Given a black box model trained to solve a supervised learning
problem(X --> Y, where X is the input and Y is the output), post-hoc interpretation can be thought of as a
function(f) 'g' with input data(D) and a predictive model. The function 'g' returns a visual or textual
representation which helps in understanding the inner working of the model or why a certain outcome is more
favorable than the other. It could also be called inspecting the black box or reverse engineering.

2. natively interpretable models: Given a supervised learning problem, the predictive model(explanator function)
has a transparent design and is interpretable both globally and locally without any further explanations.

Skater provides the ability to interpret the model in both ways(we are actively working on
implementing other algorithms, `issues <https://github.com/datascienceinc/Skater/issues?utf8=%E2%9C%93&q=is%3Aopen+>`_)

+---------+---------+-----+-----------+-----------+--------------+--------------+
| Scope of Interpretation |            Algorithms                               |
+=========+=========+=====+===========+===========+==============+==============+
| Global Interpretation   | Model agnostic Feature Importance                   | 
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Global Interpretation   | Model agnostic Partial Dependence Plots             |
+---------+---------+-----+-----------+-----------+--------------+--------------+
| Local Interpretation    | Local Interpretable Model Explanation(LIME)         |
+---------+---------+-----+-----------+-----------------------------------------+
| Local Interpretation    | DNNs      | - Layer-wise Relevance Propagation      |
|                         |           |   (e-LRP): image                        |
|                         |           |                                         |
|                         |           | - Integrated Gradient: image and text   |
|                         |           |                                         |
+---------+---------+-----+-----------+-----------------------------------------+
| Global and Local        | - Scalable Bayesian Rule Lists                      |
| Interpretation          |                                                     |
|                         | - Tree Surrogates                                   |
+---------+---------+-----+-----------+-----------+--------------+--------------+

