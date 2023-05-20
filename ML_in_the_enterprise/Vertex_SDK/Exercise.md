
# Vertex SDK: Custom Training Tabular Regression Models for Online Prediction and Explainability

Overview

This lab demonstrates how to use the Vertex SDK to train and deploy a custom tabular regression model for online prediction with explanation.

Learning objectives

In this lab, you learn how to create a custom model from a Python script in a Google prebuilt Docker container using the Vertex SDK, and then do a prediction with explanations on the deployed model by sending data. You can alternatively use the Google Cloud CLI or the Google Cloud Console to create custom models using.

You perform the following tasks:

Create a Vertex custom job for training a model.

Train a TensorFlow model.

Retrieve and load the model artifacts.

View the model evaluation.

Set explanation parameters.

Upload the model as a Vertex Model resource.

Deploy the Model resource to a serving Endpoint resource.

Make a prediction with explanation.

Undeploy the Model resource.

Setup and requirements
For each lab, you get a new Google Cloud project and set of resources for a fixed time at no cost.

Sign in to Qwiklabs using an incognito window.

Note the lab's access time (for example, 1:15:00), and make sure you can finish within that time.
There is no pause feature. You can restart if needed, but you have to start at the beginning.

When ready, click Start lab.

Note your lab credentials (Username and Password). You will use them to sign in to the Google Cloud Console.

Click Open Google Console.

Click Use another account and copy/paste credentials for this lab into the prompts.
If you use other credentials, you'll receive errors or incur charges.

Accept the terms and skip the recovery resource page.

Note: Do not click End Lab unless you have finished the lab or want to restart it. This clears your work and removes the project.

Enable the Vertex AI API
In the Google Cloud Console, on the Navigation menu, click Vertex AI > Dashboard, and then click Enable Vertex AI API.

Task 1. Launch a Vertex AI Notebooks instance
In the Google Cloud Console, on the Navigation Menu, click Vertex AI > Workbench. Select User-Managed Notebooks.

On the Notebook instances page, click New Notebook > TensorFlow Enterprise > TensorFlow Enterprise 2.6 (with LTS) > Without GPUs.

In the New notebook instance dialog, confirm the name of the deep learning VM, if you don’t want to change the region and zone, leave all settings as they are and then click Create. The new VM will take 2-3 minutes to start.

Click Open JupyterLab.
A JupyterLab window will open in a new tab.

You will see “Build recommended” pop up, click Build. If you see the build failed, ignore it.

Task 2. Clone a course repo within your Vertex AI Notebooks instance
To clone the training-data-analyst notebook in your JupyterLab instance:

In JupyterLab, to open a new terminal, click the Terminal icon.

At the command-line prompt, run the following command:

git clone https://github.com/GoogleCloudPlatform/training-data-analyst
Copied!
To confirm that you have cloned the repository, double-click on the training-data-analyst directory and ensure that you can see its contents.
The files for all the Jupyter notebook-based labs throughout this course are available in this directory.

Task 3. Train and deploy a custom tabular regression model for online prediction with explanation
In the notebook interface, navigate to training-data-analyst > courses > machine_learning > deepdive2 > machine_learning_in_the_enterprise > labs, and open sdk_custom_tabular_regression_online_explain.ipynb.

In the notebook interface, click Edit > Clear All Outputs.

Carefully read through the notebook instructions and fill in lines marked with #TODO where you need to complete the code.

Tip: To run the current cell, click the cell and press SHIFT+ENTER.

To view the complete solution, navigate to training-data-analyst > courses > machine_learning > deepdive2 > machine_learning_in_the_enterprise > solutions, and open sdk_custom_tabular_regression_online_explain.ipynb.

End your lab
When you have completed your lab, click End Lab. Qwiklabs removes the resources you’ve used and cleans the account for you.

You will be given an opportunity to rate the lab experience. Select the applicable number of stars, type a comment, and then click Submit.

The number of stars indicates the following:

1 star = Very dissatisfied
2 stars = Dissatisfied
3 stars = Neutral
4 stars = Satisfied
5 stars = Very satisfied
You can close the dialog box if you don't want to provide feedback.

For feedback, suggestions, or corrections, please use the Support tab.

Copyright 2022 Google LLC All rights reserved. Google and the Google logo are trademarks of Google LLC. All other company and product names may be trademarks of the respective companies with which they are associated.
