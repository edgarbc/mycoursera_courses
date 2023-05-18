
titanic-package: This is your working directory. Inside this folder you will have your package and code related to the Titanic survivor classifier.

setup.py: The setup file specifies how to build your distribution package. It includes information such as the package name, version, and any other packages that you might need for your training job and which are not included by default in GCP's pre-built training containers.

trainer: The folder that contains the training code. This is also a Python package. What makes it a package is the empty __init__.py file that is inside the folder.

__init__.py: Empty file called __init__.py. It signifies that the folder that it belongs to is a package.

task.py: The task.py is a package module. Here is the entry point of your code and it also accepts CLI parameters for model training. You can include your training code in this module as well or you can create additional modules inside your package. This is entirely up to you and how you want to structure your code. Now that you have an understanding of the structure, we can clarify that the names used for the package and module do not have to be "trainer" and "task.py". We are using this naming convention in this lab so that it aligns with our online documentation, but you can in fact pick the names that suit you.
