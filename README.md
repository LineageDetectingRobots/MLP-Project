# MLP-Project
Group 50's 4th Year MLP group project.


## Set-up Conda

**NOTE:** Always say yes if an option is given

1. Download and install conda [here](https://docs.anaconda.com/anaconda/install/)
2. First create an environment

````bash
conda create -n mlp-proj python=3.7
````
3. Go into environment

````bash
conda activate mlp-proj
````
4. Run the following. Please note, when you add more dependencies add to the `requirements.txt` file 

````bash
pip install requirements.txt
````

5. Run the following command to install requirements for MLP cluster.

````bash
bash install.sh
````

6. Run the following.

````bash
python setup.py develop
````

NOTE: When adding new **folders** please add a `__init__.py` file.

## Using GitHub

NOTE: **Pushing to master is not allowed**

### GitHub Workflow

You want to work on a new feature, or update existing code.

1. Create a new branch

````bash
git checkout -b branch-name
````
2. Use the following command. You will see that you are on the branch `branch-name` (whatever you called it)

````bash
git status
````
3. Now use the command. This will push the branch to global repo. Currently the repo was just on you machine

````bash
git push -u origin branch-name
````
4. Start coding.....
5. When you have finished a part of the code. Do the following

````bash
git st  # Check what files have been changed
git add file-name file-name2  # Add files that you want to commit next
# Commit the files you just added.
# The commit message should be informative of what you have just done.
git commit -m "commit message"
git push # Push the files to the global repo
````
6. When all the work you wanted to do for this branch is complete. Go onto the GitHub website and create a pull request. Someone will review it or if the code is fine merge the code to the master branch.

## Using MLP Cluster

The MLP repo has a great tutorial branch on how to use the MLP cluster. [Link here](https://github.com/CSTR-Edinburgh/mlpractical/tree/mlp2019-20/mlp_cluster_tutorial).

**NOTE:** When logging into the MLP cluster for the first time you will have to recreate the conda environment.

For a getting started guide to using the MLP cluster please use [this link](https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2019-20/mlp_cluster_tutorial/mlp_cluster_quick_start_up.md)

**NOTE:** The guide seems good. No point re-inventing the wheel