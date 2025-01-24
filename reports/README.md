# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X ] Create a git repository (M5)
* [X ] Make sure that all team members have write access to the GitHub repository (M5)
* [X ] Create a dedicated environment for you project to keep track of your packages (M2)
* [X ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X ] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [X ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X ] Setup version control for your data or part of your data (M8)
* [X ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X ] Construct one or multiple docker files for your code (M10)
* [X ] Build the docker files locally and make sure they work as intended (M10)
* [X ] Write one or multiple configurations files for your experiments (M11)
* [X ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [X ] Use logging to log important events in your code (M14)
* [X ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [X ] Consider running a hyperparameter optimization sweep (M14)
* [X ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X ] Write unit tests related to the data part of your code (M16)
* [X ] Write unit tests related to model construction and or model training (M16)
* [X ] Calculate the code coverage (M16)
* [X ] Get some continuous integration running on the GitHub repository (M17)
* [X ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X ] Add a linting step to your continuous integration (M17)
* [X ] Add pre-commit hooks to your version control setup (M18)
* [X ] Add a continues workflow that triggers when data changes (M19)
* [X ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X ] Create a trigger workflow for automatically building your docker images (M21)
* [X ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X ] Create a FastAPI application that can do inference using your model (M22)
* [X ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X ] Write API tests for your application and setup continues integration for these (M24)
* [X ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X ] Create a frontend for your API (M26)

### Week 3

* [X ] Check how robust your model is towards data drifting (M27)
* [X ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [X ] Setup cloud monitoring of your instrumented application (M28)
* [X ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [X ] Write some documentation for your application (M32)
* [X ] Publish the documentation to GitHub Pages (M32)
* [X ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [X ] Make sure all group members have an understanding about all parts of the project
* [X ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 45 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *s243296, s242965, s242962*
>
> Answer:

--- s249246, attila, matyas  ---

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used Albumentations in our project to perform advanced image augmentations and effectively extend the variability of the dataset. Also used Optuna for hyperparameter optimization to fine-tune model architecture, learning rates, and training configurations. Lastly we used TIMM in our project to access a variety of pretrained models and well-known architectures, from which we finally selected Resnet as our base architecture ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- All the projects necessary dependencies for running the pipeline can be found in the requirements.txt, except the Pytorch which is device dependant.
This requirements.txt was autogenerated and updated periodically using pipreqs, by just running

pipreqs . --force

The usage of this package allows us to maintain a cleaner requirements.txt, with just the imported packages in our code files. Afterwards, pytorch is removed to avoid incompatibilities, as it strictly depends on the current hardware setup of the user.

Separatedly, a requirements_dev.txt has been periodically updated manually, with the required packages for testing or debugging that not always appear on the imports of the code files, hence they will be missing from the requirements.txt.

As explained in the main README.md file of our project, the reccommended way of installing is:

1. Create a conda environment, if desired.
2. Install pytorch according to the current hardware. For example: `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118`
3. Install everyting (including dev dependencies) by just intalling the package with `pip install -e .`
4. Otherwise, just execute `pip install -r requirements.txt`

 ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

---

We preserved most of the cookiecutter template, with some slight modifications. We deleted the notebooks folder, as we didn't use any notebook, and added two additional folders:

- /scripts: includes two bash scripts that may serve useful to run all the complete pipeline (including the preprocessing from the very raw data) and another one that is used in the build dockerfile as entrypoint.
- /raw: this folder is not tracked, so it doesn't apear in the repository (same happens with data/ and models/ which are tracked via dvc) but if downloaded it contains the raw dataset data. The data/ folder, on the other side, contains the preprocesed data well-prepared and separated in /train, /validation and /test blocks.

Additionally, we include in our repository two dvc files: one for tracking the preprocessed data (data.dvc) and the models (models.dvc). These files are complemented with the .dvc folder, from which we just sync in github the config file with the information of our remote data.

The rest of the structure follows the provided template:

- /.github for workflow files
- /configs for storing different model setups, and parameter sweep configurations
- /dockerfiles for storing different dockerfiles (build, train, data, api, frontend, drift and an additional run_all)
- /reports includes this report
- /src/dog_breed_classifier includes the source code for each step: data, model, train, api, frontend and data_drift. We have not used either evaluate.py nor visualize.py, so we have deleted it.
- /tests for unit tests

We also used the pyproject.toml approach for managing dependencies, both requirement files, .gitignore, the pre-commit configuration yaml and a proper README.md file.

 ---

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

--- question 6 fill here

We used several approaches on this, automating different checks throughout the pipeline so that we would be "forced" to take care of this. Decided this approach as it is convenient and strict.

- Installed flake8 and added a first Github Actions task to be performed on every push that checks it.
- Integrated ruff as well in our pre-commit file, as it detects formatting style misalignments and actually corrects them before committing.

It is super important because it keeps the code readable and makes easier to identify issues. If each one followed their own style it would be very difficult to collaborate and understand each part.

---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- question 7 fill here

used pytest. 14 tests, three for data, five for model, three for train, two for the api, an additional for  model speed

We have implemented a total of total tests on the most crucial parts of our model:

- Three tests for the data processing code, checking that the images are preprocessed step by step as specified, that the albumentations transformations are valid and the data is correctly split.
- Five tests for just the model architecture, checking its initialization, forward pass, reaction to invalid inputs, validity of parameters and correct backbone loading. We focused most here as we consider it is the most difficult part to detect failures, as they may not "appear" while running the code but rather lead directly to poor results coming from mistakes in this architecture.
- Three tests for the train script, to verify its initialization, the dataset loading operation and the complete train loop. We used the "local" variant (without wandb) to ease up the test and go straight to the key parts.
- Two tests for the API, checking its robustness against different inputs.

Additionally we implemented a performance test of the model, to ensure that inference takes a reasonable time. We established 5 seconds as a limit, and it is correctly working.

---

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- question 8 fill here

Name                                     Stmts   Miss  Cover   Missing
----------------------------------------------------------------------
src/dog_breed_classifier/__init__.py         0      0   100%
src/dog_breed_classifier/api.py            107     35    67%   44, 56, 73-85, 89-92, 101-105, 120-133, 137, 147, 174-185
src/dog_breed_classifier/data.py           100     27    73%   17-38, 69-71, 92-93, 125, 167-171, 174
src/dog_breed_classifier/data_drift.py      45     45     0%   1-85
src/dog_breed_classifier/frontend.py        19     19     0%   1-31
src/dog_breed_classifier/model.py           20      6    70%   47-56
src/dog_breed_classifier/train.py          139     34    76%   23-25, 30, 47, 54, 107, 152-162, 167-169, 174-176, 181-197, 202-203, 206
----------------------------------------------------------------------
TOTAL                                      430    166    61%


The final coverage we have achieved is 61%, being quite far from the 100% mainly because we have two files where we have not implemented it: the frontend and the data_drift code, which is not crucial for the main pipeline. Without counting them, we have achieved pretty good coverages in the files we have treated, achieving a 67% in the API as the lower, and a 76% coverage in the train code.

Even though, 100% coverage does not ensure perfect performance, as this type of unit testing only allows to handle specific logic predictable cases: output validity, case handling, correctness of different inputs/outputs. There are a lot of types of failures and malfunctions that lay deeper inside the code and cannot be detected with this kind of testing.

---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- question 9 fill here

Yes, we used both. To be honest, we started with just the main branch, but as code started to get bigger we inmediately switched to another approach, where we would work on separate branches for development and do a pull request to the main branch just when we had a functional version of what we were doing at the moment. Additionally, it was be on this pull request where we performed the toughest, more exhaustive tests in github Actions (such as OS compatibility), to ensure that the merged version is perfectly usable at the moment.

---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- question 10 fill here

Yes, we used DVC to keep track of our pre-processed data and our model, even though later on we added wandb for model logging. We started using google Drive and then switched to a Cloud bucket, as it eased up working within the GCloud environment.

We had two separate dvc files: one for the data/processed folder, and another one for the models folder. In github we just keep track of these two folders and of the config file included in the .dvc folder, containing the most important configuration to be able to connect to the remote storage and download the desired files. This way, the user just has to `dvc pull` the desired data right after cloning the repo, without further need to setup manually other dvc configurations.

In practice, it helped us as we avoided wasting time and local storage space by downloading and running the data.py script every time we installed the repo, as we could just pull the processed data straightforwardly. It was also very useful for model tracking, as we would only push the good versions we tested, and this way with each pull we ensured we had the latest, best version.

---

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

--- question 11 fill here

We have organized our continuous integration setup around Github Actions, in separate files aimed at different purposes.

On the first place, we have two minor workflows aimed at checking basic functionality, that are conditionally run on every push action to the development branch.

The first one is the tests_data.yaml workflow, which checks that the DVC system is correctly setup in our repository and each new user can pull the data easily without failures. Specifically, this one is just triggered when there is a change in the data files (in the .dvc config or .dvc files), to avoid unnecessary checks.

The other one is the model_speed.yaml workflow, which rather than just performing a "speed check" serves as a similar check as the tests_data workflow: it verifies that the latest model can be synced from WANDB and run correctly within the speed limits. We added this check after adding wandb support to our code, as we wanted to verify that the wandb setup works. For this reason, we had to provide the wandb API key and model name as environment variables in the code loaded from Github Secrets, to avoid integrating this confidential information inside the code, as can be seen here:

```
  test_model:
    name: Test Model Performance
    runs-on: ubuntu-latest
    env:
      WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      WANDB_MODEL_NAME: ${{ secrets.WANDB_MODEL_NAME }}
```

Then, we have our main, toughest workflow, tests.yaml, which is only executed on pull requests for the main branch. It first checks some linting with flake8, which is quite strict (at the end we decided to skip some code errors that we considered negligible), and if it succeeds, it verifies the whole pipeline in the main three operating systems: Mac OS, Windows and Linux. It basically performs these actions:

- Checkout the branch
- Install the environment
- Pull the preprocessed data and the models from the dvc
- Runs all the unit tests

For ensuring that our workflow file would run on all operating systems, we implemented two key solutions:
- We first forced all the actions to use the Linux Bash as command line, so that we could write all the commands we need to execute in the same format (Bash)
- We introduced a conditional to perform the pytorch installation properly depending on the system.

This can be seen in the first part of the installation & setup task:

      - name: Install dependencies
        run: |
          if [[ "$RUNNER_OS" == "Linux" ]]; then
            echo "Installing PyTorch with CUDA for Linux"
            pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
          else
            echo "Installing CPU-only PyTorch for Windows/macOS"
            pip install torch torchvision
          fi
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements_dev.txt
          pip install tomli
          pip install coverage
          pip install pytest
          pip install -e .
          pip list
          pip install dvc
          pip install dv[gs]
        shell: bash

Additionally, we setup our google Cloud credentials via github Secrets, as well as the wandb ones, as we check them as well during the model performance test (as explained before).

---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- question 12 fill here
We used a very simple approach, where we take advantage of both config files and wandb logging.

First of all, our training script is able to run in three different modes:
- local, with no wandb connection, used to do quick local tests.
- wandb-run, single training run (one set of parameters), logging both the parameters setup and the model to wandb.
- sweep, taking a sweep.yaml file that configures up a set of parameters to try. Each run is also logged into wandb.


This way, when someone wanted to run a "public" experiment, he just logged into wandb setting up his own WANDB_API_KEY and our WANDB_MODEL_NAME, established the parameters in the corresponding YAML files (either single-run, config.yaml, or sweep, sweep.yaml) and executed the run with the desired argument. For example, to run a sweep: `python src/dog_breed_classifier/train.py sweep`.

With this approach, as the model is logged via wandb, and after the experiment we could access to whatever specific run/version of the model we wanted, checking both its performance in terms of loss/accuracy, iterations run and other hyperparameters of the model.
 ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- question 13 fill here

As explained in the previous question, we took complete advantage of the wandb model registry to keep track of all our model iterations, versions and experiments. As literally all the experiments we wanted were logged there, in case we wanted to reproduce a specific version we would just:

- Go to our wandb project page
- By viewing the graphs and the logs, select which model we wanted to reproduce. For example, the one that has proven to get the highest accuracy with the lowest number of iterations.
- Check the model version and parameters.
- We could just download that version of the model, but if we want to reproduce that experiment, it is as easy as filling our configs/config.yaml file and execute a train run on our code, either locally (`local`) or with wandb logging as well (`wandb-run`).

---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here

As explained before, we have completely taken advantage of the wandb logging features. As can be seen in the figure, we have run different experiments varying basic hyperparameters during training: batch size, number of epochs, learning rate and model (we tested both resnet50 and resnet18).

[figure1](figures/wandb-experiments.png)


We tracked both training and validation loss and accuracy, as well as runtime for contrasing efficiency. An example of these metrics from the last experiments can be seen in the following figure:

[figure2](figures/wandb_graphs.png)

We contrasted consistently the difference in both loss and accuracy between both sets, discarding those iterations with a higher training performance
but a significant difference with respect to the validation dataset, trying to minimize the overfitting present in our models. For this reason we also started testing with a pretty low number of iterations, as we verified that for the best setups after the fifth iteration or so the model just overfits, with no further improvement in validation metrics.

 ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 31 fill here ---
