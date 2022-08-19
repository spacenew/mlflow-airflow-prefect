# mlflow-prefect

# 1. How to Run

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

1.1. **Installation**

Clone the repo

```bash
git clone https://github.com/spacenew/mlflow-prefect.git
```
cd into the project root folder

```bash
cd mlflow-prefect
```
*Needed Packages*

python=3.9.9  
mlflow  
prefect=2.0.3  
pandas  
scikit-learn  
xgboost  
optuna  

1.2 Installing Prefect  

```bash
pip install prefect
```

1.3 Installing Mlflow

```bash
pip install mlflow
```

# 2. Run MLFlow locally 

MLflow setup:  
  
- tracking server: yes, local server  
- backend store: sqlite database  
- artifacts store: local filesystem  

```bash
mlflow server --backend-store-uri=sqlite:///mlflow-locally.db --default-artifact-root ./artifacts
```
Open mlflow at http://127.0.0.1:5000

# 3. Run Prefect locally

Execute a prefect flow one time:

```bash
python flow.py
```

3.1 Run prefect ui
```bash
prefect orion start
```

Open prefect at http://127.0.0.1:4200  

3.2 Create workqueue
```bash
prefect work-queue create -t "ny-taxi-trip-queue" ny-taxi-dev
```

3.3 Build deployment
```bash
prefect deployment build ./flow.py:run --name "ny-taxi-trip-pred" -t ny-taxi-dev -o ny-taxi-prediction.yaml
```

3.4 Creating schedules through the deployment YAML file

Open ny-taxi-prediction.yaml in root folder and add "interval: 600" into block "schedule:"

interval in seconds.

OR change schedule in prefect UI

Prefect supports several types of schedules that cover a wide range of use cases and offer a large degree of customization:    
  
- Cron is most appropriate for users who are already familiar with cron from previous use.    
- Interval is best suited for deployments that need to run at some consistent cadence that isn't related to absolute time.    
- RRule is best suited for deployments that rely on calendar logic for simple recurring schedules, irregular intervals, exclusions, or day-of-month adjustments.    


3.5 Apply deployment
```bash
prefect deployment apply ny-taxi-prediction.yaml
```

3.6 Prefect Agent Start 

3.6.1 Command Line

Find ID work-queue
```bash
prefect work-queue ls
```

3.6.1 Prefect UI
Go to work queues, click ny-taxi-trip-queue

Agent Start
```bash
prefect agent start 'ID'
```
Example: prefect agent start 8c5c65fb-beaa-4d5d-8c9c-19d771a888ce
