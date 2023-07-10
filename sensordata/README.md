# MLOps Architecture

## Assumptions

1. Historical Data is already gathered and stored in data warehouse/lake based storage.
2. Infrastructure is available to process Real time data and made available to be processed by model.
4. High volume of data is expected.
5. Model training/evaluation should be either manually triggered or triggered based on schedule/event.
5. Model Prediction triggers are request/scheduled/event based.
6. MLOps infrastructure needs to be setup from scratch.



## Environments

**LOCAL → STAGING → PRODUCTION**



## Architecture

![Architecture for MLOps](https://github.com/rajs1006/DataScience/blob/main/sensordata/docs/ARCHITECTURE_ZEISS.png)



## Blogs
	I have written some blogs on similar topic and some are in progress.
	
	- [Overview: Building an Efficient and Scalable MLOps Workflow with Feast and ZenML](https://sourabhraj.net/2023/06/07/overview-building-an-efficient-and-scalable-mlops-workflow-with-feast-and-zenml/)
	- [MLOps: Empowering the Machine Learning Lifecycle](https://sourabhraj.net/2023/05/23/mlops-empowering-the-machine-learning-lifecycle/)



## Tools

In this architecture I have tried to use mostly Open Source tools except some of them but, using a cloud provider could also be used.

#### Using Cloud Provider vs Ensemble tools

##### Comparison		
		|    Cloud Provider                    |      Ensembel tools
		:-------------------------------------: :-------------------------------------------:
		| Less maintanance		       |  More maintance required      
		| Useful with less people in the team  |  Useful with a dedicated team
		| Could be more expensive              |  Can be planned wisely to be less expensive
		| Less suitable for specific use cases |  Can be architected specific to the use case

##### AZURE
			- Data Storage : Azure Blobs
			- Database : Microft SQL/T-SQL
			- Servers : AKS(Azure Kubernetes Service)
			- Data pipelines : Microsoft Fabrics
			- Model Developemnt and Deployment :  Azure Machine Learning 
##### AWS
			- Data Storage : S3
			- Database :  Athena or Redshift
			- Servers : ECS or EKS 
			- Data pipelines : AWS Glue
			- Model Developemnt and Deployment :  Azure Sagemaker 

##### Ensemble
			- Data Storage : Hadoop/S3 or even MongoDB(depending on usecase)
			- Database :  Postgres
			- Cache DB : Redis
			- Servers : Kubernetes cluster/EKS/AKS 
			- Data pipelines : Airflow/Dagster/Prefect
			- Model Developemnt and Deployment :  Jupyter | MLFlow, DVC | Airflow/Dagster/ZenML    
			

## Details:

**Feature Store**

A Feature Store is essential for ingesting large volumes of data, computing features, and storing them. _Feast_ is a feature store tool that integrates seamlessly with _Kubernetes_ ensuring reproducibility and eliminating discrepancies between training and inference features.

**Orchestrating Pipeline**

An Orchestrator plays a vital role in planning and executing workflows, encompassing data filtering, transformation, model training, prediction, and result processing. It also manages inter dependency between models.
By leveraging tools like _Airflow_, _ZenML_, _Dagster_ one can enhances flexibility and robustness, enabling efficient management of the machine learning process.

**Model Deployment and Serving**

Once model development is complete, the deployment phase becomes pivotal. _Containerizing_ the models using _Docker_ facilitating smooth deployment and serving of models on _Kubernetes_. 

**Model Monitoring**

Monitoring the performance of deployed models is vital for analysis and explanation.  _Evidently API_ for model monitoring, allowing organizations to track and analyze results effectively. There are Paid tools as well like _Arize_, _Aporia_, _FiddlerAI_
	

**CI/CD**

Continuous Integration and Delivery is very important as it helps minimize a developer's time to release the model. A tagging based process can be introduced to enable the A/B testing.

_Example_: 

	Model 1 can be release as github tag A, which further will create a container image, a display name with A and if needed resources with named A
   	Model 2 can be release as github tag B, which further will create a container image, a display name with B and if needed resources with named B
   
   	When the model needed to be swicthed jut redirect the traffiic or switch off the first model entirely. 



## Explanation:

**Local/Dev**

In this environment, data scientists are empowered to conduct numerous experiments and execute multiple pipelines.

For data analysis and deriving valuable insights, _Tableau_ can be employed. Data scientists can also create code and pipelines to perform feature selection and engineering. These can subsequently be deployed as transformation steps for the feature store in other environments. 

Effective versioning of data and pipelines can be achieved with tools like _DVC_, ensuring reproducibility. Additionally, to ensure efficient tracking and management, experiments and models can be securely saved in _MLflow_.

**Staging**

The staging environment allows data scientists to evaluate finalized transformations, hyper-parameters, and models in a real-world setting. This step is essential to identify any performance discrepancies between historical and latest data before moving to production. 

_Airflow_, _ZenML_, _Dagster_, are great tools to orchestrate the pipelines and then host the Models on  _Kubernetes_, enabling testing and validation in a controlled environment.

**Production**

After finalizing the model in the staging environment, _CI/CD_ processes can be triggered to move the models and pipelines to production. In this environment, training and inference can be manually triggered, event-driven, REST-based, or scheduled. 
